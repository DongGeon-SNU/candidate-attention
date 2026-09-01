"""Counterfactual branch construction with strict sequence invariants.

This module deliberately does not implement a cache.  ``exact_forward`` always
invokes the frozen model with ``use_cache=False`` and never accepts a
``past_key_values`` argument, so cached decoding states cannot leak into a
counterfactual measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class Candidate:
    """A concrete candidate token inserted at one currently-masked position."""

    position: int
    token_id: int


def _candidate_tuple(candidate: Candidate | tuple[int, int]) -> Candidate:
    if isinstance(candidate, Candidate):
        return candidate
    return Candidate(*candidate)


def validate_candidates(
    candidates: Iterable[Candidate | tuple[int, int]],
    sequence_length: int,
) -> tuple[Candidate, ...]:
    """Validate bounds and enforce the one-candidate-per-position invariant."""

    normalized = tuple(_candidate_tuple(candidate) for candidate in candidates)
    positions = [candidate.position for candidate in normalized]
    if len(positions) != len(set(positions)):
        raise ValueError("A branch may contain at most one candidate per position.")
    if any(candidate.position < 0 or candidate.position >= sequence_length for candidate in normalized):
        raise IndexError("Candidate position is outside the sequence.")
    return normalized


def _clone_ids(input_ids: Any) -> Any:
    if hasattr(input_ids, "clone"):
        return input_ids.clone()
    return [list(row) for row in input_ids]


def _shape_2d(input_ids: Any) -> tuple[int, int]:
    if hasattr(input_ids, "shape"):
        shape = tuple(int(value) for value in input_ids.shape)
        if len(shape) != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        return shape
    if not input_ids or not isinstance(input_ids[0], Sequence):
        raise ValueError("input_ids must be a nonempty 2-D batch.")
    return len(input_ids), len(input_ids[0])


def _token_at(input_ids: Any, position: int) -> int:
    return int(input_ids[0, position] if hasattr(input_ids, "shape") else input_ids[0][position])


def _set_token(input_ids: Any, position: int, token_id: int) -> None:
    if hasattr(input_ids, "shape"):
        input_ids[0, position] = token_id
    else:
        input_ids[0][position] = token_id


def make_branch(
    base_input_ids: Any,
    candidates: Iterable[Candidate | tuple[int, int]],
    *,
    mask_token_id: int,
) -> Any:
    """Return an independent batch-1 branch with only requested masks changed.

    The function rejects an attempt to overwrite a non-mask token.  It is
    intentionally batch-1: callers that want branch parallelism should stack
    independent return values, never concatenate branches along sequence length.
    """

    batch_size, sequence_length = _shape_2d(base_input_ids)
    if batch_size != 1:
        raise ValueError("A decoding state must be represented as one sequence (batch size 1).")
    branch_candidates = validate_candidates(candidates, sequence_length)
    for candidate in branch_candidates:
        if _token_at(base_input_ids, candidate.position) != mask_token_id:
            raise ValueError("Candidates may only replace positions that are masks in the base state.")
    branch = _clone_ids(base_input_ids)
    for candidate in branch_candidates:
        _set_token(branch, candidate.position, candidate.token_id)
    return branch


def make_leave_one_out_branches(
    base_input_ids: Any,
    candidate_set: Iterable[Candidate | tuple[int, int]],
    *,
    mask_token_id: int,
) -> Mapping[Candidate, Any]:
    """Build one exact branch per omitted candidate from a valid candidate set."""

    candidate_tuple = validate_candidates(candidate_set, _shape_2d(base_input_ids)[1])
    if len(candidate_tuple) < 2:
        raise ValueError("Leave-one-out requires a set with at least two candidates.")
    return {
        omitted: make_branch(
            base_input_ids,
            (candidate for candidate in candidate_tuple if candidate != omitted),
            mask_token_id=mask_token_id,
        )
        for omitted in candidate_tuple
    }


def exact_forward(
    model: Any,
    branch_input_ids: Any,
    *,
    attention_mask: Any,
    position_ids: Any,
) -> Any:
    """Run one no-cache frozen forward pass and return logits.

    ``attention_mask`` and ``position_ids`` are passed through unchanged from
    the base state.  The model is put in eval mode on every call; no gradients
    or KV cache are retained.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on GPU runtime
        raise RuntimeError("exact_forward requires the project's PyTorch environment.") from error

    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=branch_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    if not hasattr(output, "logits"):
        raise TypeError("Model output must expose .logits.")
    return output.logits
