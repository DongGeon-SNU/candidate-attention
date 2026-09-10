"""Pure exact-top-1 VCCC headroom-oracle calculations.

The GPU runner supplies one full-vocabulary margin query for each pool subset.
This module turns those cached queries into the exact all-order certificate
and the two requested oracle policies without importing model code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Mapping, Sequence

from src.vccc_oracle import query_passes


@dataclass(frozen=True)
class SetCertificate:
    """Exact all-order certificate result for one subset of a fixed pool."""

    selected_mask: int
    gamma: float
    passes: bool
    certificate_margin: float | None
    query_count: int
    violating_indices: tuple[int, ...]
    first_failure_target_index: int | None
    first_failure_revealed_mask: int | None
    first_failure_margin: float | None
    first_failure_top1_matches: bool | None
    first_failure_is_tie: bool | None
    minimum_margin_target_index: int | None
    minimum_margin_revealed_mask: int | None
    missing_query_count: int
    vacuous: bool


def mask_indices(mask: int, size: int) -> tuple[int, ...]:
    """Return selected pool indices in deterministic ascending order."""

    return tuple(index for index in range(int(size)) if int(mask) & (1 << index))


def iter_submasks(mask: int) -> tuple[int, ...]:
    """Enumerate every subset once in stable ascending numeric order."""

    return tuple(candidate for candidate in range(int(mask) + 1) if (candidate & ~int(mask)) == 0)


def first_target_failure(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    selected_mask: int,
    target_index: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any] | None:
    """Return the canonical first failed all-order query for one token.

    The returned subset is deliberately a *query witness*, not a claim that
    its prefix is an observed or policy-valid decoding trajectory.  Callers
    can turn its pool indices into a canonical completion order for auditing.
    """

    selected_mask = int(selected_mask)
    target_index = int(target_index)
    target_bit = 1 << target_index
    if not (selected_mask & target_bit):
        raise ValueError("target_index is not selected by selected_mask")
    for revealed_mask in iter_submasks(selected_mask ^ target_bit):
        passes, margin, matches, is_tie = query_passes(
            margins,
            target_index,
            revealed_mask,
            float(gamma),
            tolerance=float(tolerance),
        )
        if not passes:
            return {
                "target_index": target_index,
                "revealed_mask": int(revealed_mask),
                "logit_margin": None if margin is None else float(margin),
                "top1_matches_assignment": bool(matches),
                "is_logit_tie": bool(is_tie),
            }
    return None


def exact_set_certificate(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    selected_mask: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> SetCertificate:
    """Evaluate the stated exact all-order certificate for ``selected_mask``.

    A global pool bit is used both for the selected set and each revealed
    context.  For target ``i``, every subset of ``S minus {i}`` is queried.  This
    is precisely the requested ``min_i min_A`` construction and avoids any
    permutation enumeration.
    """

    selected_mask = int(selected_mask)
    if selected_mask < 0:
        raise ValueError("selected_mask must be non-negative")
    if selected_mask == 0:
        return SetCertificate(
            selected_mask=0,
            gamma=float(gamma),
            passes=True,
            certificate_margin=None,
            query_count=0,
            violating_indices=(),
            first_failure_target_index=None,
            first_failure_revealed_mask=None,
            first_failure_margin=None,
            first_failure_top1_matches=None,
            first_failure_is_tie=None,
            minimum_margin_target_index=None,
            minimum_margin_revealed_mask=None,
            missing_query_count=0,
            vacuous=True,
        )

    values: list[tuple[float, int, int]] = []
    violating: set[int] = set()
    first_failure: tuple[int, int, float | None, bool, bool] | None = None
    missing = 0
    query_count = 0
    bit = 1
    target = 0
    while bit <= selected_mask:
        if selected_mask & bit:
            other = selected_mask ^ bit
            for revealed_mask in iter_submasks(other):
                passes, margin, matches, is_tie = query_passes(
                    margins,
                    target,
                    revealed_mask,
                    float(gamma),
                    tolerance=float(tolerance),
                )
                query_count += 1
                if margin is None:
                    missing += 1
                else:
                    values.append((float(margin), target, revealed_mask))
                if not passes:
                    violating.add(target)
                    if first_failure is None:
                        first_failure = (target, revealed_mask, margin, matches, is_tie)
        bit <<= 1
        target += 1

    minimum = min(values, key=lambda value: (value[0], value[1], value[2])) if values else None
    return SetCertificate(
        selected_mask=selected_mask,
        gamma=float(gamma),
        passes=not violating,
        certificate_margin=None if minimum is None else float(minimum[0]),
        query_count=query_count,
        violating_indices=tuple(sorted(violating)),
        first_failure_target_index=None if first_failure is None else first_failure[0],
        first_failure_revealed_mask=None if first_failure is None else first_failure[1],
        first_failure_margin=None if first_failure is None or first_failure[2] is None else float(first_failure[2]),
        first_failure_top1_matches=None if first_failure is None else bool(first_failure[3]),
        first_failure_is_tie=None if first_failure is None else bool(first_failure[4]),
        minimum_margin_target_index=None if minimum is None else minimum[1],
        minimum_margin_revealed_mask=None if minimum is None else minimum[2],
        missing_query_count=missing,
        vacuous=False,
    )


def confidence_selected_mask(probabilities: Sequence[float], threshold: float) -> int:
    """Select exactly the pool positions whose fixed top-1 p is at least tau."""

    return sum(
        1 << index
        for index, probability in enumerate(probabilities)
        if float(probability) >= float(threshold)
    )


def select_top_probability_margin_positions(
    assignments: Mapping[int, Mapping[str, Any]], k: int
) -> tuple[int, ...]:
    """Return the current masked top-``k`` positions ranked by ``p1 - p2``.

    This is deliberately a *probability* margin, not a raw-logit margin or a
    top-1 confidence ranking.  Ties are resolved by physical position so the
    candidate pool is deterministic even when a model returns equal values.
    When fewer than ``k`` positions are supplied, every supplied position is
    returned.  The tuple is in candidate-rank order, which makes its prefix
    directly reusable for a K sweep.
    """

    if int(k) < 1:
        raise ValueError("k must be at least one")
    ranked: list[tuple[float, int]] = []
    for raw_position, payload in assignments.items():
        try:
            position = int(raw_position)
            top1 = float(payload["top1_probability"])
            top2 = float(payload["top2_probability"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"position {raw_position!r} lacks finite top-1/top-2 probabilities") from error
        margin = top1 - top2
        if not math.isfinite(margin):
            raise ValueError(f"position {position} has a non-finite p1-p2 probability margin")
        ranked.append((margin, position))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(position for _margin, position in ranked[: int(k)])


def set_log_probability(mask: int, probabilities: Sequence[float]) -> float:
    """Stable tie-break score for a selected set of fixed top-1 values."""

    result = 0.0
    for index, probability in enumerate(probabilities):
        if int(mask) & (1 << index):
            result += math.log(max(float(probability), sys.float_info.min))
    return result


def _mask_position_tuple(mask: int, positions: Sequence[int]) -> tuple[int, ...]:
    # Pool order starts with B and then confidence-ranked extras, so it is not
    # necessarily physical position order.  The protocol's final tie-break is
    # explicitly by deterministic position order, not by pool construction.
    return tuple(sorted(int(position) for index, position in enumerate(positions) if int(mask) & (1 << index)))


def choose_largest_safe_mask(
    certificates: Mapping[int, SetCertificate],
    probabilities: Sequence[float],
    positions: Sequence[int],
    *,
    required_mask: int = 0,
    tie_scores: Sequence[float] | None = None,
) -> int | None:
    """Choose a maximum-cardinality exact-safe mask with prescribed ties.

    By default, historic headroom callers retain their summed fixed-top-1
    log-probability tie break.  A caller can provide one ``tie_score`` per
    pool position instead; the new top-``K`` rollout oracle uses p1-p2 scores
    so ties among equally large safe sets preserve the requested ranking
    signal.  Both variants end with ascending lexicographic physical position
    order for a deterministic fallback.
    """

    required_mask = int(required_mask)
    if len(probabilities) != len(positions):
        raise ValueError("probabilities and positions must have equal length")
    if tie_scores is not None:
        if len(tie_scores) != len(positions):
            raise ValueError("tie_scores and positions must have equal length")
        normalized_tie_scores = tuple(float(value) for value in tie_scores)
        if any(not math.isfinite(value) for value in normalized_tie_scores):
            raise ValueError("tie_scores must be finite")
    else:
        normalized_tie_scores = None
    eligible = [
        mask
        for mask, certificate in certificates.items()
        if (int(mask) & required_mask) == required_mask and certificate.passes
    ]
    if not eligible:
        return None

    def score(mask: int) -> float:
        if normalized_tie_scores is None:
            return set_log_probability(int(mask), probabilities)
        return math.fsum(
            value for index, value in enumerate(normalized_tie_scores) if int(mask) & (1 << index)
        )

    return sorted(
        eligible,
        key=lambda mask: (
            -int(mask).bit_count(),
            -score(int(mask)),
            _mask_position_tuple(int(mask), positions),
        ),
    )[0]


def certificate_cache_for_gamma(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    pool_size: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> dict[int, SetCertificate]:
    """Evaluate every possible pool subset once from cached GPU margin rows."""

    return {
        mask: exact_set_certificate(margins, mask, float(gamma), tolerance=float(tolerance))
        for mask in range(1 << int(pool_size))
    }
