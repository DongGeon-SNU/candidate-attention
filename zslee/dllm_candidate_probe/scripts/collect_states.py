"""Confidence-threshold decoding-state collection for frozen LLaDA.

State collection follows Fast-dLLM v1's confidence-threshold transfer rule:
unmask positions at/above threshold and always transfer the highest-confidence
remaining mask.  This script stores only input tokens and top-k scalar summaries;
it never serializes attention maps, hidden states, KV tensors, or vocabulary
probability tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DecodingState:
    prompt_index: int
    step: int
    input_ids: Any
    mask_positions: list[int]
    position_summaries: dict[int, dict[str, Any]]
    transferred_positions: list[int]
    remaining_mask_count: int
    low_parallel: bool

    def public_record(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "step": self.step,
            "token_sequence": self.input_ids[0].detach().cpu().tolist(),
            "mask_positions": self.mask_positions,
            "position_summaries": {str(k): v for k, v in self.position_summaries.items()},
            "baseline_transferred_positions": self.transferred_positions,
            "remaining_mask_count": self.remaining_mask_count,
            "low_parallel": self.low_parallel,
        }


def mask_token_id(model: Any, tokenizer: Any) -> int:
    value = getattr(tokenizer, "mask_token_id", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "mask_token_id", None)
    if value is None:
        raise RuntimeError("The loaded LLaDA tokenizer/model did not expose mask_token_id.")
    return int(value)


def tokenized_prompt(tokenizer: Any, prompt: str, device: Any) -> Any:
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
    except (AttributeError, ValueError):
        rendered = prompt
    return tokenizer(rendered, return_tensors="pt").input_ids.to(device)


def _summary_for_position(top_ids: Any, top_probs: Any) -> dict[str, Any]:
    return {
        "top1_confidence": float(top_probs[0].item()),
        "top5_token_ids": [int(value) for value in top_ids.tolist()],
        "top5_probabilities": [float(value) for value in top_probs.tolist()],
        "top5_cumulative_probability": float(top_probs.sum().item()),
    }


def collect_states(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    prompt_index: int,
    generation_length: int,
    steps: int,
    threshold: float,
    use_cache: bool,
    low_parallel_max_transferred: int,
    low_parallel_remaining_fraction: float,
) -> list[DecodingState]:
    """Collect low-parallel states while using only frozen inference forwards."""

    import torch

    model.eval()
    mask_id = mask_token_id(model, tokenizer)
    prompt_ids = tokenized_prompt(tokenizer, prompt, next(model.parameters()).device)
    x = torch.cat(
        [prompt_ids, torch.full((1, generation_length), mask_id, device=prompt_ids.device, dtype=prompt_ids.dtype)], dim=1
    )
    collected: list[DecodingState] = []
    for step in range(min(steps, generation_length)):
        mask_index = x.eq(mask_id)
        positions = torch.where(mask_index[0])[0]
        if positions.numel() == 0:
            break
        with torch.inference_mode():
            # State collection may use model caching. Counterfactual code never does.
            logits = model(x, use_cache=use_cache).logits
        mask_logits = logits[0, positions]
        probabilities = torch.softmax(mask_logits.float(), dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=min(5, probabilities.shape[-1]), dim=-1)
        proposed_ids = top_ids[:, 0]
        confidence = top_probs[:, 0]
        transfer = confidence.ge(threshold)
        transfer[torch.argmax(confidence)] = True
        transferred_positions = positions[transfer]
        remaining_count = int(positions.numel())
        transfer_count = int(transfer.sum().item())
        low_parallel = (
            transfer_count <= low_parallel_max_transferred
            or transfer_count / max(remaining_count, 1) <= low_parallel_remaining_fraction
        )
        summaries = {
            int(position.item()): _summary_for_position(ids, probs)
            for position, ids, probs in zip(positions, top_ids, top_probs, strict=True)
        }
        if low_parallel:
            collected.append(
                DecodingState(
                    prompt_index=prompt_index,
                    step=step,
                    input_ids=x.clone(),
                    mask_positions=[int(position.item()) for position in positions],
                    position_summaries=summaries,
                    transferred_positions=[int(position.item()) for position in transferred_positions],
                    remaining_mask_count=remaining_count,
                    low_parallel=True,
                )
            )
        x[0, transferred_positions] = proposed_ids[transfer]
    return collected
