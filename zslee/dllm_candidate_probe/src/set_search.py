"""Small, deterministic set beam search with position exclusivity."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence

from .branching import Candidate


@dataclass(frozen=True)
class ScoredSet:
    candidates: tuple[Candidate, ...]
    score: float


def valid_candidate_sets(
    candidates: Sequence[Candidate],
    size: int,
) -> Iterable[tuple[Candidate, ...]]:
    for candidate_set in combinations(candidates, size):
        positions = {candidate.position for candidate in candidate_set}
        if len(positions) == size:
            yield candidate_set


def beam_sets(
    candidates: Sequence[Candidate],
    size: int,
    *,
    pair_score: Callable[[Candidate, Candidate], float],
    beam_width: int,
) -> list[ScoredSet]:
    """Keep the highest mean-pair-score position-valid sets only."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive.")
    scored: list[ScoredSet] = []
    for candidate_set in valid_candidate_sets(candidates, size):
        pair_scores = [pair_score(a, b) for a, b in combinations(candidate_set, 2)]
        score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        scored.append(ScoredSet(candidate_set, score))
    return sorted(scored, key=lambda item: (-item.score, item.candidates))[:beam_width]
