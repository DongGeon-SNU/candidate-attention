"""Deterministic pair-graph clique generation for the hard-set audit.

The generators have two deliberately different meanings. ``policy_audit_sets``
is a randomized sequential extension of the anchor set and approximates the
distribution induced by a pair-only policy. ``stress_mined_sets`` ranks valid
cliques for counterexample discovery; it must never be interpreted as an
unbiased estimate of an operational failure rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from random import Random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class AuditCandidate:
    """One proposed token at one masked position in a single decoding state."""

    state_id: str
    position: int
    token_id: int
    kind: str  # ``anchor`` or ``unstable``
    token: str = field(compare=False)
    base_probability: float = field(compare=False)
    base_confidence: float = field(compare=False)

    @property
    def node_id(self) -> str:
        return f"{self.position}:{self.token_id}"


@dataclass(frozen=True)
class PairMetric:
    a: AuditCandidate
    b: AuditCandidate
    q2: float
    a_to_b_lift: float
    b_to_a_lift: float
    residual_mean_tv: float | None = None

    @property
    def l_min(self) -> float:
        return min(self.a_to_b_lift, self.b_to_a_lift)

    @property
    def asymmetry(self) -> float:
        return abs(self.a_to_b_lift - self.b_to_a_lift)


def pair_key(a: AuditCandidate, b: AuditCandidate) -> tuple[str, str]:
    if a.state_id != b.state_id:
        raise ValueError("Pair candidates must come from the same decoding state.")
    return tuple(sorted((a.node_id, b.node_id)))


def set_signature(candidates: Iterable[AuditCandidate]) -> tuple[str, ...]:
    """A canonical set identity; same-position alternatives can never alias."""

    return tuple(sorted(candidate.node_id for candidate in candidates))


def has_unique_positions(candidates: Iterable[AuditCandidate]) -> bool:
    positions = [candidate.position for candidate in candidates]
    return len(positions) == len(set(positions))


def edge_exists(metric: PairMetric | None, *, graph: str, pair_threshold: float, lift_delta: float) -> bool:
    if metric is None or metric.q2 < pair_threshold:
        return False
    if graph == "stability_only":
        return True
    if graph == "strict_mutual_support":
        return metric.l_min >= lift_delta
    raise ValueError(f"Unknown graph: {graph}")


def is_clique(
    candidates: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    graph: str,
    pair_threshold: float,
    lift_delta: float,
) -> bool:
    if not has_unique_positions(candidates):
        return False
    return all(
        edge_exists(pair_metrics.get(pair_key(a, b)), graph=graph, pair_threshold=pair_threshold, lift_delta=lift_delta)
        for a, b in combinations(candidates, 2)
    )


def anchor_conflicts(
    anchors: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    graph: str,
    pair_threshold: float,
    lift_delta: float,
) -> list[PairMetric | None]:
    """Return every missing or failed anchor–anchor edge for one condition."""

    failures: list[PairMetric | None] = []
    for a, b in combinations(anchors, 2):
        metric = pair_metrics.get(pair_key(a, b))
        if not edge_exists(metric, graph=graph, pair_threshold=pair_threshold, lift_delta=lift_delta):
            failures.append(metric)
    return failures


def _feasible_extensions(
    current: Sequence[AuditCandidate],
    unstable: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    graph: str,
    pair_threshold: float,
    lift_delta: float,
) -> list[AuditCandidate]:
    used_positions = {candidate.position for candidate in current}
    return [
        candidate
        for candidate in unstable
        if candidate.position not in used_positions
        and all(
            edge_exists(
                pair_metrics.get(pair_key(candidate, member)),
                graph=graph,
                pair_threshold=pair_threshold,
                lift_delta=lift_delta,
            )
            for member in current
        )
    ]


def _policy_choice(feasible: Sequence[AuditCandidate], rng: Random) -> AuditCandidate:
    """Score by marginal probability, while randomizing only a top-score tie band."""

    ranked = sorted(
        feasible,
        key=lambda candidate: (-candidate.base_probability, candidate.position, candidate.token_id),
    )
    best = ranked[0].base_probability
    # The tolerance avoids deterministic collapse for numerically near-identical candidates.
    tie_band = [candidate for candidate in ranked if candidate.base_probability >= best * 0.98]
    return tie_band[rng.randrange(len(tie_band))]


def _sample_one_policy_set(
    anchors: Sequence[AuditCandidate],
    unstable: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    target_size: int,
    graph: str,
    pair_threshold: float,
    lift_delta: float,
    rng: Random,
) -> tuple[AuditCandidate, ...] | None:
    if len(anchors) > target_size or not has_unique_positions(anchors):
        return None
    current = list(anchors)
    while len(current) < target_size:
        feasible = _feasible_extensions(
            current, unstable, pair_metrics,
            graph=graph, pair_threshold=pair_threshold, lift_delta=lift_delta,
        )
        if not feasible:
            return None
        current.append(_policy_choice(feasible, rng))
    result = tuple(sorted(current))
    return result if is_clique(result, pair_metrics, graph=graph, pair_threshold=pair_threshold, lift_delta=lift_delta) else None


def policy_audit_sets(
    anchors: Sequence[AuditCandidate],
    unstable: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    target_size: int,
    graph: str,
    pair_threshold: float,
    lift_delta: float,
    seed: int,
    count: int,
    attempts_per_set: int = 40,
) -> list[tuple[AuditCandidate, ...]]:
    """Generate deduplicated sequential-policy cliques with reproducible randomness."""

    rng = Random(seed)
    results: list[tuple[AuditCandidate, ...]] = []
    seen: set[tuple[str, ...]] = set()
    maximum_attempts = max(count * attempts_per_set, attempts_per_set)
    for _ in range(maximum_attempts):
        proposal = _sample_one_policy_set(
            anchors, unstable, pair_metrics,
            target_size=target_size, graph=graph, pair_threshold=pair_threshold,
            lift_delta=lift_delta, rng=rng,
        )
        if proposal is None:
            continue
        signature = set_signature(proposal)
        if signature in seen:
            continue
        seen.add(signature)
        results.append(proposal)
        if len(results) >= count:
            break
    return results


def stress_score(
    candidate_set: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    pair_threshold: float,
) -> float:
    """Higher means closer to a pair boundary or more asymmetric/residual-heavy."""

    metrics = [pair_metrics[pair_key(a, b)] for a, b in combinations(candidate_set, 2)]
    if not metrics:
        return 0.0
    boundary = sum(1.0 - min(1.0, max(0.0, (metric.q2 - pair_threshold) / max(1e-12, 1.0 - pair_threshold))) for metric in metrics)
    asymmetry = sum(metric.asymmetry for metric in metrics)
    residual = sum(metric.residual_mean_tv or 0.0 for metric in metrics)
    return 0.25 * len(candidate_set) + boundary / len(metrics) + asymmetry / len(metrics) + residual / len(metrics)


def stress_mined_sets(
    anchors: Sequence[AuditCandidate],
    unstable: Sequence[AuditCandidate],
    pair_metrics: Mapping[tuple[str, str], PairMetric],
    *,
    target_size: int,
    graph: str,
    pair_threshold: float,
    lift_delta: float,
    seed: int,
    count: int,
) -> list[tuple[AuditCandidate, ...]]:
    """Mine high-risk valid cliques; this is intentionally not uniform sampling."""

    proposals = policy_audit_sets(
        anchors, unstable, pair_metrics,
        target_size=target_size, graph=graph, pair_threshold=pair_threshold,
        lift_delta=lift_delta, seed=seed, count=max(count * 8, count), attempts_per_set=80,
    )
    return sorted(
        proposals,
        key=lambda proposal: (-stress_score(proposal, pair_metrics, pair_threshold=pair_threshold), set_signature(proposal)),
    )[:count]
