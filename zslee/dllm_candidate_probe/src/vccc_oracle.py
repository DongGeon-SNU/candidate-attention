"""Pure exact-certificate logic for the frozen VCCC oracle audit.

The GPU runner supplies full-vocabulary, exact/no-cache margins for every
``(target, revealed-subset)`` query.  This module deliberately contains no
model, tokenizer, cache, or dataset code: it makes the all-order and
existential definitions testable with small synthetic margin maps.

The bit convention is stable throughout this module: bit ``k`` represents the
candidate at ``positions[k]``.  A margin query for target ``i`` is valid only
when bit ``i`` is *not* set in its revealed-subset mask.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Iterable, Mapping, Sequence


EPSILON = 1.0e-12


@dataclass(frozen=True)
class CertificateResult:
    """Exact U/E/LOO result for one fixed current-top-1 assignment set."""

    all_order_margin: float | None
    existential_margin: float | None
    final_loo_margin: float | None
    all_pass: bool
    existential_pass: bool
    final_loo_pass: bool
    category: str
    witness_order: tuple[int, ...] | None
    maximum_bottleneck_order: tuple[int, ...] | None
    tie_query_count: int
    missing_query_count: int


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _query(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    target: int,
    revealed_mask: int,
) -> tuple[float | None, bool, bool]:
    """Return ``(logit_margin, deterministic_top1_matches, was_tie)``.

    Float-only synthetic maps are treated as deterministic matches whenever
    their margin is non-negative; production rows always provide the explicit
    ``top1_matches_assignment`` flag.  A tie is never silently treated as a
    valid preservation event when the deterministic argmax chose another
    token.
    """

    raw = margins.get((int(target), int(revealed_mask)))
    if raw is None:
        return None, False, False
    if isinstance(raw, Mapping):
        margin = _finite(raw.get("logit_margin"))
        match = bool(raw.get("top1_matches_assignment", False))
        tie = bool(raw.get("is_logit_tie", False))
    else:
        margin = _finite(raw)
        match = bool(margin is not None and margin >= 0.0)
        tie = bool(margin is not None and abs(margin) <= EPSILON)
    return margin, match, tie


def query_passes(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    target: int,
    revealed_mask: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> tuple[bool, float | None, bool, bool]:
    """Apply the stated certificate rule to a single exact margin query.

    ``margin >= gamma - tolerance`` implements the requested inclusive margin
    threshold.  We additionally require the deterministic full-vocabulary
    argmax to equal the fixed assignment, which resolves the otherwise
    ambiguous ``gamma=0`` tie case without device-dependent top-k ordering.
    """

    margin, matches, tie = _query(margins, target, revealed_mask)
    if margin is None:
        return False, None, matches, tie
    return bool(matches and margin >= float(gamma) - float(tolerance)), margin, matches, tie


def _valid_query_keys(size: int) -> Iterable[tuple[int, int]]:
    for target in range(size):
        for mask in range(1 << size):
            if not (mask & (1 << target)):
                yield target, mask


def _minimum_margin(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float], size: int
) -> tuple[float | None, int, int]:
    values: list[float] = []
    ties = missing = 0
    for target, mask in _valid_query_keys(size):
        value, _match, tie = _query(margins, target, mask)
        ties += int(tie)
        if value is None:
            missing += 1
        else:
            values.append(value)
    return (min(values) if values else None), ties, missing


def _final_loo_margin(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float], size: int
) -> tuple[float | None, int, int]:
    values: list[float] = []
    ties = missing = 0
    full = (1 << size) - 1
    for target in range(size):
        value, _match, tie = _query(margins, target, full ^ (1 << target))
        ties += int(tie)
        if value is None:
            missing += 1
        else:
            values.append(value)
    return (min(values) if values else None), ties, missing


def _all_order_passes(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    size: int,
    gamma: float,
    tolerance: float,
) -> bool:
    return all(
        query_passes(margins, target, mask, gamma, tolerance=tolerance)[0]
        for target, mask in _valid_query_keys(size)
    )


def existential_dp(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    size: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> tuple[bool, tuple[int, ...] | None]:
    """Return whether a safe order exists and one deterministic witness.

    Dynamic programming is over revealed sets, not permutations.  A parent is
    recorded only on the first deterministic transition into a reachable set;
    this makes the stored witness reproducible while preserving the exact
    existential predicate.
    """

    parents: dict[int, tuple[int, int]] = {}
    reachable = {0}
    full = (1 << size) - 1
    for mask in range(1 << size):
        if mask not in reachable:
            continue
        for target in range(size):
            bit = 1 << target
            if mask & bit:
                continue
            passes, _margin, _matches, _tie = query_passes(
                margins, target, mask, gamma, tolerance=tolerance
            )
            next_mask = mask | bit
            if passes and next_mask not in reachable:
                reachable.add(next_mask)
                parents[next_mask] = (mask, target)
    if full not in reachable:
        return False, None
    order: list[int] = []
    cursor = full
    while cursor:
        parent, target = parents[cursor]
        order.append(target)
        cursor = parent
    order.reverse()
    return True, tuple(order)


def maximum_bottleneck_order(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float], size: int
) -> tuple[float | None, tuple[int, ...] | None]:
    """Exact max-min order with subset DP and deterministic tie breaking."""

    # score[mask] = largest possible minimum margin along a path to mask.
    scores: dict[int, float] = {0: math.inf}
    parents: dict[int, tuple[int, int]] = {}
    for mask in range(1 << size):
        current = scores.get(mask)
        if current is None:
            continue
        for target in range(size):
            bit = 1 << target
            if mask & bit:
                continue
            margin, _matches, _tie = _query(margins, target, mask)
            if margin is None:
                continue
            candidate = min(current, margin)
            next_mask = mask | bit
            previous = scores.get(next_mask)
            # Candidate targets are visited ascending and masks ascending, so
            # retain the first equal score as the canonical witness.
            if previous is None or candidate > previous + EPSILON:
                scores[next_mask] = candidate
                parents[next_mask] = (mask, target)
    full = (1 << size) - 1
    score = scores.get(full)
    if score is None:
        return None, None
    order: list[int] = []
    cursor = full
    while cursor:
        parent, target = parents[cursor]
        order.append(target)
        cursor = parent
    order.reverse()
    return score, tuple(order)


def classify_certificate(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    size: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> CertificateResult:
    """Compute mutually exclusive ALL/EXISTS/NO-SAFE categories exactly."""

    all_margin, all_ties, all_missing = _minimum_margin(margins, size)
    loo_margin, loo_ties, loo_missing = _final_loo_margin(margins, size)
    all_pass = _all_order_passes(margins, size, gamma, tolerance)
    exists_pass, witness = existential_dp(margins, size, gamma, tolerance=tolerance)
    existential_margin, bottleneck_order = maximum_bottleneck_order(margins, size)
    final_loo_pass = all(
        query_passes(
            margins, target, ((1 << size) - 1) ^ (1 << target), gamma, tolerance=tolerance
        )[0]
        for target in range(size)
    )
    if all_pass:
        category = "ALL_PASS"
    elif exists_pass:
        category = "EXISTS_ONLY"
    else:
        category = "NO_SAFE_ORDER"
    return CertificateResult(
        all_order_margin=all_margin,
        existential_margin=existential_margin,
        final_loo_margin=loo_margin,
        all_pass=all_pass,
        existential_pass=exists_pass,
        final_loo_pass=final_loo_pass,
        category=category,
        witness_order=witness,
        maximum_bottleneck_order=bottleneck_order,
        tie_query_count=all_ties + loo_ties,
        missing_query_count=all_missing + loo_missing,
    )


def residual_rows(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float], size: int
) -> list[dict[str, Any]]:
    """Return full exact subset residuals for every target and subset."""

    rows: list[dict[str, Any]] = []
    for target in range(size):
        base, _base_match, _base_tie = _query(margins, target, 0)
        if base is None:
            continue
        singleton: dict[int, float] = {}
        for source in range(size):
            if source == target:
                continue
            value, _match, _tie = _query(margins, target, 1 << source)
            if value is not None:
                singleton[source] = value - base
        for mask in range(1 << size):
            if mask & (1 << target):
                continue
            value, match, tie = _query(margins, target, mask)
            if value is None:
                continue
            members = [source for source in range(size) if mask & (1 << source)]
            if any(source not in singleton for source in members):
                continue
            additive = base + sum(singleton[source] for source in members)
            rows.append(
                {
                    "target_index": target,
                    "revealed_mask": mask,
                    "revealed_count": len(members),
                    "base_margin": base,
                    "exact_margin": value,
                    "additive_predicted_margin": additive,
                    "residual": value - additive,
                    "pairwise_safe_at_query": bool(match and value >= 0.0),
                    "is_logit_tie": tie,
                }
            )
    return rows


def pairwise_certificate_margin(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float], size: int
) -> float | None:
    """Conservative singleton lower-bound score before residual calibration."""

    target_scores: list[float] = []
    for target in range(size):
        base, _match, _tie = _query(margins, target, 0)
        if base is None:
            return None
        directed: list[float] = []
        for source in range(size):
            if source == target:
                continue
            singleton, _singleton_match, _singleton_tie = _query(margins, target, 1 << source)
            if singleton is None:
                return None
            directed.append(singleton - base)
        target_scores.append(base + sum(min(0.0, effect) for effect in directed))
    return min(target_scores) if target_scores else None


def pairwise_safe(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    size: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Whether base plus every directed singleton query passes a threshold."""

    for target in range(size):
        if not query_passes(margins, target, 0, gamma, tolerance=tolerance)[0]:
            return False
        for source in range(size):
            if source == target:
                continue
            if not query_passes(margins, target, 1 << source, gamma, tolerance=tolerance)[0]:
                return False
    return True


def directional_dependency_graph(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    size: int,
    gamma: float,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Construct precedence edges ``i -> j`` when revealing j first harms i.

    An edge ``i -> j`` means i must precede j: once j is revealed, i fails the
    singleton context check.  This orientation makes a graph topological order
    directly interpretable as a cheap candidate witness order.
    """

    edges: list[tuple[int, int]] = []
    for target in range(size):
        for source in range(size):
            if source == target:
                continue
            if not query_passes(margins, target, 1 << source, gamma, tolerance=tolerance)[0]:
                edges.append((target, source))
    topo = topological_order(size, edges)
    components = strongly_connected_components(size, edges)
    cyclic = any(len(component) > 1 for component in components) or any(left == right for left, right in edges)
    return {
        "edges": tuple(edges),
        "edge_count": len(edges),
        "edge_density": len(edges) / max(size * (size - 1), 1),
        "topological_order": topo,
        "is_dag": topo is not None,
        "scc_count": len(components),
        "largest_scc_size": max((len(component) for component in components), default=0),
        "has_cycle": cyclic,
    }


def topological_order(size: int, edges: Iterable[tuple[int, int]]) -> tuple[int, ...] | None:
    outgoing: dict[int, list[int]] = {node: [] for node in range(size)}
    indegree = [0] * size
    for left, right in edges:
        if right not in outgoing[left]:
            outgoing[left].append(right)
            indegree[right] += 1
    queue = deque(sorted(node for node in range(size) if indegree[node] == 0))
    result: list[int] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return tuple(result) if len(result) == size else None


def strongly_connected_components(size: int, edges: Iterable[tuple[int, int]]) -> list[tuple[int, ...]]:
    """Tarjan SCCs with stable node ordering, without a networkx dependency."""

    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    output: list[tuple[int, ...]] = []

    def visit(node: int) -> None:
        nonlocal index
        indices[node] = lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for child in sorted(adjacency[node]):
            if child not in indices:
                visit(child)
                lowlink[node] = min(lowlink[node], lowlink[child])
            elif child in on_stack:
                lowlink[node] = min(lowlink[node], indices[child])
        if lowlink[node] == indices[node]:
            component: list[int] = []
            while True:
                child = stack.pop()
                on_stack.remove(child)
                component.append(child)
                if child == node:
                    break
            output.append(tuple(sorted(component)))

    for node in range(size):
        if node not in indices:
            visit(node)
    return sorted(output)


def order_bottleneck(
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float], order: Sequence[int]
) -> float | None:
    mask = 0
    values: list[float] = []
    for target in order:
        value, _match, _tie = _query(margins, int(target), mask)
        if value is None:
            return None
        values.append(value)
        mask |= 1 << int(target)
    return min(values) if values else None


def quantile(values: Sequence[float], probability: float) -> float | None:
    """Deterministic linear quantile used for calibration metadata."""

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    probability = min(1.0, max(0.0, float(probability)))
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def fixed_confidence_bin(probability: float | None) -> str:
    value = _finite(probability)
    if value is None:
        return "missing"
    if value < 0.7:
        return "p1<0.7"
    if value < 0.9:
        return "0.7<=p1<0.9"
    if value < 0.95:
        return "0.9<=p1<0.95"
    if value < 0.99:
        return "0.95<=p1<0.99"
    return "p1>=0.99"


def fixed_numeric_bin(value: float | None, edges: Sequence[float], *, label: str) -> str:
    numeric = _finite(value)
    if numeric is None:
        return "missing"
    if not edges:
        return "all"
    ordered = sorted(float(edge) for edge in edges)
    for index, upper in enumerate(ordered):
        if numeric < upper or (index == len(ordered) - 1 and numeric <= upper):
            lower = "-inf" if index == 0 else f"{ordered[index - 1]:.6g}"
            return f"{label}[{lower},{upper:.6g})"
    return f"{label}[{ordered[-1]:.6g},inf)"


__all__ = [
    "CertificateResult",
    "classify_certificate",
    "directional_dependency_graph",
    "existential_dp",
    "fixed_confidence_bin",
    "fixed_numeric_bin",
    "maximum_bottleneck_order",
    "order_bottleneck",
    "pairwise_certificate_margin",
    "pairwise_safe",
    "quantile",
    "residual_rows",
    "strongly_connected_components",
    "topological_order",
]
