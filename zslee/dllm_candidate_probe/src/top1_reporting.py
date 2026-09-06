"""Dependency-light reporting helpers for the top-1 dynamics audit.

This module deliberately consumes compact public records rather than model
tensors.  It is therefore safe to run on a CPU-only analysis host after a GPU
collection job has written trajectories, transitions, eventual-token
observations, and exact-audit records.  The public functions accept ordinary
``Mapping`` objects (including the ``public_record``/``as_dict`` output of the
other audit modules) and avoid a hard dependency on NumPy, pandas, sklearn,
pyarrow, or matplotlib.

The primary uncertainty estimate is a *prompt-clustered* percentile bootstrap:
whole prompts are resampled, never individual masked positions.  The default
of 10,000 draws follows the audit protocol, but every entry point exposes an
``iterations`` argument for smoke tests or cost-constrained analyses.

Typical runner usage::

    flip_summary = aggregate_trajectory_flips(transitions)
    future_rows = future_token_topk_table(eventual_observations)
    metric_rows = predictiveness_table(
        transition_features,
        score_fields=("entropy", "one_minus_p1", "logit_margin"),
        outcome_field="top1_flip",
        score_directions={"logit_margin": "lower"},
    )
    cohort = sample_matched_event_cohort(transition_features)
    write_artifact_tables(output_root / "tables", {
        "flip_rates": grouped_flip_rate_table(transitions, group_fields=("dataset",)),
        "future_token_rank": future_rows,
        "metric_predictiveness": metric_rows,
    })

Records may use nested (``"source.entropy"``) field paths.  Missing scalar
values are excluded from the particular calculation and counted in the result
where useful; they are never silently converted into negative examples.
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias


DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260902
DEFAULT_TOP_KS: tuple[int, ...] = (1, 2, 3, 5, 10)
DEFAULT_CONFIDENCE_BINS: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
MISSING = object()

Record: TypeAlias = Mapping[str, Any]
Weighting: TypeAlias = Literal["macro", "micro"]


class OptionalDependencyError(RuntimeError):
    """Raised only when a caller explicitly requests an optional artifact type."""


class ArtifactWriteError(RuntimeError):
    """A clear wrapper for malformed output records or unavailable writers."""


def _record_mapping(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Make a shallow, serialization-friendly mapping from a public record.

    The runner normally passes dictionaries.  Supporting dataclasses and the
    neighbouring modules' ``public_record``/``as_dict`` methods keeps this
    file useful in small interactive smoke tests as well.
    """

    if isinstance(record, Mapping):
        return dict(record)
    for method_name in ("public_record", "as_dict", "dict"):
        method = getattr(record, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return dict(result)
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        result = dataclasses.asdict(record)
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError("records must be mappings, dataclasses, or expose public_record()/as_dict().")


def _records(records: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
    return [_record_mapping(record) for record in records]


def field_value(record: Mapping[str, Any] | Any, field: str, default: Any = MISSING) -> Any:
    """Get a direct or dot-separated nested field without conflating null/missing.

    ``field_value(row, "previous_distribution.entropy")`` works for nested
    dictionaries and simple objects.  Literal keys take precedence over path
    traversal so a CSV-like record containing ``"a.b"`` still behaves as
    expected.
    """

    if not isinstance(field, str) or not field:
        raise ValueError("field must be a non-empty string.")
    current: Any = record
    if isinstance(current, Mapping) and field in current:
        return current[field]
    for part in field.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def _first_field(record: Mapping[str, Any], fields: Sequence[str], default: Any = MISSING) -> Any:
    for field in fields:
        value = field_value(record, field, MISSING)
        if value is not MISSING:
            return value
    return default


def _as_finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field!r} must contain numeric values; got {value!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"{field!r} must contain finite numeric values; got {value!r}.")
    return result


def _optional_finite_float(value: Any) -> float | None:
    if value is MISSING or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_binary(value: Any, *, field: str) -> bool:
    """Interpret explicit binary values while rejecting accidental truthiness."""

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item) and not isinstance(value, (str, bytes)):
        try:
            scalar = scalar_item()
        except (TypeError, ValueError, RuntimeError):
            scalar = value
        if scalar is not value:
            return _as_binary(scalar, field=field)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1", "flip", "event"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", "non_flip", "control"}:
            return False
    raise ValueError(f"{field!r} must be boolean/0/1; got {value!r}.")


def _freeze_group_value(value: Any) -> Hashable:
    """Turn a possibly nested group value into a deterministic hashable key."""

    if value is MISSING:
        return "<missing>"
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_group_value(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_group_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _stable_key(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=str)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    index = (len(sorted_values) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(sorted_values[low])
    fraction = index - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")


def _bootstrap_from_clusters(
    cluster_values: Mapping[Hashable, Sequence[float]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    weighting: Weighting = "macro",
) -> dict[str, Any]:
    """Cluster-resample scalar values and return percentile confidence limits."""

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    _validate_confidence(confidence)
    if weighting not in {"macro", "micro"}:
        raise ValueError("weighting must be 'macro' or 'micro'.")

    cleaned: list[tuple[Hashable, list[float]]] = []
    for cluster, values in cluster_values.items():
        numeric = [_as_finite_float(value, field="cluster value") for value in values]
        if numeric:
            cleaned.append((cluster, numeric))
    cleaned.sort(key=lambda item: _stable_key(item[0]))
    if not cleaned:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "cluster_count": 0,
            "observation_count": 0,
            "iterations": iterations,
            "confidence": confidence,
            "weighting": weighting,
        }

    means = [math.fsum(values) / len(values) for _, values in cleaned]
    sums = [math.fsum(values) for _, values in cleaned]
    counts = [len(values) for _, values in cleaned]
    if weighting == "macro":
        estimate = math.fsum(means) / len(means)
    else:
        estimate = math.fsum(sums) / math.fsum(counts)

    generator = random.Random(seed)
    n_clusters = len(cleaned)
    draws: list[float] = []
    for _ in range(iterations):
        selected = [generator.randrange(n_clusters) for _ in range(n_clusters)]
        if weighting == "macro":
            draws.append(math.fsum(means[index] for index in selected) / n_clusters)
        else:
            denominator = math.fsum(counts[index] for index in selected)
            draws.append(math.fsum(sums[index] for index in selected) / denominator)
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    low = _percentile(draws, alpha)
    high = _percentile(draws, 1.0 - alpha)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        # Kept as convenience aliases because 95% is the study's primary CI.
        "ci95_low": low if confidence == 0.95 else None,
        "ci95_high": high if confidence == 0.95 else None,
        "cluster_count": n_clusters,
        "observation_count": sum(counts),
        "iterations": iterations,
        "confidence": confidence,
        "weighting": weighting,
    }


def clustered_bootstrap_mean(
    cluster_values: Mapping[Hashable, Sequence[float]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    weighting: Weighting = "macro",
) -> dict[str, Any]:
    """Return a deterministic clustered-bootstrap mean from pre-grouped values.

    This mirrors the scalar helper in :mod:`src.dynamics_metrics`, while being
    available to analysis-only jobs that import reporting utilities directly.
    """

    return _bootstrap_from_clusters(
        cluster_values,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
        weighting=weighting,
    )


def prompt_clustered_bootstrap(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    value_field: str,
    prompt_field: str = "prompt_index",
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    weighting: Weighting = "macro",
    allow_missing_prompt: bool = False,
) -> dict[str, Any]:
    """Estimate a mean/rate by resampling whole prompts.

    ``weighting='macro'`` is the primary prompt-average estimate.  ``micro``
    retains each selected prompt's within-prompt observation count and is
    provided as a secondary transition-weighted estimate.
    """

    clusters: dict[Hashable, list[float]] = defaultdict(list)
    omitted_value_count = 0
    for index, record in enumerate(_records(records)):
        value = _optional_finite_float(field_value(record, value_field, MISSING))
        if value is None:
            omitted_value_count += 1
            continue
        prompt = field_value(record, prompt_field, MISSING)
        if prompt is MISSING:
            if not allow_missing_prompt:
                raise ValueError(f"record {index} is missing required prompt field {prompt_field!r}.")
            # Treat absent IDs as distinct clusters, never as one synthetic prompt.
            prompt = ("<missing-prompt>", index)
        clusters[_freeze_group_value(prompt)].append(value)
    result = _bootstrap_from_clusters(
        clusters,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
        weighting=weighting,
    )
    result.update({
        "prompt_field": prompt_field,
        "value_field": value_field,
        "omitted_value_count": omitted_value_count,
    })
    return result


def prompt_clustered_rate(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    outcome_field: str,
    prompt_field: str = "prompt_index",
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    weighting: Weighting = "macro",
) -> dict[str, Any]:
    """Prompt-clustered bootstrap for an explicit binary outcome field."""

    converted: list[dict[str, Any]] = []
    for record in _records(records):
        value = field_value(record, outcome_field, MISSING)
        if value is MISSING or value is None:
            converted.append(record)
            continue
        updated = dict(record)
        updated[outcome_field] = float(_as_binary(value, field=outcome_field))
        converted.append(updated)
    return prompt_clustered_bootstrap(
        converted,
        value_field=outcome_field,
        prompt_field=prompt_field,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
        weighting=weighting,
    )


def _validated_binary_pairs(labels: Sequence[Any], scores: Sequence[Any]) -> list[tuple[bool, float]]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    pairs: list[tuple[bool, float]] = []
    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        if label is None or score is None:
            continue
        pairs.append((_as_binary(label, field=f"labels[{index}]"), _as_finite_float(score, field=f"scores[{index}]")))
    return pairs


def binary_ranking_metrics(
    labels: Sequence[Any],
    scores: Sequence[Any],
    *,
    higher_is_positive: bool = True,
) -> dict[str, Any]:
    """Compute tie-aware AUROC and average-precision AUPRC without sklearn.

    ``auprc`` uses the common threshold-grouped average-precision convention,
    rather than an arbitrary row ordering within tied scores.  ``pr_auc`` is
    also returned for consumers that prefer trapezoidal interpolation.
    ``None`` signals an undefined AUROC (one class absent) instead of a
    deceptively perfect score.
    """

    pairs = _validated_binary_pairs(labels, scores)
    if not higher_is_positive:
        pairs = [(label, -score) for label, score in pairs]
    n = len(pairs)
    positive_count = sum(label for label, _ in pairs)
    negative_count = n - positive_count
    prevalence = positive_count / n if n else None
    if not n:
        return {
            "observation_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "prevalence": None,
            "auroc": None,
            "auprc": None,
            "average_precision": None,
            "pr_auc": None,
            "higher_is_positive": higher_is_positive,
        }

    # Mann--Whitney U, using average ranks for all exact score ties.
    ascending = sorted(pairs, key=lambda item: item[1])
    rank_sum_positive = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and ascending[end][1] == ascending[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(label for label, _ in ascending[start:end])
        start = end
    if positive_count and negative_count:
        u_positive = rank_sum_positive - positive_count * (positive_count + 1) / 2.0
        auroc: float | None = u_positive / (positive_count * negative_count)
    else:
        auroc = None

    # Descending score threshold groups make AP invariant to tied-record order.
    descending = sorted(pairs, key=lambda item: item[1], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    previous_precision = 1.0
    average_precision = 0.0 if positive_count else None
    pr_auc = 0.0 if positive_count else None
    start = 0
    while start < n:
        end = start + 1
        while end < n and descending[end][1] == descending[start][1]:
            end += 1
        positives = sum(label for label, _ in descending[start:end])
        group_size = end - start
        true_positive += positives
        false_positive += group_size - positives
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positive_count if positive_count else 0.0
        if positive_count:
            assert average_precision is not None and pr_auc is not None
            average_precision += (recall - previous_recall) * precision
            pr_auc += (recall - previous_recall) * (precision + previous_precision) / 2.0
        previous_recall = recall
        previous_precision = precision
        start = end
    return {
        "observation_count": n,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": average_precision,
        "average_precision": average_precision,
        "pr_auc": pr_auc,
        "higher_is_positive": higher_is_positive,
    }


def _extract_binary_score_records(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    score_field: str,
    outcome_field: str,
    prompt_field: str,
    allow_missing_prompt: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Extract valid rows while reporting omissions rather than guessing labels."""

    selected: list[dict[str, Any]] = []
    omitted = 0
    for index, record in enumerate(_records(records)):
        score = _optional_finite_float(field_value(record, score_field, MISSING))
        raw_outcome = field_value(record, outcome_field, MISSING)
        if score is None or raw_outcome is MISSING or raw_outcome is None:
            omitted += 1
            continue
        prompt = field_value(record, prompt_field, MISSING)
        if prompt is MISSING and not allow_missing_prompt:
            raise ValueError(f"record {index} is missing required prompt field {prompt_field!r}.")
        selected.append({
            "record": record,
            "score": score,
            "outcome": _as_binary(raw_outcome, field=outcome_field),
            "prompt": _freeze_group_value(("<missing-prompt>", index) if prompt is MISSING else prompt),
        })
    return selected, omitted


def prompt_clustered_ranking_bootstrap(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    score_field: str,
    outcome_field: str,
    prompt_field: str = "prompt_index",
    higher_is_positive: bool = True,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Prompt-clustered percentile CIs for AUROC and AUPRC.

    This intentionally recomputes tie-aware ranking statistics in every draw,
    because resampling prompts changes the multiplicity of score ties.  For a
    very large audit, use this on a pre-specified cohort or lower ``iterations``
    during smoke tests; the returned metadata makes that choice explicit.
    """

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    _validate_confidence(confidence)
    selected, omitted = _extract_binary_score_records(
        records,
        score_field=score_field,
        outcome_field=outcome_field,
        prompt_field=prompt_field,
    )
    clusters: dict[Hashable, list[tuple[bool, float]]] = defaultdict(list)
    for row in selected:
        clusters[row["prompt"]].append((row["outcome"], row["score"]))
    ordered_clusters = [clusters[key] for key in sorted(clusters, key=_stable_key)]
    point = binary_ranking_metrics(
        [row["outcome"] for row in selected],
        [row["score"] for row in selected],
        higher_is_positive=higher_is_positive,
    )
    if not ordered_clusters:
        return {
            **point,
            "auroc_ci95_low": None,
            "auroc_ci95_high": None,
            "auprc_ci95_low": None,
            "auprc_ci95_high": None,
            "bootstrap_valid_draw_count": 0,
            "iterations": iterations,
            "cluster_count": 0,
            "omitted_record_count": omitted,
        }

    generator = random.Random(seed)
    auroc_draws: list[float] = []
    auprc_draws: list[float] = []
    cluster_count = len(ordered_clusters)
    for _ in range(iterations):
        labels: list[bool] = []
        scores: list[float] = []
        for _cluster_draw in range(cluster_count):
            cluster = ordered_clusters[generator.randrange(cluster_count)]
            for label, score in cluster:
                labels.append(label)
                scores.append(score)
        metric = binary_ranking_metrics(labels, scores, higher_is_positive=higher_is_positive)
        if metric["auroc"] is not None:
            auroc_draws.append(float(metric["auroc"]))
        if metric["auprc"] is not None:
            auprc_draws.append(float(metric["auprc"]))
    auroc_draws.sort()
    auprc_draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        **point,
        "auroc_ci95_low": _percentile(auroc_draws, alpha) if confidence == 0.95 else None,
        "auroc_ci95_high": _percentile(auroc_draws, 1.0 - alpha) if confidence == 0.95 else None,
        "auprc_ci95_low": _percentile(auprc_draws, alpha) if confidence == 0.95 else None,
        "auprc_ci95_high": _percentile(auprc_draws, 1.0 - alpha) if confidence == 0.95 else None,
        "auroc_ci_low": _percentile(auroc_draws, alpha),
        "auroc_ci_high": _percentile(auroc_draws, 1.0 - alpha),
        "auprc_ci_low": _percentile(auprc_draws, alpha),
        "auprc_ci_high": _percentile(auprc_draws, 1.0 - alpha),
        "bootstrap_valid_auroc_draw_count": len(auroc_draws),
        "bootstrap_valid_auprc_draw_count": len(auprc_draws),
        "iterations": iterations,
        "confidence": confidence,
        "cluster_count": cluster_count,
        "omitted_record_count": omitted,
    }


def _quantile_edges(scores: Sequence[float], bin_count: int) -> list[float]:
    """Return unique interior edges so a tie never gets split across bins."""

    if bin_count < 1:
        raise ValueError("bin_count must be at least one.")
    if not scores:
        return []
    ordered = sorted(scores)
    return sorted({
        edge for edge in (
            _percentile(ordered, index / bin_count) for index in range(1, bin_count)
        )
        if edge is not None
    })


def quantile_bin_table(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    score_field: str,
    outcome_field: str,
    prompt_field: str = "prompt_index",
    bin_count: int = 10,
    bootstrap_iterations: int | None = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    include_empty: bool = False,
) -> list[dict[str, Any]]:
    """Empirical outcome rates in tie-safe score-quantile bins.

    Values equal to a quantile edge are kept in one bin, so a degenerate score
    distribution may yield fewer than ``bin_count`` non-empty rows.  This is
    preferable to pretending tied logits identify different risk deciles.
    ``bootstrap_iterations=None`` skips CI calculation (handy for unit tests).
    """

    if bin_count < 1:
        raise ValueError("bin_count must be at least one.")
    if bootstrap_iterations is not None and int(bootstrap_iterations) <= 0:
        raise ValueError("bootstrap_iterations must be positive or None.")
    selected, omitted = _extract_binary_score_records(
        records,
        score_field=score_field,
        outcome_field=outcome_field,
        prompt_field=prompt_field,
    )
    if not selected:
        return []
    edges = _quantile_edges([row["score"] for row in selected], bin_count)
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        # ``bisect_right`` assigns all values exactly at an edge to its upper
        # bin.  No equal score can cross the resulting boundary.
        bins[bisect.bisect_right(edges, row["score"])].append(row)

    output: list[dict[str, Any]] = []
    actual_bin_count = len(edges) + 1
    for bin_index in range(actual_bin_count):
        members = bins.get(bin_index, [])
        if not members and not include_empty:
            continue
        scores = [member["score"] for member in members]
        outcomes = [member["outcome"] for member in members]
        total = len(members)
        flips = sum(outcomes)
        row: dict[str, Any] = {
            "score": score_field,
            "outcome": outcome_field,
            "bin_index": bin_index + 1,
            "requested_bin_count": bin_count,
            "actual_bin_count": actual_bin_count,
            "quantile_low": bin_index / actual_bin_count,
            "quantile_high": (bin_index + 1) / actual_bin_count,
            "score_lower": min(scores) if scores else None,
            "score_upper": max(scores) if scores else None,
            "score_mean": math.fsum(scores) / total if total else None,
            "observation_count": total,
            "positive_count": flips,
            "empirical_probability": flips / total if total else None,
            "omitted_record_count": omitted,
        }
        if bootstrap_iterations is None or not members:
            row.update({
                "prompt_macro_probability": None if not members else None,
                "ci95_low": None,
                "ci95_high": None,
                "bootstrap_iterations": bootstrap_iterations,
            })
        else:
            bootstrap_records = [
                {prompt_field: member["prompt"], outcome_field: member["outcome"]}
                for member in members
            ]
            macro = prompt_clustered_bootstrap(
                bootstrap_records,
                value_field=outcome_field,
                prompt_field=prompt_field,
                iterations=int(bootstrap_iterations),
                seed=bootstrap_seed + bin_index,
                confidence=confidence,
                weighting="macro",
            )
            row.update({
                "prompt_macro_probability": macro["estimate"],
                "ci95_low": macro["ci95_low"],
                "ci95_high": macro["ci95_high"],
                "bootstrap_iterations": int(bootstrap_iterations),
                "prompt_cluster_count": macro["cluster_count"],
            })
        output.append(row)
    return output


def predictiveness_table(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    score_fields: Iterable[str],
    outcome_field: str = "top1_flip",
    prompt_field: str = "prompt_index",
    score_directions: Mapping[str, Literal["higher", "lower"] | bool] | None = None,
    quantile_bins: int = 10,
    bootstrap_iterations: int | None = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    include_ranking_ci: bool = True,
    include_quantile_rows: bool = False,
) -> list[dict[str, Any]]:
    """Report single-score flip predictiveness, optionally plus bin-level rows.

    Direction values are ``"higher"``/``True`` when a larger value predicts a
    flip and ``"lower"``/``False`` when a smaller value predicts one.  The
    output intentionally contains a ``row_type`` column so the summary and
    quantile rows can share one CSV when desired.
    """

    materialized = _records(records)
    direction_map = dict(score_directions or {})
    rows: list[dict[str, Any]] = []
    for score_index, score_field in enumerate(dict.fromkeys(str(item) for item in score_fields)):
        configured_direction = direction_map.get(score_field, "higher")
        if configured_direction in {"higher", True}:
            higher_is_positive = True
        elif configured_direction in {"lower", False}:
            higher_is_positive = False
        else:
            raise ValueError(f"Unknown score direction for {score_field!r}: {configured_direction!r}.")
        selected, omitted = _extract_binary_score_records(
            materialized,
            score_field=score_field,
            outcome_field=outcome_field,
            prompt_field=prompt_field,
        )
        point = binary_ranking_metrics(
            [item["outcome"] for item in selected],
            [item["score"] for item in selected],
            higher_is_positive=higher_is_positive,
        )
        summary: dict[str, Any] = {
            "row_type": "metric",
            "score": score_field,
            "outcome": outcome_field,
            "score_direction": "higher" if higher_is_positive else "lower",
            "omitted_record_count": omitted,
            **point,
        }
        if include_ranking_ci and bootstrap_iterations is not None and selected:
            bootstrap = prompt_clustered_ranking_bootstrap(
                materialized,
                score_field=score_field,
                outcome_field=outcome_field,
                prompt_field=prompt_field,
                higher_is_positive=higher_is_positive,
                iterations=int(bootstrap_iterations),
                seed=bootstrap_seed + score_index,
            )
            summary.update({
                key: bootstrap[key]
                for key in (
                    "auroc_ci95_low", "auroc_ci95_high", "auprc_ci95_low", "auprc_ci95_high",
                    "bootstrap_valid_auroc_draw_count", "bootstrap_valid_auprc_draw_count",
                    "cluster_count", "iterations",
                )
            })
        else:
            summary.update({
                "auroc_ci95_low": None,
                "auroc_ci95_high": None,
                "auprc_ci95_low": None,
                "auprc_ci95_high": None,
                "bootstrap_valid_auroc_draw_count": 0,
                "bootstrap_valid_auprc_draw_count": 0,
                "cluster_count": len({item["prompt"] for item in selected}),
                "iterations": bootstrap_iterations,
            })
        rows.append(summary)
        if include_quantile_rows:
            for bin_row in quantile_bin_table(
                materialized,
                score_field=score_field,
                outcome_field=outcome_field,
                prompt_field=prompt_field,
                bin_count=quantile_bins,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed + score_index * 101,
            ):
                rows.append({"row_type": "quantile_bin", **bin_row})
    return rows


def _fixed_bin(value: float, boundaries: Sequence[float]) -> str:
    """Label a numeric value by inclusive-upper fixed cut points."""

    if not boundaries:
        return "all"
    ordered = tuple(float(item) for item in boundaries)
    if tuple(sorted(ordered)) != ordered:
        raise ValueError("bin boundaries must be sorted ascending.")
    for index, upper in enumerate(ordered):
        if value <= upper:
            lower = "-inf" if index == 0 else f"{ordered[index - 1]:g}"
            return f"({lower},{upper:g}]"
    # A malformed confidence outside the configured range remains visible as a
    # distinct stratum instead of being silently folded into the highest bin.
    return f"({ordered[-1]:g},inf)"


def _stratum_key(
    record: Mapping[str, Any],
    *,
    dataset_field: str,
    phase_field: str,
    confidence_field: str,
    anchor_count_field: str,
    confidence_bins: Sequence[float],
    extra_fields: Sequence[str],
) -> tuple[Hashable, ...] | None:
    dataset = field_value(record, dataset_field, MISSING)
    phase = field_value(record, phase_field, MISSING)
    confidence = _optional_finite_float(field_value(record, confidence_field, MISSING))
    anchors = field_value(record, anchor_count_field, MISSING)
    if dataset is MISSING or phase is MISSING or confidence is None or anchors is MISSING:
        return None
    try:
        anchor_count = int(anchors)
    except (TypeError, ValueError):
        return None
    extra: list[Hashable] = []
    for field in extra_fields:
        value = field_value(record, field, MISSING)
        if value is MISSING:
            return None
        extra.append(_freeze_group_value(value))
    return (
        _freeze_group_value(dataset),
        _freeze_group_value(phase),
        _fixed_bin(confidence, confidence_bins),
        anchor_count,
        *extra,
    )


def sample_matched_event_cohort(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    event_field: str = "top1_flip",
    dataset_field: str = "dataset",
    phase_field: str = "phase",
    confidence_field: str = "previous_top1_probability",
    anchor_count_field: str = "source_anchor_count",
    confidence_bins: Sequence[float] = DEFAULT_CONFIDENCE_BINS,
    controls_per_event: int = 1,
    extra_match_fields: Sequence[str] = (),
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replace_controls: bool = False,
) -> dict[str, Any]:
    """Sample flip/non-flip cohorts matched on dataset/phase/confidence/anchors.

    Controls are sampled within exact strata.  When controls are scarce and
    replacement is disabled, the function drops a seed-determined subset of
    events rather than weakening the requested match.  Diagnostics make every
    dropped record and stratum visible to the handoff report.
    """

    controls_per_event = int(controls_per_event)
    if controls_per_event < 1:
        raise ValueError("controls_per_event must be at least one.")
    if not confidence_bins:
        raise ValueError("confidence_bins must not be empty.")
    boundaries = tuple(float(item) for item in confidence_bins)
    if tuple(sorted(boundaries)) != boundaries:
        raise ValueError("confidence_bins must be sorted ascending.")

    events_by_stratum: dict[tuple[Hashable, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    controls_by_stratum: dict[tuple[Hashable, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    missing_stratum_count = 0
    missing_event_count = 0
    for index, record in enumerate(_records(records)):
        raw_event = field_value(record, event_field, MISSING)
        if raw_event is MISSING or raw_event is None:
            missing_event_count += 1
            continue
        stratum = _stratum_key(
            record,
            dataset_field=dataset_field,
            phase_field=phase_field,
            confidence_field=confidence_field,
            anchor_count_field=anchor_count_field,
            confidence_bins=boundaries,
            extra_fields=extra_match_fields,
        )
        if stratum is None:
            missing_stratum_count += 1
            continue
        if _as_binary(raw_event, field=event_field):
            events_by_stratum[stratum].append((index, record))
        else:
            controls_by_stratum[stratum].append((index, record))

    generator = random.Random(seed)
    selected_events: list[dict[str, Any]] = []
    selected_controls: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    all_strata = sorted(set(events_by_stratum) | set(controls_by_stratum), key=_stable_key)
    labels = (dataset_field, phase_field, f"{confidence_field}_bin", anchor_count_field, *extra_match_fields)
    for stratum in all_strata:
        event_rows = list(events_by_stratum.get(stratum, []))
        control_rows = list(controls_by_stratum.get(stratum, []))
        generator.shuffle(event_rows)
        generator.shuffle(control_rows)
        event_available = len(event_rows)
        control_available = len(control_rows)
        if replace_controls:
            event_take = event_available if control_available else 0
        else:
            event_take = min(event_available, control_available // controls_per_event)
        used_events = event_rows[:event_take]
        if replace_controls and control_available:
            used_controls = [control_rows[generator.randrange(control_available)] for _ in range(event_take * controls_per_event)]
        else:
            used_controls = control_rows[: event_take * controls_per_event]
        for _, record in used_events:
            annotated = dict(record)
            annotated.update({"cohort_role": "event", "match_stratum": list(stratum)})
            selected_events.append(annotated)
        for _, record in used_controls:
            annotated = dict(record)
            annotated.update({"cohort_role": "control", "match_stratum": list(stratum)})
            selected_controls.append(annotated)
        diagnostics.append({
            **dict(zip(labels, stratum, strict=True)),
            "available_event_count": event_available,
            "available_control_count": control_available,
            "selected_event_count": len(used_events),
            "selected_control_count": len(used_controls),
            "dropped_event_count": event_available - len(used_events),
            "unmatched": bool(event_available and not used_events),
        })
    # Keep the return stable for artifact diffs while retaining independent
    # random selection inside each stratum.
    selected_events.sort(key=lambda row: _stable_key((row.get("match_stratum"), _stable_key(row))))
    selected_controls.sort(key=lambda row: _stable_key((row.get("match_stratum"), _stable_key(row))))
    return {
        "event_records": selected_events,
        "control_records": selected_controls,
        "records": [*selected_events, *selected_controls],
        "diagnostics": diagnostics,
        "selected_event_count": len(selected_events),
        "selected_control_count": len(selected_controls),
        "available_event_count": sum(len(value) for value in events_by_stratum.values()),
        "available_control_count": sum(len(value) for value in controls_by_stratum.values()),
        "missing_event_field_count": missing_event_count,
        "missing_stratum_field_count": missing_stratum_count,
        "controls_per_event": controls_per_event,
        "replace_controls": replace_controls,
        "match_fields": list(labels),
        "confidence_bins": list(boundaries),
        "seed": seed,
    }


def _transition_identity(
    record: Mapping[str, Any],
    *,
    prompt_field: str,
    position_field: str,
) -> tuple[Hashable, Hashable] | None:
    prompt = field_value(record, prompt_field, MISSING)
    position = field_value(record, position_field, MISSING)
    if prompt is MISSING or position is MISSING:
        return None
    return _freeze_group_value(prompt), _freeze_group_value(position)


def _binary_transition_rows(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    flip_field: str,
    prompt_field: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    omitted = 0
    for index, record in enumerate(_records(records)):
        raw_flip = field_value(record, flip_field, MISSING)
        if raw_flip is MISSING or raw_flip is None:
            omitted += 1
            continue
        prompt = field_value(record, prompt_field, MISSING)
        if prompt is MISSING:
            raise ValueError(f"transition record {index} is missing prompt field {prompt_field!r}.")
        rows.append({"record": record, "prompt": _freeze_group_value(prompt), "flip": _as_binary(raw_flip, field=flip_field)})
    return rows, omitted


def _flip_rate_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int | None,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if iterations is None:
        return None, None
    clusters: dict[Hashable, list[float]] = defaultdict(list)
    for row in rows:
        clusters[row["prompt"]].append(float(row["flip"]))
    return (
        _bootstrap_from_clusters(clusters, iterations=int(iterations), seed=seed, weighting="macro"),
        _bootstrap_from_clusters(clusters, iterations=int(iterations), seed=seed + 1, weighting="micro"),
    )


def grouped_flip_rate_table(
    transitions: Iterable[Mapping[str, Any] | Any],
    *,
    group_fields: Sequence[str] = (),
    flip_field: str = "top1_flip",
    prompt_field: str = "prompt_index",
    bootstrap_iterations: int | None = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return micro and prompt-macro flip-rate rows for arbitrary strata.

    Missing group values form an explicit ``"<missing>"`` bucket.  This keeps
    an incomplete logging schema visible rather than causing hidden changes to
    denominators in a phase/dataset table.
    """

    rows, omitted = _binary_transition_rows(transitions, flip_field=flip_field, prompt_field=prompt_field)
    grouped: dict[tuple[Hashable, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_freeze_group_value(field_value(row["record"], field, MISSING)) for field in group_fields)
        grouped[key].append(row)
    if not group_fields and not grouped:
        grouped[()] = []
    result: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(grouped, key=_stable_key)):
        members = grouped[key]
        transition_count = len(members)
        flip_count = sum(member["flip"] for member in members)
        per_prompt: dict[Hashable, list[bool]] = defaultdict(list)
        for member in members:
            per_prompt[member["prompt"]].append(bool(member["flip"]))
        prompt_rates = [math.fsum(values) / len(values) for values in per_prompt.values() if values]
        macro, micro = _flip_rate_bootstrap(
            members,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + group_index * 11,
        )
        result.append({
            **dict(zip(group_fields, key, strict=True)),
            "transition_count": transition_count,
            "flip_count": flip_count,
            "transition_flip_rate": flip_count / transition_count if transition_count else None,
            "prompt_count": len(per_prompt),
            "prompt_macro_flip_rate": math.fsum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
            "prompt_macro_ci95_low": macro["ci95_low"] if macro else None,
            "prompt_macro_ci95_high": macro["ci95_high"] if macro else None,
            "micro_clustered_ci95_low": micro["ci95_low"] if micro else None,
            "micro_clustered_ci95_high": micro["ci95_high"] if micro else None,
            "bootstrap_iterations": bootstrap_iterations,
            "omitted_transition_count": omitted,
        })
    return result


def _position_trajectory_summary(
    members: Sequence[Mapping[str, Any]],
    *,
    source_step_field: str,
    target_step_field: str,
    previous_top1_field: str,
    next_top1_field: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Reconstruct one position's observed top-1 path from aligned transitions."""

    edges: list[tuple[int, int, Any, Any, bool]] = []
    malformed = 0
    for member in members:
        record = member["record"]
        source = _optional_finite_float(field_value(record, source_step_field, MISSING))
        target = _optional_finite_float(field_value(record, target_step_field, MISSING))
        previous = field_value(record, previous_top1_field, MISSING)
        current = field_value(record, next_top1_field, MISSING)
        if source is None or target is None or previous is MISSING or current is MISSING:
            malformed += 1
            continue
        if int(source) != source or int(target) != target:
            malformed += 1
            continue
        edges.append((int(source), int(target), previous, current, bool(member["flip"])))
    edges.sort(key=lambda item: (item[0], item[1], _stable_key(item[2]), _stable_key(item[3])))
    # Duplicated edge records are data-quality errors.  Retain one deterministic
    # copy for the path diagnostic while accounting for every transition in the
    # aggregate rate above.
    unique_edges: list[tuple[int, int, Any, Any, bool]] = []
    seen_edge_coordinates: set[tuple[int, int]] = set()
    duplicate_edge_count = 0
    for edge in edges:
        coordinate = edge[:2]
        if coordinate in seen_edge_coordinates:
            duplicate_edge_count += 1
            continue
        seen_edge_coordinates.add(coordinate)
        unique_edges.append(edge)

    tokens: dict[int, Any] = {}
    token_conflict_count = 0
    for source, target, previous, current, _ in unique_edges:
        for step, token in ((source, previous), (target, current)):
            if step in tokens and tokens[step] != token:
                token_conflict_count += 1
            else:
                tokens.setdefault(step, token)

    flipback_count = 0
    flipback_eligible_count = 0
    for left, right in zip(unique_edges, unique_edges[1:]):
        source, target, old, middle, _ = left
        next_source, _next_target, repeated_middle, new, _ = right
        if target != next_source or middle != repeated_middle:
            continue
        if old != middle:
            flipback_eligible_count += 1
            if old == new:
                flipback_count += 1

    # Runs are only joined across consecutive observed state indices.  A gap
    # represents a commit/unobserved state and must not become fake survival.
    run_lengths: list[int] = []
    prior_step: int | None = None
    prior_token: Any = MISSING
    current_run = 0
    for step in sorted(tokens):
        token = tokens[step]
        if prior_step is None or step != prior_step + 1 or token != prior_token:
            if current_run:
                run_lengths.append(current_run)
            current_run = 1
        else:
            current_run += 1
        prior_step, prior_token = step, token
    if current_run:
        run_lengths.append(current_run)

    horizon_rows: dict[str, dict[str, int]] = {}
    for horizon in horizons:
        eligible = 0
        retained = 0
        for start in tokens:
            end = start + horizon
            if all(step in tokens for step in range(start, end + 1)):
                eligible += 1
                retained += int(tokens[start] == tokens[end])
        horizon_rows[str(horizon)] = {"eligible_count": eligible, "retained_count": retained}
    return {
        "transition_count": len(members),
        "flip_count": sum(bool(member["flip"]) for member in members),
        "ever_flip": any(bool(member["flip"]) for member in members),
        "observed_state_count": len(tokens),
        "flipback_count": flipback_count,
        "flipback_eligible_count": flipback_eligible_count,
        "mean_top1_run_length_states": math.fsum(run_lengths) / len(run_lengths) if run_lengths else None,
        "mean_top1_survival_steps": (
            math.fsum(length - 1 for length in run_lengths) / len(run_lengths) if run_lengths else None
        ),
        "horizons": horizon_rows,
        "malformed_path_transition_count": malformed,
        "duplicate_edge_count": duplicate_edge_count,
        "token_conflict_count": token_conflict_count,
    }


def aggregate_trajectory_flips(
    transitions: Iterable[Mapping[str, Any] | Any],
    *,
    prompt_field: str = "prompt_index",
    position_field: str = "position",
    flip_field: str = "top1_flip",
    source_step_field: str = "source_step",
    target_step_field: str = "target_step",
    previous_top1_field: str = "previous_top1_token_id",
    next_top1_field: str = "next_top1_token_id",
    horizons: Sequence[int] = (1, 2, 3, 5),
    position_universe: Iterable[Mapping[str, Any] | Any] | None = None,
    bootstrap_iterations: int | None = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Aggregate transition, position, and prompt flip statistics.

    The input should contain only natural transitions whose target position was
    masked in both states.  A ``position_universe`` may additionally include
    positions with no eligible transition (for example, ones committed at the
    first state), which makes the denominator used for ever-flip prevalence
    explicit rather than inferred.
    """

    requested_horizons = tuple(sorted({int(item) for item in horizons}))
    if not requested_horizons or any(item < 1 for item in requested_horizons):
        raise ValueError("horizons must contain positive integers.")
    rows, omitted = _binary_transition_rows(transitions, flip_field=flip_field, prompt_field=prompt_field)
    by_position: dict[tuple[Hashable, Hashable], list[dict[str, Any]]] = defaultdict(list)
    malformed_identity_count = 0
    for row in rows:
        identity = _transition_identity(row["record"], prompt_field=prompt_field, position_field=position_field)
        if identity is None:
            malformed_identity_count += 1
            continue
        by_position[identity].append(row)

    universe: set[tuple[Hashable, Hashable]] = set(by_position)
    if position_universe is not None:
        for record in _records(position_universe):
            identity = _transition_identity(record, prompt_field=prompt_field, position_field=position_field)
            if identity is not None:
                universe.add(identity)
    position_rows: list[dict[str, Any]] = []
    for identity in sorted(universe, key=_stable_key):
        prompt, position = identity
        summary = _position_trajectory_summary(
            by_position.get(identity, []),
            source_step_field=source_step_field,
            target_step_field=target_step_field,
            previous_top1_field=previous_top1_field,
            next_top1_field=next_top1_field,
            horizons=requested_horizons,
        ) if identity in by_position else {
            "transition_count": 0,
            "flip_count": 0,
            "ever_flip": False,
            "observed_state_count": 0,
            "flipback_count": 0,
            "flipback_eligible_count": 0,
            "mean_top1_run_length_states": None,
            "mean_top1_survival_steps": None,
            "horizons": {str(horizon): {"eligible_count": 0, "retained_count": 0} for horizon in requested_horizons},
            "malformed_path_transition_count": 0,
            "duplicate_edge_count": 0,
            "token_conflict_count": 0,
        }
        position_rows.append({prompt_field: prompt, position_field: position, **summary})

    transition_count = len(rows)
    flip_count = sum(row["flip"] for row in rows)
    prompt_rows: list[dict[str, Any]] = []
    by_prompt: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt"]].append(row)
    for prompt in sorted(by_prompt, key=_stable_key):
        prompt_members = by_prompt[prompt]
        prompt_flip_count = sum(member["flip"] for member in prompt_members)
        prompt_rows.append({
            prompt_field: prompt,
            "transition_count": len(prompt_members),
            "flip_count": prompt_flip_count,
            "transition_flip_rate": prompt_flip_count / len(prompt_members) if prompt_members else None,
            "ever_flip": bool(prompt_flip_count),
        })
    prompt_rates = [row["transition_flip_rate"] for row in prompt_rows if row["transition_flip_rate"] is not None]
    macro_bootstrap, micro_bootstrap = _flip_rate_bootstrap(
        rows,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    position_count = len(position_rows)
    ever_flip_positions = sum(bool(row["ever_flip"]) for row in position_rows)
    flipback_count = sum(int(row["flipback_count"]) for row in position_rows)
    flipback_eligible_count = sum(int(row["flipback_eligible_count"]) for row in position_rows)
    horizon_summary: dict[str, dict[str, Any]] = {}
    for horizon in requested_horizons:
        eligible = sum(row["horizons"][str(horizon)]["eligible_count"] for row in position_rows)
        retained = sum(row["horizons"][str(horizon)]["retained_count"] for row in position_rows)
        horizon_summary[str(horizon)] = {
            "horizon": horizon,
            "eligible_count": eligible,
            "retained_count": retained,
            "retention_rate": retained / eligible if eligible else None,
        }
    return {
        "micro": {
            "transition_count": transition_count,
            "flip_count": flip_count,
            "transition_flip_rate": flip_count / transition_count if transition_count else None,
            "clustered_ci95_low": micro_bootstrap["ci95_low"] if micro_bootstrap else None,
            "clustered_ci95_high": micro_bootstrap["ci95_high"] if micro_bootstrap else None,
        },
        "macro": {
            "prompt_count": len(prompt_rows),
            "prompt_mean_transition_flip_rate": math.fsum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
            "clustered_ci95_low": macro_bootstrap["ci95_low"] if macro_bootstrap else None,
            "clustered_ci95_high": macro_bootstrap["ci95_high"] if macro_bootstrap else None,
        },
        "position": {
            "position_count": position_count,
            "ever_flip_position_count": ever_flip_positions,
            "position_ever_flip_rate": ever_flip_positions / position_count if position_count else None,
            "mean_flips_per_position": flip_count / position_count if position_count else None,
        },
        "flipback": {
            "flipback_count": flipback_count,
            "flipback_eligible_count": flipback_eligible_count,
            "flipback_rate": flipback_count / flipback_eligible_count if flipback_eligible_count else None,
        },
        "horizon_retention": horizon_summary,
        "position_rows": position_rows,
        "prompt_rows": prompt_rows,
        "omitted_transition_count": omitted,
        "malformed_identity_transition_count": malformed_identity_count,
        "bootstrap_iterations": bootstrap_iterations,
        # Flat aliases make summary templating less error-prone for runners.
        "transition_flip_rate": flip_count / transition_count if transition_count else None,
        "position_ever_flip_rate": ever_flip_positions / position_count if position_count else None,
    }


def _topk_membership(
    record: Mapping[str, Any],
    *,
    prefix: str,
    rank_field: str,
    k: int,
) -> bool | None:
    """Read stored top-k membership or derive it from a one-based rank."""

    value = field_value(record, f"{prefix}{k}", MISSING)
    if value is not MISSING and value is not None:
        return _as_binary(value, field=f"{prefix}{k}")
    rank = _optional_finite_float(field_value(record, rank_field, MISSING))
    if rank is None or rank < 1:
        return None
    return rank <= k


def _group_records(
    records: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
) -> dict[tuple[Hashable, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Hashable, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(_freeze_group_value(field_value(record, field, MISSING)) for field in group_fields)
        groups[key].append(record)
    if not group_fields and not groups:
        groups[()] = []
    return groups


def _coverage_row(
    members: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    group_key: tuple[Hashable, ...],
    kind: str,
    k: int,
    prefix: str,
    rank_field: str,
    prompt_field: str,
    bootstrap_iterations: int | None,
    bootstrap_seed: int,
) -> dict[str, Any]:
    covered_records: list[dict[str, Any]] = []
    unavailable = 0
    for record in members:
        covered = _topk_membership(record, prefix=prefix, rank_field=rank_field, k=k)
        if covered is None:
            unavailable += 1
            continue
        prompt = field_value(record, prompt_field, MISSING)
        if prompt is MISSING:
            raise ValueError(f"future-token record is missing prompt field {prompt_field!r}.")
        covered_records.append({prompt_field: _freeze_group_value(prompt), "covered": covered})
    count = len(covered_records)
    covered_count = sum(item["covered"] for item in covered_records)
    if bootstrap_iterations is None or not covered_records:
        macro = None
        micro = None
    else:
        macro = prompt_clustered_bootstrap(
            covered_records,
            value_field="covered",
            prompt_field=prompt_field,
            iterations=int(bootstrap_iterations),
            seed=bootstrap_seed,
            weighting="macro",
        )
        micro = prompt_clustered_bootstrap(
            covered_records,
            value_field="covered",
            prompt_field=prompt_field,
            iterations=int(bootstrap_iterations),
            seed=bootstrap_seed + 1,
            weighting="micro",
        )
    return {
        **dict(zip(group_fields, group_key, strict=True)),
        "row_type": "topk_coverage",
        "coverage_kind": kind,
        "k": k,
        "observation_count": count,
        "covered_count": covered_count,
        "coverage_rate": covered_count / count if count else None,
        "prompt_macro_coverage": macro["estimate"] if macro else None,
        "prompt_macro_ci95_low": macro["ci95_low"] if macro else None,
        "prompt_macro_ci95_high": macro["ci95_high"] if macro else None,
        "micro_clustered_ci95_low": micro["ci95_low"] if micro else None,
        "micro_clustered_ci95_high": micro["ci95_high"] if micro else None,
        "unavailable_count": unavailable,
        "bootstrap_iterations": bootstrap_iterations,
    }


def future_token_topk_table(
    observations: Iterable[Mapping[str, Any] | Any],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    group_fields: Sequence[str] = (),
    prompt_field: str = "prompt_index",
    eventual_prefix: str = "eventual_in_top",
    eventual_rank_field: str = "eventual_token_rank",
    next_prefix: str = "next_step_top1_in_current_top",
    next_rank_field: str = "next_step_top1_rank_in_current",
    include_next_step: bool = True,
    bootstrap_iterations: int | None = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Create eventual-token and next-step top-k coverage tables.

    Eventual-token rows include all recorded masked states; next-step rows
    automatically exclude observations with no next state (for example, a
    position committed on that step).  Membership booleans from trajectory
    replay are preferred, with one-based ranks as a portable fallback.
    """

    requested_ks = tuple(sorted({int(item) for item in top_ks}))
    if not requested_ks or any(item < 1 for item in requested_ks):
        raise ValueError("top_ks must contain positive integers.")
    records = _records(observations)
    groups = _group_records(records, group_fields)
    result: list[dict[str, Any]] = []
    row_number = 0
    for group_key in sorted(groups, key=_stable_key):
        members = groups[group_key]
        for k in requested_ks:
            result.append(_coverage_row(
                members,
                group_fields=group_fields,
                group_key=group_key,
                kind="eventual_committed_token",
                k=k,
                prefix=eventual_prefix,
                rank_field=eventual_rank_field,
                prompt_field=prompt_field,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed + row_number * 7,
            ))
            row_number += 1
            if include_next_step:
                result.append(_coverage_row(
                    members,
                    group_fields=group_fields,
                    group_key=group_key,
                    kind="next_step_top1",
                    k=k,
                    prefix=next_prefix,
                    rank_field=next_rank_field,
                    prompt_field=prompt_field,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed + row_number * 7,
                ))
                row_number += 1
    return result


def future_token_rank_table(
    observations: Iterable[Mapping[str, Any] | Any],
    *,
    group_fields: Sequence[str] = (),
    rank_field: str = "eventual_token_rank",
    current_matches_field: str = "current_top1_matches_eventual",
    prompt_field: str = "prompt_index",
    bootstrap_iterations: int | None = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Describe where the eventual committed token sits when current top-1 disagrees.

    ``rank_exact`` rows are mutually exclusive (2, 3, 4-5, and >5); the
    cumulative rows answer the protocol's top-2/top-3/top-5/outside-top-5
    questions directly.  They are deliberately labeled as eventual baseline
    tokens, not ground-truth tokens.
    """

    records = _records(observations)
    groups = _group_records(records, group_fields)
    rows: list[dict[str, Any]] = []
    categories: tuple[tuple[str, Callable[[int], bool]], ...] = (
        ("rank_2", lambda rank: rank == 2),
        ("rank_3", lambda rank: rank == 3),
        ("rank_4_to_5", lambda rank: 4 <= rank <= 5),
        ("outside_top5", lambda rank: rank > 5),
        ("within_top2_cumulative", lambda rank: rank <= 2),
        ("within_top3_cumulative", lambda rank: rank <= 3),
        ("within_top5_cumulative", lambda rank: rank <= 5),
    )
    sequence = 0
    for group_key in sorted(groups, key=_stable_key):
        mismatch_rows: list[tuple[dict[str, Any], int]] = []
        unavailable = 0
        for record in groups[group_key]:
            matches = field_value(record, current_matches_field, MISSING)
            rank_value = _optional_finite_float(field_value(record, rank_field, MISSING))
            if matches is MISSING or matches is None or rank_value is None or rank_value < 1:
                unavailable += 1
                continue
            if _as_binary(matches, field=current_matches_field):
                continue
            mismatch_rows.append((record, int(rank_value)))
        for category, predicate in categories:
            binary_records: list[dict[str, Any]] = []
            for record, rank in mismatch_rows:
                prompt = field_value(record, prompt_field, MISSING)
                if prompt is MISSING:
                    raise ValueError(f"future-token record is missing prompt field {prompt_field!r}.")
                binary_records.append({prompt_field: _freeze_group_value(prompt), "member": predicate(rank)})
            count = len(binary_records)
            member_count = sum(item["member"] for item in binary_records)
            macro = prompt_clustered_bootstrap(
                binary_records,
                value_field="member",
                prompt_field=prompt_field,
                iterations=int(bootstrap_iterations),
                seed=bootstrap_seed + sequence,
                weighting="macro",
            ) if bootstrap_iterations is not None and binary_records else None
            rows.append({
                **dict(zip(group_fields, group_key, strict=True)),
                "row_type": "eventual_rank_when_current_top1_differs",
                "rank_category": category,
                "observation_count": count,
                "member_count": member_count,
                "rate": member_count / count if count else None,
                "prompt_macro_rate": macro["estimate"] if macro else None,
                "ci95_low": macro["ci95_low"] if macro else None,
                "ci95_high": macro["ci95_high"] if macro else None,
                "unavailable_count": unavailable,
                "bootstrap_iterations": bootstrap_iterations,
            })
            sequence += 1
    return rows


def counterfactual_attribution_table(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    classification_field: str = "classification",
    group_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Summarize exact-audit classes without assuming a particular runner class.

    Records emitted by ``CounterfactualAttribution.as_dict()`` work directly.
    A reconstruction mismatch is kept as its own class/flag and must not be
    interpreted as a causal attribution result.
    """

    groups = _group_records(_records(records), group_fields)
    output: list[dict[str, Any]] = []
    for group_key in sorted(groups, key=_stable_key):
        members = groups[group_key]
        classes: Counter[str] = Counter()
        reconstruction_mismatch = 0
        for record in members:
            classification = field_value(record, classification_field, MISSING)
            classes[str(classification) if classification is not MISSING and classification is not None else "missing"] += 1
            matched = field_value(record, "reconstruction_matches", MISSING)
            if matched is not MISSING and matched is not None and not _as_binary(matched, field="reconstruction_matches"):
                reconstruction_mismatch += 1
        total = len(members)
        for classification in sorted(classes):
            count = classes[classification]
            output.append({
                **dict(zip(group_fields, group_key, strict=True)),
                "classification": classification,
                "event_count": count,
                "total_event_count": total,
                "classification_rate": count / total if total else None,
                "reconstruction_mismatch_count": reconstruction_mismatch,
                "reconstruction_mismatch_rate": reconstruction_mismatch / total if total else None,
            })
    return output


def _replay_state_key(record: Mapping[str, Any], state_fields: Sequence[str]) -> tuple[Hashable, ...]:
    # Parent indexes are only a fallback: including them alongside a real
    # event/state ID would split already-flattened order rows into fake states.
    ordinary = tuple(
        (field, _freeze_group_value(value))
        for field in state_fields
        if field != "_report_parent_index"
        for value in (field_value(record, field, MISSING),)
        if value is not MISSING
    )
    if ordinary:
        return ordinary
    parent_index = field_value(record, "_report_parent_index", MISSING)
    if parent_index is not MISSING:
        return ("_report_parent_index", _freeze_group_value(parent_index))
    # A no-ID record cannot be compared across rows without risking a false
    # order effect.  Its unique serialized representation is conservative.
    return ("<unidentified>", _stable_key(record))


def _anchor_positions_from_replay(record: Mapping[str, Any]) -> set[int]:
    """Read original anchor positions from either public replay representation."""

    order = field_value(record, "order", [])
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        return set()
    positions: set[int] = set()
    for item in order:
        if isinstance(item, Mapping):
            value = field_value(item, "position", MISSING)
        elif hasattr(item, "position"):
            value = getattr(item, "position")
        else:
            value = item
        try:
            positions.add(int(value))
        except (TypeError, ValueError):
            # A malformed order should not make an unrelated table writer
            # crash; retain all tracked positions in that rare fallback.
            return set()
    return positions


def _non_anchor_tracked_signature(
    step: Mapping[str, Any], anchor_positions: set[int]
) -> tuple[tuple[int, Any], ...]:
    """Canonical non-anchor tracked decisions for one replay prefix.

    At a given prefix, different orders have committed different original
    anchors.  Comparing those disappearing keys would manufacture an order
    effect even if every genuinely comparable decision is identical.
    """

    tracked = field_value(step, "tracked_top1", MISSING)
    if not isinstance(tracked, Mapping):
        return ()
    values: list[tuple[int, Any]] = []
    for raw_position, token in tracked.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        if position not in anchor_positions:
            values.append((position, _freeze_group_value(token)))
    return tuple(sorted(values, key=lambda item: item[0]))


def _fixed_replay_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order-invariant fixed-replay signature for meaningful decisions only."""

    steps = field_value(record, "steps", [])
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return ()
    anchor_positions = _anchor_positions_from_replay(record)
    anchor_decisions: list[tuple[Any, ...]] = []
    non_anchor_trajectories: list[tuple[Any, ...]] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        # Compare the decision encountered *when the same original anchor is
        # about to be committed*, not the arbitrary sequence slot in which it
        # appeared.  This directly answers whether an order changed whether
        # the original token remained top-1 / threshold eligible.
        anchor_decisions.append((
            _freeze_group_value(field_value(step, "committed_position", MISSING)),
            _freeze_group_value(field_value(step, "top1_token_id", MISSING)),
            _freeze_group_value(field_value(step, "original_token_is_top1", MISSING)),
            _freeze_group_value(field_value(step, "original_token_threshold_eligible", MISSING)),
        ))
        non_anchor_trajectories.append((
            _freeze_group_value(field_value(step, "prefix_size", MISSING)),
            _non_anchor_tracked_signature(step, anchor_positions),
        ))
    return (
        tuple(sorted(anchor_decisions, key=_stable_key)),
        tuple(sorted(non_anchor_trajectories, key=_stable_key)),
    )


def _adaptive_final_assignment_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    assignments = field_value(record, "final_assignments", MISSING)
    if assignments is not MISSING and assignments is not None:
        return (_freeze_group_value(assignments),)
    steps = field_value(record, "steps", [])
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return ()
    return tuple(sorted((
        (
            _freeze_group_value(field_value(step, "committed_position", MISSING)),
            _freeze_group_value(_first_field(step, ("selected_token_id", "token_id"), MISSING)),
        )
        for step in steps if isinstance(step, Mapping)
    ), key=_stable_key))


def _adaptive_replay_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Include both adaptive final assignments and non-anchor trajectories."""

    steps = field_value(record, "steps", [])
    anchor_positions = _anchor_positions_from_replay(record)
    trajectory = ()
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        trajectory = tuple(
            (
                _freeze_group_value(field_value(step, "prefix_size", MISSING)),
                _non_anchor_tracked_signature(step, anchor_positions),
            )
            for step in steps if isinstance(step, Mapping)
        )
    return _adaptive_final_assignment_signature(record), trajectory


def flatten_order_replays(
    audits_or_replays: Iterable[Mapping[str, Any] | Any],
) -> list[dict[str, Any]]:
    """Flatten ``OrderReplayAudit.as_dict()`` records into per-order rows.

    Passing already-flattened fixed/adaptive rows is also supported.  Parent
    metadata (for example dataset, prompt, phase, or an event ID added by the
    runner) is copied onto each child so table stratification remains possible.
    ``_report_parent_index`` is an internal stable grouping fallback when a
    caller did not provide a state/event identifier.
    """

    output: list[dict[str, Any]] = []
    for parent_index, raw in enumerate(audits_or_replays):
        parent = _record_mapping(raw)
        fixed = field_value(parent, "fixed_replays", MISSING)
        adaptive = field_value(parent, "adaptive_replays", MISSING)
        if not isinstance(fixed, Sequence) or isinstance(fixed, (str, bytes)):
            fixed = []
        if not isinstance(adaptive, Sequence) or isinstance(adaptive, (str, bytes)):
            adaptive = []
        if fixed or adaptive:
            metadata = {
                key: value for key, value in parent.items()
                if key not in {"fixed_replays", "adaptive_replays"}
            }
            for mode, child_rows in (("fixed", fixed), ("adaptive", adaptive)):
                for child in child_rows:
                    row = {**metadata, **_record_mapping(child)}
                    row["mode"] = mode
                    row.setdefault("_report_parent_index", parent_index)
                    output.append(row)
            continue
        row = dict(parent)
        row.setdefault("_report_parent_index", parent_index)
        output.append(row)
    return output


def order_sensitivity_table(
    replays: Iterable[Mapping[str, Any] | Any],
    *,
    state_fields: Sequence[str] = (
        "event_id", "state_id", "_report_parent_index", "prompt_index", "source_step", "target_position", "position",
    ),
    group_fields: Sequence[str] = (),
    mode_field: str = "mode",
) -> list[dict[str, Any]]:
    """Summarize whether different anchor orders change intermediate decisions.

    Fixed replay compares ordered prefix decisions; adaptive replay compares
    final assignments (or per-step selected assignments if the former is not
    stored).  It never calls fixed replay's final-context equality an order
    effect, consistent with the experiment's interpretation rule.
    """

    grouped: dict[tuple[tuple[Hashable, ...], str], dict[tuple[Hashable, ...], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in flatten_order_replays(replays):
        mode = str(field_value(record, mode_field, "unknown"))
        group_key = tuple(_freeze_group_value(field_value(record, field, MISSING)) for field in group_fields)
        grouped[(group_key, mode)][_replay_state_key(record, state_fields)].append(record)
    output: list[dict[str, Any]] = []
    for (group_key, mode) in sorted(grouped, key=lambda item: (_stable_key(item[0]), item[1])):
        states = grouped[(group_key, mode)]
        evaluated_states = 0
        sensitive_states = 0
        final_assignment_sensitive_states = 0
        decision_trajectory_sensitive_states = 0
        total_orders = 0
        for _state_key, records_for_state in states.items():
            # One order cannot establish order sensitivity, so it is recorded
            # but excluded from the denominator.
            total_orders += len(records_for_state)
            if len(records_for_state) < 2:
                continue
            evaluated_states += 1
            signatures = {
                _fixed_replay_signature(record) if mode == "fixed" else _adaptive_replay_signature(record)
                for record in records_for_state
            }
            if len(signatures) > 1:
                sensitive_states += 1
            if mode == "adaptive":
                final_signatures = {
                    _adaptive_final_assignment_signature(record) for record in records_for_state
                }
                trajectory_signatures = {
                    _adaptive_replay_signature(record)[1] for record in records_for_state
                }
                final_assignment_sensitive_states += int(len(final_signatures) > 1)
                decision_trajectory_sensitive_states += int(len(trajectory_signatures) > 1)
        output.append({
            **dict(zip(group_fields, group_key, strict=True)),
            "mode": mode,
            "state_count": len(states),
            "evaluated_state_count": evaluated_states,
            "order_sensitive_state_count": sensitive_states,
            "order_sensitivity_rate": sensitive_states / evaluated_states if evaluated_states else None,
            "order_replay_count": total_orders,
            "insufficient_order_state_count": len(states) - evaluated_states,
            "final_assignment_order_sensitive_state_count": (
                final_assignment_sensitive_states if mode == "adaptive" else None
            ),
            "final_assignment_order_sensitivity_rate": (
                final_assignment_sensitive_states / evaluated_states if evaluated_states and mode == "adaptive" else None
            ),
            "decision_trajectory_order_sensitive_state_count": (
                decision_trajectory_sensitive_states if mode == "adaptive" else None
            ),
            "decision_trajectory_order_sensitivity_rate": (
                decision_trajectory_sensitive_states / evaluated_states if evaluated_states and mode == "adaptive" else None
            ),
        })
    return output


def _json_safe(value: Any) -> Any:
    """Recursively turn public-record values into strict JSON-safe values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    # NumPy/PyTorch scalar-like values commonly expose ``item``.  Do not import
    # either dependency just to serialize a scalar.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _tabular_value(value: Any) -> Any:
    value = _json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _materialized_table_rows(records: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
    return [_record_mapping(record) for record in records]


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any] | Any]) -> Path:
    """Write compact JSONL with strict JSON values and UTF-8 text."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(_record_mapping(record)), ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
    return output


def write_csv(path: str | Path, records: Iterable[Mapping[str, Any] | Any]) -> Path:
    """Write heterogeneous report rows as a deterministic union-column CSV."""

    output = Path(path)
    rows = _materialized_table_rows(records)
    fields = sorted({str(field) for row in rows for field in row})
    if not fields:
        fields = ["status"]
        rows = [{"status": "no_records"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _tabular_value(row.get(field)) for field in fields})
    return output


def write_parquet(path: str | Path, records: Iterable[Mapping[str, Any] | Any]) -> Path:
    """Write a compact Parquet table, failing clearly if ``pyarrow`` is absent."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OptionalDependencyError(
            "Parquet output requires the optional 'pyarrow' dependency. "
            "Install pyarrow or request CSV/JSONL artifacts instead."
        ) from error
    output = Path(path)
    rows = _materialized_table_rows(records)
    if not rows:
        rows = [{"status": "no_records"}]
    tabular_rows = [{str(key): _tabular_value(value) for key, value in row.items()} for row in rows]
    try:
        table = pa.Table.from_pylist(tabular_rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output)
    except Exception as error:  # pragma: no cover - pyarrow version/schema dependent
        raise ArtifactWriteError(f"Could not write Parquet artifact {output}: {error}") from error
    return output


def write_artifact_tables(
    output_dir: str | Path,
    tables: Mapping[str, Iterable[Mapping[str, Any] | Any]],
    *,
    formats: Sequence[Literal["csv", "jsonl", "parquet"]] = ("csv",),
) -> dict[str, list[Path]]:
    """Write named report tables in one or more explicit artifact formats.

    The function never silently substitutes CSV for requested Parquet: a
    missing optional dependency is an actionable provenance/configuration
    failure, not a successful run with an unexpected artifact type.
    """

    destination = Path(output_dir)
    requested = tuple(dict.fromkeys(formats))
    if not requested:
        raise ValueError("formats must contain at least one artifact format.")
    invalid = set(requested).difference({"csv", "jsonl", "parquet"})
    if invalid:
        raise ValueError(f"Unsupported artifact formats: {sorted(invalid)!r}.")
    output: dict[str, list[Path]] = {}
    for name, records in tables.items():
        stem = str(name)
        candidate = Path(stem)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"table name must be a relative artifact stem, got {name!r}.")
        rows = _materialized_table_rows(records)
        paths: list[Path] = []
        for format_name in requested:
            path = destination / f"{stem}.{format_name}"
            if format_name == "csv":
                paths.append(write_csv(path, rows))
            elif format_name == "jsonl":
                paths.append(write_jsonl(path, rows))
            else:
                paths.append(write_parquet(path, rows))
        output[stem] = paths
    return output


# A semantic alias reads naturally in a runner that also writes raw JSONL
# artifacts.  Keeping one implementation prevents subtle format divergence.
write_report_tables = write_artifact_tables


def _matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OptionalDependencyError(
            "Figure generation requires the optional 'matplotlib' dependency. "
            "Install matplotlib or omit figure generation."
        ) from error
    return pyplot


def _first_present_field(records: Sequence[Mapping[str, Any]], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if any(_optional_finite_float(field_value(record, candidate, MISSING)) is not None for record in records):
            return candidate
    return None


def _save_figure(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return path


def write_basic_figures(
    output_dir: str | Path,
    *,
    transitions: Iterable[Mapping[str, Any] | Any] | None = None,
    future_token_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    quantile_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    counterfactual_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    order_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    flip_field: str = "top1_flip",
    prompt_field: str = "prompt_index",
) -> dict[str, Any]:
    """Write a small, headless-safe set of figures when compatible data exists.

    This is intentionally a baseline visualization layer, not a plotting
    framework.  It covers the report's most decision-relevant plots (flip
    rates, score distributions/quantiles, future-token coverage, causal class,
    and order sensitivity) and safely skips a plot whose input is unavailable.
    """

    destination = Path(output_dir)
    transition_rows = _records(transitions or [])
    future_rows = _records(future_token_rows or [])
    quantiles = _records(quantile_rows or [])
    causal_rows = _records(counterfactual_rows or [])
    orders = _records(order_rows or [])
    if not any((transition_rows, future_rows, quantiles, causal_rows, orders)):
        return {"written": [], "skipped": ["No records available for figure generation."]}
    pyplot = _matplotlib_pyplot()
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[str] = []

    if transition_rows:
        dataset_field = "dataset" if any(field_value(row, "dataset", MISSING) is not MISSING for row in transition_rows) else None
        rate_rows = grouped_flip_rate_table(
            transition_rows,
            group_fields=(dataset_field,) if dataset_field else (),
            flip_field=flip_field,
            prompt_field=prompt_field,
            bootstrap_iterations=None,
        )
        if rate_rows:
            overall_rows = grouped_flip_rate_table(
                transition_rows,
                group_fields=(),
                flip_field=flip_field,
                prompt_field=prompt_field,
                bootstrap_iterations=None,
            )
            display_rows = [
                *(({**overall_rows[0], dataset_field: "all"},) if dataset_field and overall_rows else ()),
                *rate_rows,
            ]
            labels = [str(row.get(dataset_field, "all")) if dataset_field else "all" for row in display_rows]
            values = [row["transition_flip_rate"] or 0.0 for row in display_rows]
            figure, axis = pyplot.subplots(figsize=(max(4.0, len(labels) * 1.1), 3.4))
            axis.bar(labels, values, color="#3b82f6")
            axis.set_ylabel("top-1 flip rate")
            axis.set_ylim(0.0, max(0.05, min(1.0, max(values, default=0.0) * 1.15)))
            axis.set_title("Natural top-1 flip rate")
            written.append(_save_figure(figure, destination / "flip_rates.png"))
            pyplot.close(figure)

        phase_field = "phase" if any(field_value(row, "phase", MISSING) is not MISSING for row in transition_rows) else None
        if phase_field:
            phase_rows = grouped_flip_rate_table(
                transition_rows,
                group_fields=(phase_field,),
                flip_field=flip_field,
                prompt_field=prompt_field,
                bootstrap_iterations=None,
            )
            labels = [str(row[phase_field]) for row in phase_rows]
            values = [row["transition_flip_rate"] or 0.0 for row in phase_rows]
            figure, axis = pyplot.subplots(figsize=(max(4.0, len(labels) * 1.1), 3.4))
            axis.plot(labels, values, marker="o", color="#2563eb")
            axis.set_ylim(bottom=0.0)
            axis.set_ylabel("top-1 flip rate")
            axis.set_title("Flip rate by decoding phase")
            written.append(_save_figure(figure, destination / "flip_rate_by_phase.png"))
            pyplot.close(figure)

        feature_field = _first_present_field(
            transition_rows,
            ("entropy", "previous_entropy", "source_entropy", "normalized_entropy", "logit_margin", "previous_logit_margin"),
        )
        if feature_field is not None:
            flip_values: list[float] = []
            stable_values: list[float] = []
            for row in transition_rows:
                feature = _optional_finite_float(field_value(row, feature_field, MISSING))
                raw_flip = field_value(row, flip_field, MISSING)
                if feature is None or raw_flip is MISSING or raw_flip is None:
                    continue
                (flip_values if _as_binary(raw_flip, field=flip_field) else stable_values).append(feature)
            if flip_values or stable_values:
                figure, axis = pyplot.subplots(figsize=(5.0, 3.4))
                if stable_values:
                    axis.hist(stable_values, bins="auto", density=True, alpha=0.55, label="non-flip", color="#94a3b8")
                if flip_values:
                    axis.hist(flip_values, bins="auto", density=True, alpha=0.55, label="flip", color="#ef4444")
                axis.set_xlabel(feature_field)
                axis.set_ylabel("density")
                axis.set_title(f"{feature_field}: flip vs non-flip")
                axis.legend()
                written.append(_save_figure(figure, destination / "flip_feature_distribution.png"))
                pyplot.close(figure)

    if future_rows:
        coverage = [
            row for row in future_rows
            if row.get("coverage_kind") == "eventual_committed_token" and row.get("coverage_rate") is not None
        ]
        # Prefer the all-data group if grouped rows are present.  Otherwise
        # retain the first available group, which is still explicitly plotted.
        if coverage:
            all_rows = [row for row in coverage if row.get("dataset") is None]
            if all_rows:
                coverage = all_rows
            coverage.sort(key=lambda row: int(row.get("k", 0)))
            figure, axis = pyplot.subplots(figsize=(4.8, 3.4))
            axis.plot([row["k"] for row in coverage], [row["coverage_rate"] for row in coverage], marker="o", color="#16a34a")
            axis.set_xticks([row["k"] for row in coverage])
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("current top-K")
            axis.set_ylabel("eventual-token coverage")
            axis.set_title("Eventual committed token in current top-K")
            written.append(_save_figure(figure, destination / "future_token_topk_coverage.png"))
            pyplot.close(figure)

    if quantiles:
        bin_rows = [row for row in quantiles if row.get("row_type") in {None, "quantile_bin"} and row.get("empirical_probability") is not None]
        if bin_rows:
            first_score = str(bin_rows[0].get("score", "score"))
            selected = [row for row in bin_rows if str(row.get("score", "score")) == first_score]
            selected.sort(key=lambda row: int(row.get("bin_index", 0)))
            figure, axis = pyplot.subplots(figsize=(5.0, 3.4))
            axis.plot(
                [row["bin_index"] for row in selected],
                [row["empirical_probability"] for row in selected],
                marker="o", color="#7c3aed",
            )
            axis.set_xlabel("score quantile bin")
            axis.set_ylabel("empirical flip probability")
            axis.set_title(f"Future-flip probability by {first_score} bin")
            written.append(_save_figure(figure, destination / "flip_probability_by_quantile.png"))
            pyplot.close(figure)

    if causal_rows:
        all_rows = [row for row in causal_rows if row.get("dataset") is None]
        if all_rows:
            causal_rows = all_rows
        counts: Counter[str] = Counter()
        for row in causal_rows:
            label = row.get("classification")
            if label is not None:
                counts[str(label)] += int(row.get("event_count", 1))
        if counts:
            labels = sorted(counts)
            figure, axis = pyplot.subplots(figsize=(max(5.0, len(labels) * 1.15), 3.4))
            axis.bar(labels, [counts[label] for label in labels], color="#f59e0b")
            axis.tick_params(axis="x", rotation=30)
            axis.set_ylabel("exact-audit events")
            axis.set_title("Counterfactual flip attribution")
            written.append(_save_figure(figure, destination / "counterfactual_attribution.png"))
            pyplot.close(figure)

    if orders:
        all_rows = [row for row in orders if row.get("anchor_set_size") is None]
        if all_rows:
            orders = all_rows
        evaluable = [row for row in orders if row.get("order_sensitivity_rate") is not None]
        if evaluable:
            labels = [str(row.get("mode", "unknown")) for row in evaluable]
            values = [float(row["order_sensitivity_rate"]) for row in evaluable]
            figure, axis = pyplot.subplots(figsize=(max(4.0, len(labels) * 1.2), 3.4))
            axis.bar(labels, values, color="#db2777")
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("order-sensitive state rate")
            axis.set_title("Anchor-order sensitivity")
            written.append(_save_figure(figure, destination / "order_sensitivity.png"))
            pyplot.close(figure)

    if not written:
        skipped.append("No compatible non-empty table was available for a baseline figure.")
    return {"written": [str(path) for path in written], "skipped": skipped}
