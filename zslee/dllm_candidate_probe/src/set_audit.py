"""Artifact validation, scalar caching, and statistics for hard-set audits.

This module is intentionally framework-light so raw pilot artifacts can be
validated before a GPU model is loaded.  Cached entries hold only scalar target
probabilities keyed by the complete counterfactual description; they never hold
logits, hidden states, attention, or KV data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

from src.hard_set_generation import AuditCandidate, PairMetric, pair_key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required pilot artifact is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path}:{line_number}.")
            records.append(payload)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    if not fields:
        fields = ["status"]
        rows = [{"status": "no_records"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state_id(record: Mapping[str, Any]) -> str:
    """Stable identity tied to full token IDs and the base mask positions."""

    descriptor = {
        "prompt_index": record["prompt_index"],
        "step": record["step"],
        "token_sequence": record["token_sequence"],
        "mask_positions": record["mask_positions"],
    }
    packed = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()[:20]


def _require_fields(record: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def validate_pilot_artifacts(probe_root: Path, reuse: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fail closed on artifacts that cannot support a position-aware audit."""

    states = read_jsonl(probe_root / str(reuse["states"]))
    pairs = read_jsonl(probe_root / str(reuse["pairs"]))
    _ = read_jsonl(probe_root / str(reuse["sets"]))  # existence is part of provenance validation
    result_path = probe_root / str(reuse["result"])
    if not result_path.exists():
        raise FileNotFoundError(f"Required pilot result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "success":
        raise ValueError(f"Pilot result status is not success: {result.get('status')!r}")
    if not states or not pairs:
        raise ValueError("Pilot artifacts have no states or pairs to audit.")
    for record in states:
        _require_fields(record, ("prompt_index", "step", "token_sequence", "mask_positions", "position_summaries"), "pilot state")
    for record in pairs:
        _require_fields(record, ("prompt_index", "step", "a", "b", "pair_stability_q2", "a_to_b_lift", "b_to_a_lift"), "pilot pair")
        _require_fields(record["a"], ("position", "token_id"), "pilot pair a")
        _require_fields(record["b"], ("position", "token_id"), "pilot pair b")
    return states, pairs, result


def provenance_validation(probe_root: Path, reuse: Mapping[str, Any]) -> dict[str, Any]:
    """Check the pilot summary pin when available without conflating it with raw data."""

    summary_path = probe_root / "outputs" / "summary.md"
    expected_fast = str(reuse["expected_fast_dllm_commit"])
    expected_source = str(reuse["expected_probe_source_commit"])
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    return {
        "summary_path": str(summary_path),
        "fast_dllm_pin_match": expected_fast in summary,
        "pilot_source_commit_match": expected_source in summary,
        "warning": None if expected_fast in summary and expected_source in summary else (
            "Pilot raw schema is valid but outputs/summary.md does not confirm every expected provenance pin; "
            "reuse is recorded as a limitation."
        ),
    }


def candidates_from_state(record: Mapping[str, Any], *, threshold: float, tau_mass: float) -> tuple[list[AuditCandidate], list[AuditCandidate]]:
    anchors: list[AuditCandidate] = []
    unstable: list[AuditCandidate] = []
    identifier = state_id(record)
    for position_text, summary in record["position_summaries"].items():
        position = int(position_text)
        _require_fields(summary, ("top1_confidence", "top5_cumulative_probability", "top5_token_ids", "top5_probabilities"), "position summary")
        token_ids = summary["top5_token_ids"]
        probabilities = summary["top5_probabilities"]
        if not token_ids or not probabilities:
            continue
        confidence = float(summary["top1_confidence"])
        mass = float(summary["top5_cumulative_probability"])
        if confidence >= threshold:
            anchors.append(AuditCandidate(identifier, position, int(token_ids[0]), "anchor", "", float(probabilities[0]), confidence))
        elif mass >= tau_mass:
            for token_id, probability in zip(token_ids[:5], probabilities[:5]):
                unstable.append(AuditCandidate(identifier, position, int(token_id), "unstable", "", float(probability), confidence))
    return sorted(anchors), sorted(unstable)


def candidate_with_token(candidate: AuditCandidate, token: str) -> AuditCandidate:
    return AuditCandidate(
        candidate.state_id, candidate.position, candidate.token_id, candidate.kind, token,
        candidate.base_probability, candidate.base_confidence,
    )


def _candidate_from_pair(item: Mapping[str, Any], record: Mapping[str, Any], kind: str) -> AuditCandidate:
    identifier = state_id(record)
    summary = record["position_summaries"][str(int(item["position"]))]
    return AuditCandidate(
        identifier, int(item["position"]), int(item["token_id"]), kind,
        str(item.get("token", "")), 0.0, float(summary["top1_confidence"]),
    )


def pilot_pair_metrics(states: Iterable[Mapping[str, Any]], pair_rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], PairMetric]:
    by_coordinate = {(int(row["prompt_index"]), int(row["step"])): row for row in states}
    metrics: dict[tuple[str, str], PairMetric] = {}
    for row in pair_rows:
        state = by_coordinate.get((int(row["prompt_index"]), int(row["step"])))
        if state is None:
            continue
        a = _candidate_from_pair(row["a"], state, "unstable")
        b = _candidate_from_pair(row["b"], state, "unstable")
        metric = PairMetric(
            a, b, float(row["pair_stability_q2"]), float(row["a_to_b_lift"]),
            float(row["b_to_a_lift"]), float(row["residual_mean_tv"]) if row.get("residual_mean_tv") is not None else None,
        )
        key = pair_key(a, b)
        if key not in metrics:
            metrics[key] = metric
    return metrics


def scalar_cache_key(
    *, input_token_ids: list[int], mask_positions: list[int], insertions: Iterable[AuditCandidate],
    target: AuditCandidate, model_revision: str, dtype: str,
) -> str:
    descriptor = {
        "input_token_ids": [int(item) for item in input_token_ids],
        "mask_positions": sorted(int(item) for item in mask_positions),
        "insertions": sorted((int(item.position), int(item.token_id)) for item in insertions),
        "target": (int(target.position), int(target.token_id)),
        "model_revision": model_revision,
        "dtype": dtype,
        "use_cache": False,
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ExactScalarCache:
    """Append-safe scalar cache, readable after a job interruption."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, float] = {}
        if path.exists():
            for record in read_jsonl(path):
                if isinstance(record.get("key"), str) and isinstance(record.get("probability"), (int, float)):
                    self.entries[record["key"]] = float(record["probability"])

    def get(self, key: str) -> float | None:
        return self.entries.get(key)

    def put(self, key: str, probability: float, metadata: Mapping[str, Any]) -> None:
        if key in self.entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "probability": float(probability), "use_cache": False, **dict(metadata)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.entries[key] = float(probability)


def candidate_record(candidate: AuditCandidate) -> dict[str, Any]:
    return {
        "position": candidate.position, "token_id": candidate.token_id, "token": candidate.token,
        "kind": candidate.kind, "base_probability": candidate.base_probability,
        "base_confidence": candidate.base_confidence,
    }


def pair_record(metric: PairMetric) -> dict[str, Any]:
    return {
        "a": candidate_record(metric.a), "b": candidate_record(metric.b), "q2": metric.q2,
        "a_to_b_lift": metric.a_to_b_lift, "b_to_a_lift": metric.b_to_a_lift,
        "l_min": metric.l_min, "asymmetry": metric.asymmetry, "residual_mean_tv": metric.residual_mean_tv,
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize_failures(rows: Iterable[Mapping[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        failures = sum(bool(item["pair_safe_set_unsafe"]) for item in group)
        n = len(group)
        low, high = wilson_interval(failures, n)
        result.append({
            **dict(zip(group_fields, key)), "set_count": n, "failure_count": failures,
            "failure_rate": failures / n, "ci95_low": low, "ci95_high": high,
            "rule_of_three_upper_95": 3 / n if failures == 0 else None,
            "mean_gap": sum(float(item["gap"]) for item in group) / n,
        })
    return result
