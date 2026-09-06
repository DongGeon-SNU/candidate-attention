"""Public, provenance-preserving prompt loaders for the top-1 dynamics audit.

The audit needs natural decoding trajectories from several public benchmarks,
but it must never silently replace a failed benchmark download with unrelated
examples.  This module therefore has a deliberately small API:

* explicit local files are preferred;
* an explicit persistent Hugging Face ``cache_dir`` is tried before a public
  download;
* a public download is attempted only when ``allow_remote=True``;
* built-in examples are available only with ``allow_fallback=True`` and are
  labelled ``status='fallback'`` / ``source_type='built_in_fallback'``.

The module imports :mod:`datasets` lazily.  It remains usable for local JSONL,
JSON, and CSV files (and for CPU-only tests) before that optional dependency is
installed.  Callers should pass a cache path on the project persistent volume;
the loader intentionally does not fall back to a container-home cache.

No Hugging Face token is read, printed, or required for these public datasets.
No remote dataset code is enabled by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    """Stable public identifier and expected schema for one benchmark."""

    name: str
    hub_repo: str
    config: str | None
    split: str
    default_limit: int | None
    local_file_stems: tuple[str, ...]
    revision: str = "main"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        name="gsm8k",
        hub_repo="openai/gsm8k",
        config="main",
        split="test",
        default_limit=200,
        local_file_stems=("gsm8k", "gsm8k_test"),
    ),
    "humaneval": DatasetSpec(
        name="humaneval",
        hub_repo="openai/openai_humaneval",
        config=None,
        split="test",
        default_limit=None,
        local_file_stems=("humaneval", "openai_humaneval", "humaneval_test"),
    ),
    "ifeval": DatasetSpec(
        name="ifeval",
        hub_repo="google/IFEval",
        config=None,
        split="train",
        default_limit=200,
        local_file_stems=("ifeval", "ifeval_train"),
    ),
}

_ALIASES = {
    "gsm8k": "gsm8k",
    "openai/gsm8k": "gsm8k",
    "human_eval": "humaneval",
    "humaneval": "humaneval",
    "openai_humaneval": "humaneval",
    "openai/openai_humaneval": "humaneval",
    "ifeval": "ifeval",
    "google/ifeval": "ifeval",
}


@dataclass(frozen=True)
class PromptExample:
    """One benchmark prompt with only public, JSON-serializable provenance."""

    dataset: str
    example_id: str
    prompt: str
    reference: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    def public_record(self) -> dict[str, Any]:
        """Return a safe record suitable for a run manifest or trajectory row."""

        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "prompt": self.prompt,
            "reference": _json_safe(self.reference) if self.reference is not None else None,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkLoadResult:
    """Result of a non-interactive benchmark-load attempt.

    ``status='unavailable'`` always has zero examples.  A caller must opt into
    fallback data explicitly and must keep its source type separate from real
    benchmark results in every table/figure.
    """

    dataset: str
    status: str  # success | fallback | unavailable
    examples: tuple[PromptExample, ...]
    provenance: Mapping[str, Any]
    attempts: tuple[Mapping[str, str], ...]

    @property
    def is_benchmark(self) -> bool:
        return self.status == "success"

    def manifest_record(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "status": self.status,
            "example_count": len(self.examples),
            "provenance": _json_safe(self.provenance),
            "attempts": [_json_safe(item) for item in self.attempts],
            "example_ids": [item.example_id for item in self.examples],
        }


class DatasetSchemaError(ValueError):
    """A public file was found but does not match the requested benchmark."""


def canonical_dataset_name(name: str) -> str:
    """Normalize supported public names without accepting arbitrary datasets."""

    normalized = name.strip().lower().replace("-", "_")
    try:
        return _ALIASES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported benchmark {name!r}; supported: {supported}.") from error


def dataset_spec(name: str) -> DatasetSpec:
    """Return the immutable public source definition for a supported benchmark."""

    return DATASET_SPECS[canonical_dataset_name(name)]


def load_benchmark(
    name: str,
    *,
    limit: int | None = None,
    seed: int = 20260902,
    cache_dir: str | Path | None = None,
    local_path: str | Path | None = None,
    allow_remote: bool = False,
    allow_fallback: bool = False,
    revision: str | None = None,
) -> BenchmarkLoadResult:
    """Load a public benchmark in a deterministic, provenance-visible way.

    Parameters
    ----------
    name:
        One of GSM8K, HumanEval, or IFEval (common aliases are accepted).
    limit:
        Number of examples after deterministic hash-based selection.  ``None``
        uses the benchmark default: 200 for GSM8K/IFEval and all HumanEval
        problems.
    cache_dir:
        Required for Hugging Face cache or remote stages.  It should be beneath
        the persistent project volume, e.g. ``cache/huggingface/datasets``.
    local_path:
        An explicit JSONL/JSON/CSV file, or a directory containing a file named
        after the requested benchmark.  Explicit local data is attempted first.
    allow_remote:
        Permit unauthenticated public Hugging Face download *after* the local
        cache attempt.  This does not request or log credentials.
    allow_fallback:
        Return tiny deterministic built-in prompts if no real dataset could be
        loaded.  It is false by default so an experiment cannot mistake them
        for GSM8K/HumanEval/IFEval results.
    revision:
        Optional immutable Hugging Face revision.  The default ``main`` is
        recorded in provenance, along with the runtime dataset fingerprint.
    """

    spec = dataset_spec(name)
    requested_limit = spec.default_limit if limit is None else limit
    if requested_limit is not None and requested_limit < 1:
        raise ValueError("limit must be a positive integer or None.")

    cache = Path(cache_dir) if cache_dir is not None else None
    requested_revision = revision or spec.revision
    attempts: list[dict[str, str]] = []
    rows: list[dict[str, Any]] | None = None
    provenance: dict[str, Any] | None = None

    if local_path is not None:
        try:
            local_file = _resolve_local_file(Path(local_path), spec)
            rows = _read_local_records(local_file, split=spec.split)
            provenance = {
                "source_type": "local_file",
                "path": str(local_file),
                "requested_hub_repo": spec.hub_repo,
                "requested_config": spec.config,
                "requested_split": spec.split,
                "requested_revision": requested_revision,
            }
            attempts.append({"stage": "local_file", "status": "success"})
        except Exception as error:  # converted below without exposing credentials/URLs
            attempts.append(_attempt_failure("local_file", error))

    if rows is None and cache is not None:
        try:
            rows, fingerprint = _load_hf_rows(
                spec, cache_dir=cache, revision=requested_revision, local_files_only=True
            )
            provenance = _hf_provenance(spec, cache, requested_revision, "huggingface_cache", fingerprint)
            attempts.append({"stage": "huggingface_cache", "status": "success"})
        except Exception as error:
            attempts.append(_attempt_failure("huggingface_cache", error))
    elif rows is None:
        attempts.append({
            "stage": "huggingface_cache",
            "status": "skipped",
            "message": "cache_dir was not supplied; refusing a non-persistent default cache.",
        })

    if rows is None and allow_remote:
        if cache is None:
            attempts.append({
                "stage": "huggingface_public_download",
                "status": "skipped",
                "message": "cache_dir is required before a public download.",
            })
        else:
            try:
                rows, fingerprint = _load_hf_rows(
                    spec, cache_dir=cache, revision=requested_revision, local_files_only=False
                )
                provenance = _hf_provenance(
                    spec, cache, requested_revision, "huggingface_public_download", fingerprint
                )
                attempts.append({"stage": "huggingface_public_download", "status": "success"})
            except Exception as error:
                attempts.append(_attempt_failure("huggingface_public_download", error))

    if rows is not None and provenance is not None:
        try:
            examples = _select_examples(_examples_from_rows(spec, rows), requested_limit, seed)
        except Exception as error:
            attempts.append(_attempt_failure("schema_validation", error))
        else:
            return BenchmarkLoadResult(
                dataset=spec.name,
                status="success",
                examples=examples,
                provenance={
                    **provenance,
                    "available_examples": len(rows),
                    "selected_examples": len(examples),
                    "selection_algorithm": "sha256(seed|dataset|example_id), first N, then id sort",
                    "selection_seed": int(seed),
                    "is_benchmark": True,
                },
                attempts=tuple(attempts),
            )

    if allow_fallback:
        examples = _select_examples(_fallback_examples(spec), requested_limit, seed)
        return BenchmarkLoadResult(
            dataset=spec.name,
            status="fallback",
            examples=examples,
            provenance={
                "source_type": "built_in_fallback",
                "requested_hub_repo": spec.hub_repo,
                "requested_config": spec.config,
                "requested_split": spec.split,
                "requested_revision": requested_revision,
                "available_examples": len(_fallback_examples(spec)),
                "selected_examples": len(examples),
                "selection_algorithm": "sha256(seed|dataset|example_id), first N, then id sort",
                "selection_seed": int(seed),
                "is_benchmark": False,
                "warning": (
                    "Built-in prompts are smoke-only fallback data, not an official benchmark subset; "
                    "do not combine them with benchmark rates."
                ),
            },
            attempts=tuple(attempts),
        )

    return BenchmarkLoadResult(
        dataset=spec.name,
        status="unavailable",
        examples=(),
        provenance={
            "source_type": "unavailable",
            "requested_hub_repo": spec.hub_repo,
            "requested_config": spec.config,
            "requested_split": spec.split,
            "requested_revision": requested_revision,
            "is_benchmark": False,
            "next_action": (
                "Provide an explicit local public benchmark file, install the optional datasets dependency, "
                "or enable allow_remote with a persistent cache_dir. No credential is requested for these sources."
            ),
        },
        attempts=tuple(attempts),
    )


def load_benchmarks(
    names: Iterable[str],
    *,
    limits: Mapping[str, int | None] | None = None,
    seed: int = 20260902,
    cache_dir: str | Path | None = None,
    local_paths: Mapping[str, str | Path] | None = None,
    allow_remote: bool = False,
    allow_fallback: bool = False,
    revisions: Mapping[str, str] | None = None,
) -> dict[str, BenchmarkLoadResult]:
    """Load several requested sources without allowing one failure to hide another."""

    normalized_limits = {canonical_dataset_name(key): value for key, value in (limits or {}).items()}
    normalized_paths = {canonical_dataset_name(key): value for key, value in (local_paths or {}).items()}
    normalized_revisions = {canonical_dataset_name(key): value for key, value in (revisions or {}).items()}
    result: dict[str, BenchmarkLoadResult] = {}
    for requested_name in names:
        canonical = canonical_dataset_name(requested_name)
        result[canonical] = load_benchmark(
            canonical,
            limit=normalized_limits.get(canonical),
            seed=seed,
            cache_dir=cache_dir,
            local_path=normalized_paths.get(canonical),
            allow_remote=allow_remote,
            allow_fallback=allow_fallback,
            revision=normalized_revisions.get(canonical),
        )
    return result


def _resolve_local_file(path: Path, spec: DatasetSpec) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for stem in spec.local_file_stems:
            for suffix in (".jsonl", ".json", ".csv", ".parquet"):
                candidate = path / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(
        f"No explicit local {spec.name} file found at {path}. Expected one of "
        f"{', '.join(f'{stem}.jsonl' for stem in spec.local_file_stems)} (or .json/.csv/.parquet)."
    )


def _read_local_records(path: Path, *, split: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetSchemaError(f"Invalid JSONL at {path}:{line_number}.") from error
                if not isinstance(value, Mapping):
                    raise DatasetSchemaError(f"Expected an object at {path}:{line_number}.")
                rows.append(dict(value))
        return rows
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DatasetSchemaError(f"Invalid JSON in {path}.") from error
        if isinstance(payload, Mapping):
            payload = payload.get(split, payload.get("data", payload.get("records", payload)))
        if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
            raise DatasetSchemaError(f"{path} must contain an object list, a split object, or a data/records list.")
        return [dict(item) for item in payload]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(item) for item in csv.DictReader(handle)]
    if suffix == ".parquet":
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Local Parquet support requires the optional 'datasets' package (which installs pyarrow)."
            ) from error
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return [dict(item) for item in dataset]
    raise DatasetSchemaError(f"Unsupported local data extension {path.suffix!r}; use JSONL, JSON, CSV, or Parquet.")


def _load_hf_rows(
    spec: DatasetSpec, *, cache_dir: Path, revision: str, local_files_only: bool
) -> tuple[list[dict[str, Any]], str | None]:
    """Use public dataset data only; callers choose whether network use is allowed."""

    try:
        from datasets import DownloadConfig, load_dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Public benchmark loading requires the optional 'datasets' package. "
            "Install it in the project virtual environment; no token is needed for these public datasets."
        ) from error

    args: list[str] = [spec.hub_repo]
    if spec.config is not None:
        args.append(spec.config)
    kwargs: dict[str, Any] = {
        "split": spec.split,
        "cache_dir": str(cache_dir),
        "revision": revision,
    }
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    # Do not pass trust_remote_code=True or a token.  These repositories expose
    # public data and modern Datasets reads their standard data files directly.
    dataset = load_dataset(*args, **kwargs)
    return [dict(item) for item in dataset], getattr(dataset, "_fingerprint", None)


def _hf_provenance(
    spec: DatasetSpec, cache_dir: Path, revision: str, source_type: str, fingerprint: str | None
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "hub_repo": spec.hub_repo,
        "config": spec.config,
        "split": spec.split,
        "requested_revision": revision,
        "dataset_fingerprint": fingerprint,
        "cache_dir": str(cache_dir),
    }


def _examples_from_rows(spec: DatasetSpec, rows: Sequence[Mapping[str, Any]]) -> tuple[PromptExample, ...]:
    if not rows:
        raise DatasetSchemaError(f"{spec.name} source has zero rows.")
    adapters = {
        "gsm8k": _gsm8k_example,
        "humaneval": _humaneval_example,
        "ifeval": _ifeval_example,
    }
    examples = tuple(adapters[spec.name](row) for row in rows)
    ids = [item.example_id for item in examples]
    if len(set(ids)) != len(ids):
        raise DatasetSchemaError(f"{spec.name} source contains duplicate example IDs; refusing ambiguous selection.")
    return examples


def _gsm8k_example(row: Mapping[str, Any]) -> PromptExample:
    question = _required_text(row, "question", "GSM8K")
    answer = _required_text(row, "answer", "GSM8K")
    identifier = str(row.get("id") or f"gsm8k-{_short_hash(question)}")
    final_answer = answer.rsplit("####", 1)[-1].strip() if "####" in answer else answer.strip()
    return PromptExample(
        dataset="gsm8k",
        example_id=identifier,
        prompt=question,
        reference={"final_answer": final_answer},
        metadata={"task_type": "math_word_problem", "raw_answer_has_final_delimiter": "####" in answer},
    )


def _humaneval_example(row: Mapping[str, Any]) -> PromptExample:
    prompt = _required_text(row, "prompt", "HumanEval")
    task_id = str(row.get("task_id") or f"humaneval-{_short_hash(prompt)}")
    canonical_solution = row.get("canonical_solution")
    if canonical_solution is not None and not isinstance(canonical_solution, str):
        raise DatasetSchemaError("HumanEval field 'canonical_solution' must be a string when present.")
    entry_point = row.get("entry_point")
    if entry_point is not None and not isinstance(entry_point, str):
        raise DatasetSchemaError("HumanEval field 'entry_point' must be a string when present.")
    return PromptExample(
        dataset="humaneval",
        example_id=task_id,
        prompt=prompt,
        reference={"canonical_solution": canonical_solution} if canonical_solution is not None else None,
        metadata={"task_type": "code_generation", "entry_point": entry_point},
    )


def _ifeval_example(row: Mapping[str, Any]) -> PromptExample:
    prompt = _required_text(row, "prompt", "IFEval")
    key = row.get("key")
    identifier = str(key) if key is not None else f"ifeval-{_short_hash(prompt)}"
    instruction_ids = row.get("instruction_id_list", [])
    kwargs = row.get("kwargs", [])
    if not isinstance(instruction_ids, list) or not isinstance(kwargs, list):
        raise DatasetSchemaError("IFEval fields 'instruction_id_list' and 'kwargs' must be lists when present.")
    return PromptExample(
        dataset="ifeval",
        example_id=identifier,
        prompt=prompt,
        reference=None,
        metadata={
            "task_type": "instruction_following",
            "instruction_id_list": _json_safe(instruction_ids),
            "kwargs": _json_safe(kwargs),
        },
    )


def _required_text(row: Mapping[str, Any], field: str, source_name: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetSchemaError(f"{source_name} row is missing non-empty string field {field!r}.")
    return value


def _select_examples(
    examples: Sequence[PromptExample], limit: int | None, seed: int
) -> tuple[PromptExample, ...]:
    ordered = sorted(examples, key=lambda item: item.example_id)
    if limit is None or limit >= len(ordered):
        return tuple(ordered)
    ranked = sorted(
        ordered,
        key=lambda item: hashlib.sha256(
            f"{int(seed)}|{item.dataset}|{item.example_id}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(sorted(ranked[:limit], key=lambda item: item.example_id))


def _fallback_examples(spec: DatasetSpec) -> tuple[PromptExample, ...]:
    """Tiny deterministic smoke-only prompts, intentionally not benchmark copies."""

    data: dict[str, tuple[PromptExample, ...]] = {
        "gsm8k": (
            PromptExample(
                "gsm8k", "fallback-gsm8k-001",
                "If 3 notebooks cost $6, what is the cost of 5 notebooks? Explain briefly.",
                {"final_answer": "10"},
                {"task_type": "math_word_problem", "fallback": True},
            ),
            PromptExample(
                "gsm8k", "fallback-gsm8k-002",
                "A train travels 60 km each hour for 3 hours. How far does it travel?",
                {"final_answer": "180"},
                {"task_type": "math_word_problem", "fallback": True},
            ),
        ),
        "humaneval": (
            PromptExample(
                "humaneval", "fallback-humaneval-001",
                "def add(a, b):\n    \"\"\"Return the sum of two integers.\"\"\"\n",
                {"canonical_solution": "    return a + b\n"},
                {"task_type": "code_generation", "entry_point": "add", "fallback": True},
            ),
            PromptExample(
                "humaneval", "fallback-humaneval-002",
                "def is_even(n):\n    \"\"\"Return True when n is an even integer.\"\"\"\n",
                {"canonical_solution": "    return n % 2 == 0\n"},
                {"task_type": "code_generation", "entry_point": "is_even", "fallback": True},
            ),
        ),
        "ifeval": (
            PromptExample(
                "ifeval", "fallback-ifeval-001",
                "Write exactly two bullet points about reproducible experiments.",
                None,
                {"task_type": "instruction_following", "instruction_id_list": ["fallback:two_bullets"], "kwargs": [], "fallback": True},
            ),
            PromptExample(
                "ifeval", "fallback-ifeval-002",
                "Reply with the single lowercase word blue.",
                None,
                {"task_type": "instruction_following", "instruction_id_list": ["fallback:lowercase_word"], "kwargs": [], "fallback": True},
            ),
        ),
    }
    return data[spec.name]


def _attempt_failure(stage: str, error: Exception) -> dict[str, str]:
    return {"stage": stage, "status": "failed", "message": _safe_error(error)}


def _safe_error(error: Exception) -> str:
    """Report a useful blocker without retaining an exception that may contain a URL/token."""

    if isinstance(error, ModuleNotFoundError):
        return str(error).split(";", 1)[0]
    if isinstance(error, FileNotFoundError):
        return "Requested local/cache artifact was not found."
    if isinstance(error, DatasetSchemaError):
        return str(error)
    return (
        f"{type(error).__name__} while loading public data. No credential was requested or logged; "
        "inspect the timestamped command log for the environment-level failure."
    )


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _json_safe(value: Any) -> Any:
    """Convert common dataset scalar/container values to JSON-safe public metadata."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    return str(value)
