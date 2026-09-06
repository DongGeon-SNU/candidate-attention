# Frozen LLaDA candidate-interaction probe

This project measures frozen-model counterfactual effects only. It does not train a head, change the backbone, add a referee/predictor, or use quantization. The state collector follows Fast-dLLM v1's confidence-threshold transfer rule; every base, singleton, pair, full-set, and leave-one-out metric is a separate LLaDA forward with `use_cache=False`.

## Shared-GPU placement

Do the following on the GPU job, not on the local coding machine. The code must first be copied or checked out beneath an already-mounted durable volume:

```bash
export PERSISTENT_ROOT=/the/persistent/volume/root
cd "$PERSISTENT_ROOT"
mkdir -p zslee
# Place this source tree at $PERSISTENT_ROOT/zslee/dllm_candidate_probe.
cd "$PERSISTENT_ROOT/zslee/dllm_candidate_probe"
```

`scripts/setup.sh` rejects every other location. It writes `outputs/environment_report.md` first, classifies the mount, and refuses to download anything until all of these are true:

- the project is on a likely-persistent mount (or the job owner has verified it and set `PERSISTENCE_CONFIRMED=1`);
- there are at least 40 GiB free at the persistent root;
- one GPU has at least 20,000 MiB free, a conservative no-quantization BF16 budget for the roughly 16 GiB LLaDA checkpoint plus runtime memory.

`GSAI-ML/LLaDA-8B-Instruct` is public, so setup first attempts an unauthenticated download. Only if Hugging Face returns an access error should a Portainer/Kubernetes secret named `HF_TOKEN` be configured; do not use interactive Hugging Face login. All Hugging Face and pip caches stay under `cache/` in this project. No service, deployment, or background process is created.

## Commands

```bash
export PERSISTENT_ROOT=/the/persistent/volume/root
bash scripts/setup.sh
bash scripts/run_smoke.sh
bash scripts/run_pilot.sh
```

Each command creates a timestamped file under `logs/`. The smoke stage uses one prompt and one low-parallel decoding state with branch microbatch one. It measures GPU peak allocation and timing and writes `outputs/resource_estimate.md`. The pilot command will refuse to start unless that estimate is at most 30 minutes; it then uses at most ten prompts and two collected states per prompt.

For an ephemeral Kubernetes job, mount the same persistent volume at the same path for every stage and run one of the commands above as the job command. Do not rely on the container filesystem for code, caches, logs, or results.

## Hard-set audit (pilot continuation)

The hard-set audit is a continuation of a successful pilot, not a replacement
for it. It reads `outputs/raw/pilot_states.jsonl`, `pilot_pairs.jsonl`,
`pilot_sets.jsonl`, and `outputs/pilot_result.json` from the same persistent
project directory. The artifacts must remain in place; they are deliberately
Git-ignored experiment results.

```bash
export PERSISTENT_ROOT=/the/persistent/volume/root
cd "$PERSISTENT_ROOT/zslee/dllm_candidate_probe"
bash scripts/run_hard_set_smoke.sh
bash scripts/run_hard_set_audit.sh
```

The smoke run validates two stored states, attempts five sets for each target
size (3, 4, 6, 8), checks all generated sets are cliques with unique positions,
and compares reusable pilot pair quantities to newly measured no-cache
singleton scalars (Q2 absolute tolerance `1e-4`, directed-lift absolute
tolerance `1e-2` to account for BF16 log amplification). It writes a
timestamped log under `logs/` and results under `outputs/hard_set/`.

The audit examines two definitions of a hard set: stability-only pair cliques
at `q2 >= tau`, and strict cliques which additionally require both directed
lifts to exceed `delta`. It includes policy-style sequential construction
starting with all anchors, plus a separately labelled stress-mined cohort.
Anchor-pair conflicts are reported and excluded from the policy primary rate;
the stress cohort is never presented as a population rate.

The fixed grid is budgeted before it runs: the policy primary (Graph 1,
`tau=0.7`) aims for 500 audited sets per size; its stress cohort aims for 100;
each other predeclared graph/threshold/delta cell aims for 25. These counts are
configuration values, not post-hoc threshold selection, and can be increased
only up to the declared 1,000-per-size cap after reviewing the resource report.

Before any leave-one-out audit, the full grid is planned and its exact-forward
time is projected using newly measured scalar-forward timing. If it exceeds
30 minutes, `run_hard_set_audit.sh` writes
`outputs/hard_set/audit_resource_estimate.json` and stops without running the
full LOO workload. Reduce the declared target only after reviewing that report;
do not bypass the guard. The scalar cache at
`outputs/hard_set/raw/exact_scalar_cache.jsonl` is restart-safe and keys every
entry on full input IDs, mask positions, insertion set, target, model revision,
dtype, and `use_cache=False`.

## H100 top-1 dynamics audit

The server handoff, including the H100 decision record, first-checkout steps,
verified 16-mask/16-step scope, and review artifacts, is in
[`H100_TOP1_DYNAMICS_AUDIT_HANDOFF.md`](H100_TOP1_DYNAMICS_AUDIT_HANDOFF.md).

This is a separate, frozen-model audit of whether the current top-1 token at a
masked position changes across the natural dLLM trajectory. It first records a
natural trajectory using the existing threshold-plus-fallback collector, then
replays every reported state and every causal branch from complete input IDs
with `use_cache=False`. It does not train a predictor, alter decoding, or save
full vocabulary distributions.

Run it from the persistent H100 project location after `scripts/setup.sh`:

```bash
export PERSISTENT_ROOT=/workspace/zslee/code/candidate-attention
cd "$PERSISTENT_ROOT/zslee/dllm_candidate_probe"
# Required only if the scheduler has not already set it and this job can see
# multiple physical GPUs. Use the H100 assigned by the scheduler; PyTorch
# logical cuda:0 will then be that device.
# export CUDA_VISIBLE_DEVICES=<assigned-H100-index>

# Optional but recommended: one prompt per public benchmark and exact replay.
bash scripts/run_top1_dynamics_smoke.sh

# Resource-gated primary run plus threshold sensitivity and, when budget permits,
# exact anchor counterfactuals and order replays.
bash scripts/run_top1_dynamics_audit.sh
```

The audit wrapper opts into downloading the declared public benchmark datasets
and writes a timestamped run below `outputs/top1_dynamics_audit/`, plus a
timestamped log under `logs/`. Its default `auto` mode always performs its own
small smoke collection, then stops safely if the measured projection exceeds
the configured 80 GiB peak-VRAM, four-hour, or 50 GiB additional-disk limits.
The projection includes a smoke-calibrated estimate for the mandated 10,000
prompt-clustered bootstrap analysis, not only GPU forwards. It also refuses to
run if the checked-out Fast-dLLM commit or the historical raw-logit
threshold/fallback collector semantics do not match the declared audit.
Use `TOP1_DYNAMICS_MODE=observational bash scripts/run_top1_dynamics_audit.sh`
to omit anchor counterfactuals and order replays; use `TOP1_DYNAMICS_MODE=all`
to request them, still subject to the same resource gate. Do not add
`--skip-resource-gate` to unattended jobs.

Each completed run contains `run_manifest.json`, `resource_estimate.json`,
`handoff.md`, raw scalar/Parquet evidence, CSV tables, figures, and
representative cases. A resource-gated run still preserves its smoke evidence
and manifest for diagnosis.

The pinned, verified collector currently uses a 16-mask / 16-step compatible
trajectory. The config records the requested 512-token / 64-block target only
as metadata because no corresponding block-decoding behavior has yet been
verified in the pinned Fast-dLLM source; same-block results are therefore
reported as unavailable rather than inferred from an artificial partition.

## Output policy

Only scalar statistics, IDs, and top-k probabilities are written. Full-vocabulary distributions are reduced on GPU and never saved; attention maps, hidden states, and KV tensors are never stored. Runtime outputs, virtual environments, caches, and vendor source are ignored by Git. `outputs/summary.md` is generated after smoke/pilot and records the model, dtype, source pin, seed, metrics, and result paths.

## Local coding validation

The pure branch-semantics test can be run without a GPU or PyTorch:

```bash
python -m unittest zslee/dllm_candidate_probe/tests/test_branch_semantics.py -v
python -m unittest zslee/dllm_candidate_probe/tests/test_hard_set_generation.py -v
```

The setup stage runs the additional Torch-level tests. The real LLaDA smoke is intentionally not run on a local coding machine.
