# H100 top-1 dynamics audit handoff

## Decision record

This repository extends the frozen LLaDA candidate-interaction pilot into a
top-1 decoding-dynamics audit. The intended execution environment is an
assigned **NVIDIA H100 server**, not the local development machine. This
handoff contains code, configuration, tests, and documentation only: no
benchmark result, model cache, or generated runtime artifact has been
committed. Results must be produced and evaluated from a server run.

The experiment preserves the pilot's historical threshold-plus-fallback
collector. It records the natural trajectory first, then uses full-input,
`use_cache=False` replays for reported state checks and causal branches. It
does not train a model, change decoding actions, or persist full-vocabulary
distributions.

## Delivered scope

- `configs/top1_dynamics_audit.yaml`: frozen model, benchmark, resource, and
  reporting policy.
- `scripts/run_top1_dynamics_smoke.sh`: small H100 smoke run with exact replay.
- `scripts/run_top1_dynamics_audit.sh`: primary resource-gated run.
- `scripts/run_top1_dynamics_audit.py` and `src/top1_*.py`: natural
  trajectories; scalar distribution and transition metrics; matched
  counterfactuals; fixed/adaptive order replays; clustered-bootstrap reports;
  tables, figures, representative cases, and a handoff report.

The active trajectory is 16 masks over 16 steps, the compatible behavior
verified in the pinned Fast-dLLM source. The requested 512-token/64-block
setting is retained as metadata only, not presented as measured same-block
behavior. Do not infer an unverified block-decoding result from this audit.

## First server checkout

The GitHub branch for this handoff is `codex/top1-dynamics-audit`. After it
has been pushed, a fresh H100 server checkout is:

```bash
mkdir -p /workspace/zslee/code
git clone --branch codex/top1-dynamics-audit \
  https://github.com/DongGeon-SNU/candidate-attention.git \
  /workspace/zslee/code/candidate-attention

export PERSISTENT_ROOT=/workspace/zslee/code/candidate-attention
cd "$PERSISTENT_ROOT/zslee/dllm_candidate_probe"
```

For a pre-existing checkout, use `git fetch origin`, `git switch
codex/top1-dynamics-audit`, and `git pull --ff-only` instead of cloning again.
For a private repository, use a server GitHub SSH key (or an approved
credential helper); do not put an access token in shell history.

## H100 execution

```bash
# Set this only when the scheduler has not already restricted visible GPUs.
# The selected device must be the H100 assigned to this job.
export CUDA_VISIBLE_DEVICES=<assigned-H100-index>

bash scripts/setup.sh
bash scripts/run_top1_dynamics_smoke.sh
bash scripts/run_top1_dynamics_audit.sh
```

Use `TOP1_DYNAMICS_MODE=observational` only when an observational run is
intended; it omits anchor counterfactuals and order replays. Do not bypass the
resource gate for unattended work. It validates the pinned vendor revision,
collector compatibility, selected GPU, peak-VRAM projection (<80 GiB),
estimated runtime (<4 hours), and additional-disk projection (<50 GiB).

## What to return for review

Each run writes a fresh directory below `outputs/top1_dynamics_audit/` and a
timestamped log under `logs/`. Share the run's `run_manifest.json`,
`resource_estimate.json`, and `handoff.md` for result validation and
interpretation. These files distinguish a completed audit from an
observational-only or resource-gated run.
