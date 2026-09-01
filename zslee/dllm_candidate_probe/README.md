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
- `HF_TOKEN` is present without printing it; and
- one GPU has at least 20,000 MiB free, a conservative no-quantization BF16 budget for the roughly 16 GiB LLaDA checkpoint plus runtime memory.

If model access is rejected, configure a Portainer/Kubernetes secret named `HF_TOKEN`; do not use interactive Hugging Face login. All Hugging Face and pip caches stay under `cache/` in this project. No service, deployment, or background process is created.

## Commands

```bash
export PERSISTENT_ROOT=/the/persistent/volume/root
export HF_TOKEN=...  # inject this from the platform secret mechanism; never put it in a file
bash scripts/setup.sh
bash scripts/run_smoke.sh
bash scripts/run_pilot.sh
```

Each command creates a timestamped file under `logs/`. The smoke stage uses one prompt and one low-parallel decoding state with branch microbatch one. It measures GPU peak allocation and timing and writes `outputs/resource_estimate.md`. The pilot command will refuse to start unless that estimate is at most 30 minutes; it then uses at most ten prompts and two collected states per prompt.

For an ephemeral Kubernetes job, mount the same persistent volume at the same path for every stage and run one of the commands above as the job command. Do not rely on the container filesystem for code, caches, logs, or results.

## Output policy

Only scalar statistics, IDs, and top-k probabilities are written. Full-vocabulary distributions are reduced on GPU and never saved; attention maps, hidden states, and KV tensors are never stored. Runtime outputs, virtual environments, caches, and vendor source are ignored by Git. `outputs/summary.md` is generated after smoke/pilot and records the model, dtype, source pin, seed, metrics, and result paths.

## Local coding validation

The pure branch-semantics test can be run without a GPU or PyTorch:

```bash
python -m unittest zslee/dllm_candidate_probe/tests/test_branch_semantics.py -v
```

The setup stage runs the additional Torch-level tests. The real LLaDA smoke is intentionally not run on a local coding machine.
