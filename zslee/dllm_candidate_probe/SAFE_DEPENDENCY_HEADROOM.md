# Safe Dependency Headroom / Terminal Token Agreement

This is a pre-validity experiment: when an official dependency-avoidance
decoder declines to co-commit a position, does that position actually finish
with a different token from its same-input Fast-dLLM rollout? It is **not** an
implementation of VCCC, a sequential-reveal oracle, a new heuristic, or
predictor training. DAPD and DEMASK are each compared independently with
Fast-dLLM; they are not compared with one another.

## Frozen and shared conditions

Every active baseline for an input shares the following, recorded in the
per-run manifest:

- `GSAI-ML/LLaDA-8B-Instruct` at immutable revision
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`, the same tokenizer, rendered
  prompt, initial all-mask suffix, and absolute position convention;
- the same fixed generation length and source-native one-block decoding
  configuration (`steps == block_length == generation_length`);
- deterministic greedy transfer (`temperature=0`, `top_p=null`), fixed seed,
  and fixed-length/no-early-EOS policy; and
- the native upstream cache behavior. The upstream generation APIs do not
  expose a cache option, so a request to change it fails closed instead of
  pretending that `use_cache` was honored.

The setup pins Fast-dLLM to `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` and
DAPD to `05727b08da4cb4008a275123d7d9885dd5714f7c`. It applies two small,
tracked, default-off observation-hook patches under `patches/`. The patched
functions retain the upstream selection and decoding logic; callbacks only
receive already-computed primitive values. Before smoke data are accepted, the
runner calls each decoder once without observers and once with observers under
the same seed/input. It fails on a terminal-output or Fast native-forward-count
change, requires each observed trace to reconstruct the terminal output, and
also checks DAPD's observed selected positions against its native step history.
The five-prompt smoke config performs this A/B check for all five prompts.

`scripts/setup.sh` is idempotent for those patches: it accepts exactly the
pristine pinned source or the same patch already applied, and stops on any
other vendor-tree state rather than overwriting it. It writes
`outputs/safe_dependency_headroom_source_manifest.json` with source commits
and patch/file hashes. The runner rereads that manifest and compares the live
patched source-file hashes before model loading, so a later vendor edit fails
closed. Hugging Face and pip caches remain in the project `cache/` directory on
the persistent volume.

For the declared H100 environment, setup installs the pinned CUDA wheel
`torch==2.5.1+cu121` from the CUDA 12.1 wheel index and Fast-dLLM's pinned
`transformers==4.49.0`. An intentional alternative wheel can be requested only
through `SAFE_DEPENDENCY_TORCH_VERSION` and `SAFE_DEPENDENCY_TORCH_INDEX_URL`;
setup and the run manifest record it, and the runner refuses a later runtime
version drift from setup's dependency freeze.

## Declared decoding values

The YAML files are the authoritative, copied-into-provenance configuration.
For quick review, both v2 configurations declare `remasking=low_confidence`,
Fast-dLLM `threshold=0.9`, `mask_id=126336`, `temperature=0.0`, and
`top_p=null`. DAPD uses its published `dapd_direct` algorithm with
`layer_ratio=0.3`, `tau_min=0.01`, and `tau_max=0.05`; the actual per-step
threshold is logged rather than inferred afterwards. The smoke run uses 16
positions/steps/block length and seed `20260911`; the full run uses 256 for
each of those length fields and the same seed. The prompt bootstrap uses 2,000
draws for smoke and 10,000 for full, with bootstrap seed `20260911`.

## What is counted

For DAPD, a rejection event is one actual candidate rejection from the
published independent-set selector, with its **first decisive selected
blocker**. The detailed pair table is an event projection, not a collection of
independent pair rejections: rows retain `event_id`, candidate rank, selected
set before the decision, raw attention dependency, normalized selector score,
threshold, and selection stage. A candidate that is eventually added by a
separate direct/fallback stage is retained for provenance but excluded from the
primary "not co-committed" headroom denominator.

Each eligible event compares the DAPD terminal token at candidate and blocker
positions against the corresponding Fast-dLLM terminal output. It is assigned
exactly one category: `2/2 match`, `1/2 match`, or `0/2 match`. Bootstrap
resampling is by the full input-prompt cohort, including prompts with zero
rejection events. Replicates with no sampled events are marked undefined, not
silently converted to zero.

The run writes durable JSONL raw records as prompts complete and materializes
CSV/Parquet-compatible tabular logs below:

```text
results/raw/<run_id>/
results/summary/<run_id>/
```

The summary directory contains baseline JSON/CSV summaries, a manifest,
prompt-clustered bootstrap intervals, and a DAPD normalized-dependency-bin
agreement plot. It also contains a DEMASK blocker artifact where applicable.

## H100 commands

Run only from the persistent project checkout:

```bash
export PERSISTENT_ROOT=/workspace/zslee/code/candidate-attention
cd "$PERSISTENT_ROOT/zslee/dllm_candidate_probe"

# Fetch pinned sources, apply/verify the observation patches, and prepare the
# persistent Hugging Face cache. This is safe to rerun after a git update.
bash scripts/setup.sh

# Five prompts, 16 generated positions; also executes source-vs-observed
# equivalence checks before accepting data.
bash scripts/run_safe_dependency_headroom_smoke.sh

# Primary run: uses the checked-in 20-prompt JSONL cohort.
bash scripts/run_safe_dependency_headroom_full.sh
```

The full configuration uses the checked-in, versioned
`prompts/safe_dependency_headroom_primary.jsonl` cohort (20 prompt objects with
stable IDs, SHA-256 `5b426ecd0d7c7c56623142b597073fa96bf6806b399ffdff7a1102edae5ecb14`).
The runner validates that declared hash, copies the cohort into raw provenance,
and hashes it in the source manifest. To use a different research cohort, make a separately
versioned config and prompt JSONL rather than editing a completed run's input.

## DEMASK status

DEMASK is explicitly requested in the configuration metadata but is not an
active baseline in this checkout. There is no compatible official DEMASK
predictor source/checkpoint, supported backbone hidden-state interface, or
official `tau`/selection configuration available here. The runner therefore
writes `blocked_no_official_predictor_or_checkpoint` and does **not** substitute
attention scores, a learned surrogate, an oracle, or a new threshold rule.

To activate DEMASK, supply all four official assets/configuration pieces above
and add `demask` to `baselines.active`; preflight will otherwise stop rather
than emitting an invalid comparison.

## Status of earlier v1 smoke logs

Any earlier `results/safe_dependency_headroom_*` result is exploratory only.
That runner reconstructed decoder loops and expanded blocker-pair rows after
the fact, so its "rejection event" counts and bootstrap denominator do not
meet this v2 protocol. Do not pool it with, or cite it as, a v2 terminal-token
agreement estimate. Re-run the smoke command above after updating the source.
