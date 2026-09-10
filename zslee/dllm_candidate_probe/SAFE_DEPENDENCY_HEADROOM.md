# Safe Dependency Headroom / Terminal Token Agreement

`scripts/run_safe_dependency_headroom.py` compares two independent rollouts on
the same frozen `GSAI-ML/LLaDA-8B-Instruct` backbone, tokenizer, rendered
prompt, initial all-mask suffix, absolute positions, fixed generation length,
and deterministic (`temperature=0`, `top_p=null`) decoding policy. Neither
path uses a KV cache; both use fixed-length decoding with no early EOS stop.

The Fast-dLLM path calls the official vendored DAPD release's Fast-dLLM
`get_transfer_index` implementation. The DAPD path calls that release's
`AttentionCaptureHook`, `build_dependency_graph`, and
`select_independent_set`. Its trace is generated after selection and replays the
published greedy ordering; an assertion aborts if replay and actual selection
differ. Thus instrumentation cannot silently change the selected set.

On the H100 persistent volume, first run `bash scripts/setup.sh`. It pins
DAPD to `05727b08da4cb4008a275123d7d9885dd5714f7c` and records both source
revisions. Then run:

```bash
bash scripts/run_safe_dependency_headroom_smoke.sh
./.venv/bin/python scripts/run_safe_dependency_headroom.py \
  --config configs/safe_dependency_headroom_full.yaml --allow-missing-demask
```

The smoke config has five fixed prompts and 16 generated positions. The full
config requires a versioned JSONL prompt file (`{"prompt": "..."}` per line)
before it will run, preventing an accidental primary run with an unstated
cohort. Results live under the configured run timestamp in `results/raw/` and
`results/summary/`: step logs, excluded dependency edges, terminal outputs,
CSV/JSON summaries, and the DAPD dependency-bin plot.

## DEMASK blocker

This checkout contains no DEMASK predictor implementation or checkpoint, and
the public paper/abstract does not provide a retrievable official artifact.
Consequently, this runner does **not** fabricate a score, threshold, or
selection rule. To enable DEMASK, provide (1) the official predictor source,
(2) its exact checkpoint, (3) the supported backbone/revision and hidden-state
interface, and (4) the paper's `tau`/selection configuration. Until then,
DEMASK is recorded as `blocked_no_official_predictor_or_checkpoint` in the run
manifest and no DEMASK comparison is claimed.
