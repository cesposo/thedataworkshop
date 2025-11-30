# Experiments & Benchmarks

The experiment runner executes multiple configs, collects end-of-run status, and writes a CSV summary.

Usage:
```
dist-llm-train run-experiments \
  --mode ml \
  --glob 'configs/*_demo.yaml' \
  --repeats 3 \
  --window 20 \
  --output runs.csv \
  --profile no-netem
```

CSV columns:
- config: path to config file
- mode: ml or basic
- profile: label for external conditions (e.g., netem profile)
- window: EWMA window size used for telemetry smoothing
- repeat: 1..N
- duration_s: wall time for run
- num_workers, num_completed_tasks, num_pending_tasks
- mean_ewma_tokens_per_sec, mean_ewma_step_time_s: averaged across workers

Reproducibility:
- Keep configs under version control.
- Capture environment (TELEMETRY_ALPHA) or use `--window` for determinism.
- Document external network conditions (netem), if applied.
