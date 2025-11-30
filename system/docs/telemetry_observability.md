# Telemetry & Observability

What is collected:
- Per-batch tokens/sec and step time (TaskExecutor).
- EWMA rollups per worker on the controller (avg_tokens_per_sec, avg_step_time_s).
- Task metrics (final loss, num batches/epochs), assignments, and status snapshots.

How it’s used:
- Capability scheduler uses EWMA throughput to prefer faster workers, adapting to changing conditions.
- The `metrics` CLI prints a readable summary of task metrics and last telemetry.
- Experiments aggregate run-level metrics, including average EWMA across workers.

Configuring EWMA:
- Environment variable `TELEMETRY_ALPHA` (default 0.2) controls smoothing.
- Experiment runner can set an EWMA window `N` and maps it to `alpha = 2/(N+1)` per run.

Example CLI:
```
dist-llm-train metrics --controller http://localhost:8000
```

Status payload (selected):
- workers, workers_detail, pending_tasks, completed_tasks
- telemetry (last report), telemetry_rollups (EWMA), assignments
- tasks_detail (full task descriptors)
