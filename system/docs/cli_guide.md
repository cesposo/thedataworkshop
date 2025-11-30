# CLI Guide

Entrypoint: `dist-llm-train`

Global:
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

Commands:
- `demo-ml --config <file> [--state-db <db>] [--ps-checkpoint <pt>]`
  Run ML training simulation using the provided config. Saves PS checkpoint if path provided.
- `demo-basic --config <file> [--state-db <db>] [--ps-checkpoint <pt>]`
  Run basic scheduling/heartbeat simulation. Saves PS checkpoint if path provided.
- `submit-job --config <file>`
  In-process toy demo that submits a basic job to a controller.
- `status --controller <rpc>`
  Prints raw controller status JSON.
- `metrics --controller <rpc>`
  Pretty-prints task metrics and worker telemetry.
- `run-experiments --mode {ml|basic} (--configs ... | --glob pattern) [--repeats N] [--window N] [--output path] [--profile label]`
  Runs a batch of experiments and writes a CSV summary. `--window` maps to EWMA alpha via 2/(N+1).
- `resume --state-db <db> [--ps-checkpoint <pt>] [--host H] [--port P] [--scheduler name]`
  Resumes a controller from the last snapshot; requeues non-completed tasks; optionally loads PS checkpoint.
