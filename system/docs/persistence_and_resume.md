# Persistence & Resume

Two persistence channels enable long-lived experiments and recovery:

1) Controller state snapshots (SQLite)
- A compact JSON status snapshot is stored after key events (worker register, task added, assignment, completion, telemetry).
- Use `--state-db state.db` in demos to enable persistence.
- Snapshots include tasks_detail, assignments, telemetry, and metrics.

2) Parameter Server checkpoint
- PS exposes `ps_save(path)` and `ps_load(path)` RPCs to save/restore model parameters and version.
- Use `--ps-checkpoint ps.pt` to save at the end of a run; the demo will load if path exists.

Resume flow:
- `dist-llm-train resume --state-db state.db --ps-checkpoint ps.pt --host localhost --port 8000 --scheduler capability`
- The CLI starts a new controller, optionally loads the PS checkpoint, and requeues all non-completed tasks from the last snapshot.
- Workers then re-register and training/scheduling continues.
