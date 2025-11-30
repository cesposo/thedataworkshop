import argparse
import sys
from dist_llm_train.logging_utils import configure_logging


def main():
    """Entry point for the CLI binary `dist-llm-train`."""
    parser = argparse.ArgumentParser(
        prog="dist-llm-train",
        description="Distributed LLM Training demos and utilities",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_basic = sub.add_parser("demo-basic", help="Run the basic scheduling/heartbeat simulation")
    p_basic.add_argument("--config", default=None, help="Path to YAML/JSON config (optional)")
    p_basic.add_argument("--state-db", default=None, help="Path to SQLite DB for controller state snapshots")
    p_basic.add_argument("--ps-checkpoint", default=None, help="Path to PS checkpoint file (load if exists, save on exit)")
    p_basic.add_argument("--netem-profile", default=None, help="Netem profile name or inline JSON/YAML dict for WAN impairment (optional)")
    p_basic.add_argument("--netem-seed", type=int, default=None, help="Seed for netem randomness (optional)")

    p_ml = sub.add_parser("demo-ml", help="Run the ML training simulation with PyTorch")
    p_ml.add_argument("--config", default="config.yaml", help="Path to YAML/JSON config (default: config.yaml)")
    p_ml.add_argument("--state-db", default=None, help="Path to SQLite DB for controller state snapshots")
    p_ml.add_argument("--ps-checkpoint", default=None, help="Path to PS checkpoint file (load if exists, save on exit)")
    p_ml.add_argument("--netem-profile", default=None, help="Netem profile name or inline JSON/YAML dict for WAN impairment (optional)")
    p_ml.add_argument("--netem-seed", type=int, default=None, help="Seed for netem randomness (optional)")

    p_submit = sub.add_parser("submit-job", help="Submit a demo job from a JSON/YAML config (runs in-process demo)")
    p_submit.add_argument("--config", required=True, help="Path to job config (JSON or YAML)")

    p_status = sub.add_parser("status", help="Fetch status from a running controller RPC endpoint")
    p_status.add_argument("--controller", required=True, help="Controller XML-RPC address, e.g., http://localhost:8000")

    p_metrics = sub.add_parser("metrics", help="Fetch metrics and telemetry from controller and pretty-print")
    p_metrics.add_argument("--controller", required=True, help="Controller XML-RPC address, e.g., http://localhost:8000")

    p_exp = sub.add_parser("run-experiments", help="Run multiple simulations and export a CSV summary")
    p_exp.add_argument("--mode", choices=["ml", "basic"], default="ml", help="Which simulation to run for each config")
    group = p_exp.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", nargs="+", help="List of config files")
    group.add_argument("--glob", help="Glob for config files, e.g., 'configs/*_demo.yaml'")
    p_exp.add_argument("--repeats", type=int, default=1, help="Repeats per config")
    p_exp.add_argument("--output", default="experiments.csv", help="Output CSV path")
    p_exp.add_argument("--window", type=int, default=0, help="EWMA window size (maps to alpha = 2/(N+1))")
    p_exp.add_argument("--profile", default="", help="Profile name label (informational)")
    p_exp.add_argument("--netem-profile", default=None, help="Netem profile name or inline JSON/YAML dict to apply to runs")
    p_exp.add_argument("--netem-seed", type=int, default=None, help="Seed for netem randomness (optional)")

    # resume subcommand
    p_resume = sub.add_parser("resume", help="Start a controller from the last saved snapshot and requeue tasks")
    p_resume.add_argument("--state-db", required=True, help="Path to SQLite DB with snapshots")
    p_resume.add_argument("--host", default="localhost", help="Controller host bind")
    p_resume.add_argument("--port", type=int, default=8000, help="Controller port bind")
    p_resume.add_argument("--scheduler", default="gale-shapley", help="Scheduler name")
    p_resume.add_argument("--ps-checkpoint", default=None, help="Optional PS checkpoint to load on resume")

    args = parser.parse_args()

    # Initialize logging early
    configure_logging(args.log_level)

    if args.cmd == "demo-basic":
        from simulation import run_simulation

        _ = run_simulation(args.config, state_db_path=args.state_db, ps_checkpoint_path=args.ps_checkpoint, netem_profile=args.netem_profile, netem_seed=args.netem_seed)
        return 0

    if args.cmd == "demo-ml":
        from ml_training_simulation import run_ml_training_simulation

        _ = run_ml_training_simulation(args.config, state_db_path=args.state_db, ps_checkpoint_path=args.ps_checkpoint, netem_profile=args.netem_profile, netem_seed=args.netem_seed)
        return 0

    if args.cmd == "submit-job":
        # Very simple in-process demonstration using the same pattern as simulation.py
        from dist_llm_train.controller.main_controller import MainController
        from dist_llm_train.worker.node import WorkerNode
        from dist_llm_train.controller.job_manager import JobManager
        from dist_llm_train.config import load_config
        import time as _time

        cfg = load_config(args.config)

        host = cfg.get('controller', {}).get('host', 'localhost')
        port = int(cfg.get('controller', {}).get('port', 8200))
        controller = MainController(host=host, port=port)

        workers_cfg = cfg.get('workers', [
            {'name': 'w1', 'memory': 16, 'flops': 100, 'bandwidth': 1000, 'port': 8201},
            {'name': 'w2', 'memory': 32, 'flops': 200, 'bandwidth': 1000, 'port': 8202},
        ])
        workers = []
        for wc in workers_cfg:
            w = WorkerNode(
                name=wc.get('name', 'w'),
                memory=float(wc.get('memory', 16)),
                flops_per_second=int(wc.get('flops', 100)),
                network_bandwidth=int(wc.get('bandwidth', 1000)),
                host=host,
                port=int(wc.get('port', 0)),
                controller_address=f"http://{host}:{port}",
            )
            workers.append(w)
            controller.register_worker(
                w.id,
                f"http://{w.communicator.host}:{w.communicator.port}",
                w,
            )

        jm = JobManager(controller)
        job_cfg = cfg.get('job', {'name': 'demo-job', 'num_tasks': 4})
        jm.submit_job(job_cfg)

        # Kick a few scheduling cycles
        for _ in range(int(cfg.get('cycles', 3))):
            controller.run_scheduling_cycle()
            _time.sleep(2)

        status = controller.get_system_status()
        print(status)

        # Cleanup
        controller.communicator.stop_server()
        for w in workers:
            w.communicator.stop_server()
        return 0

    if args.cmd == "status":
        from xmlrpc.client import ServerProxy
        with ServerProxy(args.controller) as proxy:
            status = proxy.get_status()
            print(status)
        return 0

    if args.cmd == "metrics":
        from xmlrpc.client import ServerProxy
        with ServerProxy(args.controller) as proxy:
            status = proxy.get_status()
        # Pretty-print metrics and telemetry
        print("Task Metrics:")
        metrics = status.get('metrics', {}) or {}
        if not metrics:
            print("  (none)")
        else:
            for tid, m in metrics.items():
                worker = m.get('worker_id')
                final_loss = m.get('final_loss')
                n_batches = m.get('num_batches')
                n_epochs = m.get('num_epochs')
                print(f"  - {tid} | worker={worker} loss={final_loss} batches={n_batches} epochs={n_epochs}")

        print("\nWorker Telemetry (last report per worker):")
        telem = status.get('telemetry', {}) or {}
        if not telem:
            print("  (none)")
        else:
            for wid, t in telem.items():
                mode = t.get('mode')
                tps = t.get('tokens_per_sec')
                step = t.get('step_time_s')
                epoch = t.get('epoch')
                batch = t.get('batch')
                print(f"  - {wid} | mode={mode} tps={tps} step_s={step} epoch={epoch} batch={batch}")
        netem = status.get('netem_stats') or {}
        if netem:
            print("\nNetem Stats:")
            ctrl = netem.get('controller') or {}
            if ctrl:
                print(f"  controller: sent={ctrl.get('sent')} drop={ctrl.get('dropped')} dup={ctrl.get('duplicated')} bytes={ctrl.get('bytes')} p50ms={ctrl.get('p50_latency_ms')} p95ms={ctrl.get('p95_latency_ms')}")
            workers = netem.get('workers') or {}
            if not workers:
                print("  (no worker netem stats)")
            else:
                for wid, s in workers.items():
                    print(f"  {wid}: sent={s.get('sent')} drop={s.get('dropped')} dup={s.get('duplicated')} bytes={s.get('bytes')} p50ms={s.get('p50_latency_ms')} p95ms={s.get('p95_latency_ms')}")
        return 0

    if args.cmd == "run-experiments":
        from dist_llm_train.experiments.runner import run_experiments, expand_glob

        cfgs = args.configs or expand_glob(args.glob)
        if not cfgs:
            print("No configs resolved")
            return 1
        out = run_experiments(
            cfgs,
            mode=args.mode,
            repeats=args.repeats,
            output_csv=args.output,
            profile=args.profile,
            window=args.window,
            netem_profile=args.netem_profile,
            netem_seed=args.netem_seed,
        )
        print(f"Wrote {out} with {len(cfgs) * args.repeats} rows")
        return 0

    if args.cmd == "resume":
        # Read last snapshot and recreate controller, requeue tasks (workers should re-register)
        from dist_llm_train.persistence.state_store import StateStore
        from dist_llm_train.controller.main_controller import MainController
        from dist_llm_train.task.training_task import TrainingTask
        import os as _os

        store = StateStore(args.state_db)
        snap = store.load_last_status()
        if not snap:
            print("No snapshot found in state DB")
            return 1

        ctrl = MainController(host=args.host, port=args.port, scheduler_name=args.scheduler, state_db_path=args.state_db)
        # Optionally load PS
        if args.ps_checkpoint and _os.path.exists(args.ps_checkpoint):
            try:
                ok = ctrl.ps_load(args.ps_checkpoint)
                print(f"PS load {'ok' if ok else 'failed'} from {args.ps_checkpoint}")
            except Exception as e:
                print(f"PS load error: {e}")

        # Requeue tasks not completed
        tasks_detail = (snap.get('tasks_detail') or {})
        completed = set(snap.get('completed_tasks') or [])
        for tid, tinfo in tasks_detail.items():
            if tid in completed:
                continue
            task = TrainingTask(
                task_id=tinfo.get('id', tid),
                model_name=tinfo.get('model_name', 'model'),
                model_layer=int(tinfo.get('model_layer', 0)),
                model_shard_size=float(tinfo.get('model_shard_size', 0.0)),
                data_size=float(tinfo.get('data_size', 0.0)),
                required_flops=int(tinfo.get('required_flops', 0)),
                optimizer_state_size=float(tinfo.get('optimizer_state_size', 0.0)),
                gradient_size=float(tinfo.get('gradient_size', 0.0)),
                activation_size=float(tinfo.get('activation_size', 0.0)),
                model_config=tinfo.get('model_config', {}),
                training_config=tinfo.get('training_config', {}),
                priority=int(tinfo.get('priority', 0)),
            )
            ctrl.add_task(task)

        print("Controller resumed. Waiting for workers to register. Current status:")
        print(ctrl.get_system_status())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
"""Command-line interface for dist-llm-train.

Provides demo simulations, job submission, status/metrics inspection,
an experiment runner, and a resume command.
"""
