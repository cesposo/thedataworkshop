import time
import threading
from typing import Optional
import os

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.controller.job_manager import JobManager
from dist_llm_train.config import load_config
from dist_llm_train.logging_utils import configure_logging


def run_simulation(
    config_path: Optional[str] = None,
    state_db_path: Optional[str] = None,
    ps_checkpoint_path: Optional[str] = None,
    netem_profile: Optional[str] = None,
    netem_seed: Optional[int] = None,
):
    """
    Runs a basic scheduling/heartbeat simulation.

    If config_path is provided, reads controller/workers/job/cycles from YAML/JSON.
    """
    configure_logging()

    if config_path:
        cfg = load_config(config_path)
        ctrl_cfg = cfg.get('controller', {})
        host = ctrl_cfg.get('host', 'localhost')
        port = int(ctrl_cfg.get('port', 8000))
        scheduler = ctrl_cfg.get('scheduler', 'gale-shapley')
        net_cfg = cfg.get('network', {})
        profile = netem_profile if netem_profile is not None else net_cfg.get('profile')
        seed = netem_seed if netem_seed is not None else net_cfg.get('seed')

        controller = MainController(host=host, port=port, scheduler_name=scheduler, state_db_path=state_db_path, netem_profile=profile, netem_seed=seed)

        # Optionally load PS
        if ps_checkpoint_path and os.path.exists(ps_checkpoint_path):
            try:
                controller.ps_load(ps_checkpoint_path)
            except Exception:
                pass

        workers_cfg = cfg.get('workers', [])
        workers = []
        for wc in workers_cfg:
            w = WorkerNode(
                name=wc.get('name', 'w'),
                memory=float(wc.get('memory', 16)),
                flops_per_second=int(wc.get('flops_per_second', wc.get('flops', 100))),
                network_bandwidth=int(wc.get('network_bandwidth', wc.get('bandwidth', 1000))),
                host=wc.get('host', host),
                port=int(wc.get('port', 0)),
                controller_address=f"http://{host}:{controller.communicator.port}",
                netem_profile=profile,
                netem_seed=seed,
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

        sim_cfg = cfg.get('simulation', {})
        heartbeat_cycles = int(sim_cfg.get('heartbeat_cycles', 3))
        heartbeat_interval = float(sim_cfg.get('heartbeat_interval', 1))
        scheduling_cycles = int(sim_cfg.get('scheduling_cycles', cfg.get('cycles', 3)))
        scheduling_interval = float(sim_cfg.get('scheduling_interval', 2))

        # Heartbeats (finite cycles)
        def send_heartbeats(worker):
            for _ in range(heartbeat_cycles):
                worker.send_heartbeat()
                time.sleep(heartbeat_interval)

        hb_threads = []
        for w in workers:
            t = threading.Thread(target=send_heartbeats, args=(w,))
            t.daemon = True
            t.start()
            hb_threads.append(t)

        # Scheduling cycles
        for _ in range(scheduling_cycles):
            controller.run_scheduling_cycle()
            time.sleep(scheduling_interval)

        status = controller.get_system_status()
        print(status)

        # Optionally save PS
        if ps_checkpoint_path:
            try:
                controller.ps_save(ps_checkpoint_path)
            except Exception:
                pass

        # Cleanup
        try:
            controller.communicator.stop_server()
        except Exception:
            pass
        for w in workers:
            try:
                w.communicator.stop_server()
            except Exception:
                pass
        return status

    # Fallback to legacy behavior when no config provided
    controller = MainController(host='localhost', port=8000, netem_profile=netem_profile, netem_seed=netem_seed)

    worker1 = WorkerNode(name='worker-1', memory=16, flops_per_second=100, network_bandwidth=1000,
                         host='localhost', port=8001, controller_address='http://localhost:8000', netem_profile=netem_profile, netem_seed=netem_seed)
    worker2 = WorkerNode(name='worker-2', memory=32, flops_per_second=200, network_bandwidth=1000,
                         host='localhost', port=8002, controller_address='http://localhost:8000', netem_profile=netem_profile, netem_seed=netem_seed)

    controller.register_worker(worker1.id, f'http://{worker1.communicator.host}:{worker1.communicator.port}', worker1)
    controller.register_worker(worker2.id, f'http://{worker2.communicator.host}:{worker2.communicator.port}', worker2)

    job_manager = JobManager(controller)
    job_manager.submit_job({'name': 'my-training-job', 'num_tasks': 4})

    def send_heartbeats(worker):
        for _ in range(3):
            worker.send_heartbeat()
            time.sleep(1)

    threading.Thread(target=send_heartbeats, args=(worker1,), daemon=True).start()
    threading.Thread(target=send_heartbeats, args=(worker2,), daemon=True).start()

    for i in range(5):
        controller.run_scheduling_cycle()
        time.sleep(2)

    status = controller.get_system_status()
    print(status)

    controller.communicator.stop_server()
    worker1.communicator.stop_server()
    worker2.communicator.stop_server()
    return status


if __name__ == '__main__':
    run_simulation()
