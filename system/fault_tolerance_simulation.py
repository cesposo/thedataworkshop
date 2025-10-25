import time
from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask

def run_fault_tolerance_simulation():
    """
    Runs a simulation to test the system's fault tolerance.
    """
    print("--- Starting Fault Tolerance Simulation ---")

    # 1. Initialize the controller with a short timeout for testing
    controller = MainController(heartbeat_timeout=2)

    # 2. Create and register workers
    workers = [
        WorkerNode(name="W1-Reliable", memory=24, flops_per_second=300, network_bandwidth=1000),
        WorkerNode(name="W2-Unreliable", memory=16, flops_per_second=150, network_bandwidth=800),
        WorkerNode(name="W3-Backup", memory=12, flops_per_second=80, network_bandwidth=500),
    ]
    for w in workers:
        controller.register_worker(w)

    # 3. Create and add tasks
    tasks = [
        TrainingTask(task_id="TaskA", model_shard_size=14, data_size=2, required_flops=200),
        TrainingTask(task_id="TaskB", model_shard_size=10, data_size=2, required_flops=120),
    ]
    for t in tasks:
        controller.add_task(t)

    # Initial system status
    controller.get_system_status()

    # --- Simulation Cycle 1: Initial Assignment ---
    print("\n>>> Running Scheduling Cycle 1 <<<")
    controller.run_scheduling_cycle()
    controller.get_system_status()

    # --- Simulate a Worker Failure ---
    print("\n>>> Simulating Worker Failure <<<")

    # All workers send heartbeats initially
    for worker in controller.workers.values():
        worker.send_heartbeat()
    print("All workers sent a heartbeat.")

    # Wait for a period longer than the timeout
    print(f"Waiting for {controller.heartbeat_timeout + 1} seconds to simulate a timeout...")
    time.sleep(controller.heartbeat_timeout + 1)

    # Now, only the reliable workers send another heartbeat
    for w_id, worker in controller.workers.items():
        if "Unreliable" not in w_id:
            worker.send_heartbeat()
            print(f"Worker {w_id} sent a heartbeat.")

    # The "Unreliable" worker has now missed its heartbeat.
    # The controller should detect this and requeue its task.
    controller.check_worker_health()
    controller.get_system_status()

    # --- Simulation Cycle 2: Re-scheduling the orphaned task ---
    print("\n>>> Running Scheduling Cycle 2 (after failure detection) <<<")
    controller.run_scheduling_cycle()
    controller.get_system_status()

    print("--- Simulation Finished ---")

if __name__ == '__main__':
    run_fault_tolerance_simulation()