"""
Simulation of distributed training with real PyTorch models.
"""

import time
import threading
from dist_llm_train.config import load_config
from dist_llm_train.logging_utils import configure_logging
import os

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.models.model_loader import ModelLoader


def run_ml_training_simulation(config_path: str = 'config.yaml', state_db_path: str = None, ps_checkpoint_path: str = None):
    """
    Runs a simulation with real ML training using PyTorch.
    """
    print("="*80)
    print("ML TRAINING SIMULATION - Using Real PyTorch Models")
    print("="*80)

    # Configure logging and load configuration
    configure_logging()
    config = load_config(config_path)
    controller_config = config['controller']
    workers_config = config['workers']
    model_config_name = config['model']['name']
    training_config = config['training']
    tasks_config = config['tasks']
    simulation_config = config['simulation']

    # Create the main controller
    controller = MainController(
        host=controller_config['host'],
        port=controller_config['port'],
        scheduler_name=controller_config.get('scheduler', 'gale-shapley'),
        state_db_path=state_db_path,
    )

    # Optionally load parameter server checkpoint
    if ps_checkpoint_path and os.path.exists(ps_checkpoint_path):
        try:
            controller.ps_load(ps_checkpoint_path)
        except Exception:
            pass

    # Create and register worker nodes
    workers = []
    for worker_conf in workers_config:
        worker = WorkerNode(
            name=worker_conf['name'],
            memory=worker_conf['memory'],
            flops_per_second=worker_conf['flops_per_second'],
            network_bandwidth=worker_conf['network_bandwidth'],
            host=worker_conf['host'],
            port=worker_conf['port'],
            controller_address=f"http://{controller_config['host']}:{controller_config['port']}"
        )
        workers.append(worker)
        controller.register_worker(
            worker.id,
            f'http://{worker.communicator.host}:{worker.communicator.port}',
            worker
        )

    # Get model configuration
    model_config = ModelLoader.get_model_config(model_config_name)

    # Create ML training tasks
    print("\n[Simulation] Creating ML training tasks...")
    tasks = []
    for i, task_config in enumerate(tasks_config):
        task = TrainingTask(
            task_id=task_config['id'],
            model_name=model_config['type'],
            model_layer=i,
            model_shard_size=task_config['model_shard_size'],
            data_size=task_config['data_size'],
            required_flops=task_config['required_flops'],
            priority=task_config['priority'],
            model_config=model_config,
            training_config=training_config
        )
        tasks.append(task)
        controller.add_task(task)

    print(f"[Simulation] Created {len(tasks)} ML training tasks")
    print(f"[Simulation] Model: {model_config['type']}")
    print(f"[Simulation] Training config: {training_config}")

    # Start sending heartbeats from workers
    def send_heartbeats(worker):
        for _ in range(simulation_config['heartbeat_cycles']):
            worker.send_heartbeat()
            time.sleep(simulation_config['heartbeat_interval'])

    heartbeat_threads = []
    for worker in workers:
        thread = threading.Thread(target=send_heartbeats, args=(worker,))
        thread.daemon = True
        thread.start()
        heartbeat_threads.append(thread)

    # Run scheduling cycles
    print("\n[Simulation] Starting scheduling cycles...")
    for i in range(simulation_config['scheduling_cycles']):
        print(f"\n--- Scheduling Cycle {i+1} ---")
        controller.run_scheduling_cycle()
        time.sleep(simulation_config['scheduling_interval'])

    # Wait for tasks to complete
    print("\n[Simulation] Waiting for tasks to complete...")
    while len(controller.completed_tasks) < len(tasks):
        time.sleep(5)

    # Get final system status
    print("\n" + "="*80)
    print("FINAL SYSTEM STATUS")
    print("="*80)
    status = controller.get_system_status()
    print(status)

    # Print training metrics
    print("\n" + "="*80)
    print("TRAINING METRICS")
    print("="*80)
    for task in controller.completed_tasks:
        if hasattr(task, 'metrics') and task.metrics:
            print(f"\nTask {task.id}:")
            print(f"  Final Loss: {task.metrics.get('final_loss', 'N/A'):.4f}")
            print(f"  Num Batches: {task.metrics.get('num_batches', 'N/A')}")
            print(f"  Num Epochs: {task.metrics.get('num_epochs', 'N/A')}")

    # Optionally save parameter server checkpoint
    if ps_checkpoint_path:
        try:
            controller.ps_save(ps_checkpoint_path)
        except Exception:
            pass

    # Stop the servers
    print("\n[Simulation] Stopping servers...")
    controller.communicator.stop_server()
    for worker in workers:
        worker.communicator.stop_server()

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    return status


if __name__ == '__main__':
    run_ml_training_simulation()
