"""
Simulation of distributed training with real PyTorch models.
"""

import time
import threading

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.models.model_loader import ModelLoader


def run_ml_training_simulation():
    """
    Runs a simulation with real ML training using PyTorch.
    """
    print("="*80)
    print("ML TRAINING SIMULATION - Using Real PyTorch Models")
    print("="*80)

    # Create the main controller
    controller = MainController(host='localhost', port=8000)

    # Create worker nodes
    worker1 = WorkerNode(
        name='worker-1',
        memory=16,
        flops_per_second=100,
        network_bandwidth=1000,
        host='localhost',
        port=8001,
        controller_address='http://localhost:8000'
    )
    worker2 = WorkerNode(
        name='worker-2',
        memory=32,
        flops_per_second=200,
        network_bandwidth=1000,
        host='localhost',
        port=8002,
        controller_address='http://localhost:8000'
    )

    # Register workers with the controller
    controller.register_worker(
        worker1.id,
        f'http://{worker1.communicator.host}:{worker1.communicator.port}',
        worker1
    )
    controller.register_worker(
        worker2.id,
        f'http://{worker2.communicator.host}:{worker2.communicator.port}',
        worker2
    )

    # Get model configuration
    model_config = ModelLoader.get_model_config('tiny-lstm')

    # Create training configuration
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 8,
        'num_epochs': 2,
        'num_samples': 50,  # Small for quick testing
        'seq_length': 32
    }

    # Create ML training tasks
    print("\n[Simulation] Creating ML training tasks...")
    tasks = []
    for i in range(2):
        task = TrainingTask(
            task_id=f"ml-task-{i}",
            model_name=model_config['type'],
            model_layer=i,
            model_shard_size=0.5,  # GB
            data_size=0.1,  # GB
            required_flops=1000,
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
        for _ in range(30):  # Run for 30 heartbeat cycles
            worker.send_heartbeat()
            time.sleep(5)

    heartbeat_thread1 = threading.Thread(target=send_heartbeats, args=(worker1,))
    heartbeat_thread2 = threading.Thread(target=send_heartbeats, args=(worker2,))
    heartbeat_thread1.daemon = True
    heartbeat_thread2.daemon = True
    heartbeat_thread1.start()
    heartbeat_thread2.start()

    # Run scheduling cycles
    print("\n[Simulation] Starting scheduling cycles...")
    for i in range(5):
        print(f"\n--- Scheduling Cycle {i+1} ---")
        controller.run_scheduling_cycle()
        time.sleep(15)  # Wait longer for ML training to complete

    # Get final system status
    print("\n" + "="*80)
    print("FINAL SYSTEM STATUS")
    print("="*80)
    controller.get_system_status()

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

    # Stop the servers
    print("\n[Simulation] Stopping servers...")
    controller.communicator.stop_server()
    worker1.communicator.stop_server()
    worker2.communicator.stop_server()

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    run_ml_training_simulation()
