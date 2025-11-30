"""
Demonstration of lattice-aware scheduling for distributed LLM training.

This script shows how to use the stable matching lattice for:
1. Exploring multiple stable matchings
2. Fast reconfiguration via rotations
3. Learned preferences from historical data
"""

import time
import logging
from typing import List

from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.scheduler.gale_shapley import GaleShapleyScheduler
from dist_llm_train.scheduler.preference_learning import PerformancePredictor, TrainingRun
from dist_llm_train.scheduler.learned_preferences import LearnedPreferenceBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_tasks() -> List[TrainingTask]:
    """Create sample training tasks."""
    tasks = [
        TrainingTask(
            task_id='task-1',
            model_name='gpt-small',
            model_layer=0,
            model_shard_size=2.0,
            data_size=0.5,
            required_flops=10000,
            priority=1
        ),
        TrainingTask(
            task_id='task-2',
            model_name='gpt-small',
            model_layer=1,
            model_shard_size=2.0,
            data_size=0.5,
            required_flops=10000,
            priority=2
        ),
        TrainingTask(
            task_id='task-3',
            model_name='gpt-small',
            model_layer=2,
            model_shard_size=2.0,
            data_size=0.5,
            required_flops=10000,
            priority=3
        ),
    ]
    return tasks


def create_demo_workers() -> List[dict]:
    """Create sample worker nodes."""
    # Note: In real usage, these would be actual WorkerNode instances with RPC servers
    # For this demo, we create mock worker dictionaries
    workers = []

    for i in range(3):
        # Create a mock worker info object with required attributes
        class MockWorkerInfo:
            def __init__(self, worker_id, memory, flops, bandwidth):
                self.id = worker_id
                self.memory = memory
                self.flops_per_second = flops
                self.network_bandwidth = bandwidth
                self.gpu_count = 1
                self.status = 'available'
                self.assigned_task_id = None
                self.preferences = []

            def calculate_preferences(self, tasks):
                # Simple preference: prefer tasks that fit and have higher priority
                scores = {}
                for task in tasks:
                    if task.get_total_memory_req() > self.memory:
                        scores[task.id] = -1
                    else:
                        scores[task.id] = task.priority + (self.flops_per_second / 1000.0)
                self.preferences = [t for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True) if s > 0]

        worker_info = MockWorkerInfo(
            worker_id=f'worker-{i+1}',
            memory=16.0 + i * 8,  # Varying memory: 16, 24, 32 GB
            flops=5000 + i * 2000,  # Varying compute: 5000, 7000, 9000
            bandwidth=1000 + i * 500  # Varying bandwidth: 1000, 1500, 2000
        )

        workers.append({
            'info': worker_info,
            'address': f'http://localhost:800{i+1}',
            'status': 'available',
            'last_heartbeat': time.time()
        })

    return workers


def demo_basic_lattice():
    """Demonstrate basic lattice computation."""
    logger.info("=== Demo 1: Basic Lattice Computation ===")

    tasks = create_demo_tasks()
    workers = create_demo_workers()

    # Create scheduler with lattice mode
    scheduler = GaleShapleyScheduler(
        tasks,
        workers,
        compute_lattice=True,
        max_lattice_size=20
    )

    # Build lattice
    logger.info("Building stable matching lattice...")
    lattice = scheduler.build_lattice()

    logger.info(f"Lattice contains {lattice.size()} stable matchings")
    logger.info(f"Rotation poset has {len(lattice.rotation_poset.rotations)} rotations")

    # Show task-optimal matching
    task_optimal = lattice.find_matching_by_id(lattice.task_optimal_id)
    logger.info(f"Task-optimal matching: {task_optimal}")

    # Show worker-optimal matching (if different)
    if lattice.worker_optimal_id != lattice.task_optimal_id:
        worker_optimal = lattice.find_matching_by_id(lattice.worker_optimal_id)
        logger.info(f"Worker-optimal matching: {worker_optimal}")

    # Show all matchings
    logger.info("\nAll stable matchings:")
    for i, matching in enumerate(lattice.enumerate_all_matchings()):
        logger.info(f"  Matching {i}: {matching}")

    return lattice


def demo_rotation_reconfiguration(lattice):
    """Demonstrate fast reconfiguration via rotations."""
    logger.info("\n=== Demo 2: Rotation-Based Reconfiguration ===")

    tasks = create_demo_tasks()
    workers = create_demo_workers()

    scheduler = GaleShapleyScheduler(tasks, workers)
    scheduler.lattice = lattice  # Use pre-built lattice

    # Start with task-optimal matching
    current_matching = lattice.find_matching_by_id(lattice.task_optimal_id)
    logger.info(f"Current matching: {current_matching}")

    # Simulate worker failure
    failed_workers = {'worker-2'}
    logger.info(f"Worker(s) failed: {failed_workers}")

    # Reconfigure via rotations
    logger.info("Attempting reconfiguration via rotation path...")
    new_matching = scheduler.reconfigure_via_rotations(current_matching, failed_workers)

    if new_matching:
        logger.info(f"Successfully reconfigured to: {new_matching}")
        logger.info("Reconfiguration completed via rotation operations")
    else:
        logger.warning("No valid reconfiguration found in lattice")


def demo_learned_preferences():
    """Demonstrate learned preference construction."""
    logger.info("\n=== Demo 3: Learned Preferences ===")

    # Create synthetic training history
    logger.info("Creating synthetic training history...")
    training_data = []

    for i in range(20):
        # Simulate different task-worker combinations with varying performance
        task_features = {
            'model_shard_size': 2.0,
            'data_size': 0.5,
            'required_flops': 10000 + i * 100,
            'total_memory_req': 2.5,
            'priority': 1
        }

        worker_features = {
            'flops_per_second': 5000 + (i % 3) * 2000,
            'memory': 16.0 + (i % 3) * 8,
            'network_bandwidth': 1000,
            'gpu_count': 1
        }

        # Simulate throughput (higher FLOPS = better performance)
        throughput = worker_features['flops_per_second'] / 100.0 + (i % 5) * 2

        run = TrainingRun(
            task_id=f'task-{i}',
            worker_id=f'worker-{i % 3}',
            task_features=task_features,
            worker_features=worker_features,
            completion_time_s=100.0 / throughput,
            throughput_tokens_per_sec=throughput,
            success=True,
            timestamp=time.time()
        )
        training_data.append(run)

    # Train predictor
    logger.info("Training performance predictor...")
    predictor = PerformancePredictor(model_type='baseline')  # Use baseline (no sklearn needed)
    metrics = predictor.fit(training_data)
    logger.info(f"Training metrics: {metrics}")

    # Build preferences using learned model
    logger.info("Building learned preferences...")
    tasks = create_demo_tasks()
    workers_list = [w['info'] for w in create_demo_workers()]

    pref_builder = LearnedPreferenceBuilder(predictor=predictor, objective='throughput')

    for task in tasks:
        prefs = pref_builder.build_task_preferences(task, workers_list)
        logger.info(f"Task {task.id} learned preferences: {prefs}")


def demo_alternative_matchings():
    """Demonstrate sampling alternative stable matchings."""
    logger.info("\n=== Demo 4: Alternative Matchings for Robustness ===")

    tasks = create_demo_tasks()
    workers = create_demo_workers()

    scheduler = GaleShapleyScheduler(tasks, workers, compute_lattice=True, max_lattice_size=20)
    lattice = scheduler.build_lattice()

    # Get alternative matchings for fallback
    alternatives = scheduler.get_alternative_matchings(count=3)

    logger.info(f"Found {len(alternatives)} alternative stable matchings:")
    for i, matching in enumerate(alternatives):
        logger.info(f"  Alternative {i+1}: {matching}")

    logger.info("\nThese can be used as fallback options if primary matching fails")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Lattice-Aware Scheduling Demonstration")
    print("="*70 + "\n")

    # Demo 1: Basic lattice computation
    lattice = demo_basic_lattice()

    # Demo 2: Rotation-based reconfiguration
    demo_rotation_reconfiguration(lattice)

    # Demo 3: Learned preferences
    demo_learned_preferences()

    # Demo 4: Alternative matchings
    demo_alternative_matchings()

    print("\n" + "="*70)
    print("Demonstration Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Lattice structure provides multiple stable matching options")
    print("2. Rotation operations enable fast reconfiguration")
    print("3. Learned preferences improve matching quality over time")
    print("4. Alternative matchings provide robustness to failures")
    print("\nSee docs/lattice_implementation_progress.md for integration details")


if __name__ == '__main__':
    main()
