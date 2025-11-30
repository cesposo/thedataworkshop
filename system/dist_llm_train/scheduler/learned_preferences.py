"""
Learned preference construction from performance predictions.

This module uses performance predictors to construct preference rankings
for tasks and workers based on predicted outcomes.
"""

import logging
from typing import List, Dict, Optional
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.worker.node import WorkerNode
from .preference_learning import PerformancePredictor

logger = logging.getLogger("dist_llm_train.scheduler.learned_preferences")


class LearnedPreferenceBuilder:
    """
    Constructs preference rankings from performance predictions.

    This implements preference construction as a data-driven problem:
    instead of hand-coded heuristics, preferences are derived from
    learned models that predict task-worker performance.
    """

    def __init__(self,
                 predictor: Optional[PerformancePredictor] = None,
                 objective: str = 'throughput'):
        """
        Initialize learned preference builder.

        Args:
            predictor: Trained PerformancePredictor (if None, uses baseline)
            objective: Optimization objective ('throughput', 'latency', 'balanced')
        """
        self.predictor = predictor or PerformancePredictor(model_type='baseline')
        self.objective = objective
        self.preference_cache = {}  # Cache for (task_id, timestamp) -> preferences
        self.cache_ttl = 60.0  # Cache validity in seconds

    def build_task_preferences(self,
                               task: TrainingTask,
                               workers: List[WorkerNode],
                               use_cache: bool = True) -> List[str]:
        """
        Rank workers by predicted performance for this task.

        Args:
            task: Training task to build preferences for
            workers: List of available workers
            use_cache: Whether to use cached preferences

        Returns:
            List of worker IDs ranked from best to worst
        """
        # Extract task features
        task_features = {
            'model_shard_size': task.model_shard_size,
            'data_size': task.data_size,
            'required_flops': task.required_flops,
            'total_memory_req': task.get_total_memory_req(),
            'priority': task.priority
        }

        # Score each worker
        scores = {}
        for worker in workers:
            if worker.status != 'available':
                scores[worker.id] = float('-inf')
                continue

            # Extract worker features
            worker_features = {
                'flops_per_second': worker.flops_per_second,
                'memory': worker.memory,
                'network_bandwidth': worker.network_bandwidth,
                'gpu_count': worker.gpu_count,
                'avg_tokens_per_sec': 0.0,  # TODO: get from telemetry
                'avg_step_time_s': 0.0
            }

            # Predict score based on objective
            if self.objective == 'throughput':
                scores[worker.id] = self.predictor.predict_throughput(
                    task_features, worker_features
                )
            elif self.objective == 'latency':
                # Lower latency is better, so negate
                predicted_time = self.predictor.predict_completion_time(
                    task_features, worker_features
                )
                scores[worker.id] = -predicted_time
            elif self.objective == 'balanced':
                # Combination of throughput and latency
                throughput = self.predictor.predict_throughput(
                    task_features, worker_features
                )
                latency = self.predictor.predict_completion_time(
                    task_features, worker_features
                )
                scores[worker.id] = throughput * 0.7 - latency * 0.3
            else:
                logger.warning(f"Unknown objective '{self.objective}', using throughput")
                scores[worker.id] = self.predictor.predict_throughput(
                    task_features, worker_features
                )

        # Sort by score (descending)
        ranked_workers = sorted(scores.keys(), key=lambda w: scores[w], reverse=True)

        # Filter out workers with negative scores (incompatible)
        ranked_workers = [w for w in ranked_workers if scores[w] > 0]

        logger.debug(f"Task {task.id} preferences: {ranked_workers[:3]}... (showing top 3)")
        return ranked_workers

    def build_worker_preferences(self,
                                 worker: WorkerNode,
                                 tasks: List[TrainingTask]) -> List[str]:
        """
        Rank tasks by predicted utilization efficiency for this worker.

        Worker preferences are based on how well they can utilize their
        resources for each task (higher utilization = higher preference).

        Args:
            worker: Worker node to build preferences for
            tasks: List of available tasks

        Returns:
            List of task IDs ranked from best to worst
        """
        # Extract worker features
        worker_features = {
            'flops_per_second': worker.flops_per_second,
            'memory': worker.memory,
            'network_bandwidth': worker.network_bandwidth,
            'gpu_count': worker.gpu_count,
            'avg_tokens_per_sec': 0.0,
            'avg_step_time_s': 0.0
        }

        # Score each task based on utilization
        scores = {}
        for task in tasks:
            # Check if task fits in memory
            if task.get_total_memory_req() > worker.memory:
                scores[task.id] = float('-inf')
                continue

            task_features = {
                'model_shard_size': task.model_shard_size,
                'data_size': task.data_size,
                'required_flops': task.required_flops,
                'total_memory_req': task.get_total_memory_req(),
                'priority': task.priority
            }

            # Predict performance
            throughput = self.predictor.predict_throughput(task_features, worker_features)

            # Add utilization bonus: prefer tasks that use resources efficiently
            memory_utilization = task.get_total_memory_req() / worker.memory
            compute_utilization = min(1.0, task.required_flops / worker.flops_per_second)

            utilization_score = (memory_utilization + compute_utilization) / 2.0
            scores[task.id] = throughput * (0.7 + 0.3 * utilization_score)

        # Sort by score (descending)
        ranked_tasks = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

        # Filter out incompatible tasks
        ranked_tasks = [t for t in ranked_tasks if scores[t] > 0]

        logger.debug(f"Worker {worker.id} preferences: {ranked_tasks[:3]}... (showing top 3)")
        return ranked_tasks

    def apply_preferences_to_tasks_and_workers(self,
                                               tasks: List[TrainingTask],
                                               workers: List[WorkerNode]):
        """
        Apply learned preferences to all tasks and workers in-place.

        This modifies the 'preferences' attribute of tasks and workers.

        Args:
            tasks: List of training tasks
            workers: List of worker nodes
        """
        # Build preferences for each task
        for task in tasks:
            task.preferences = self.build_task_preferences(task, workers)

        # Build preferences for each worker
        for worker in workers:
            worker.preferences = self.build_worker_preferences(worker, tasks)

        logger.info(f"Applied learned preferences to {len(tasks)} tasks and {len(workers)} workers")

    def set_objective(self, objective: str):
        """
        Change the optimization objective.

        Args:
            objective: New objective ('throughput', 'latency', 'balanced')
        """
        if objective not in ['throughput', 'latency', 'balanced']:
            logger.warning(f"Unknown objective '{objective}', keeping current")
            return

        self.objective = objective
        self.preference_cache.clear()  # Clear cache when objective changes
        logger.info(f"Set preference objective to: {objective}")

    def update_predictor(self, predictor: PerformancePredictor):
        """
        Update the performance predictor.

        Args:
            predictor: New trained predictor
        """
        self.predictor = predictor
        self.preference_cache.clear()  # Clear cache when predictor changes
        logger.info("Updated performance predictor")
