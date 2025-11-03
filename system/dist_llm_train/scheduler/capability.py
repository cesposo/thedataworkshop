"""Capability-based scheduler.

Greedy assignment of tasks to workers using a composite score combining
compute capacity (FLOPS), bandwidth, memory headroom, and (optionally)
telemetry EWMA tokens/sec.
"""

from typing import List, Dict

from dist_llm_train.task.training_task import TrainingTask
from .base import BaseScheduler
from .interfaces import SchedulerInput, SchedulerOutput


class CapabilityScheduler(BaseScheduler):
    """Greedy, telemetry-aware scheduler.

    Chooses the highest scoring worker per task with:
    score = FLOPS + 0.1*bandwidth + 0.01*headroom + 0.01*tokens/sec (EWMA preferred).
    Returns mapping `{task_id: worker_id}`.
    """

    SUPPORTS_STRUCTURED_INPUT = True

    def __init__(self, tasks: List[TrainingTask], workers: List[Dict]):
        self.tasks = tasks
        self.workers = workers

    def schedule(self) -> Dict[str, str]:
        # Collect available workers
        available = [w for w in self.workers if w.get('status') == 'available']

        # Helper to score worker for a task
        def score(worker, task: TrainingTask) -> float:
            info = worker['info']
            # Must fit in memory
            if info.memory < task.get_total_memory_req():
                return -1.0
            # Weighted combination: prioritize flops, then bandwidth, slight memory headroom
            flops = float(getattr(info, 'flops_per_second', 0))
            bw = float(getattr(info, 'network_bandwidth', 0))
            headroom = max(0.0, float(info.memory - task.get_total_memory_req()))
            return flops * 1.0 + bw * 0.1 + headroom * 0.01

        # Greedy: iterate tasks (higher priority first), assign best-scoring worker
        tasks_sorted = sorted(self.tasks, key=lambda t: t.priority, reverse=True)
        matches: Dict[str, str] = {}

        for task in tasks_sorted:
            if not available:
                break
            # Pick best worker for this task
            best = None
            best_score = -1.0
            for w in available:
                s = score(w, task)
                if s > best_score:
                    best_score = s
                    best = w
            if best is not None and best_score >= 0:
                matches[task.id] = best['info'].id
                available.remove(best)

        return matches

    def schedule_from_input(self, sched_input: SchedulerInput) -> SchedulerOutput:
        """Structured scheduling API using normalized task/worker/telemetry input."""
        available = [w for w in sched_input.workers if w.status == 'available']

        def score(worker, task) -> float:
            if worker.memory < task.total_memory_req:
                return -1.0
            flops = float(worker.flops_per_second)
            bw = float(worker.network_bandwidth)
            headroom = max(0.0, float(worker.memory - task.total_memory_req))
            # Favor workers with recent higher tokens/sec if available
            tps_bonus = 0.0
            if worker.telemetry:
                if worker.telemetry.avg_tokens_per_sec is not None:
                    tps_bonus = float(worker.telemetry.avg_tokens_per_sec) * 0.01
                elif worker.telemetry.tokens_per_sec is not None:
                    tps_bonus = float(worker.telemetry.tokens_per_sec) * 0.01
            return flops * 1.0 + bw * 0.1 + headroom * 0.01 + tps_bonus

        tasks_sorted = sorted(sched_input.tasks, key=lambda t: t.priority, reverse=True)
        matches: Dict[str, str] = {}
        for task in tasks_sorted:
            if not available:
                break
            best = None
            best_score = -1.0
            for w in available:
                s = score(w, task)
                if s > best_score:
                    best_score = s
                    best = w
            if best is not None and best_score >= 0:
                matches[task.id] = best.id
                available.remove(best)
        return SchedulerOutput(assignments=matches)
