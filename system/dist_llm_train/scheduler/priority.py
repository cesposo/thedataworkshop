from typing import List, Dict

from dist_llm_train.task.training_task import TrainingTask
from .base import BaseScheduler
from .interfaces import SchedulerInput, SchedulerOutput

class PriorityScheduler(BaseScheduler):
    """
    A simple scheduler that assigns tasks based on their priority.
    """
    SUPPORTS_STRUCTURED_INPUT = True
    def __init__(self, tasks: List[TrainingTask], workers: List[Dict]):
        self.tasks = tasks
        self.workers = workers

    def schedule(self) -> Dict[str, str]:
        """
        Sorts tasks by priority and assigns them to available workers.

        Returns:
            A dictionary representing the matching, where keys are
            task IDs and values are the assigned worker IDs.
        """
        # Sort tasks by priority in descending order
        sorted_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)

        # Get available workers
        available_workers = [w for w in self.workers if w['status'] == 'available']

        matches = {}
        for task in sorted_tasks:
            if not available_workers:
                break  # No more available workers

            # Find a worker that can accommodate the task
            for i, worker in enumerate(available_workers):
                if worker['info'].memory >= task.get_total_memory_req():
                    matches[task.id] = worker['info'].id
                    available_workers.pop(i)  # Worker is now assigned
                    break
        
        return matches

    # Optional structured-input API
    def schedule_from_input(self, sched_input: SchedulerInput) -> SchedulerOutput:
        """Structured scheduling API that greedily assigns by task priority.

        Only memory fit is enforced; telemetry is ignored in this policy.
        """
        # Sort tasks by priority
        sorted_tasks = sorted(sched_input.tasks, key=lambda t: t.priority, reverse=True)
        available = [w for w in sched_input.workers if w.status == 'available']

        matches: Dict[str, str] = {}
        for task in sorted_tasks:
            if not available:
                break
            # pick first worker that fits memory
            for i, w in enumerate(available):
                if w.memory >= task.total_memory_req:
                    matches[task.id] = w.id
                    available.pop(i)
                    break
        return SchedulerOutput(assignments=matches)
