from typing import List, Dict, Tuple

from dist_llm_train.task.training_task import TrainingTask
from .base import BaseScheduler

class GaleShapleyScheduler(BaseScheduler):
    """
    Implements the Gale-Shapley stable matching algorithm to assign tasks to workers.
    In this implementation, tasks "propose" to workers.
    """
    def __init__(self, tasks: List[TrainingTask], workers: List[Dict]):
        self.tasks = {task.id: task for task in tasks}
        self.workers = {worker['info'].id: worker for worker in workers}

        # A dictionary to track which worker each task has proposed to.
        # The value is the index in the task's preference list.
        self.proposals_made: Dict[str, int] = {task.id: 0 for task in tasks}

        # Initialize the free tasks list with all task IDs.
        self.free_tasks: List[str] = list(self.tasks.keys())

        # This will hold the final matching: {worker_id: task_id}
        self.matches: Dict[str, str] = {}

    def _prepare_for_matching(self):
        """Calculates the preference lists for all tasks and workers."""
        print("Calculating preferences for tasks and workers...")
        worker_nodes = [w['info'] for w in self.workers.values()]
        for task in self.tasks.values():
            task.calculate_preferences(worker_nodes)

        for worker in self.workers.values():
            worker['info'].calculate_preferences(list(self.tasks.values()))

    def schedule(self) -> Dict[str, str]:
        """
        Performs the stable matching algorithm.

        Returns:
            A dictionary representing the stable matching, where keys are
            worker IDs and values are the assigned task IDs.
        """
        self._prepare_for_matching()

        while self.free_tasks:
            task_id = self.free_tasks[0]
            task = self.tasks[task_id]

            if self.proposals_made[task_id] >= len(task.preferences):
                # This task has proposed to all possible workers and been rejected.
                # It will remain unassigned.
                self.free_tasks.pop(0)
                continue

            # Get the next worker on the task's preference list.
            worker_id = task.preferences[self.proposals_made[task_id]]
            worker = self.workers[worker_id]['info']

            print(f"Task {task_id} proposes to Worker {worker_id}.")

            # The task has now made a proposal to this worker.
            self.proposals_made[task_id] += 1

            current_assignment_id = worker.assigned_task_id

            if current_assignment_id is None:
                # Worker is free, accepts the proposal.
                print(f"  - Worker {worker_id} is free and accepts.")
                worker.assigned_task_id = task_id
                task.assigned_worker_id = worker_id
                self.free_tasks.pop(0)
            else:
                # Worker is already assigned, check its preferences.
                try:
                    current_assignment_rank = worker.preferences.index(current_assignment_id)
                    new_proposal_rank = worker.preferences.index(task_id)

                    if new_proposal_rank < current_assignment_rank:
                        # Worker prefers the new task.
                        print(f"  - Worker {worker_id} prefers Task {task_id} over {current_assignment_id}.")

                        # The old task becomes free again.
                        old_task = self.tasks[current_assignment_id]
                        old_task.assigned_worker_id = None
                        self.free_tasks.append(current_assignment_id)

                        # Assign the new task to the worker.
                        worker.assigned_task_id = task_id
                        task.assigned_worker_id = worker_id
                        self.free_tasks.pop(0)
                    else:
                        # Worker rejects the proposal.
                        print(f"  - Worker {worker_id} rejects Task {task_id}.")
                except ValueError:
                    # This case should ideally not be reached if preferences are well-formed.
                    print(f"  - Worker {worker_id} has no preference for Task {task_id}, rejects.")

        # Compile the final list of matches from the worker assignments.
        self.matches = {w['info'].id: w['info'].assigned_task_id for w in self.workers.values() if w['info'].assigned_task_id is not None}
        return self.matches