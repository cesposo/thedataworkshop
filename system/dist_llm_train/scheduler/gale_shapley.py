from typing import List, Dict, Tuple, Optional, Set
import logging

from dist_llm_train.task.training_task import TrainingTask
from .base import BaseScheduler
from .lattice import StableMatchingLattice, LatticeBuilder

class GaleShapleyScheduler(BaseScheduler):
    """
    Implements the Gale-Shapley stable matching algorithm to assign tasks to workers.
    In this implementation, tasks "propose" to workers.

    Enhanced with lattice support for multi-matching exploration and
    rotation-based reconfiguration.
    """
    def __init__(self, tasks: List[TrainingTask], workers: List[Dict],
                 compute_lattice: bool = False, max_lattice_size: int = 100):
        """
        Initialize Gale-Shapley scheduler.

        Args:
            tasks: List of training tasks
            workers: List of worker dictionaries
            compute_lattice: If True, build complete stable matching lattice
            max_lattice_size: Maximum number of matchings to enumerate
        """
        self.tasks = {task.id: task for task in tasks}
        self.workers = {worker['info'].id: worker for worker in workers}

        # A dictionary to track which worker each task has proposed to.
        # The value is the index in the task's preference list.
        self.proposals_made: Dict[str, int] = {task.id: 0 for task in tasks}

        # Initialize the free tasks list with all task IDs.
        self.free_tasks: List[str] = list(self.tasks.keys())

        # This will hold the final matching: {worker_id: task_id}
        self.matches: Dict[str, str] = {}

        # Lattice support
        self.compute_lattice = compute_lattice
        self.max_lattice_size = max_lattice_size
        self.lattice: Optional[StableMatchingLattice] = None

    def _prepare_for_matching(self):
        """Calculates the preference lists for all tasks and workers."""
        self.logger.debug("Calculating preferences for tasks and workers…")
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
            task IDs and values are the assigned worker IDs.
        """
        self._prepare_for_matching()
        self.reset()

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

            self.logger.debug(f"Task {task_id} proposes to Worker {worker_id}.")

            # The task has now made a proposal to this worker.
            self.proposals_made[task_id] += 1

            current_assignment_id = worker.assigned_task_id

            if current_assignment_id is None:
                # Worker is free, accepts the proposal.
                self.logger.debug(f"  - Worker {worker_id} is free and accepts.")
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
                        self.logger.debug(f"  - Worker {worker_id} prefers Task {task_id} over {current_assignment_id}.")

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
                        self.logger.debug(f"  - Worker {worker_id} rejects Task {task_id}.")
                except ValueError:
                    # This case should ideally not be reached if preferences are well-formed.
                    self.logger.debug(f"  - Worker {worker_id} has no preference for Task {task_id}, rejects.")

        # Compile the final list of matches from the task assignments.
        self.matches = {task.id: task.assigned_worker_id for task in self.tasks.values() if task.assigned_worker_id is not None}
        return self.matches

    def reset(self):
        """Resets the state of the scheduler, tasks, and workers."""
        self.proposals_made = {task.id: 0 for task in self.tasks.values()}
        self.free_tasks = list(self.tasks.keys())
        self.matches = {}
        for task in self.tasks.values():
            task.assigned_worker_id = None
        for worker in self.workers.values():
            worker['info'].assigned_task_id = None

    def build_lattice(self) -> StableMatchingLattice:
        """
        Compute the complete lattice of stable matchings.

        Returns:
            StableMatchingLattice containing all stable matchings
        """
        # First get task-optimal matching
        task_optimal = self.schedule()

        # Extract preference lists
        task_prefs = {task_id: task.preferences.copy() for task_id, task in self.tasks.items()}
        worker_prefs = {worker_id: worker['info'].preferences.copy()
                       for worker_id, worker in self.workers.items()}

        # Build lattice
        builder = LatticeBuilder(max_size=self.max_lattice_size)
        self.lattice = builder.build_lattice(task_optimal, task_prefs, worker_prefs)

        self.logger.info(f"Built lattice with {self.lattice.size()} stable matchings")
        return self.lattice

    def schedule_with_lattice(self, objective: str = 'task_optimal') -> Dict[str, str]:
        """
        Return a stable matching from the lattice optimizing the given objective.

        Args:
            objective: One of 'task_optimal', 'worker_optimal', or 'balanced'

        Returns:
            Matching dictionary {task_id: worker_id}
        """
        if self.lattice is None:
            self.build_lattice()

        if objective == 'task_optimal':
            return self.lattice.find_matching_by_id(self.lattice.task_optimal_id)
        elif objective == 'worker_optimal':
            if self.lattice.worker_optimal_id is not None:
                return self.lattice.find_matching_by_id(self.lattice.worker_optimal_id)
            return self.lattice.find_matching_by_id(self.lattice.task_optimal_id)
        elif objective == 'balanced':
            # Return a matching in the middle of the lattice
            all_ids = list(self.lattice.matchings.keys())
            mid_id = all_ids[len(all_ids) // 2] if all_ids else 0
            return self.lattice.find_matching_by_id(mid_id)
        else:
            self.logger.warning(f"Unknown objective '{objective}', using task_optimal")
            return self.lattice.find_matching_by_id(self.lattice.task_optimal_id)

    def get_alternative_matchings(self, count: int = 5) -> List[Dict[str, str]]:
        """
        Sample diverse stable matchings for fallback options.

        Args:
            count: Number of alternative matchings to return

        Returns:
            List of matching dictionaries
        """
        if self.lattice is None:
            self.build_lattice()

        alternatives = []
        all_ids = list(self.lattice.matchings.keys())

        # Sample evenly across the lattice
        step = max(1, len(all_ids) // count)
        for i in range(0, min(count, len(all_ids))):
            idx = min(i * step, len(all_ids) - 1)
            matching = self.lattice.find_matching_by_id(all_ids[idx])
            if matching:
                alternatives.append(matching)

        return alternatives

    def reconfigure_via_rotations(self,
                                   current_matching: Dict[str, str],
                                   failed_workers: Set[str]) -> Optional[Dict[str, str]]:
        """
        Find a nearby stable matching avoiding failed workers via rotation path.

        Args:
            current_matching: Current assignment
            failed_workers: Set of worker IDs that have failed

        Returns:
            New matching avoiding failed workers, or None if not possible
        """
        if self.lattice is None:
            self.logger.warning("Lattice not built, cannot reconfigure via rotations")
            return None

        # Find a matching that avoids failed workers
        result = self.lattice.get_matching_avoiding_workers(failed_workers)

        if result is None:
            self.logger.warning("No stable matching avoids all failed workers")
            return None

        matching_id, new_matching = result
        self.logger.info(f"Reconfigured to matching {matching_id} avoiding {failed_workers}")
        return new_matching

    logger = logging.getLogger("dist_llm_train.scheduler.gale_shapley")
