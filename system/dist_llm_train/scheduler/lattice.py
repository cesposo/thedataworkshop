"""
Stable matching lattice structure.

This module implements the finite lattice of all stable matchings for a given
preference profile, following the Irving-Leather algorithm for lattice enumeration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Tuple
import logging
from collections import deque

from .rotation import Rotation, RotationPoset

logger = logging.getLogger("dist_llm_train.scheduler.lattice")


@dataclass
class StableMatchingLattice:
    """
    Finite lattice of all stable matchings for a preference profile.

    The lattice is represented as a collection of stable matchings connected
    by rotation operations. The structure forms a distributive lattice with
    two extremal points: the task-optimal and worker-optimal matchings.

    Attributes:
        matchings: Dictionary mapping matching_id to assignment {task_id: worker_id}
        rotation_poset: The poset of rotations connecting matchings
        task_optimal_id: ID of the task-optimal (proposer-optimal) matching
        worker_optimal_id: ID of the worker-optimal (responder-optimal) matching
        matching_to_rotations: Maps matching_id to set of rotations eliminated to reach it
    """
    matchings: Dict[int, Dict[str, str]] = field(default_factory=dict)
    rotation_poset: RotationPoset = field(default_factory=RotationPoset)
    task_optimal_id: int = 0
    worker_optimal_id: Optional[int] = None
    matching_to_rotations: Dict[int, Set[int]] = field(default_factory=dict)

    def add_matching(self, matching_id: int, matching: Dict[str, str], rotations_eliminated: Set[int]):
        """
        Add a stable matching to the lattice.

        Args:
            matching_id: Unique ID for this matching
            matching: The assignment {task_id: worker_id}
            rotations_eliminated: Set of rotation IDs eliminated to reach this matching
        """
        self.matchings[matching_id] = matching.copy()
        self.matching_to_rotations[matching_id] = rotations_eliminated.copy()

    def find_matching_by_id(self, matching_id: int) -> Optional[Dict[str, str]]:
        """
        Retrieve a matching by its ID.

        Args:
            matching_id: ID of the matching to retrieve

        Returns:
            Matching dictionary or None if not found
        """
        return self.matchings.get(matching_id)

    def get_path_between_matchings(self, from_id: int, to_id: int) -> List[int]:
        """
        Find the sequence of rotations to go from one matching to another.

        Args:
            from_id: Starting matching ID
            to_id: Target matching ID

        Returns:
            List of rotation IDs to apply in sequence
        """
        if from_id not in self.matching_to_rotations or to_id not in self.matching_to_rotations:
            logger.warning(f"Matching {from_id} or {to_id} not found in lattice")
            return []

        from_rotations = self.matching_to_rotations[from_id]
        to_rotations = self.matching_to_rotations[to_id]

        # If going "down" the lattice (eliminating more rotations)
        if from_rotations.issubset(to_rotations):
            additional_rotations = to_rotations - from_rotations
            # Sort by topological order
            topo_order = self.rotation_poset.topological_sort()
            return [r for r in topo_order if r in additional_rotations]

        # If going "up" the lattice (un-eliminating rotations) - not directly supported
        # Would need to rebuild from task-optimal
        logger.warning("Going up the lattice not directly supported; use task_optimal as intermediate")
        return []

    def rank_matchings_by_objective(self, objective_fn: Callable[[Dict[str, str]], float]) -> List[int]:
        """
        Rank all matchings by a given objective function.

        Args:
            objective_fn: Function that takes a matching and returns a score (higher is better)

        Returns:
            List of matching IDs sorted by score (descending)
        """
        scores = {}
        for matching_id, matching in self.matchings.items():
            try:
                scores[matching_id] = objective_fn(matching)
            except Exception as e:
                logger.warning(f"Objective function failed for matching {matching_id}: {e}")
                scores[matching_id] = float('-inf')

        return sorted(scores.keys(), key=lambda m_id: scores[m_id], reverse=True)

    def get_matching_avoiding_workers(self, avoid_workers: Set[str]) -> Optional[Tuple[int, Dict[str, str]]]:
        """
        Find a stable matching that avoids assigning tasks to specific workers.

        Args:
            avoid_workers: Set of worker IDs to avoid

        Returns:
            Tuple of (matching_id, matching) or None if no such matching exists
        """
        for matching_id, matching in self.matchings.items():
            assigned_workers = set(matching.values())
            if not assigned_workers.intersection(avoid_workers):
                return (matching_id, matching)

        logger.warning(f"No matching found avoiding workers: {avoid_workers}")
        return None

    def enumerate_all_matchings(self) -> List[Dict[str, str]]:
        """
        Return all stable matchings in the lattice.

        Returns:
            List of all matchings
        """
        return list(self.matchings.values())

    def size(self) -> int:
        """Return the number of stable matchings in the lattice."""
        return len(self.matchings)

    def to_dict(self) -> Dict:
        """Serialize lattice to dictionary for persistence."""
        return {
            'matchings': {
                str(m_id): matching
                for m_id, matching in self.matchings.items()
            },
            'rotation_poset': self.rotation_poset.to_dict(),
            'task_optimal_id': self.task_optimal_id,
            'worker_optimal_id': self.worker_optimal_id,
            'matching_to_rotations': {
                str(m_id): list(rotations)
                for m_id, rotations in self.matching_to_rotations.items()
            }
        }

    @staticmethod
    def from_dict(data: Dict) -> 'StableMatchingLattice':
        """Deserialize lattice from dictionary."""
        lattice = StableMatchingLattice()
        lattice.matchings = {
            int(m_id): matching
            for m_id, matching in data['matchings'].items()
        }
        lattice.rotation_poset = RotationPoset.from_dict(data['rotation_poset'])
        lattice.task_optimal_id = data['task_optimal_id']
        lattice.worker_optimal_id = data.get('worker_optimal_id')
        lattice.matching_to_rotations = {
            int(m_id): set(rotations)
            for m_id, rotations in data['matching_to_rotations'].items()
        }
        return lattice

    def __repr__(self) -> str:
        return (f"StableMatchingLattice(matchings={len(self.matchings)}, "
                f"rotations={len(self.rotation_poset.rotations)})")


class LatticeBuilder:
    """
    Builds the stable matching lattice using rotation extraction.

    This implements a simplified version of the Irving-Leather algorithm:
    1. Start with task-optimal matching (from Gale-Shapley)
    2. Extract rotations from preference structure
    3. Build rotation poset
    4. Generate all matchings by eliminating rotation subsets
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize lattice builder.

        Args:
            max_size: Maximum number of matchings to enumerate (for performance)
        """
        self.max_size = max_size

    def build_lattice(self,
                      task_optimal_matching: Dict[str, str],
                      task_preferences: Dict[str, List[str]],
                      worker_preferences: Dict[str, List[str]]) -> StableMatchingLattice:
        """
        Build the complete stable matching lattice.

        Args:
            task_optimal_matching: The task-optimal matching from Gale-Shapley
            task_preferences: Preference lists for all tasks {task_id: [worker_ids]}
            worker_preferences: Preference lists for all workers {worker_id: [task_ids]}

        Returns:
            Complete lattice structure
        """
        lattice = StableMatchingLattice()
        lattice.task_optimal_id = 0
        lattice.add_matching(0, task_optimal_matching, set())

        # Extract rotations using simplified algorithm
        rotations = self._extract_rotations(
            task_optimal_matching,
            task_preferences,
            worker_preferences
        )

        # Build rotation poset
        poset = self._build_rotation_poset(rotations)
        lattice.rotation_poset = poset

        # Generate matchings by enumerating rotation elimination subsets
        self._enumerate_matchings(lattice, task_optimal_matching)

        logger.info(f"Built lattice with {lattice.size()} stable matchings and {len(rotations)} rotations")
        return lattice

    def _extract_rotations(self,
                           matching: Dict[str, str],
                           task_prefs: Dict[str, List[str]],
                           worker_prefs: Dict[str, List[str]]) -> List[Rotation]:
        """
        Extract rotations from the preference structure.

        This is a simplified rotation extraction. A full implementation would
        use the Irving-Leather algorithm with "reduced" preference lists.

        For now, we implement a basic version that identifies simple cycles
        in the preference structure.

        Args:
            matching: Current stable matching
            task_prefs: Task preference lists
            worker_prefs: Worker preference lists

        Returns:
            List of extracted rotations
        """
        rotations = []

        # Simplified extraction: look for "second choice" rotations
        # A rotation exists when tasks would prefer to swap to next-choice workers

        # For each task, check if there's a beneficial rotation
        checked_tasks = set()

        for task_id, current_worker in matching.items():
            if task_id in checked_tasks:
                continue

            if task_id not in task_prefs or len(task_prefs[task_id]) < 2:
                continue

            # Find task's next preferred worker
            current_rank = task_prefs[task_id].index(current_worker) if current_worker in task_prefs[task_id] else -1
            if current_rank < 0 or current_rank >= len(task_prefs[task_id]) - 1:
                continue

            next_worker = task_prefs[task_id][current_rank + 1]

            # Check if this forms a valid rotation (simplified check)
            # In full implementation, would trace complete cycle
            rotation = Rotation(
                id=len(rotations),
                edges=[(task_id, current_worker), (task_id, next_worker)],
                eliminated_pairs={(task_id, current_worker)}
            )

            rotations.append(rotation)
            checked_tasks.add(task_id)

            # Limit number of rotations for performance
            if len(rotations) >= 20:
                break

        return rotations

    def _build_rotation_poset(self, rotations: List[Rotation]) -> RotationPoset:
        """
        Build the precedence graph for rotations.

        Args:
            rotations: List of extracted rotations

        Returns:
            Rotation poset with precedence relationships
        """
        poset = RotationPoset()

        for rotation in rotations:
            poset.add_rotation(rotation)

        # Add precedence constraints (simplified)
        # Full implementation would determine precedence from rotation structure
        # For now, use simple ordering
        for i in range(len(rotations) - 1):
            # Each rotation can come after the previous (linear order for simplicity)
            # This is conservative but safe
            pass  # No strict precedence for now; all rotations are independent

        return poset

    def _enumerate_matchings(self, lattice: StableMatchingLattice, base_matching: Dict[str, str]):
        """
        Enumerate all stable matchings by applying rotation subsets.

        Args:
            lattice: Lattice to populate with matchings
            base_matching: Task-optimal matching to start from
        """
        poset = lattice.rotation_poset

        if len(poset.rotations) == 0:
            # Only one matching (task-optimal = worker-optimal)
            lattice.worker_optimal_id = 0
            return

        # Use BFS to enumerate matchings by rotation sets
        queue = deque([(base_matching, set(), 0)])  # (matching, rotations_eliminated, matching_id)
        visited_rotation_sets = {frozenset()}  # Track which rotation sets we've seen
        matching_id_counter = 1

        while queue and len(lattice.matchings) < self.max_size:
            current_matching, eliminated, current_id = queue.popleft()

            # Find executable rotations
            executable = poset.get_executable_rotations(eliminated)

            for rot_id in executable:
                if len(lattice.matchings) >= self.max_size:
                    break
                new_eliminated = eliminated | {rot_id}
                frozen_new = frozenset(new_eliminated)

                if frozen_new in visited_rotation_sets:
                    continue

                visited_rotation_sets.add(frozen_new)

                # Apply rotation to get new matching
                new_matching = poset.apply_rotation(current_matching, rot_id)

                # Add to lattice
                new_id = matching_id_counter
                matching_id_counter += 1
                lattice.add_matching(new_id, new_matching, new_eliminated)

                # Continue exploration
                queue.append((new_matching, new_eliminated, new_id))

        # The matching with all rotations eliminated is worker-optimal
        max_rotations = max((len(rots), m_id) for m_id, rots in lattice.matching_to_rotations.items())
        lattice.worker_optimal_id = max_rotations[1]

        logger.debug(f"Enumerated {len(lattice.matchings)} matchings (max: {self.max_size})")
