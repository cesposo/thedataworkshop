"""
Rotation data structures for stable matching lattices.

This module implements the mathematical structures needed to represent
and manipulate rotations in the stable matching lattice, following the
Irving-Leather algorithm for rotation extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger("dist_llm_train.scheduler.rotation")


@dataclass
class Rotation:
    """
    Represents a minimal cycle of reassignments between stable matchings.

    A rotation is a sequence of (task, worker) pairs that can be "eliminated"
    from the preference structure to move from one stable matching to another.

    In the context of stable matching theory, eliminating a rotation ρ means:
    - Each task in the rotation becomes matched to its next choice
    - Certain pairs are removed from preference lists
    - The result is a new stable matching

    Attributes:
        id: Unique identifier for this rotation
        edges: List of (task_id, worker_id) pairs forming the rotation cycle
        eliminated_pairs: Set of (task_id, worker_id) pairs removed when applying rotation
    """
    id: int
    edges: List[Tuple[str, str]] = field(default_factory=list)
    eliminated_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    def __repr__(self) -> str:
        edge_str = " -> ".join([f"({t},{w})" for t, w in self.edges])
        return f"Rotation({self.id}: {edge_str})"

    def to_dict(self) -> Dict:
        """Serialize rotation to dictionary for persistence."""
        return {
            'id': self.id,
            'edges': self.edges,
            'eliminated_pairs': list(self.eliminated_pairs)
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Rotation':
        """Deserialize rotation from dictionary."""
        return Rotation(
            id=data['id'],
            edges=data['edges'],
            eliminated_pairs=set(tuple(p) for p in data['eliminated_pairs'])
        )


@dataclass
class RotationPoset:
    """
    Directed acyclic graph (DAG) of rotation precedence relationships.

    The rotation poset captures the partial order on rotations: rotation ρ₁
    precedes ρ₂ if ρ₁ must be eliminated before ρ₂ can be eliminated while
    maintaining stability.

    This structure allows efficient navigation through the lattice of stable
    matchings by following valid paths in the poset.

    Attributes:
        rotations: Dictionary mapping rotation ID to Rotation object
        precedence_graph: Adjacency list of precedence relationships
                         (rotation_id -> set of rotation IDs that must come before it)
    """
    rotations: Dict[int, Rotation] = field(default_factory=dict)
    precedence_graph: Dict[int, Set[int]] = field(default_factory=dict)

    def add_rotation(self, rotation: Rotation):
        """Add a rotation to the poset."""
        self.rotations[rotation.id] = rotation
        if rotation.id not in self.precedence_graph:
            self.precedence_graph[rotation.id] = set()

    def add_precedence(self, from_id: int, to_id: int):
        """
        Add precedence constraint: from_id must be eliminated before to_id.

        Args:
            from_id: ID of rotation that must come first
            to_id: ID of rotation that depends on from_id
        """
        if to_id not in self.precedence_graph:
            self.precedence_graph[to_id] = set()
        self.precedence_graph[to_id].add(from_id)

    def get_executable_rotations(self, eliminated: Set[int]) -> List[int]:
        """
        Get list of rotation IDs that can be executed next.

        A rotation is executable if all its precedence constraints are satisfied
        (i.e., all rotations it depends on have already been eliminated).

        Args:
            eliminated: Set of rotation IDs that have already been eliminated

        Returns:
            List of rotation IDs that can be executed next
        """
        executable = []
        for rot_id in self.rotations:
            if rot_id in eliminated:
                continue
            # Check if all predecessors have been eliminated
            predecessors = self.precedence_graph.get(rot_id, set())
            if predecessors.issubset(eliminated):
                executable.append(rot_id)
        return executable

    def apply_rotation(self, matching: Dict[str, str], rotation_id: int) -> Dict[str, str]:
        """
        Apply a rotation to a matching to produce a new stable matching.

        Args:
            matching: Current matching {task_id: worker_id}
            rotation_id: ID of rotation to apply

        Returns:
            New matching after applying rotation
        """
        if rotation_id not in self.rotations:
            raise ValueError(f"Rotation {rotation_id} not found in poset")

        rotation = self.rotations[rotation_id]
        new_matching = matching.copy()

        # Apply the rotation cycle
        # Each task in the rotation gets matched to the next worker in the cycle
        if not rotation.edges:
            return new_matching

        # Build cycle mapping: task -> next_worker
        task_to_next_worker = {}
        for i, (task_id, worker_id) in enumerate(rotation.edges):
            next_idx = (i + 1) % len(rotation.edges)
            next_worker = rotation.edges[next_idx][1]
            task_to_next_worker[task_id] = next_worker

        # Apply the cycle
        for task_id, next_worker in task_to_next_worker.items():
            new_matching[task_id] = next_worker

        logger.debug(f"Applied rotation {rotation_id}: {rotation.edges}")
        return new_matching

    def topological_sort(self) -> List[int]:
        """
        Return a topological ordering of rotations respecting precedence.

        This gives a valid sequence in which rotations can be eliminated
        to traverse from task-optimal to worker-optimal matching.

        Returns:
            List of rotation IDs in topological order
        """
        # Kahn's algorithm for topological sort
        in_degree = {rot_id: 0 for rot_id in self.rotations}

        # Calculate in-degrees
        for rot_id, predecessors in self.precedence_graph.items():
            in_degree[rot_id] = len(predecessors)

        # Queue of nodes with no incoming edges
        queue = [rot_id for rot_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort for determinism
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            # Find all nodes that depend on current
            for rot_id, predecessors in self.precedence_graph.items():
                if current in predecessors:
                    in_degree[rot_id] -= 1
                    if in_degree[rot_id] == 0:
                        queue.append(rot_id)

        # Check for cycles
        if len(result) != len(self.rotations):
            logger.warning("Cycle detected in rotation poset")

        return result

    def get_path_length(self, from_rotations: Set[int], to_rotations: Set[int]) -> int:
        """
        Calculate the minimum number of rotation steps between two sets.

        Args:
            from_rotations: Set of rotations already eliminated
            to_rotations: Set of rotations to eliminate

        Returns:
            Number of rotation steps (difference in cardinality)
        """
        return len(to_rotations - from_rotations)

    def to_dict(self) -> Dict:
        """Serialize poset to dictionary for persistence."""
        return {
            'rotations': {rot_id: rot.to_dict() for rot_id, rot in self.rotations.items()},
            'precedence_graph': {
                rot_id: list(predecessors)
                for rot_id, predecessors in self.precedence_graph.items()
            }
        }

    @staticmethod
    def from_dict(data: Dict) -> 'RotationPoset':
        """Deserialize poset from dictionary."""
        poset = RotationPoset()
        poset.rotations = {
            int(rot_id): Rotation.from_dict(rot_data)
            for rot_id, rot_data in data['rotations'].items()
        }
        poset.precedence_graph = {
            int(rot_id): set(predecessors)
            for rot_id, predecessors in data['precedence_graph'].items()
        }
        return poset

    def __repr__(self) -> str:
        return f"RotationPoset(rotations={len(self.rotations)}, edges={sum(len(p) for p in self.precedence_graph.values())})"
