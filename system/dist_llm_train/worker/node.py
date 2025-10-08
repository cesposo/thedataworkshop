import uuid
import time
from typing import List, Dict, TYPE_CHECKING

# By using TYPE_CHECKING, we can import the TrainingTask for type hints
# without causing a circular import error at runtime.
if TYPE_CHECKING:
    from dist_llm_train.task.training_task import TrainingTask

class WorkerNode:
    """
    Represents a worker node in the distributed training cluster.
    It has a specific resource profile and can rank its preference for tasks.
    """
    def __init__(self, name: str, memory: float, flops_per_second: int, network_bandwidth: int):
        """
        Initializes a WorkerNode.

        Args:
            name (str): A human-readable name for the worker.
            memory (float): Total available memory in GB.
            flops_per_second (int): A measure of the worker's compute capability.
            network_bandwidth (int): Network bandwidth in Mbps.
        """
        self.id = f"{name}-{str(uuid.uuid4())[:4]}"
        self.memory = memory
        self.flops_per_second = flops_per_second
        self.network_bandwidth = network_bandwidth

        self.status = 'available'  # Can be 'available', 'busy', 'offline'
        self.preferences: List[str] = [] # A ranked list of task IDs
        self.assigned_task_id: str = None

        self.last_heartbeat = time.time()

    def __repr__(self) -> str:
        return f"WorkerNode(id={self.id}, mem={self.memory}GB, status='{self.status}')"

    def send_heartbeat(self):
        """Simulates the worker sending a heartbeat to the controller."""
        self.last_heartbeat = time.time()

    def get_compute_score(self) -> float:
        """A simple score to represent the worker's overall compute power."""
        # Only available workers can provide a score.
        if self.status != 'available':
            return -1.0
        return self.flops_per_second

    def calculate_preferences(self, tasks: List['TrainingTask']):
        """
        Calculates and ranks its preference for a list of tasks.
        """
        scores = {}
        for task in tasks:
            if self.status != 'available' or self.memory < task.get_total_memory_req():
                score = -1.0
            else:
                score = task.required_flops
            scores[task.id] = score

        sorted_task_ids = sorted(scores.keys(), key=lambda t_id: scores[t_id], reverse=True)
        self.preferences = [t_id for t_id in sorted_task_ids if scores[t_id] > 0]

    def handle_proposal(self, task_id: str) -> bool:
        """
        Handles a proposal from a task.
        """
        if self.assigned_task_id is None:
            self.assigned_task_id = task_id
            return True

        try:
            current_assignment_rank = self.preferences.index(self.assigned_task_id)
            new_proposal_rank = self.preferences.index(task_id)

            if new_proposal_rank < current_assignment_rank:
                self.assigned_task_id = task_id
                return True
            else:
                return False
        except ValueError:
            return False