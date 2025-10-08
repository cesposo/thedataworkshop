from typing import List, TYPE_CHECKING

# By using TYPE_CHECKING, we can import the WorkerNode for type hints
# without causing a circular import error at runtime.
if TYPE_CHECKING:
    from dist_llm_train.worker.node import WorkerNode

class TrainingTask:
    """
    Represents a single training task (e.g., a microbatch for a model shard).
    It holds information about its computational needs and can rank its
    preferences for available workers.
    """
    def __init__(self, task_id: str, model_shard_size: float, data_size: float, required_flops: int):
        """
        Initializes a TrainingTask.

        Args:
            task_id (str): A unique identifier for the task.
            model_shard_size (float): The memory required for the model shard (in GB).
            data_size (float): The size of the data for this task (in GB).
            required_flops (int): The estimated floating-point operations required.
        """
        self.id = task_id
        self.model_shard_size = model_shard_size
        self.data_size = data_size
        self.required_flops = required_flops

        self.preferences: List[str] = []  # A ranked list of worker IDs
        self.assigned_worker_id: str = None

    def __repr__(self) -> str:
        return f"TrainingTask(id={self.id}, model_gb={self.model_shard_size}, flops={self.required_flops})"

    def get_total_memory_req(self) -> float:
        """Calculates the total memory requirement for the task."""
        return self.model_shard_size + self.data_size

    def calculate_preferences(self, workers: List['WorkerNode']):
        """
        Calculates and ranks its preference for a list of workers.

        A task's preference is determined by how well a worker's resources match
        the task's requirements. A worker that can process the task faster and
        has sufficient memory is preferred.

        Args:
            workers (List[WorkerNode]): The list of available worker nodes to rank.
        """
        scores = {}
        for worker in workers:
            if worker.status != 'available' or worker.memory < self.get_total_memory_req():
                score = -1.0
            else:
                score = worker.get_compute_score()
            scores[worker.id] = score

        sorted_worker_ids = sorted(scores.keys(), key=lambda w_id: scores[w_id], reverse=True)
        self.preferences = [w_id for w_id in sorted_worker_ids if scores[w_id] > 0]