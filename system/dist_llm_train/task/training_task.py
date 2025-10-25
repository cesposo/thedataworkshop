from typing import List, Dict, Any, Optional, TYPE_CHECKING

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
    def __init__(self, task_id: str, model_name: str, model_layer: int,
                 model_shard_size: float, data_size: float, required_flops: int,
                 optimizer_state_size: float = 0, gradient_size: float = 0, activation_size: float = 0,
                 model_config: Optional[Dict[str, Any]] = None,
                 training_config: Optional[Dict[str, Any]] = None):
        """
        Initializes a TrainingTask.

        Args:
            task_id (str): A unique identifier for the task.
            model_name (str): The name of the model being trained.
            model_layer (int): The specific layer of the model this task is for.
            model_shard_size (float): The memory for the model shard (in GB).
            data_size (float): The size of the data for this task (in GB).
            required_flops (int): The estimated floating-point operations required.
            optimizer_state_size (float): Memory for the optimizer state (in GB).
            gradient_size (float): Memory for gradients (in GB).
            activation_size (float): Memory for activations (in GB).
            model_config (dict, optional): Configuration for the model (type, size, etc.)
            training_config (dict, optional): Configuration for training (lr, batch_size, etc.)
        """
        self.id = task_id
        self.model_name = model_name
        self.model_layer = model_layer
        self.model_shard_size = model_shard_size
        self.data_size = data_size
        self.required_flops = required_flops
        self.optimizer_state_size = optimizer_state_size
        self.gradient_size = gradient_size
        self.activation_size = activation_size

        # ML-specific configurations
        self.model_config = model_config or {}
        self.training_config = training_config or {}

        # Runtime state
        self.preferences: List[str] = []  # A ranked list of worker IDs
        self.assigned_worker_id: str = None
        self.metrics: Dict[str, Any] = {}  # Training metrics (loss, accuracy, etc.)

    def __repr__(self) -> str:
        return (f"TrainingTask(id={self.id}, model={self.model_name}, layer={self.model_layer}, "
                f"mem={self.get_total_memory_req():.2f}GB, flops={self.required_flops})")

    def get_total_memory_req(self) -> float:
        """Calculates the total memory requirement for the task."""
        return (self.model_shard_size + self.data_size + self.optimizer_state_size +
                self.gradient_size + self.activation_size)

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