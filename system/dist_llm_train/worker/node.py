import uuid
import time
import threading
import logging
from typing import List, Dict, TYPE_CHECKING, Optional

from dist_llm_train.communication.rpc import RPCCommunicator
from dist_llm_train.communication.chaos import build_communicator
from .task_executor import TaskExecutor
from dist_llm_train.communication.communicator import Communicator

# By using TYPE_CHECKING, we can import the TrainingTask for type hints
# without causing a circular import error at runtime.
if TYPE_CHECKING:
    from dist_llm_train.task.training_task import TrainingTask

class WorkerNode:
    """
    Represents a worker node in the distributed training cluster.
    It has a specific resource profile and can rank its preference for tasks.
    """
    def __init__(self, name: str, memory: float, flops_per_second: int, network_bandwidth: int,
                 host: str, port: int, controller_address: str,
                 gpu_type: str = 'N/A', gpu_count: int = 0, pci_e_lanes: int = 0, nvlink_version: str = 'N/A',
                 communicator: Optional[Communicator] = None, netem_profile: Optional[dict] = None, netem_seed: Optional[int] = None):
        """
        Initializes a WorkerNode.

        Args:
            name (str): A human-readable name for the worker.
            memory (float): Total available memory in GB.
            flops_per_second (int): A measure of the worker's compute capability.
            network_bandwidth (int): Network bandwidth in Mbps.
            host (str): The host address for the worker's RPC server.
            port (int): The port for the worker's RPC server.
            controller_address (str): The address of the main controller.
            gpu_type (str): The type of GPU (e.g., 'A100', 'V100').
            gpu_count (int): The number of GPUs available.
            pci_e_lanes (int): The number of PCIe lanes.
            nvlink_version (str): The NVLink version.
        """
        self.id = f"{name}-{str(uuid.uuid4())[:4]}"
        self.memory = memory
        self.flops_per_second = flops_per_second
        self.network_bandwidth = network_bandwidth
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.pci_e_lanes = pci_e_lanes
        self.nvlink_version = nvlink_version
        self.controller_address = controller_address

        self.status = 'available'  # Can be 'available', 'busy', 'offline'
        self.preferences: List[str] = [] # A ranked list of task IDs
        self.assigned_task_id: str = None

        self.last_heartbeat = time.time()

        if communicator is not None:
            self.communicator = communicator
        else:
            self.communicator = build_communicator(host, port, netem_profile=netem_profile, seed=netem_seed)
        self.communicator.register_function(self.receive_task, 'receive_task')
        if hasattr(self.communicator, "start_server"):
            self.communicator.start_server()

        self.executor = TaskExecutor(
            self.id,
            self.communicator,
            self.controller_address,
            status_callback=self._on_task_finished,
        )
        self.logger = logging.getLogger(f"dist_llm_train.worker.{self.id}")

    def __repr__(self) -> str:
        return (f"WorkerNode(id={self.id}, mem={self.memory}GB, gpu='{self.gpu_type}', "
                f"gpu_count={self.gpu_count}, status='{self.status}')")

    def send_heartbeat(self):
        """Sends a heartbeat to the controller."""
        self.logger.info("Sending heartbeat to controller.")
        try:
            self.communicator.send(self.controller_address, {
                'method': 'heartbeat',
                'params': [self.id, self.status]
            })
            self.last_heartbeat = time.time()
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {e}")

    def receive_task(self, task_dict: Dict):
        """
        Receives a task from the controller and starts processing it.

        Args:
            task_dict: Dictionary representation of the task
        """
        from dist_llm_train.task.training_task import TrainingTask

        # Reconstruct TrainingTask from dict
        task = TrainingTask(
            task_id=task_dict['id'],
            model_name=task_dict['model_name'],
            model_layer=task_dict['model_layer'],
            model_shard_size=task_dict['model_shard_size'],
            data_size=task_dict['data_size'],
            required_flops=task_dict['required_flops'],
            optimizer_state_size=task_dict.get('optimizer_state_size', 0),
            gradient_size=task_dict.get('gradient_size', 0),
            activation_size=task_dict.get('activation_size', 0),
            model_config=task_dict.get('model_config', {}),
            training_config=task_dict.get('training_config', {}),
            priority=task_dict.get('priority', 0)
        )

        self.logger.info(f"Received task: {task.id}")
        self.assigned_task_id = task.id
        self.status = 'busy'

        # Execute the task in a new thread to avoid blocking the RPC server
        task_thread = threading.Thread(target=self.executor.execute_task, args=(task,))
        task_thread.start()

    def _on_task_finished(self, task: 'TrainingTask', success: bool):
        """Reset worker state after a task finishes."""
        self.status = 'available'
        self.assigned_task_id = None
        self.last_heartbeat = time.time()
        result = "completed" if success else "failed"
        self.logger.info(f"Task {task.id} {result}; worker marked available.")

    def get_compute_score(self) -> float:
        """A simple score to represent the worker's overall compute power."""
        # Only available workers can provide a score.
        if self.status != 'available':
            return -1.0
        return self.flops_per_second

    def calculate_preferences(self, tasks: List['TrainingTask']):
        """
        Calculates and ranks its preference for a list of tasks.

        A worker's preference is determined by how well a task matches
        the worker's capabilities. A worker prefers tasks that fit within
        its memory and can be computed efficiently.

        Args:
            tasks (List[TrainingTask]): The list of training tasks to rank.
        """
        scores = {}
        for task in tasks:
            # Check if the task can fit in memory
            if task.get_total_memory_req() > self.memory:
                score = -1.0
            else:
                # Prefer tasks that use resources efficiently
                # Higher score for tasks that require more compute (better utilization)
                memory_efficiency = task.get_total_memory_req() / self.memory
                compute_match = min(task.required_flops / self.flops_per_second, 1.0)
                score = (compute_match + memory_efficiency) / 2.0
            scores[task.id] = score

        # Sort tasks by score (higher is better)
        sorted_task_ids = sorted(scores.keys(), key=lambda t_id: scores[t_id], reverse=True)
        self.preferences = [t_id for t_id in sorted_task_ids if scores[t_id] > 0]
