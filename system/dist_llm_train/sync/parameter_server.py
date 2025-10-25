"""Parameter server for distributed training."""

import torch
import threading
from typing import Dict, Any, Optional
import copy


class ParameterServer:
    """
    A simple parameter server that maintains the global model state
    and coordinates gradient updates from multiple workers.
    """

    def __init__(self, initial_model_state: Optional[Dict[str, torch.Tensor]] = None):
        """
        Initialize parameter server.

        Args:
            initial_model_state: Initial model state dictionary
        """
        self.model_state = initial_model_state or {}
        self.gradient_buffer: Dict[str, list] = {}
        self.lock = threading.Lock()
        self.version = 0  # Track model version

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.

        Returns:
            Dictionary of current model parameters
        """
        with self.lock:
            return {
                'parameters': copy.deepcopy(self.model_state),
                'version': self.version
            }

    def push_gradients(self, worker_id: str, gradients: Dict[str, torch.Tensor]):
        """
        Push gradients from a worker to the parameter server.

        Args:
            worker_id: ID of the worker
            gradients: Dictionary of gradients
        """
        with self.lock:
            if worker_id not in self.gradient_buffer:
                self.gradient_buffer[worker_id] = []

            self.gradient_buffer[worker_id].append(gradients)

    def aggregate_and_update(self, learning_rate: float = 0.001) -> bool:
        """
        Aggregate gradients and update parameters.

        Args:
            learning_rate: Learning rate for parameter update

        Returns:
            True if update was successful, False otherwise
        """
        with self.lock:
            if not self.gradient_buffer:
                return False

            # Aggregate gradients (simple averaging)
            aggregated_grads = {}

            for worker_id, grad_list in self.gradient_buffer.items():
                if not grad_list:
                    continue

                # Use most recent gradients
                worker_grads = grad_list[-1]

                for param_name, grad in worker_grads.items():
                    if param_name not in aggregated_grads:
                        aggregated_grads[param_name] = []
                    aggregated_grads[param_name].append(grad)

            # Average and apply to model state
            for param_name, grad_list in aggregated_grads.items():
                if param_name in self.model_state:
                    avg_grad = torch.mean(torch.stack(grad_list), dim=0)
                    self.model_state[param_name] -= learning_rate * avg_grad

            # Clear buffer and increment version
            self.gradient_buffer.clear()
            self.version += 1

            return True

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters (useful for initialization).

        Args:
            parameters: Dictionary of model parameters
        """
        with self.lock:
            self.model_state = copy.deepcopy(parameters)
            self.version += 1

    def get_version(self) -> int:
        """Get current model version."""
        with self.lock:
            return self.version

    def get_num_workers(self) -> int:
        """Get number of workers that have pushed gradients."""
        with self.lock:
            return len(self.gradient_buffer)


class SimpleSyncCoordinator:
    """
    Coordinates synchronous training across workers.

    Workers wait at barriers until all have completed a step,
    then gradients are aggregated and parameters are updated.
    """

    def __init__(self, num_workers: int):
        """
        Initialize sync coordinator.

        Args:
            num_workers: Expected number of workers
        """
        self.num_workers = num_workers
        self.barrier_count = 0
        self.waiting_workers = set()
        self.condition = threading.Condition()
        self.gradients_ready = {}

    def wait_for_all(self, worker_id: str) -> bool:
        """
        Worker waits until all workers have reached the barrier.

        Args:
            worker_id: ID of the worker

        Returns:
            True when all workers are ready
        """
        with self.condition:
            self.waiting_workers.add(worker_id)

            if len(self.waiting_workers) == self.num_workers:
                # All workers ready, release them
                self.waiting_workers.clear()
                self.barrier_count += 1
                self.condition.notify_all()
                return True
            else:
                # Wait for other workers
                self.condition.wait()
                return True

    def submit_gradients(self, worker_id: str, gradients: Dict[str, Any]):
        """
        Submit gradients at barrier.

        Args:
            worker_id: ID of the worker
            gradients: Gradient dictionary
        """
        with self.condition:
            self.gradients_ready[worker_id] = gradients

    def get_aggregated_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get aggregated gradients from all workers.

        Returns:
            Aggregated gradients
        """
        with self.condition:
            if len(self.gradients_ready) != self.num_workers:
                return {}

            # Aggregate
            aggregated = {}
            param_names = list(next(iter(self.gradients_ready.values())).keys())

            for param_name in param_names:
                grads = [worker_grads[param_name]
                         for worker_grads in self.gradients_ready.values()]
                aggregated[param_name] = torch.mean(torch.stack(grads), dim=0)

            self.gradients_ready.clear()
            return aggregated
