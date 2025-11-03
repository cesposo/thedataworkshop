"""Parameter server for distributed training."""

import torch
import threading
from typing import Dict, Any, Optional
import copy
import os
import json
import time


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

    def aggregate_and_update(self, learning_rate: float = 0.001, rule: str = 'mean', trim_ratio: float = 0.0) -> bool:
        """Aggregate gradients and update parameters.

        Args:
            learning_rate: Learning rate for parameter update.
            rule: Aggregation rule. 'mean' or 'trimmed_mean'.
            trim_ratio: Fraction [0,1] of gradients to trim in total (half from each side) for 'trimmed_mean'.

        Returns:
            True if update was successful, False otherwise.
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

            # Robust aggregation
            def _aggregate(param_name: str, grad_list):
                stack = torch.stack(grad_list)
                if rule == 'mean' or len(grad_list) < 3:
                    return torch.mean(stack, dim=0)
                if rule == 'trimmed_mean':
                    # Trim extremes per element
                    # k is number to trim from each side; trim_ratio is total fraction trimmed
                    k = int((trim_ratio * len(grad_list)) // 2)
                    if k == 0:
                        return torch.mean(stack, dim=0)
                    # Sort along worker dimension for each element
                    sorted_vals, _ = torch.sort(stack, dim=0)
                    trimmed = sorted_vals[k:len(grad_list)-k]
                    if trimmed.numel() == 0:
                        return torch.mean(stack, dim=0)
                    return torch.mean(trimmed, dim=0)
                # Fallback for unsupported rules
                return torch.mean(stack, dim=0)

            # Apply update
            for param_name, grad_list in aggregated_grads.items():
                if param_name in self.model_state:
                    agg = _aggregate(param_name, grad_list)
                    self.model_state[param_name] -= learning_rate * agg

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

    # --- Persistence helpers ---
    def save_to_file(self, path: str) -> bool:
        """Save current parameters and version to a file using torch.save."""
        with self.lock:
            try:
                data = {
                    'version': self.version,
                    'model_state': {k: v.cpu() for k, v in self.model_state.items()},
                }
                torch.save(data, path)
                return True
            except Exception:
                return False

    def load_from_file(self, path: str) -> bool:
        """Load parameters and version from a file created by save_to_file."""
        try:
            data = torch.load(path, map_location='cpu')
            params = data.get('model_state', {})
            with self.lock:
                self.model_state = copy.deepcopy(params)
                self.version = int(data.get('version', self.version))
            return True
        except Exception:
            return False

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

    def __init__(self, num_workers: int, window_size: int = None, max_wait_s: float = None):
        """
        Initialize sync coordinator.

        Args:
            num_workers: Expected number of workers
        """
        self.num_workers = num_workers
        self.window_size = window_size or num_workers
        self.max_wait_s = max_wait_s
        self.barrier_count = 0
        self.waiting_workers = set()
        self.condition = threading.Condition()
        self.gradients_ready = {}
        self._first_arrival_ts = None

    def wait_for_all(self, worker_id: str) -> bool:
        """
        Worker waits until all workers have reached the barrier.

        Args:
            worker_id: ID of the worker

        Returns:
            True when all workers are ready
        """
        with self.condition:
            start_round = self.barrier_count
            now = time.time()
            self.waiting_workers.add(worker_id)
            if self._first_arrival_ts is None:
                self._first_arrival_ts = now

            def _release_barrier():
                self.waiting_workers.clear()
                self._first_arrival_ts = None
                self.barrier_count += 1
                self.condition.notify_all()

            while True:
                if self.barrier_count > start_round:
                    return True

                if len(self.waiting_workers) >= max(1, self.window_size):
                    _release_barrier()
                    return True

                wait_timeout = None
                if self.max_wait_s is not None and self._first_arrival_ts is not None:
                    elapsed = time.time() - self._first_arrival_ts
                    remaining = self.max_wait_s - elapsed
                    if remaining <= 0:
                        _release_barrier()
                        return True
                    wait_timeout = remaining

                self.condition.wait(timeout=wait_timeout)

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
