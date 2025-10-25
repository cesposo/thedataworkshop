"""Gradient aggregation strategies for distributed training."""

import torch
from typing import List, Dict, Any
import numpy as np


class GradientAggregator:
    """
    Aggregates gradients from multiple workers for distributed training.

    Supports different aggregation strategies (average, sum, etc.)
    """

    def __init__(self, strategy: str = 'average'):
        """
        Initialize gradient aggregator.

        Args:
            strategy: Aggregation strategy ('average', 'sum', 'weighted_average')
        """
        self.strategy = strategy
        self.gradient_buffer: Dict[str, List[Dict[str, Any]]] = {}

    def collect_gradients(self, worker_id: str, gradients: Dict[str, torch.Tensor]):
        """
        Collect gradients from a worker.

        Args:
            worker_id: ID of the worker
            gradients: Dictionary mapping parameter names to gradient tensors
        """
        if worker_id not in self.gradient_buffer:
            self.gradient_buffer[worker_id] = []

        # Convert tensors to serializable format if needed
        serializable_grads = {}
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                serializable_grads[name] = grad.cpu().numpy()
            else:
                serializable_grads[name] = grad

        self.gradient_buffer[worker_id].append(serializable_grads)

    def aggregate(self, param_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Aggregate collected gradients across all workers.

        Args:
            param_names: List of parameter names to aggregate

        Returns:
            Dictionary mapping parameter names to aggregated gradients
        """
        if not self.gradient_buffer:
            return {}

        aggregated = {}

        for param_name in param_names:
            # Collect all gradients for this parameter from all workers
            grads_for_param = []

            for worker_id, gradient_list in self.gradient_buffer.items():
                if gradient_list:  # Use most recent gradients
                    latest_grads = gradient_list[-1]
                    if param_name in latest_grads:
                        grads_for_param.append(latest_grads[param_name])

            if not grads_for_param:
                continue

            # Aggregate based on strategy
            if self.strategy == 'average':
                aggregated[param_name] = np.mean(grads_for_param, axis=0)
            elif self.strategy == 'sum':
                aggregated[param_name] = np.sum(grads_for_param, axis=0)
            elif self.strategy == 'weighted_average':
                # Simple uniform weighting for now
                aggregated[param_name] = np.mean(grads_for_param, axis=0)
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

        return aggregated

    def clear_buffer(self):
        """Clear the gradient buffer."""
        self.gradient_buffer.clear()

    def get_num_workers(self) -> int:
        """Returns the number of workers with collected gradients."""
        return len(self.gradient_buffer)


class AllReduceAggregator:
    """
    Simulates AllReduce-style gradient aggregation.

    This is a simplified version that doesn't require actual distributed
    communication primitives like MPI or NCCL.
    """

    @staticmethod
    def all_reduce_average(gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform AllReduce with averaging across all workers.

        Args:
            gradients_list: List of gradient dictionaries from each worker

        Returns:
            Dictionary of averaged gradients
        """
        if not gradients_list:
            return {}

        # Get parameter names from first worker
        param_names = list(gradients_list[0].keys())
        averaged_grads = {}

        for param_name in param_names:
            # Stack gradients from all workers
            grads = [worker_grads[param_name] for worker_grads in gradients_list]

            # Average
            stacked = torch.stack(grads)
            averaged_grads[param_name] = torch.mean(stacked, dim=0)

        return averaged_grads

    @staticmethod
    def synchronize_parameters(models: List[torch.nn.Module]):
        """
        Synchronize parameters across multiple model instances.

        Args:
            models: List of model instances to synchronize
        """
        if len(models) < 2:
            return

        # Use first model as source
        source_state = models[0].state_dict()

        # Copy to all other models
        for model in models[1:]:
            model.load_state_dict(source_state)
