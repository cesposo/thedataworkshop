"""Parameter server for distributed training."""

import torch
import threading
from typing import Dict, Any, Optional, Set
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

    def aggregate_and_update(self, learning_rate: float = 0.001, rule: str = 'mean', trim_ratio: float = 0.0,
                           krum_f: int = None, bulyan_f: int = None) -> bool:
        """Aggregate gradients and update parameters.

        Args:
            learning_rate: Learning rate for parameter update.
            rule: Aggregation rule. Options:
                - 'mean': Simple averaging (no Byzantine tolerance)
                - 'trimmed_mean': Trim extremes, moderate Byzantine tolerance
                - 'krum': Multi-Krum selection, strong Byzantine tolerance
                - 'bulyan': Bulyan aggregation, strongest Byzantine tolerance
            trim_ratio: Fraction [0,1] of gradients to trim in total for 'trimmed_mean'.
            krum_f: Number of Byzantine workers to tolerate for Krum (default: n//4)
            bulyan_f: Number of Byzantine workers to tolerate for Bulyan (default: n//4)

        Returns:
            True if update was successful, False otherwise.
        """
        with self.lock:
            if not self.gradient_buffer:
                return False

            # Aggregate gradients
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
                n = len(grad_list)
                stack = torch.stack(grad_list)

                if rule == 'mean' or n < 3:
                    return torch.mean(stack, dim=0)

                elif rule == 'trimmed_mean':
                    # Trim extremes per element
                    k = int((trim_ratio * n) // 2)
                    if k == 0:
                        return torch.mean(stack, dim=0)
                    sorted_vals, _ = torch.sort(stack, dim=0)
                    trimmed = sorted_vals[k:n-k]
                    if trimmed.numel() == 0:
                        return torch.mean(stack, dim=0)
                    return torch.mean(trimmed, dim=0)

                elif rule == 'krum':
                    # Multi-Krum: Select m most representative gradients
                    f = krum_f if krum_f is not None else max(1, n // 4)
                    m = n - f - 2  # Number of gradients to select
                    if m < 1:
                        return torch.mean(stack, dim=0)

                    # Flatten gradients for distance computation
                    flattened = [g.flatten() for g in grad_list]

                    # Compute pairwise distances
                    scores = []
                    for i in range(n):
                        # Sum of squared distances to n-f-2 nearest neighbors
                        distances = []
                        for j in range(n):
                            if i != j:
                                dist = torch.sum((flattened[i] - flattened[j]) ** 2).item()
                                distances.append(dist)
                        # Score = sum of n-f-2 smallest distances
                        distances.sort()
                        score = sum(distances[:n-f-2])
                        scores.append((score, i))

                    # Select m gradients with smallest scores
                    scores.sort()
                    selected_indices = [idx for _, idx in scores[:m]]
                    selected = stack[selected_indices]
                    return torch.mean(selected, dim=0)

                elif rule == 'bulyan':
                    # Bulyan: Krum selection + trimmed mean
                    f = bulyan_f if bulyan_f is not None else max(1, n // 4)
                    theta = n - 2 * f  # Number of gradients to select via Krum

                    if theta < 1 or n < 4 * f + 3:
                        # Fall back to mean if not enough workers
                        return torch.mean(stack, dim=0)

                    # Flatten gradients for distance computation
                    flattened = [g.flatten() for g in grad_list]

                    # Compute Krum scores
                    scores = []
                    for i in range(n):
                        distances = []
                        for j in range(n):
                            if i != j:
                                dist = torch.sum((flattened[i] - flattened[j]) ** 2).item()
                                distances.append(dist)
                        distances.sort()
                        score = sum(distances[:n-f-2])
                        scores.append((score, i))

                    # Select theta gradients with smallest scores
                    scores.sort()
                    selected_indices = [idx for _, idx in scores[:theta]]
                    selected_grads = [grad_list[i] for i in selected_indices]

                    # Apply trimmed mean to selected gradients
                    selected_stack = torch.stack(selected_grads)
                    beta = 2 * f  # Trim 2f gradients total
                    k = beta // 2
                    if k >= len(selected_grads) // 2:
                        return torch.mean(selected_stack, dim=0)

                    sorted_vals, _ = torch.sort(selected_stack, dim=0)
                    trimmed = sorted_vals[k:len(selected_grads)-k]
                    if trimmed.numel() == 0:
                        return torch.mean(selected_stack, dim=0)
                    return torch.mean(trimmed, dim=0)

                # Fallback
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


class BoundedAsyncCoordinator:
    """
    Coordinates asynchronous training across workers with bounded staleness.

    Workers submit gradients without barriers. Gradients that are too stale
    (based on max_staleness) are rejected to prevent divergence.
    This design handles heterogeneous and volatile environments where workers
    have different speeds and may fail intermittently.
    """

    def __init__(self, num_workers: int, max_staleness: int = 50, window_size: int = None, max_wait_s: float = None,
                 adaptive_staleness: bool = True, min_staleness: int = 5, max_staleness_multiplier: float = 10.0):
        """
        Initialize async coordinator.

        Args:
            num_workers: Expected number of workers
            max_staleness: Base maximum staleness (in steps) for gradients (default: 50, was 5)
            window_size: Unused in async mode, kept for API compatibility
            max_wait_s: Unused in async mode, kept for API compatibility
            adaptive_staleness: Enable adaptive staleness based on worker speed (default: True)
            min_staleness: Minimum staleness bound (default: 5)
            max_staleness_multiplier: Maximum multiplier for slow workers (default: 10.0)
        """
        self.num_workers = num_workers
        self.max_staleness = max_staleness  # Base staleness for normal workers
        self.window_size = window_size or num_workers  # Kept for compatibility
        self.max_wait_s = max_wait_s  # Kept for compatibility

        # Adaptive staleness configuration
        self.adaptive_staleness = adaptive_staleness
        self.min_staleness = min_staleness
        self.max_staleness_multiplier = max_staleness_multiplier

        self.global_step = 0  # Global model step counter
        self.worker_steps = {}  # {worker_id: last_step_number}
        self.gradient_buffer = []  # List of (worker_id, gradients, staleness) tuples
        self.lock = threading.Lock()

        # Worker speed tracking for adaptive staleness
        self.worker_submission_times = {}  # {worker_id: [timestamps]}
        self.worker_speed_estimates = {}  # {worker_id: speed_multiplier}
        self.worker_staleness_bounds = {}  # {worker_id: max_staleness}
        # Track all workers that have interacted (accepted or rejected) for stats
        self.seen_workers: Set[str] = set()

        # Statistics
        self.total_gradients_received = 0
        self.total_gradients_rejected = 0
        self.barrier_count = 0  # Kept for compatibility with existing code

    def wait_for_all(self, worker_id: str) -> bool:
        """
        Compatibility method for async coordinator.
        In async mode, workers don't wait at barriers - they return immediately.

        Args:
            worker_id: ID of the worker

        Returns:
            True immediately (no blocking)
        """
        # No blocking in async mode
        return True

    def _update_worker_speed(self, worker_id: str):
        """Update worker speed estimate based on submission frequency.

        Args:
            worker_id: ID of the worker
        """
        import time

        current_time = time.time()

        # Track submission times (keep last 10)
        if worker_id not in self.worker_submission_times:
            self.worker_submission_times[worker_id] = []

        self.worker_submission_times[worker_id].append(current_time)
        if len(self.worker_submission_times[worker_id]) > 10:
            self.worker_submission_times[worker_id].pop(0)

        # Estimate speed from submission frequency (need at least 3 samples)
        if len(self.worker_submission_times[worker_id]) >= 3:
            times = self.worker_submission_times[worker_id]
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_interval = sum(intervals) / len(intervals)

            # Compare to median interval across all workers
            all_intervals = []
            for wid, timestamps in self.worker_submission_times.items():
                if len(timestamps) >= 2:
                    worker_intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                    all_intervals.extend(worker_intervals)

            if all_intervals:
                # Median interval represents "normal" speed
                all_intervals.sort()
                median_interval = all_intervals[len(all_intervals) // 2]

                # Speed multiplier: fast worker has low interval, slow worker has high interval
                if median_interval > 0:
                    speed_multiplier = median_interval / max(avg_interval, 0.001)
                    # Clamp to reasonable range [0.1, 10.0]
                    speed_multiplier = max(0.1, min(10.0, speed_multiplier))
                    self.worker_speed_estimates[worker_id] = speed_multiplier

    def _get_worker_staleness_bound(self, worker_id: str) -> int:
        """Get adaptive staleness bound for a specific worker.

        Fast workers get tighter bounds, slow workers get looser bounds.

        Args:
            worker_id: ID of the worker

        Returns:
            Maximum staleness for this worker
        """
        if not self.adaptive_staleness:
            return self.max_staleness

        # If we don't have speed estimate yet, use base staleness
        if worker_id not in self.worker_speed_estimates:
            return self.max_staleness

        speed = self.worker_speed_estimates[worker_id]

        # Adaptive staleness formula:
        # Slow workers (speed < 1.0) get higher staleness tolerance
        # Fast workers (speed > 1.0) get lower staleness tolerance
        # Formula: staleness = base_staleness / speed
        adaptive_bound = int(self.max_staleness / speed)

        # Clamp to reasonable range
        adaptive_bound = max(self.min_staleness, adaptive_bound)
        adaptive_bound = min(int(self.max_staleness * self.max_staleness_multiplier), adaptive_bound)

        # Cache for statistics
        self.worker_staleness_bounds[worker_id] = adaptive_bound

        return adaptive_bound

    def submit_gradients(self, worker_id: str, gradients: Dict[str, Any], worker_step: int = None) -> bool:
        """
        Submit gradients from a worker (non-blocking).
        Gradients are checked for staleness and rejected if too old.

        With adaptive staleness, slow workers are given higher staleness tolerance.

        Args:
            worker_id: ID of the worker
            gradients: Gradient dictionary
            worker_step: Worker's local step number (optional, uses global step if not provided)

        Returns:
            True if gradients accepted, False if rejected due to staleness
        """
        with self.lock:
            self.total_gradients_received += 1
            self.seen_workers.add(worker_id)

            # If worker_step not provided, assume current step
            if worker_step is None:
                worker_step = self.global_step

            # Update worker speed estimate (for adaptive staleness)
            if self.adaptive_staleness:
                self._update_worker_speed(worker_id)

            # Calculate staleness
            staleness = self.global_step - worker_step

            # Get adaptive staleness bound for this worker
            worker_max_staleness = self._get_worker_staleness_bound(worker_id)

            # Reject if too stale
            if staleness > worker_max_staleness:
                self.total_gradients_rejected += 1
                return False

            # Accept gradients
            self.gradient_buffer.append((worker_id, gradients, staleness))
            self.worker_steps[worker_id] = worker_step

            return True

    def get_and_clear_gradients(self) -> list:
        """
        Get all accumulated gradients and clear the buffer.

        Returns:
            List of (worker_id, gradients, staleness) tuples
        """
        with self.lock:
            gradients = self.gradient_buffer[:]
            self.gradient_buffer.clear()
            if gradients:
                self.global_step += 1
                self.barrier_count += 1  # For compatibility
            else:
                # Even if no gradients, advance the global clock to reflect elapsed cycles.
                self.global_step += 1
            return gradients

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about async training.

        Returns:
            Dictionary with statistics including adaptive staleness info
        """
        with self.lock:
            rejection_rate = 0.0
            if self.total_gradients_received > 0:
                rejection_rate = self.total_gradients_rejected / self.total_gradients_received

            stats = {
                'global_step': self.global_step,
                'total_received': self.total_gradients_received,
                'total_rejected': self.total_gradients_rejected,
                'rejection_rate': rejection_rate,
                'active_workers': len(self.seen_workers) if hasattr(self, "seen_workers") else len(self.worker_steps),
                'pending_gradients': len(self.gradient_buffer),
                'adaptive_staleness_enabled': self.adaptive_staleness,
                'base_max_staleness': self.max_staleness
            }

            # Add per-worker adaptive staleness info
            if self.adaptive_staleness and self.worker_speed_estimates:
                stats['worker_speeds'] = self.worker_speed_estimates.copy()
                stats['worker_staleness_bounds'] = self.worker_staleness_bounds.copy()

            return stats


class SimpleSyncCoordinator:
    """
    Coordinates synchronous training across workers.

    Workers wait at barriers until all have completed a step,
    then gradients are aggregated and parameters are updated.

    DEPRECATED: Use BoundedAsyncCoordinator for heterogeneous/volatile environments.
    This synchronous implementation suffers from the straggler problem in WAN settings.
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
