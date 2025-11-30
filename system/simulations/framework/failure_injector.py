"""Failure injection framework for testing fault tolerance.

Simulates various failure modes:
- Node failures (crash-stop)
- Transient failures (temporary unavailability)
- Slowdowns (stragglers)
- Byzantine failures (corrupted gradients)
- Network partitions
"""

import random
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from enum import Enum


class FailureMode(Enum):
    """Types of failures that can be injected."""

    CRASH_STOP = "crash_stop"              # Node crashes and stays down
    TRANSIENT = "transient"                # Node temporarily unavailable
    SLOWDOWN = "slowdown"                  # Node becomes slow (straggler)
    BYZANTINE = "byzantine"                # Node sends corrupted data
    NETWORK_PARTITION = "network_partition"  # Network partition


@dataclass
class FailurePolicy:
    """Policy for injecting failures."""

    mode: FailureMode
    probability: float = 0.0          # Probability of failure per step
    duration_steps: int = 0            # How long failure lasts (0 = permanent)
    slowdown_factor: float = 1.0      # Multiplier for slowdown mode
    target_workers: Optional[list] = None  # Specific workers to affect (None = random)


class FailureInjector:
    """Injects failures into distributed training for testing.

    Tracks worker states and can simulate various failure modes
    to test system robustness and recovery mechanisms.
    """

    def __init__(self):
        """Initialize failure injector."""
        self.worker_states = {}  # {worker_id: state}
        self.failure_timers = {}  # {worker_id: steps_remaining}
        self.total_failures = 0
        self.failures_by_type = {mode: 0 for mode in FailureMode}

    def register_worker(self, worker_id: str):
        """Register a worker for failure injection.

        Args:
            worker_id: Worker identifier
        """
        self.worker_states[worker_id] = {
            'status': 'healthy',
            'failure_mode': None,
            'slowdown_factor': 1.0,
            'steps_failed': 0
        }

    def inject_failure(self, worker_id: str, policy: FailurePolicy) -> bool:
        """Inject a failure according to policy.

        Args:
            worker_id: Worker to affect
            policy: Failure policy

        Returns:
            True if failure was injected, False otherwise
        """
        if worker_id not in self.worker_states:
            self.register_worker(worker_id)

        # Check if already failed
        if self.worker_states[worker_id]['status'] != 'healthy':
            return False

        # Apply failure
        self.worker_states[worker_id]['status'] = 'failed'
        self.worker_states[worker_id]['failure_mode'] = policy.mode
        self.worker_states[worker_id]['slowdown_factor'] = policy.slowdown_factor

        # Set timer for transient failures
        if policy.duration_steps > 0:
            self.failure_timers[worker_id] = policy.duration_steps

        # Track statistics
        self.total_failures += 1
        self.failures_by_type[policy.mode] += 1

        return True

    def maybe_inject_failure(self, worker_id: str, policy: FailurePolicy) -> bool:
        """Probabilistically inject a failure.

        Args:
            worker_id: Worker to potentially affect
            policy: Failure policy with probability

        Returns:
            True if failure was injected, False otherwise
        """
        if random.random() < policy.probability:
            return self.inject_failure(worker_id, policy)
        return False

    def step(self):
        """Advance one time step (update failure timers)."""
        # Decrement timers
        expired = []
        for worker_id, steps_remaining in self.failure_timers.items():
            if steps_remaining <= 1:
                expired.append(worker_id)
            else:
                self.failure_timers[worker_id] -= 1

        # Recover expired failures
        for worker_id in expired:
            self.recover_worker(worker_id)
            del self.failure_timers[worker_id]

    def recover_worker(self, worker_id: str):
        """Recover a failed worker.

        Args:
            worker_id: Worker to recover
        """
        if worker_id in self.worker_states:
            self.worker_states[worker_id]['status'] = 'healthy'
            self.worker_states[worker_id]['failure_mode'] = None
            self.worker_states[worker_id]['slowdown_factor'] = 1.0

    def is_worker_available(self, worker_id: str) -> bool:
        """Check if worker is available (not crashed).

        Args:
            worker_id: Worker identifier

        Returns:
            True if worker can process requests
        """
        if worker_id not in self.worker_states:
            return True  # Unknown workers assumed healthy

        state = self.worker_states[worker_id]

        # Crash-stop failures make worker unavailable
        if state['failure_mode'] == FailureMode.CRASH_STOP:
            return False

        # Transient failures make worker temporarily unavailable
        if state['failure_mode'] == FailureMode.TRANSIENT:
            return False

        return True

    def get_worker_slowdown(self, worker_id: str) -> float:
        """Get slowdown factor for a worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Slowdown multiplier (1.0 = normal, 2.0 = half speed, etc.)
        """
        if worker_id not in self.worker_states:
            return 1.0

        state = self.worker_states[worker_id]
        if state['failure_mode'] == FailureMode.SLOWDOWN:
            return state['slowdown_factor']

        return 1.0

    def should_corrupt_gradients(self, worker_id: str) -> bool:
        """Check if worker should send corrupted gradients (Byzantine failure).

        Args:
            worker_id: Worker identifier

        Returns:
            True if gradients should be corrupted
        """
        if worker_id not in self.worker_states:
            return False

        state = self.worker_states[worker_id]
        return state['failure_mode'] == FailureMode.BYZANTINE

    def corrupt_gradients(self, gradients: Dict, corruption_type: str = 'scale') -> Dict:
        """Corrupt gradients (Byzantine behavior).

        Args:
            gradients: Gradient dictionary
            corruption_type: Type of corruption ('scale', 'flip', 'noise')

        Returns:
            Corrupted gradients
        """
        import torch
        corrupted = {}

        for name, grad in gradients.items():
            if corruption_type == 'scale':
                # Scale gradients by large factor
                corrupted[name] = grad * 1000.0
            elif corruption_type == 'flip':
                # Flip gradient sign
                corrupted[name] = -grad
            elif corruption_type == 'noise':
                # Add large noise
                noise = torch.randn_like(grad) * grad.abs().mean() * 10
                corrupted[name] = grad + noise
            else:
                corrupted[name] = grad

        return corrupted

    def get_statistics(self) -> dict:
        """Get failure injection statistics."""
        healthy_count = sum(1 for state in self.worker_states.values()
                           if state['status'] == 'healthy')
        failed_count = sum(1 for state in self.worker_states.values()
                          if state['status'] == 'failed')

        return {
            'total_workers': len(self.worker_states),
            'healthy_workers': healthy_count,
            'failed_workers': failed_count,
            'total_failures_injected': self.total_failures,
            'failures_by_type': dict(self.failures_by_type),
            'active_failure_timers': len(self.failure_timers)
        }

    def reset(self):
        """Reset all failures and statistics."""
        for worker_id in self.worker_states:
            self.recover_worker(worker_id)
        self.failure_timers.clear()
        self.total_failures = 0
        self.failures_by_type = {mode: 0 for mode in FailureMode}


class ChurnSimulator:
    """Simulates worker churn (nodes joining and leaving).

    Models realistic patterns of worker availability in volunteer
    computing or spot instance scenarios.
    """

    def __init__(self, mean_session_duration: int = 100,
                 mean_rejoin_delay: int = 50):
        """Initialize churn simulator.

        Args:
            mean_session_duration: Average steps a worker stays online
            mean_rejoin_delay: Average steps before worker rejoins
        """
        self.mean_session_duration = mean_session_duration
        self.mean_rejoin_delay = mean_rejoin_delay
        self.worker_timers = {}  # {worker_id: steps_until_event}
        self.worker_online = {}  # {worker_id: bool}

    def register_worker(self, worker_id: str, initially_online: bool = True):
        """Register a worker for churn simulation.

        Args:
            worker_id: Worker identifier
            initially_online: Whether worker starts online
        """
        self.worker_online[worker_id] = initially_online

        if initially_online:
            # Schedule departure
            duration = int(random.expovariate(1.0 / self.mean_session_duration))
            self.worker_timers[worker_id] = max(1, duration)
        else:
            # Schedule arrival
            delay = int(random.expovariate(1.0 / self.mean_rejoin_delay))
            self.worker_timers[worker_id] = max(1, delay)

    def step(self) -> dict:
        """Advance one time step.

        Returns:
            Dictionary with 'departed' and 'arrived' worker lists
        """
        departed = []
        arrived = []

        # Decrement timers
        for worker_id in list(self.worker_timers.keys()):
            self.worker_timers[worker_id] -= 1

            # Check if event should trigger
            if self.worker_timers[worker_id] <= 0:
                if self.worker_online[worker_id]:
                    # Worker departs
                    self.worker_online[worker_id] = False
                    departed.append(worker_id)
                    # Schedule rejoin
                    delay = int(random.expovariate(1.0 / self.mean_rejoin_delay))
                    self.worker_timers[worker_id] = max(1, delay)
                else:
                    # Worker arrives
                    self.worker_online[worker_id] = True
                    arrived.append(worker_id)
                    # Schedule departure
                    duration = int(random.expovariate(1.0 / self.mean_session_duration))
                    self.worker_timers[worker_id] = max(1, duration)

        return {'departed': departed, 'arrived': arrived}

    def is_worker_online(self, worker_id: str) -> bool:
        """Check if worker is currently online.

        Args:
            worker_id: Worker identifier

        Returns:
            True if worker is online
        """
        return self.worker_online.get(worker_id, True)

    def get_statistics(self) -> dict:
        """Get churn statistics."""
        online_count = sum(1 for online in self.worker_online.values() if online)
        offline_count = len(self.worker_online) - online_count

        return {
            'total_workers': len(self.worker_online),
            'online_workers': online_count,
            'offline_workers': offline_count,
            'online_percentage': online_count / max(1, len(self.worker_online)) * 100
        }
