"""Network condition simulator for WAN training experiments.

Simulates realistic network conditions including:
- Bandwidth constraints
- Latency (RTT)
- Packet loss
- Jitter
- Network partitions
"""

import time
import random
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum


class NetworkProfile(Enum):
    """Predefined network profiles."""

    # LAN environments
    DATACENTER = "datacenter"           # 10 Gbps, 0.1ms, 0% loss
    FAST_LAN = "fast_lan"               # 1 Gbps, 1ms, 0% loss

    # WAN environments
    FAST_WAN = "fast_wan"               # 100 Mbps, 50ms, 0.1% loss
    TYPICAL_WAN = "typical_wan"         # 50 Mbps, 100ms, 1% loss
    SLOW_WAN = "slow_wan"               # 10 Mbps, 200ms, 2% loss

    # Challenging environments
    EDGE = "edge"                       # 5 Mbps, 300ms, 5% loss
    MOBILE = "mobile"                   # 2 Mbps, 500ms, 10% loss
    SATELLITE = "satellite"             # 1 Mbps, 800ms, 15% loss


@dataclass
class NetworkConditions:
    """Network condition parameters."""

    bandwidth_mbps: float         # Bandwidth in Mbps
    latency_ms: float            # Base latency in ms
    packet_loss_rate: float      # Packet loss probability (0-1)
    jitter_ms: float = 0.0       # Latency jitter in ms
    partition_probability: float = 0.0  # Network partition probability

    @classmethod
    def from_profile(cls, profile: NetworkProfile) -> 'NetworkConditions':
        """Create conditions from a predefined profile."""
        profiles = {
            NetworkProfile.DATACENTER: cls(10000, 0.1, 0.0, 0.0),
            NetworkProfile.FAST_LAN: cls(1000, 1.0, 0.0, 0.0),
            NetworkProfile.FAST_WAN: cls(100, 50, 0.001, 10.0),
            NetworkProfile.TYPICAL_WAN: cls(50, 100, 0.01, 20.0),
            NetworkProfile.SLOW_WAN: cls(10, 200, 0.02, 50.0),
            NetworkProfile.EDGE: cls(5, 300, 0.05, 100.0),
            NetworkProfile.MOBILE: cls(2, 500, 0.10, 200.0),
            NetworkProfile.SATELLITE: cls(1, 800, 0.15, 300.0),
        }
        return profiles[profile]


class NetworkSimulator:
    """Simulates network conditions for distributed training.

    Can be used to inject realistic latency, bandwidth constraints,
    and failures into gradient transmission.
    """

    def __init__(self, conditions: NetworkConditions):
        """Initialize network simulator.

        Args:
            conditions: Network condition parameters
        """
        self.conditions = conditions
        self.total_bytes_transmitted = 0
        self.total_transmission_time = 0.0
        self.packets_sent = 0
        self.packets_lost = 0
        self.is_partitioned = False

    @classmethod
    def from_profile(cls, profile: NetworkProfile) -> 'NetworkSimulator':
        """Create simulator from predefined profile."""
        return cls(NetworkConditions.from_profile(profile))

    def transmit(self, data_bytes: int, simulate_delay: bool = True) -> dict:
        """Simulate transmission of data over the network.

        Args:
            data_bytes: Size of data to transmit in bytes
            simulate_delay: Whether to actually sleep (for realistic timing)

        Returns:
            Dictionary with transmission statistics:
            - success: bool (False if packet lost or partitioned)
            - latency_ms: float (actual latency including jitter)
            - bandwidth_used_mbps: float
            - transmission_time_s: float
        """
        self.packets_sent += 1

        # Check for network partition
        if random.random() < self.conditions.partition_probability:
            self.is_partitioned = True
            return {
                'success': False,
                'latency_ms': 0.0,
                'bandwidth_used_mbps': 0.0,
                'transmission_time_s': 0.0,
                'reason': 'network_partition'
            }

        # Check for packet loss
        if random.random() < self.conditions.packet_loss_rate:
            self.packets_lost += 1
            return {
                'success': False,
                'latency_ms': 0.0,
                'bandwidth_used_mbps': 0.0,
                'transmission_time_s': 0.0,
                'reason': 'packet_loss'
            }

        # Calculate transmission time based on bandwidth
        data_bits = data_bytes * 8
        bandwidth_bps = self.conditions.bandwidth_mbps * 1_000_000
        transmission_time_s = data_bits / bandwidth_bps

        # Add latency with jitter
        jitter = random.uniform(-self.conditions.jitter_ms, self.conditions.jitter_ms)
        total_latency_ms = max(0.0, self.conditions.latency_ms + jitter)
        total_time_s = transmission_time_s + (total_latency_ms / 1000.0)

        # Simulate actual delay if requested
        if simulate_delay:
            time.sleep(total_time_s)

        # Track statistics
        self.total_bytes_transmitted += data_bytes
        self.total_transmission_time += total_time_s

        return {
            'success': True,
            'latency_ms': total_latency_ms,
            'bandwidth_used_mbps': self.conditions.bandwidth_mbps,
            'transmission_time_s': total_time_s,
            'reason': 'success'
        }

    def get_statistics(self) -> dict:
        """Get cumulative transmission statistics."""
        packet_loss_rate = self.packets_lost / max(1, self.packets_sent)
        avg_bandwidth_mbps = 0.0
        if self.total_transmission_time > 0:
            avg_bandwidth_mbps = (self.total_bytes_transmitted * 8) / (self.total_transmission_time * 1_000_000)

        return {
            'total_bytes': self.total_bytes_transmitted,
            'total_time_s': self.total_transmission_time,
            'packets_sent': self.packets_sent,
            'packets_lost': self.packets_lost,
            'packet_loss_rate': packet_loss_rate,
            'avg_bandwidth_mbps': avg_bandwidth_mbps,
            'is_partitioned': self.is_partitioned
        }

    def reset(self):
        """Reset statistics."""
        self.total_bytes_transmitted = 0
        self.total_transmission_time = 0.0
        self.packets_sent = 0
        self.packets_lost = 0
        self.is_partitioned = False

    def heal_partition(self):
        """Heal network partition."""
        self.is_partitioned = False


class HeterogeneousNetwork:
    """Simulates a heterogeneous network with different conditions per worker.

    Useful for testing scenarios where workers have different network capabilities
    (e.g., datacenter GPU vs. edge device vs. mobile worker).
    """

    def __init__(self):
        """Initialize heterogeneous network."""
        self.worker_simulators = {}

    def add_worker(self, worker_id: str, conditions: NetworkConditions):
        """Add a worker with specific network conditions.

        Args:
            worker_id: Worker identifier
            conditions: Network conditions for this worker
        """
        self.worker_simulators[worker_id] = NetworkSimulator(conditions)

    def add_worker_from_profile(self, worker_id: str, profile: NetworkProfile):
        """Add a worker using a predefined profile.

        Args:
            worker_id: Worker identifier
            profile: Network profile to use
        """
        self.add_worker(worker_id, NetworkConditions.from_profile(profile))

    def transmit(self, worker_id: str, data_bytes: int, simulate_delay: bool = True) -> dict:
        """Simulate transmission for a specific worker.

        Args:
            worker_id: Worker identifier
            data_bytes: Size of data to transmit
            simulate_delay: Whether to actually sleep

        Returns:
            Transmission statistics
        """
        if worker_id not in self.worker_simulators:
            raise ValueError(f"Unknown worker: {worker_id}")

        return self.worker_simulators[worker_id].transmit(data_bytes, simulate_delay)

    def get_worker_statistics(self, worker_id: str) -> dict:
        """Get statistics for a specific worker."""
        if worker_id not in self.worker_simulators:
            raise ValueError(f"Unknown worker: {worker_id}")

        return self.worker_simulators[worker_id].get_statistics()

    def get_all_statistics(self) -> dict:
        """Get statistics for all workers."""
        return {
            worker_id: sim.get_statistics()
            for worker_id, sim in self.worker_simulators.items()
        }

    def reset_all(self):
        """Reset statistics for all workers."""
        for sim in self.worker_simulators.values():
            sim.reset()


def calculate_transmission_time(data_bytes: int, bandwidth_mbps: float, latency_ms: float) -> float:
    """Calculate transmission time for given data size and network conditions.

    Args:
        data_bytes: Size of data in bytes
        bandwidth_mbps: Bandwidth in Mbps
        latency_ms: Latency in milliseconds

    Returns:
        Total transmission time in seconds
    """
    data_bits = data_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    transmission_time_s = data_bits / bandwidth_bps
    latency_s = latency_ms / 1000.0
    return transmission_time_s + latency_s
