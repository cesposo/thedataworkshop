"""WAN challenge scenario definitions.

Progressive challenges that test different aspects of distributed training
in WAN environments, from basic bandwidth constraints to extreme adversarial
conditions with failures, churn, and Byzantine workers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.network_simulator import NetworkProfile
from framework.failure_injector import FailureMode, FailurePolicy


@dataclass
class WorkerProfile:
    """Profile for a worker in the simulation."""

    worker_id: str
    network_profile: NetworkProfile
    speed_multiplier: float = 1.0  # Relative to reference GPU
    initially_online: bool = True


@dataclass
class ChallengeConfig:
    """Configuration for a WAN challenge scenario."""

    name: str
    description: str
    difficulty: int  # 1-10

    # Worker configuration
    worker_profiles: List[WorkerProfile]
    num_workers: int = field(init=False)

    # Failure policies (optional)
    failure_policies: List[FailurePolicy] = field(default_factory=list)

    # Churn configuration
    enable_churn: bool = False
    mean_session_duration: int = 100  # Steps
    mean_rejoin_delay: int = 50  # Steps

    # Success criteria
    min_convergence_steps: int = 100
    max_rejection_rate: float = 0.3  # 30%
    min_throughput_ratio: float = 0.5  # Relative to ideal

    # Recommended configuration
    recommended_async: bool = True
    recommended_compression: str = "topk"
    recommended_compression_ratio: float = 0.01
    recommended_max_staleness: int = 5

    def __post_init__(self):
        self.num_workers = len(self.worker_profiles)


# ============================================================================
# CHALLENGE 1: BASELINE (LAN, No Failures)
# ============================================================================

CHALLENGE_1_BASELINE = ChallengeConfig(
    name="baseline",
    description="Baseline: Homogeneous LAN cluster with no failures",
    difficulty=1,

    worker_profiles=[
        WorkerProfile(f"worker_{i}", NetworkProfile.DATACENTER, speed_multiplier=1.0)
        for i in range(4)
    ],

    failure_policies=[],
    enable_churn=False,

    min_convergence_steps=50,
    max_rejection_rate=0.0,
    min_throughput_ratio=0.95,

    recommended_async=False,  # Sync is fine for homogeneous LAN
    recommended_compression="none",
    recommended_compression_ratio=1.0,
    recommended_max_staleness=0
)


# ============================================================================
# CHALLENGE 2: WAN BANDWIDTH (Bandwidth Constraints Only)
# ============================================================================

CHALLENGE_2_WAN_BANDWIDTH = ChallengeConfig(
    name="wan_bandwidth",
    description="WAN bandwidth constraints - tests compression effectiveness",
    difficulty=2,

    worker_profiles=[
        WorkerProfile(f"worker_{i}", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0)
        for i in range(4)
    ],

    failure_policies=[],
    enable_churn=False,

    min_convergence_steps=100,
    max_rejection_rate=0.05,
    min_throughput_ratio=0.7,

    recommended_async=True,
    recommended_compression="topk",
    recommended_compression_ratio=0.01,
    recommended_max_staleness=3
)


# ============================================================================
# CHALLENGE 3: WAN LATENCY (Bandwidth + High Latency)
# ============================================================================

CHALLENGE_3_WAN_LATENCY = ChallengeConfig(
    name="wan_latency",
    description="High latency WAN - tests async training benefits",
    difficulty=3,

    worker_profiles=[
        WorkerProfile("worker_0", NetworkProfile.FAST_WAN, speed_multiplier=1.0),
        WorkerProfile("worker_1", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0),
        WorkerProfile("worker_2", NetworkProfile.SLOW_WAN, speed_multiplier=1.0),
        WorkerProfile("worker_3", NetworkProfile.SLOW_WAN, speed_multiplier=1.0),
    ],

    failure_policies=[],
    enable_churn=False,

    min_convergence_steps=150,
    max_rejection_rate=0.1,
    min_throughput_ratio=0.6,

    recommended_async=True,
    recommended_compression="topk",
    recommended_compression_ratio=0.05,
    recommended_max_staleness=5
)


# ============================================================================
# CHALLENGE 4: HETEROGENEOUS (Mixed Worker Speeds)
# ============================================================================

CHALLENGE_4_HETEROGENEOUS = ChallengeConfig(
    name="heterogeneous",
    description="Heterogeneous workers (GPU/CPU/Slow) - tests straggler handling",
    difficulty=4,

    worker_profiles=[
        WorkerProfile("fast_gpu", NetworkProfile.FAST_WAN, speed_multiplier=2.0),
        WorkerProfile("standard_gpu", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0),
        WorkerProfile("slow_gpu", NetworkProfile.TYPICAL_WAN, speed_multiplier=0.5),
        WorkerProfile("cpu_worker", NetworkProfile.SLOW_WAN, speed_multiplier=0.2),
        WorkerProfile("edge_worker", NetworkProfile.EDGE, speed_multiplier=0.1),
    ],

    failure_policies=[],
    enable_churn=False,

    min_convergence_steps=200,
    max_rejection_rate=0.15,
    min_throughput_ratio=0.5,

    recommended_async=True,  # Critical for heterogeneous
    recommended_compression="topk",
    recommended_compression_ratio=0.01,
    recommended_max_staleness=10  # Higher tolerance for heterogeneity
)


# ============================================================================
# CHALLENGE 5: UNRELIABLE (Packet Loss + Transient Failures)
# ============================================================================

CHALLENGE_5_UNRELIABLE = ChallengeConfig(
    name="unreliable",
    description="Unreliable network + transient failures - tests fault tolerance",
    difficulty=5,

    worker_profiles=[
        WorkerProfile("worker_0", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0),
        WorkerProfile("worker_1", NetworkProfile.SLOW_WAN, speed_multiplier=1.0),
        WorkerProfile("worker_2", NetworkProfile.EDGE, speed_multiplier=0.8),
        WorkerProfile("worker_3", NetworkProfile.EDGE, speed_multiplier=0.8),
    ],

    failure_policies=[
        FailurePolicy(
            mode=FailureMode.TRANSIENT,
            probability=0.02,  # 2% chance per step
            duration_steps=10,  # Down for 10 steps
        ),
        FailurePolicy(
            mode=FailureMode.SLOWDOWN,
            probability=0.05,  # 5% chance per step
            duration_steps=20,
            slowdown_factor=5.0,  # 5x slower
        ),
    ],

    enable_churn=False,

    min_convergence_steps=250,
    max_rejection_rate=0.2,
    min_throughput_ratio=0.4,

    recommended_async=True,
    recommended_compression="topk",
    recommended_compression_ratio=0.01,
    recommended_max_staleness=8
)


# ============================================================================
# CHALLENGE 6: CHURN (Workers Joining/Leaving)
# ============================================================================

CHALLENGE_6_CHURN = ChallengeConfig(
    name="churn",
    description="Worker churn (volunteer/spot instances) - tests dynamic management",
    difficulty=6,

    worker_profiles=[
        WorkerProfile("worker_0", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0, initially_online=True),
        WorkerProfile("worker_1", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0, initially_online=True),
        WorkerProfile("worker_2", NetworkProfile.SLOW_WAN, speed_multiplier=0.8, initially_online=True),
        WorkerProfile("worker_3", NetworkProfile.SLOW_WAN, speed_multiplier=0.8, initially_online=False),
        WorkerProfile("worker_4", NetworkProfile.EDGE, speed_multiplier=0.5, initially_online=False),
        WorkerProfile("worker_5", NetworkProfile.EDGE, speed_multiplier=0.5, initially_online=False),
    ],

    failure_policies=[
        FailurePolicy(
            mode=FailureMode.TRANSIENT,
            probability=0.01,
            duration_steps=15,
        ),
    ],

    enable_churn=True,
    mean_session_duration=80,  # Workers stay ~80 steps
    mean_rejoin_delay=40,  # Rejoin after ~40 steps

    min_convergence_steps=300,
    max_rejection_rate=0.25,
    min_throughput_ratio=0.35,

    recommended_async=True,
    recommended_compression="topk",
    recommended_compression_ratio=0.01,
    recommended_max_staleness=10
)


# ============================================================================
# CHALLENGE 7: BYZANTINE (Malicious Workers)
# ============================================================================

CHALLENGE_7_BYZANTINE = ChallengeConfig(
    name="byzantine",
    description="Byzantine workers (corrupted gradients) - tests Byzantine tolerance",
    difficulty=7,

    worker_profiles=[
        WorkerProfile("honest_1", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0),
        WorkerProfile("honest_2", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0),
        WorkerProfile("honest_3", NetworkProfile.SLOW_WAN, speed_multiplier=0.8),
        WorkerProfile("byzantine_1", NetworkProfile.FAST_WAN, speed_multiplier=1.2),  # Fast Byzantine
        WorkerProfile("byzantine_2", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0),
    ],

    failure_policies=[
        FailurePolicy(
            mode=FailureMode.BYZANTINE,
            probability=1.0,  # Always corrupt from specific workers
            duration_steps=0,  # Permanent
            target_workers=["byzantine_1", "byzantine_2"]
        ),
    ],

    enable_churn=False,

    min_convergence_steps=350,
    max_rejection_rate=0.3,
    min_throughput_ratio=0.4,

    recommended_async=True,
    recommended_compression="topk",
    recommended_compression_ratio=0.05,
    recommended_max_staleness=5
)


# ============================================================================
# CHALLENGE 8: EXTREME (All Challenges Combined)
# ============================================================================

CHALLENGE_8_EXTREME = ChallengeConfig(
    name="extreme",
    description="EXTREME: All challenges combined - ultimate stress test",
    difficulty=10,

    worker_profiles=[
        # Fast, reliable workers
        WorkerProfile("fast_1", NetworkProfile.FAST_WAN, speed_multiplier=2.0, initially_online=True),
        WorkerProfile("fast_2", NetworkProfile.FAST_WAN, speed_multiplier=1.5, initially_online=True),

        # Typical workers with churn
        WorkerProfile("typical_1", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0, initially_online=True),
        WorkerProfile("typical_2", NetworkProfile.TYPICAL_WAN, speed_multiplier=1.0, initially_online=False),

        # Slow, unreliable workers
        WorkerProfile("slow_1", NetworkProfile.SLOW_WAN, speed_multiplier=0.5, initially_online=True),
        WorkerProfile("slow_2", NetworkProfile.SLOW_WAN, speed_multiplier=0.5, initially_online=False),

        # Edge workers with high churn
        WorkerProfile("edge_1", NetworkProfile.EDGE, speed_multiplier=0.3, initially_online=False),
        WorkerProfile("edge_2", NetworkProfile.EDGE, speed_multiplier=0.2, initially_online=False),

        # Byzantine worker (fast but malicious)
        WorkerProfile("byzantine", NetworkProfile.FAST_WAN, speed_multiplier=1.8, initially_online=True),
    ],

    failure_policies=[
        # Transient failures
        FailurePolicy(
            mode=FailureMode.TRANSIENT,
            probability=0.03,
            duration_steps=15,
        ),
        # Slowdowns
        FailurePolicy(
            mode=FailureMode.SLOWDOWN,
            probability=0.05,
            duration_steps=25,
            slowdown_factor=10.0,
        ),
        # Byzantine behavior from specific worker
        FailurePolicy(
            mode=FailureMode.BYZANTINE,
            probability=1.0,
            duration_steps=0,
            target_workers=["byzantine"]
        ),
    ],

    enable_churn=True,
    mean_session_duration=60,
    mean_rejoin_delay=30,

    min_convergence_steps=500,
    max_rejection_rate=0.4,
    min_throughput_ratio=0.25,

    recommended_async=True,
    recommended_compression="topk",
    recommended_compression_ratio=0.01,
    recommended_max_staleness=15
)


# ============================================================================
# CHALLENGE REGISTRY
# ============================================================================

ALL_CHALLENGES = {
    'baseline': CHALLENGE_1_BASELINE,
    'wan_bandwidth': CHALLENGE_2_WAN_BANDWIDTH,
    'wan_latency': CHALLENGE_3_WAN_LATENCY,
    'heterogeneous': CHALLENGE_4_HETEROGENEOUS,
    'unreliable': CHALLENGE_5_UNRELIABLE,
    'churn': CHALLENGE_6_CHURN,
    'byzantine': CHALLENGE_7_BYZANTINE,
    'extreme': CHALLENGE_8_EXTREME,
}


def get_challenge(name: str) -> ChallengeConfig:
    """Get a challenge configuration by name.

    Args:
        name: Challenge name

    Returns:
        Challenge configuration

    Raises:
        ValueError: If challenge name not found
    """
    if name not in ALL_CHALLENGES:
        raise ValueError(
            f"Unknown challenge '{name}'. "
            f"Available: {list(ALL_CHALLENGES.keys())}"
        )
    return ALL_CHALLENGES[name]


def list_challenges() -> Dict[str, ChallengeConfig]:
    """Get all available challenges.

    Returns:
        Dictionary mapping challenge names to configurations
    """
    return ALL_CHALLENGES.copy()


def print_challenge_summary():
    """Print a summary of all challenges."""
    print("=" * 120)
    print("WAN CHALLENGE SUMMARY")
    print("=" * 120)
    print(f"{'Challenge':<18} {'Workers':<10} {'Churn':<8} {'Failures':<10} {'Difficulty':<12} {'Description':<50}")
    print("-" * 120)

    for name, config in ALL_CHALLENGES.items():
        churn_str = "Yes" if config.enable_churn else "No"
        failures_str = str(len(config.failure_policies))
        diff_str = '⭐' * min(config.difficulty, 10)

        print(f"{config.name:<18} {config.num_workers:<10} {churn_str:<8} {failures_str:<10} {diff_str:<12} {config.description[:48]}")

    print("=" * 120)


def print_challenge_details(challenge: ChallengeConfig):
    """Print detailed information about a challenge.

    Args:
        challenge: Challenge configuration
    """
    print("=" * 100)
    print(f"CHALLENGE: {challenge.name.upper()}")
    print("=" * 100)
    print(f"Description: {challenge.description}")
    print(f"Difficulty: {'⭐' * challenge.difficulty} ({challenge.difficulty}/10)")
    print()

    print("Worker Profiles:")
    print("-" * 100)
    for wp in challenge.worker_profiles:
        print(f"  {wp.worker_id:<15} Network: {wp.network_profile.name:<20} "
              f"Speed: {wp.speed_multiplier:>5.1f}x  "
              f"Initially: {'Online' if wp.initially_online else 'Offline'}")
    print()

    if challenge.failure_policies:
        print("Failure Policies:")
        print("-" * 100)
        for i, fp in enumerate(challenge.failure_policies, 1):
            print(f"  {i}. {fp.mode.value}: "
                  f"probability={fp.probability:.1%}, "
                  f"duration={fp.duration_steps} steps")
            if fp.target_workers:
                print(f"     Targets: {', '.join(fp.target_workers)}")
        print()

    if challenge.enable_churn:
        print("Churn Configuration:")
        print("-" * 100)
        print(f"  Mean session duration: {challenge.mean_session_duration} steps")
        print(f"  Mean rejoin delay: {challenge.mean_rejoin_delay} steps")
        print()

    print("Success Criteria:")
    print("-" * 100)
    print(f"  Min convergence steps: {challenge.min_convergence_steps}")
    print(f"  Max rejection rate: {challenge.max_rejection_rate:.1%}")
    print(f"  Min throughput ratio: {challenge.min_throughput_ratio:.1%}")
    print()

    print("Recommended Configuration:")
    print("-" * 100)
    print(f"  Async training: {challenge.recommended_async}")
    print(f"  Compression: {challenge.recommended_compression}")
    print(f"  Compression ratio: {challenge.recommended_compression_ratio}")
    print(f"  Max staleness: {challenge.recommended_max_staleness}")
    print("=" * 100)


if __name__ == "__main__":
    # Print challenge summary
    print_challenge_summary()

    # Print details for a few interesting challenges
    print("\n")
    print_challenge_details(CHALLENGE_4_HETEROGENEOUS)
    print("\n")
    print_challenge_details(CHALLENGE_7_BYZANTINE)
    print("\n")
    print_challenge_details(CHALLENGE_8_EXTREME)
