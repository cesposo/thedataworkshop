"""Automated test harness for running WAN simulation experiments.

This module provides the main orchestration for running simulations that combine
DNN scenarios with WAN challenge configurations, collecting metrics, and evaluating
success criteria.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.dnn_scenarios import ScenarioConfig, get_scenario
from challenges.wan_challenges import ChallengeConfig, get_challenge, WorkerProfile
from framework.network_simulator import NetworkSimulator, HeterogeneousNetwork
from framework.failure_injector import FailureInjector, ChurnSimulator


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentMetrics:
    """Metrics collected during an experiment."""

    # Throughput metrics
    total_steps: int = 0
    total_time_s: float = 0.0
    steps_per_second: float = 0.0

    # Network metrics
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    average_bandwidth_mbps: float = 0.0
    average_latency_ms: float = 0.0

    # Gradient metrics
    total_gradients_submitted: int = 0
    total_gradients_accepted: int = 0
    total_gradients_rejected: int = 0
    rejection_rate: float = 0.0
    average_staleness: float = 0.0

    # Compression metrics
    total_compression_ratio: float = 1.0
    bandwidth_saved_percent: float = 0.0

    # Failure metrics
    total_failures: int = 0
    failures_by_type: Dict[str, int] = field(default_factory=dict)

    # Worker metrics
    active_workers: int = 0
    total_worker_steps: Dict[str, int] = field(default_factory=dict)
    worker_acceptance_rates: Dict[str, float] = field(default_factory=dict)

    # Convergence metrics (placeholder - would need actual model training)
    final_loss: Optional[float] = None
    converged: bool = False


@dataclass
class ExperimentResult:
    """Result of running an experiment."""

    scenario_name: str
    challenge_name: str
    status: ExperimentStatus
    metrics: ExperimentMetrics
    success_criteria_met: bool
    failure_reasons: List[str] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0

    # Configuration used
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


class SimulationTestHarness:
    """Test harness for running WAN training simulations.

    Orchestrates experiments by combining DNN scenarios with WAN challenges,
    running simulated training, and collecting comprehensive metrics.
    """

    def __init__(self, verbose: bool = True):
        """Initialize test harness.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.results: List[ExperimentResult] = []

    def run_experiment(
        self,
        scenario: ScenarioConfig,
        challenge: ChallengeConfig,
        use_async: bool = True,
        compression_method: str = "topk",
        compression_ratio: float = 0.01,
        max_staleness: int = 5,
        aggregation_rule: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        max_steps: Optional[int] = None,
    ) -> ExperimentResult:
        """Run a single experiment.

        Args:
            scenario: DNN scenario configuration
            challenge: WAN challenge configuration
            use_async: Whether to use async training
            compression_method: Gradient compression method
            compression_ratio: Compression ratio for topk
            max_staleness: Maximum gradient staleness
            aggregation_rule: Aggregation rule (mean, trimmed_mean)
            trim_ratio: Trim ratio for trimmed_mean
            max_steps: Maximum training steps (None = use scenario default)

        Returns:
            Experiment result with metrics and success evaluation
        """
        if self.verbose:
            print("=" * 100)
            print(f"EXPERIMENT: {scenario.name} × {challenge.name}")
            print("=" * 100)
            print(f"Scenario: {scenario.description}")
            print(f"Challenge: {challenge.description}")
            print(f"Config: async={use_async}, compression={compression_method}, "
                  f"staleness={max_staleness}")
            print()

        # Initialize result
        result = ExperimentResult(
            scenario_name=scenario.name,
            challenge_name=challenge.name,
            status=ExperimentStatus.RUNNING,
            metrics=ExperimentMetrics(),
            success_criteria_met=False,
            start_time=time.time(),
            config_snapshot={
                'use_async': use_async,
                'compression_method': compression_method,
                'compression_ratio': compression_ratio,
                'max_staleness': max_staleness,
                'aggregation_rule': aggregation_rule,
                'trim_ratio': trim_ratio,
            }
        )

        try:
            # Set up simulation environment
            network = self._setup_network(challenge)
            failure_injector = self._setup_failure_injector(challenge)
            churn_simulator = self._setup_churn_simulator(challenge)

            # Determine number of steps
            num_steps = max_steps if max_steps is not None else scenario.num_batches

            # Run simulation
            self._run_simulation(
                scenario=scenario,
                challenge=challenge,
                network=network,
                failure_injector=failure_injector,
                churn_simulator=churn_simulator,
                num_steps=num_steps,
                use_async=use_async,
                compression_method=compression_method,
                compression_ratio=compression_ratio,
                max_staleness=max_staleness,
                aggregation_rule=aggregation_rule,
                trim_ratio=trim_ratio,
                result=result
            )

            # Compute final metrics
            self._compute_final_metrics(result, scenario, challenge)

            # Evaluate success criteria
            self._evaluate_success_criteria(result, challenge)

            result.status = ExperimentStatus.COMPLETED

        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.failure_reasons.append(f"Exception: {str(e)}")
            if self.verbose:
                print(f"\nEXPERIMENT FAILED: {e}")

        # Record timing
        result.end_time = time.time()
        result.duration_s = result.end_time - result.start_time

        # Store result
        self.results.append(result)

        if self.verbose:
            self._print_result_summary(result)

        return result

    def _setup_network(self, challenge: ChallengeConfig) -> HeterogeneousNetwork:
        """Set up heterogeneous network simulator.

        Args:
            challenge: Challenge configuration

        Returns:
            Configured network simulator
        """
        network = HeterogeneousNetwork()
        for wp in challenge.worker_profiles:
            network.add_worker_from_profile(wp.worker_id, wp.network_profile)

        return network

    def _setup_failure_injector(self, challenge: ChallengeConfig) -> FailureInjector:
        """Set up failure injector.

        Args:
            challenge: Challenge configuration

        Returns:
            Configured failure injector
        """
        injector = FailureInjector()

        # Register workers
        for wp in challenge.worker_profiles:
            injector.register_worker(wp.worker_id)

        return injector

    def _setup_churn_simulator(self, challenge: ChallengeConfig) -> Optional[ChurnSimulator]:
        """Set up churn simulator if enabled.

        Args:
            challenge: Challenge configuration

        Returns:
            Configured churn simulator or None
        """
        if not challenge.enable_churn:
            return None

        churn = ChurnSimulator(
            mean_session_duration=challenge.mean_session_duration,
            mean_rejoin_delay=challenge.mean_rejoin_delay
        )

        # Register workers
        for wp in challenge.worker_profiles:
            churn.register_worker(wp.worker_id, initially_online=wp.initially_online)

        return churn

    def _run_simulation(
        self,
        scenario: ScenarioConfig,
        challenge: ChallengeConfig,
        network: HeterogeneousNetwork,
        failure_injector: FailureInjector,
        churn_simulator: Optional[ChurnSimulator],
        num_steps: int,
        use_async: bool,
        compression_method: str,
        compression_ratio: float,
        max_staleness: int,
        aggregation_rule: str,
        trim_ratio: float,
        result: ExperimentResult
    ):
        """Run the training simulation.

        Args:
            scenario: DNN scenario
            challenge: Challenge configuration
            network: Network simulator
            failure_injector: Failure injector
            churn_simulator: Churn simulator (optional)
            num_steps: Number of training steps
            use_async: Whether to use async training
            compression_method: Compression method
            compression_ratio: Compression ratio
            max_staleness: Max staleness
            aggregation_rule: Aggregation rule
            trim_ratio: Trim ratio
            result: Result object to populate
        """
        # Simulation state
        global_step = 0
        worker_steps = {wp.worker_id: 0 for wp in challenge.worker_profiles}
        worker_accepted = {wp.worker_id: 0 for wp in challenge.worker_profiles}
        worker_rejected = {wp.worker_id: 0 for wp in challenge.worker_profiles}

        start_time = time.time()

        for step in range(num_steps):
            # Update churn
            if churn_simulator:
                churn_events = churn_simulator.step()
                if self.verbose and (churn_events['departed'] or churn_events['arrived']):
                    print(f"  Step {step}: Departed={churn_events['departed']}, "
                          f"Arrived={churn_events['arrived']}")

            # Inject failures
            failure_injector.step()
            for policy in challenge.failure_policies:
                if policy.target_workers:
                    # Target specific workers
                    for worker_id in policy.target_workers:
                        failure_injector.maybe_inject_failure(worker_id, policy)
                else:
                    # Random workers
                    for wp in challenge.worker_profiles:
                        failure_injector.maybe_inject_failure(wp.worker_id, policy)

            # Simulate worker steps
            for wp in challenge.worker_profiles:
                worker_id = wp.worker_id

                # Check if worker is available
                if churn_simulator and not churn_simulator.is_worker_online(worker_id):
                    continue
                if not failure_injector.is_worker_available(worker_id):
                    continue

                # Determine if worker should step (based on speed)
                slowdown = failure_injector.get_worker_slowdown(worker_id)
                effective_speed = wp.speed_multiplier / slowdown

                # Probabilistic stepping based on speed
                if random.random() < effective_speed:
                    # Worker computes gradients
                    gradient_size_bytes = int(scenario.gradient_size_mb * 1024 * 1024)

                    # Apply compression
                    if compression_method == "topk":
                        compressed_size = int(gradient_size_bytes * compression_ratio)
                    elif compression_method == "quantize":
                        compressed_size = int(gradient_size_bytes * 0.25)  # 4x compression
                    elif compression_method == "fp16":
                        compressed_size = int(gradient_size_bytes * 0.5)  # 2x compression
                    else:
                        compressed_size = gradient_size_bytes

                    # Simulate network transmission
                    transmission = network.transmit(worker_id, compressed_size, simulate_delay=False)
                    result.metrics.total_bytes_sent += compressed_size

                    # Check gradient corruption (Byzantine)
                    if failure_injector.should_corrupt_gradients(worker_id):
                        # Byzantine gradients might be rejected or cause issues
                        if aggregation_rule == "trimmed_mean" and random.random() < 0.7:
                            # Trimmed mean has 70% chance to filter Byzantine
                            pass  # Gradient filtered out
                        else:
                            # Byzantine gradient might affect training
                            pass  # Would affect convergence

                    # Check staleness
                    staleness = global_step - worker_steps[worker_id]
                    accepted = True

                    if use_async:
                        if staleness > max_staleness:
                            accepted = False
                            worker_rejected[worker_id] += 1
                            result.metrics.total_gradients_rejected += 1
                        else:
                            worker_accepted[worker_id] += 1
                            result.metrics.total_gradients_accepted += 1
                            global_step += 1
                    else:
                        # Sync mode: all workers must submit before global step (simplified)
                        worker_accepted[worker_id] += 1
                        result.metrics.total_gradients_accepted += 1
                        # In sync mode, increment global step once per iteration when all workers submit
                        # For simulation simplicity, we increment once per worker submission

                    worker_steps[worker_id] += 1
                    result.metrics.total_gradients_submitted += 1

            # In sync mode, increment global step once per round
            if not use_async:
                # Count how many workers submitted this round
                workers_submitted = sum(1 for wp in challenge.worker_profiles
                                       if (not churn_simulator or churn_simulator.is_worker_online(wp.worker_id))
                                       and failure_injector.is_worker_available(wp.worker_id))
                if workers_submitted > 0:
                    global_step += 1

            # Progress indicator
            if self.verbose and step % max(1, num_steps // 10) == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {step}/{num_steps} steps, "
                      f"{rate:.2f} steps/s, "
                      f"global_step={global_step}")

        # Store final metrics
        result.metrics.total_steps = global_step
        result.metrics.total_time_s = time.time() - start_time
        result.metrics.total_worker_steps = worker_steps.copy()
        result.metrics.worker_acceptance_rates = {
            wid: worker_accepted[wid] / max(1, worker_accepted[wid] + worker_rejected[wid])
            for wid in worker_accepted
        }
        result.metrics.active_workers = len([
            wp for wp in challenge.worker_profiles
            if worker_steps[wp.worker_id] > 0
        ])

    def _compute_final_metrics(self, result: ExperimentResult, scenario: ScenarioConfig,
                                challenge: ChallengeConfig):
        """Compute final derived metrics.

        Args:
            result: Experiment result
            scenario: Scenario configuration
            challenge: Challenge configuration
        """
        m = result.metrics

        # Throughput
        if m.total_time_s > 0:
            m.steps_per_second = m.total_steps / m.total_time_s

        # Rejection rate
        if m.total_gradients_submitted > 0:
            m.rejection_rate = m.total_gradients_rejected / m.total_gradients_submitted

        # Average staleness (simplified estimate)
        if m.total_gradients_accepted > 0:
            # In reality we'd track actual staleness
            m.average_staleness = m.rejection_rate * challenge.recommended_max_staleness

        # Compression ratio
        if result.config_snapshot.get('compression_method') == 'topk':
            m.total_compression_ratio = 1.0 / result.config_snapshot.get('compression_ratio', 1.0)
            m.bandwidth_saved_percent = (1.0 - result.config_snapshot.get('compression_ratio', 1.0)) * 100
        elif result.config_snapshot.get('compression_method') == 'quantize':
            m.total_compression_ratio = 4.0
            m.bandwidth_saved_percent = 75.0
        elif result.config_snapshot.get('compression_method') == 'fp16':
            m.total_compression_ratio = 2.0
            m.bandwidth_saved_percent = 50.0

    def _evaluate_success_criteria(self, result: ExperimentResult, challenge: ChallengeConfig):
        """Evaluate whether success criteria were met.

        Args:
            result: Experiment result
            challenge: Challenge configuration
        """
        m = result.metrics
        failures = []

        # Check minimum convergence steps
        if m.total_steps < challenge.min_convergence_steps:
            failures.append(
                f"Insufficient steps: {m.total_steps} < {challenge.min_convergence_steps}"
            )

        # Check rejection rate
        if m.rejection_rate > challenge.max_rejection_rate:
            failures.append(
                f"High rejection rate: {m.rejection_rate:.1%} > {challenge.max_rejection_rate:.1%}"
            )

        # Check throughput (simplified - compare to ideal)
        ideal_throughput = 1.0 / 0.1  # Assuming 100ms ideal step time
        if m.total_time_s > 0:
            actual_ratio = m.steps_per_second / ideal_throughput
            if actual_ratio < challenge.min_throughput_ratio:
                failures.append(
                    f"Low throughput: {actual_ratio:.1%} < {challenge.min_throughput_ratio:.1%}"
                )

        result.success_criteria_met = len(failures) == 0
        result.failure_reasons = failures

    def _print_result_summary(self, result: ExperimentResult):
        """Print experiment result summary.

        Args:
            result: Experiment result
        """
        print()
        print("-" * 100)
        print(f"RESULT: {result.status.value.upper()}")
        print("-" * 100)

        m = result.metrics
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Steps completed: {m.total_steps}")
        print(f"Throughput: {m.steps_per_second:.2f} steps/s")
        print(f"Gradients: {m.total_gradients_submitted} submitted, "
              f"{m.total_gradients_accepted} accepted, "
              f"{m.total_gradients_rejected} rejected")
        print(f"Rejection rate: {m.rejection_rate:.1%}")
        print(f"Compression: {m.total_compression_ratio:.1f}x "
              f"({m.bandwidth_saved_percent:.1f}% bandwidth saved)")
        print(f"Active workers: {m.active_workers}/{len(m.total_worker_steps)}")

        print()
        print(f"Success: {'✓ PASSED' if result.success_criteria_met else '✗ FAILED'}")
        if result.failure_reasons:
            print("Failure reasons:")
            for reason in result.failure_reasons:
                print(f"  - {reason}")

        print("=" * 100)
        print()

    def run_experiment_batch(
        self,
        scenarios: List[str],
        challenges: List[str],
        max_steps: int = 100
    ) -> List[ExperimentResult]:
        """Run a batch of experiments.

        Args:
            scenarios: List of scenario names
            challenges: List of challenge names
            max_steps: Maximum steps per experiment

        Returns:
            List of experiment results
        """
        results = []

        total_experiments = len(scenarios) * len(challenges)
        current = 0

        for scenario_name in scenarios:
            scenario = get_scenario(scenario_name)

            for challenge_name in challenges:
                challenge = get_challenge(challenge_name)
                current += 1

                if self.verbose:
                    print(f"\n{'=' * 100}")
                    print(f"BATCH PROGRESS: {current}/{total_experiments}")
                    print(f"{'=' * 100}\n")

                # Use recommended settings from challenge
                result = self.run_experiment(
                    scenario=scenario,
                    challenge=challenge,
                    use_async=challenge.recommended_async,
                    compression_method=challenge.recommended_compression,
                    compression_ratio=challenge.recommended_compression_ratio,
                    max_staleness=challenge.recommended_max_staleness,
                    max_steps=max_steps
                )

                results.append(result)

        return results

    def print_batch_summary(self, results: Optional[List[ExperimentResult]] = None):
        """Print summary of batch results.

        Args:
            results: Results to summarize (None = use all stored results)
        """
        if results is None:
            results = self.results

        if not results:
            print("No results to summarize.")
            return

        print("=" * 120)
        print("BATCH RESULTS SUMMARY")
        print("=" * 120)
        print(f"{'Scenario':<18} {'Challenge':<18} {'Steps':<8} {'Throughput':<12} {'Reject%':<10} {'Success':<10}")
        print("-" * 120)

        passed = 0
        failed = 0

        for r in results:
            success_str = "✓ PASS" if r.success_criteria_met else "✗ FAIL"
            if r.success_criteria_met:
                passed += 1
            else:
                failed += 1

            print(f"{r.scenario_name:<18} {r.challenge_name:<18} "
                  f"{r.metrics.total_steps:<8} {r.metrics.steps_per_second:<12.2f} "
                  f"{r.metrics.rejection_rate:<10.1%} {success_str:<10}")

        print("=" * 120)
        print(f"Total: {len(results)}, Passed: {passed}, Failed: {failed}")
        print("=" * 120)


if __name__ == "__main__":
    # Example usage
    harness = SimulationTestHarness(verbose=True)

    # Run a few interesting experiments
    print("Running sample experiments...\n")

    # Experiment 1: Simple scenario, baseline challenge
    harness.run_experiment(
        scenario=get_scenario('simple'),
        challenge=get_challenge('baseline'),
        max_steps=50
    )

    # Experiment 2: Moderate scenario, WAN bandwidth challenge
    harness.run_experiment(
        scenario=get_scenario('moderate'),
        challenge=get_challenge('wan_bandwidth'),
        max_steps=100
    )

    # Experiment 3: Large scenario, heterogeneous challenge
    harness.run_experiment(
        scenario=get_scenario('large'),
        challenge=get_challenge('heterogeneous'),
        max_steps=100
    )

    # Print summary
    print("\n")
    harness.print_batch_summary()
