"""Results collection and analysis for simulation experiments.

This module provides tools for:
- Storing experiment results to disk (JSON, CSV)
- Loading and analyzing historical results
- Generating comparison reports
- Statistical analysis and visualization
"""

import json
import csv
import os
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_harness import ExperimentResult, ExperimentStatus


class ResultsStore:
    """Storage manager for experiment results."""

    def __init__(self, results_dir: str = "results"):
        """Initialize results store.

        Args:
            results_dir: Directory to store results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_result(self, result: ExperimentResult, filename: Optional[str] = None) -> str:
        """Save a single experiment result to JSON.

        Args:
            result: Experiment result
            filename: Optional filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.scenario_name}_{result.challenge_name}_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        # Convert result to dict
        result_dict = self._result_to_dict(result)

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        return filepath

    def save_batch_results(self, results: List[ExperimentResult],
                           filename: Optional[str] = None) -> str:
        """Save multiple experiment results to JSON.

        Args:
            results: List of experiment results
            filename: Optional filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_results_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(results),
            'results': [self._result_to_dict(r) for r in results]
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        return filepath

    def save_results_csv(self, results: List[ExperimentResult],
                         filename: Optional[str] = None) -> str:
        """Save experiment results to CSV.

        Args:
            results: List of experiment results
            filename: Optional filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)

        if not results:
            return filepath

        # Define CSV columns
        fieldnames = [
            'scenario_name', 'challenge_name', 'status', 'success',
            'total_steps', 'duration_s', 'steps_per_second',
            'total_gradients_submitted', 'total_gradients_accepted',
            'total_gradients_rejected', 'rejection_rate',
            'compression_ratio', 'bandwidth_saved_percent',
            'active_workers', 'use_async', 'compression_method',
            'max_staleness'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'scenario_name': result.scenario_name,
                    'challenge_name': result.challenge_name,
                    'status': result.status.value,
                    'success': result.success_criteria_met,
                    'total_steps': result.metrics.total_steps,
                    'duration_s': result.duration_s,
                    'steps_per_second': result.metrics.steps_per_second,
                    'total_gradients_submitted': result.metrics.total_gradients_submitted,
                    'total_gradients_accepted': result.metrics.total_gradients_accepted,
                    'total_gradients_rejected': result.metrics.total_gradients_rejected,
                    'rejection_rate': result.metrics.rejection_rate,
                    'compression_ratio': result.metrics.total_compression_ratio,
                    'bandwidth_saved_percent': result.metrics.bandwidth_saved_percent,
                    'active_workers': result.metrics.active_workers,
                    'use_async': result.config_snapshot.get('use_async', False),
                    'compression_method': result.config_snapshot.get('compression_method', 'none'),
                    'max_staleness': result.config_snapshot.get('max_staleness', 0),
                }
                writer.writerow(row)

        return filepath

    def load_result(self, filepath: str) -> ExperimentResult:
        """Load a single experiment result from JSON.

        Args:
            filepath: Path to JSON file

        Returns:
            Experiment result
        """
        with open(filepath, 'r') as f:
            result_dict = json.load(f)

        return self._dict_to_result(result_dict)

    def load_batch_results(self, filepath: str) -> List[ExperimentResult]:
        """Load multiple experiment results from JSON.

        Args:
            filepath: Path to JSON file

        Returns:
            List of experiment results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return [self._dict_to_result(r) for r in data['results']]

    def _result_to_dict(self, result: ExperimentResult) -> Dict[str, Any]:
        """Convert ExperimentResult to dictionary.

        Args:
            result: Experiment result

        Returns:
            Dictionary representation
        """
        result_dict = asdict(result)
        # Convert enum to string
        result_dict['status'] = result.status.value
        return result_dict

    def _dict_to_result(self, data: Dict[str, Any]) -> ExperimentResult:
        """Convert dictionary to ExperimentResult.

        Args:
            data: Dictionary representation

        Returns:
            Experiment result
        """
        # This is a simplified version - would need proper reconstruction
        # of nested dataclasses in production
        from experiments.test_harness import ExperimentMetrics

        metrics = ExperimentMetrics(**data['metrics'])
        status = ExperimentStatus(data['status'])

        return ExperimentResult(
            scenario_name=data['scenario_name'],
            challenge_name=data['challenge_name'],
            status=status,
            metrics=metrics,
            success_criteria_met=data['success_criteria_met'],
            failure_reasons=data['failure_reasons'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            duration_s=data['duration_s'],
            config_snapshot=data['config_snapshot']
        )


class ResultsAnalyzer:
    """Analyzer for experiment results."""

    def __init__(self, results: List[ExperimentResult]):
        """Initialize analyzer.

        Args:
            results: List of experiment results to analyze
        """
        self.results = results

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute overall statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.results:
            return {}

        stats = {
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r.success_criteria_met),
            'failed': sum(1 for r in self.results if not r.success_criteria_met),
            'success_rate': sum(1 for r in self.results if r.success_criteria_met) / len(self.results),
        }

        # Throughput statistics
        throughputs = [r.metrics.steps_per_second for r in self.results if r.metrics.steps_per_second > 0]
        if throughputs:
            stats['avg_throughput'] = sum(throughputs) / len(throughputs)
            stats['min_throughput'] = min(throughputs)
            stats['max_throughput'] = max(throughputs)

        # Rejection rate statistics
        rejection_rates = [r.metrics.rejection_rate for r in self.results]
        if rejection_rates:
            stats['avg_rejection_rate'] = sum(rejection_rates) / len(rejection_rates)
            stats['max_rejection_rate'] = max(rejection_rates)

        # Compression statistics
        compression_ratios = [r.metrics.total_compression_ratio for r in self.results]
        if compression_ratios:
            stats['avg_compression_ratio'] = sum(compression_ratios) / len(compression_ratios)

        return stats

    def compare_by_scenario(self) -> Dict[str, Dict[str, Any]]:
        """Compare results by scenario.

        Returns:
            Dictionary mapping scenario names to statistics
        """
        by_scenario = defaultdict(list)
        for r in self.results:
            by_scenario[r.scenario_name].append(r)

        comparison = {}
        for scenario_name, scenario_results in by_scenario.items():
            analyzer = ResultsAnalyzer(scenario_results)
            comparison[scenario_name] = analyzer.compute_statistics()

        return comparison

    def compare_by_challenge(self) -> Dict[str, Dict[str, Any]]:
        """Compare results by challenge.

        Returns:
            Dictionary mapping challenge names to statistics
        """
        by_challenge = defaultdict(list)
        for r in self.results:
            by_challenge[r.challenge_name].append(r)

        comparison = {}
        for challenge_name, challenge_results in by_challenge.items():
            analyzer = ResultsAnalyzer(challenge_results)
            comparison[challenge_name] = analyzer.compute_statistics()

        return comparison

    def compare_by_compression(self) -> Dict[str, Dict[str, Any]]:
        """Compare results by compression method.

        Returns:
            Dictionary mapping compression methods to statistics
        """
        by_compression = defaultdict(list)
        for r in self.results:
            method = r.config_snapshot.get('compression_method', 'none')
            by_compression[method].append(r)

        comparison = {}
        for method, method_results in by_compression.items():
            analyzer = ResultsAnalyzer(method_results)
            comparison[method] = analyzer.compute_statistics()

        return comparison

    def find_best_configuration(self) -> Optional[ExperimentResult]:
        """Find best performing configuration.

        Returns:
            Best experiment result (by throughput)
        """
        if not self.results:
            return None

        # Filter to successful experiments only
        successful = [r for r in self.results if r.success_criteria_met]
        if not successful:
            return None

        # Find highest throughput
        return max(successful, key=lambda r: r.metrics.steps_per_second)

    def print_statistics(self):
        """Print comprehensive statistics."""
        stats = self.compute_statistics()

        print("=" * 100)
        print("EXPERIMENT STATISTICS")
        print("=" * 100)
        print(f"Total experiments: {stats.get('total_experiments', 0)}")
        print(f"Successful: {stats.get('successful', 0)}")
        print(f"Failed: {stats.get('failed', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.1%}")
        print()

        if 'avg_throughput' in stats:
            print(f"Average throughput: {stats['avg_throughput']:.2f} steps/s")
            print(f"Min throughput: {stats['min_throughput']:.2f} steps/s")
            print(f"Max throughput: {stats['max_throughput']:.2f} steps/s")
            print()

        if 'avg_rejection_rate' in stats:
            print(f"Average rejection rate: {stats['avg_rejection_rate']:.1%}")
            print(f"Max rejection rate: {stats['max_rejection_rate']:.1%}")
            print()

        if 'avg_compression_ratio' in stats:
            print(f"Average compression ratio: {stats['avg_compression_ratio']:.1f}x")
            print()

        print("=" * 100)

    def print_comparison_by_scenario(self):
        """Print comparison by scenario."""
        comparison = self.compare_by_scenario()

        print("=" * 100)
        print("COMPARISON BY SCENARIO")
        print("=" * 100)
        print(f"{'Scenario':<20} {'Experiments':<12} {'Success Rate':<15} {'Avg Throughput':<18}")
        print("-" * 100)

        for scenario, stats in sorted(comparison.items()):
            print(f"{scenario:<20} {stats['total_experiments']:<12} "
                  f"{stats['success_rate']:<15.1%} "
                  f"{stats.get('avg_throughput', 0):<18.2f}")

        print("=" * 100)

    def print_comparison_by_challenge(self):
        """Print comparison by challenge."""
        comparison = self.compare_by_challenge()

        print("=" * 100)
        print("COMPARISON BY CHALLENGE")
        print("=" * 100)
        print(f"{'Challenge':<20} {'Experiments':<12} {'Success Rate':<15} {'Avg Rejection':<18}")
        print("-" * 100)

        for challenge, stats in sorted(comparison.items()):
            print(f"{challenge:<20} {stats['total_experiments']:<12} "
                  f"{stats['success_rate']:<15.1%} "
                  f"{stats.get('avg_rejection_rate', 0):<18.1%}")

        print("=" * 100)

    def print_comparison_by_compression(self):
        """Print comparison by compression method."""
        comparison = self.compare_by_compression()

        print("=" * 100)
        print("COMPARISON BY COMPRESSION METHOD")
        print("=" * 100)
        print(f"{'Method':<20} {'Experiments':<12} {'Success Rate':<15} {'Avg Compression':<18}")
        print("-" * 100)

        for method, stats in sorted(comparison.items()):
            print(f"{method:<20} {stats['total_experiments']:<12} "
                  f"{stats['success_rate']:<15.1%} "
                  f"{stats.get('avg_compression_ratio', 1.0):<18.1f}x")

        print("=" * 100)

    def generate_report(self, output_path: Optional[str] = None):
        """Generate a comprehensive text report.

        Args:
            output_path: Optional path to save report (None = print to stdout)
        """
        lines = []

        lines.append("=" * 100)
        lines.append("SIMULATION EXPERIMENT REPORT")
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total experiments: {len(self.results)}")
        lines.append("")

        # Overall statistics
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 100)
        stats = self.compute_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")

        # Best configuration
        best = self.find_best_configuration()
        if best:
            lines.append("BEST CONFIGURATION")
            lines.append("-" * 100)
            lines.append(f"Scenario: {best.scenario_name}")
            lines.append(f"Challenge: {best.challenge_name}")
            lines.append(f"Throughput: {best.metrics.steps_per_second:.2f} steps/s")
            lines.append(f"Rejection rate: {best.metrics.rejection_rate:.1%}")
            lines.append(f"Configuration: {best.config_snapshot}")
            lines.append("")

        # Individual experiment details
        lines.append("INDIVIDUAL EXPERIMENTS")
        lines.append("-" * 100)
        for i, result in enumerate(self.results, 1):
            lines.append(f"{i}. {result.scenario_name} × {result.challenge_name}")
            lines.append(f"   Status: {result.status.value}")
            lines.append(f"   Success: {result.success_criteria_met}")
            lines.append(f"   Throughput: {result.metrics.steps_per_second:.2f} steps/s")
            lines.append(f"   Rejection: {result.metrics.rejection_rate:.1%}")
            if result.failure_reasons:
                lines.append(f"   Failures: {', '.join(result.failure_reasons)}")
            lines.append("")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        else:
            print(report)


if __name__ == "__main__":
    # Example usage
    from experiments.test_harness import SimulationTestHarness
    from scenarios.dnn_scenarios import get_scenario
    from challenges.wan_challenges import get_challenge

    print("Running sample experiments for analysis demonstration...\n")

    # Run a batch of experiments
    harness = SimulationTestHarness(verbose=False)

    scenarios = ['simple', 'moderate', 'large']
    challenges = ['baseline', 'wan_bandwidth', 'heterogeneous']

    results = harness.run_experiment_batch(scenarios, challenges, max_steps=50)

    # Store results
    store = ResultsStore()
    json_path = store.save_batch_results(results)
    csv_path = store.save_results_csv(results)

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Analyze results
    print("\n")
    analyzer = ResultsAnalyzer(results)
    analyzer.print_statistics()

    print("\n")
    analyzer.print_comparison_by_scenario()

    print("\n")
    analyzer.print_comparison_by_challenge()

    print("\n")
    analyzer.print_comparison_by_compression()

    # Generate report
    report_path = os.path.join(store.results_dir, "experiment_report.txt")
    analyzer.generate_report(report_path)
