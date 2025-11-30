#!/usr/bin/env python3
"""Convenience script for running simulation experiments.

This script provides easy command-line access to the simulation suite.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenarios.dnn_scenarios import get_scenario, list_scenarios, print_scenario_summary
from challenges.wan_challenges import get_challenge, list_challenges, print_challenge_summary, print_challenge_details
from experiments.test_harness import SimulationTestHarness
from experiments.results_analyzer import ResultsAnalyzer, ResultsStore


def list_available():
    """List all available scenarios and challenges."""
    print("\n")
    print_scenario_summary()
    print("\n")
    print_challenge_summary()


def show_scenario(scenario_name):
    """Show details for a specific scenario."""
    try:
        scenario = get_scenario(scenario_name)
        print(f"\nScenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Parameters: {scenario.param_count:,}")
        print(f"Gradient size: {scenario.gradient_size_mb:.2f} MB")
        print(f"Batch size: {scenario.batch_size}")
        print(f"Complexity: {'⭐' * scenario.complexity}")
    except ValueError as e:
        print(f"Error: {e}")


def show_challenge(challenge_name):
    """Show details for a specific challenge."""
    try:
        challenge = get_challenge(challenge_name)
        print("\n")
        print_challenge_details(challenge)
    except ValueError as e:
        print(f"Error: {e}")


def run_single_experiment(args):
    """Run a single experiment."""
    try:
        scenario = get_scenario(args.scenario)
        challenge = get_challenge(args.challenge)

        print(f"\nRunning experiment: {scenario.name} × {challenge.name}\n")

        harness = SimulationTestHarness(verbose=args.verbose)

        result = harness.run_experiment(
            scenario=scenario,
            challenge=challenge,
            use_async=args.use_async,
            compression_method=args.compression,
            compression_ratio=args.compression_ratio,
            max_staleness=args.max_staleness,
            max_steps=args.max_steps
        )

        # Save result if requested
        if args.save:
            store = ResultsStore()
            json_path = store.save_result(result)
            print(f"\nResult saved to: {json_path}")

        return result

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_batch_experiments(args):
    """Run a batch of experiments."""
    scenarios = args.scenarios.split(',')
    challenges = args.challenges.split(',')

    print(f"\nRunning batch: {len(scenarios)} scenarios × {len(challenges)} challenges")
    print(f"Total experiments: {len(scenarios) * len(challenges)}\n")

    harness = SimulationTestHarness(verbose=args.verbose)

    results = harness.run_experiment_batch(
        scenarios=scenarios,
        challenges=challenges,
        max_steps=args.max_steps
    )

    # Print summary
    print("\n")
    harness.print_batch_summary()

    # Analyze
    if args.analyze:
        print("\n")
        analyzer = ResultsAnalyzer(results)
        analyzer.print_statistics()
        print("\n")
        analyzer.print_comparison_by_scenario()
        print("\n")
        analyzer.print_comparison_by_challenge()

    # Save results
    if args.save:
        store = ResultsStore()
        json_path = store.save_batch_results(results)
        csv_path = store.save_results_csv(results)
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")

        if args.report:
            report_path = os.path.join(store.results_dir, "report.txt")
            analyzer = ResultsAnalyzer(results)
            analyzer.generate_report(report_path)
            print(f"  Report: {report_path}")

    return results


def run_quick_test():
    """Run a quick test with default settings."""
    print("\nRunning quick test (3 scenarios × 3 challenges)...\n")

    harness = SimulationTestHarness(verbose=False)

    scenarios = ['simple', 'moderate', 'large']
    challenges = ['baseline', 'wan_bandwidth', 'heterogeneous']

    results = harness.run_experiment_batch(scenarios, challenges, max_steps=50)

    harness.print_batch_summary()

    # Save results
    store = ResultsStore()
    json_path = store.save_batch_results(results, "quick_test.json")
    csv_path = store.save_results_csv(results, "quick_test.csv")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WAN LLM Training Simulation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List available scenarios and challenges')

    # Show scenario command
    show_scenario_parser = subparsers.add_parser('show-scenario', help='Show scenario details')
    show_scenario_parser.add_argument('scenario', help='Scenario name')

    # Show challenge command
    show_challenge_parser = subparsers.add_parser('show-challenge', help='Show challenge details')
    show_challenge_parser.add_argument('challenge', help='Challenge name')

    # Run single experiment
    run_parser = subparsers.add_parser('run', help='Run single experiment')
    run_parser.add_argument('scenario', help='Scenario name')
    run_parser.add_argument('challenge', help='Challenge name')
    run_parser.add_argument('--max-steps', type=int, default=100, help='Maximum training steps')
    run_parser.add_argument('--use-async', action='store_true', default=True, help='Use async training')
    run_parser.add_argument('--compression', default='topk', help='Compression method')
    run_parser.add_argument('--compression-ratio', type=float, default=0.01, help='Compression ratio')
    run_parser.add_argument('--max-staleness', type=int, default=5, help='Max staleness')
    run_parser.add_argument('--save', action='store_true', help='Save results')
    run_parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Run batch experiments
    batch_parser = subparsers.add_parser('batch', help='Run batch of experiments')
    batch_parser.add_argument('--scenarios', required=True, help='Comma-separated scenario names')
    batch_parser.add_argument('--challenges', required=True, help='Comma-separated challenge names')
    batch_parser.add_argument('--max-steps', type=int, default=100, help='Maximum training steps')
    batch_parser.add_argument('--save', action='store_true', help='Save results')
    batch_parser.add_argument('--analyze', action='store_true', help='Print analysis')
    batch_parser.add_argument('--report', action='store_true', help='Generate report')
    batch_parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Quick test command
    quick_parser = subparsers.add_parser('quick-test', help='Run quick test with defaults')

    args = parser.parse_args()

    if args.command == 'list':
        list_available()
    elif args.command == 'show-scenario':
        show_scenario(args.scenario)
    elif args.command == 'show-challenge':
        show_challenge(args.challenge)
    elif args.command == 'run':
        run_single_experiment(args)
    elif args.command == 'batch':
        run_batch_experiments(args)
    elif args.command == 'quick-test':
        run_quick_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
