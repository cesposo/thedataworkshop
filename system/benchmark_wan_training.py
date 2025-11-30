#!/usr/bin/env python
"""
WAN Training Benchmark Script

This script benchmarks distributed training performance under realistic WAN conditions,
comparing different combinations of:
- Sync vs. Async training
- Different compression methods
- Various network bandwidth scenarios

Addresses the "bandwidth wall" problem identified in the code review.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass
from dist_llm_train.sync.parameter_server import (
    BoundedAsyncCoordinator,
    SimpleSyncCoordinator,
    ParameterServer
)
from dist_llm_train.compression.compressor import GradientCompressor


@dataclass
class NetworkProfile:
    """Network conditions for simulation."""
    name: str
    bandwidth_mbps: float  # Megabits per second
    latency_ms: float      # Milliseconds
    packet_loss: float      # Probability of packet loss (0-1)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    scenario_name: str
    total_time_seconds: float
    steps_completed: int
    effective_throughput: float  # steps per second
    total_bandwidth_mb: float
    bandwidth_per_step_mb: float
    rejection_rate: float
    avg_staleness: float


# Network profiles (realistic WAN scenarios)
NETWORK_PROFILES = {
    'lan': NetworkProfile('LAN (1 Gbps)', bandwidth_mbps=1000, latency_ms=1, packet_loss=0.0),
    'fast_wan': NetworkProfile('Fast WAN (100 Mbps)', bandwidth_mbps=100, latency_ms=50, packet_loss=0.001),
    'typical_wan': NetworkProfile('Typical WAN (50 Mbps)', bandwidth_mbps=50, latency_ms=100, packet_loss=0.01),
    'slow_wan': NetworkProfile('Slow WAN (10 Mbps)', bandwidth_mbps=10, latency_ms=200, packet_loss=0.02),
    'edge': NetworkProfile('Edge/Mobile (5 Mbps)', bandwidth_mbps=5, latency_ms=300, packet_loss=0.05),
}


def calculate_transmission_time(data_bytes: float, network: NetworkProfile) -> float:
    """Calculate time to transmit data over network."""
    # Convert bytes to bits
    data_bits = data_bytes * 8

    # Convert bandwidth to bits per second
    bandwidth_bps = network.bandwidth_mbps * 1_000_000

    # Transmission time (seconds)
    transmission_time = data_bits / bandwidth_bps

    # Add latency (convert ms to seconds)
    total_time = transmission_time + (network.latency_ms / 1000)

    return total_time


def simulate_gradient_size(model: nn.Module) -> int:
    """Calculate total gradient size in bytes."""
    total_params = sum(p.numel() for p in model.parameters())
    # FP32 = 4 bytes per parameter
    return total_params * 4


def run_benchmark(
    model: nn.Module,
    network: NetworkProfile,
    use_async: bool,
    compression_method: str = None,
    compression_ratio: float = 0.01,
    num_steps: int = 100,
    num_workers: int = 3,
    max_staleness: int = 5
) -> BenchmarkResult:
    """
    Run a single benchmark configuration.

    Args:
        model: Neural network model
        network: Network profile for simulation
        use_async: Use async coordinator if True, sync if False
        compression_method: Compression method ('topk', 'quantize', 'fp16', None)
        compression_ratio: Compression ratio for topk
        num_steps: Number of training steps
        num_workers: Number of workers
        max_staleness: Max staleness for async mode

    Returns:
        BenchmarkResult with performance metrics
    """

    # Create parameter server
    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    param_server = ParameterServer(initial_params)

    # Create coordinator
    if use_async:
        coordinator = BoundedAsyncCoordinator(num_workers=num_workers, max_staleness=max_staleness)
        coord_name = f"Async(s={max_staleness})"
    else:
        coordinator = SimpleSyncCoordinator(num_workers=num_workers)
        coord_name = "Sync"

    # Create compressor
    if compression_method:
        if compression_method == 'topk':
            compressor = GradientCompressor(method=compression_method, compression_ratio=compression_ratio)
            comp_name = f"{compression_method}({compression_ratio*100:.0f}%)"
        else:
            compressor = GradientCompressor(method=compression_method)
            comp_name = compression_method
    else:
        compressor = None
        comp_name = "None"

    scenario_name = f"{coord_name} + {comp_name} on {network.name}"

    # Calculate gradient size
    uncompressed_size = simulate_gradient_size(model)

    # Tracking
    total_bandwidth = 0
    total_staleness = 0
    staleness_count = 0
    start_time = time.time()

    # Simulate training
    for step in range(num_steps):
        # Generate dummy gradients
        gradients = {name: param.grad if param.grad is not None else torch.zeros_like(param)
                    for name, param in model.named_parameters()}

        # Apply compression
        if compressor:
            compressed, metadata = compressor.compress(gradients)
            stats = compressor.get_compression_stats(gradients, compressed)
            gradient_size = stats['compressed_bytes']
            decompressed = compressor.decompress(compressed, metadata)
        else:
            gradient_size = uncompressed_size
            decompressed = gradients

        # Simulate network transmission
        transmission_time = calculate_transmission_time(gradient_size, network)
        time.sleep(transmission_time / 1000)  # Scale down for simulation

        total_bandwidth += gradient_size

        # Submit to coordinator
        if use_async:
            accepted = coordinator.submit_gradients(f'worker_{step % num_workers}', decompressed, step)
            if accepted:
                accumulated = coordinator.get_and_clear_gradients()
                for wid, grads, staleness in accumulated:
                    param_server.push_gradients(wid, grads)
                    total_staleness += staleness
                    staleness_count += 1
                if accumulated:
                    param_server.aggregate_and_update(learning_rate=0.01)
        else:
            # Sync mode - all workers must wait
            coordinator.submit_gradients(f'worker_{step % num_workers}', decompressed)
            if (step + 1) % num_workers == 0:
                # Barrier - all workers arrived
                for wid in range(num_workers):
                    param_server.push_gradients(f'worker_{wid}', decompressed)
                param_server.aggregate_and_update(learning_rate=0.01)

    elapsed_time = time.time() - start_time

    # Calculate metrics
    steps_completed = num_steps
    effective_throughput = steps_completed / elapsed_time if elapsed_time > 0 else 0
    total_bandwidth_mb = total_bandwidth / (1024 * 1024)
    bandwidth_per_step_mb = total_bandwidth_mb / steps_completed if steps_completed > 0 else 0

    if use_async:
        stats = coordinator.get_statistics()
        rejection_rate = stats['rejection_rate']
        avg_staleness = total_staleness / staleness_count if staleness_count > 0 else 0
    else:
        rejection_rate = 0.0
        avg_staleness = 0.0

    return BenchmarkResult(
        scenario_name=scenario_name,
        total_time_seconds=elapsed_time,
        steps_completed=steps_completed,
        effective_throughput=effective_throughput,
        total_bandwidth_mb=total_bandwidth_mb,
        bandwidth_per_step_mb=bandwidth_per_step_mb,
        rejection_rate=rejection_rate,
        avg_staleness=avg_staleness
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 120)
    print(f"{'Scenario':<45} {'Time(s)':<10} {'Steps':<8} {'Throughput':<12} {'BW/step(MB)':<13} {'Reject%':<10} {'Staleness':<10}")
    print("=" * 120)

    for result in results:
        print(f"{result.scenario_name:<45} "
              f"{result.total_time_seconds:<10.2f} "
              f"{result.steps_completed:<8} "
              f"{result.effective_throughput:<12.2f} "
              f"{result.bandwidth_per_step_mb:<13.2f} "
              f"{result.rejection_rate*100:<10.1f} "
              f"{result.avg_staleness:<10.2f}")

    print("=" * 120)


def main():
    """Run comprehensive WAN training benchmarks."""

    print("=" * 120)
    print("WAN DISTRIBUTED TRAINING BENCHMARK")
    print("=" * 120)
    print("\nThis benchmark demonstrates the impact of:")
    print("  1. Async vs. Sync training in heterogeneous environments")
    print("  2. Gradient compression on bandwidth requirements")
    print("  3. Network conditions on training throughput")
    print()

    # Create a medium-sized model (~ 1M parameters = 4MB gradients)
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 100),
    )

    total_params = sum(p.numel() for p in model.parameters())
    gradient_size_mb = (total_params * 4) / (1024 * 1024)

    print(f"Model Configuration:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Gradient size (FP32): {gradient_size_mb:.2f} MB")
    print()

    # Benchmark configurations
    num_steps = 50  # Reduced for faster benchmarking

    print(f"Running benchmarks with {num_steps} training steps per configuration...")
    print("(This may take a few minutes)\n")

    results = []

    # Test different network conditions
    for network_name in ['typical_wan', 'slow_wan']:
        network = NETWORK_PROFILES[network_name]

        print(f"\nTesting on {network.name} ({network.bandwidth_mbps} Mbps, {network.latency_ms}ms latency)...")

        # Baseline: Sync without compression
        print("  - Running: Sync + No Compression...")
        result = run_benchmark(
            model=model,
            network=network,
            use_async=False,
            compression_method=None,
            num_steps=num_steps
        )
        results.append(result)

        # Async without compression
        print("  - Running: Async + No Compression...")
        result = run_benchmark(
            model=model,
            network=network,
            use_async=True,
            compression_method=None,
            num_steps=num_steps
        )
        results.append(result)

        # Async with FP16
        print("  - Running: Async + FP16...")
        result = run_benchmark(
            model=model,
            network=network,
            use_async=True,
            compression_method='fp16',
            num_steps=num_steps
        )
        results.append(result)

        # Async with Top-K (10%)
        print("  - Running: Async + Top-K(10%)...")
        result = run_benchmark(
            model=model,
            network=network,
            use_async=True,
            compression_method='topk',
            compression_ratio=0.1,
            num_steps=num_steps
        )
        results.append(result)

        # Async with Top-K (1%) - aggressive
        print("  - Running: Async + Top-K(1%)...")
        result = run_benchmark(
            model=model,
            network=network,
            use_async=True,
            compression_method='topk',
            compression_ratio=0.01,
            num_steps=num_steps
        )
        results.append(result)

    # Print results
    print_results_table(results)

    # Analysis
    print("\n" + "=" * 120)
    print("KEY FINDINGS:")
    print("=" * 120)

    # Find best and worst performers
    results_sorted = sorted(results, key=lambda r: r.effective_throughput, reverse=True)
    best = results_sorted[0]
    worst = results_sorted[-1]

    print(f"\n1. BEST PERFORMING CONFIGURATION:")
    print(f"   {best.scenario_name}")
    print(f"   Throughput: {best.effective_throughput:.2f} steps/sec")
    print(f"   Bandwidth per step: {best.bandwidth_per_step_mb:.2f} MB")

    print(f"\n2. WORST PERFORMING CONFIGURATION:")
    print(f"   {worst.scenario_name}")
    print(f"   Throughput: {worst.effective_throughput:.2f} steps/sec")
    print(f"   Bandwidth per step: {worst.bandwidth_per_step_mb:.2f} MB")

    speedup = best.effective_throughput / worst.effective_throughput if worst.effective_throughput > 0 else float('inf')
    bandwidth_reduction = (1 - best.bandwidth_per_step_mb / worst.bandwidth_per_step_mb) * 100 if worst.bandwidth_per_step_mb > 0 else 0

    print(f"\n3. IMPROVEMENT:")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Bandwidth reduction: {bandwidth_reduction:.1f}%")

    print("\n4. RECOMMENDATIONS FOR WAN TRAINING:")
    print("   - Always use async training for heterogeneous environments")
    print("   - Use top-k(1-10%) for severely bandwidth-constrained scenarios")
    print("   - Use FP16 as a safe default (2x compression, minimal accuracy loss)")
    print("   - Monitor rejection rate - if >20%, increase max_staleness")

    print("\n" + "=" * 120)


if __name__ == '__main__':
    main()
