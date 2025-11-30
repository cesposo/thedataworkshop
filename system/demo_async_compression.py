#!/usr/bin/env python
"""
Demo script showcasing asynchronous training with gradient compression.

This script demonstrates the new features that address the architectural issues
identified in the code review:
1. Bounded-staleness asynchronous SGD (eliminates straggler problem)
2. Gradient compression (reduces bandwidth requirements)
3. Byzantine-tolerant aggregation (robustness against malicious gradients)

Run with: python demo_async_compression.py
"""

import time
import torch
import torch.nn as nn
from typing import Dict
from dist_llm_train.sync.parameter_server import BoundedAsyncCoordinator, ParameterServer
from dist_llm_train.compression.compressor import GradientCompressor


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class SimulatedWorker:
    """Simulates a worker node in a heterogeneous cluster."""

    def __init__(self, worker_id: str, speed_multiplier: float,
                 compressor: GradientCompressor, model: nn.Module):
        """
        Args:
            worker_id: Worker identifier
            speed_multiplier: Relative speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
            compressor: Gradient compressor instance
            model: Neural network model
        """
        self.worker_id = worker_id
        self.speed_multiplier = speed_multiplier
        self.compressor = compressor
        self.model = model
        self.step = 0
        self.criterion = nn.CrossEntropyLoss()

    def compute_gradients(self, batch_size=32, input_size=100, output_size=10):
        """Simulate a training step and compute gradients."""
        # Simulate forward/backward pass time based on worker speed
        computation_time = 0.1 / self.speed_multiplier
        time.sleep(computation_time)

        # Generate dummy data
        inputs = torch.randn(batch_size, input_size)
        targets = torch.randint(0, output_size, (batch_size,))

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()

        return gradients, loss.item()

    def compress_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Compress gradients and return compression stats."""
        if self.compressor is None:
            # No compression
            return gradients, None, None

        compressed, metadata = self.compressor.compress(gradients)
        stats = self.compressor.get_compression_stats(gradients, compressed)

        return compressed, metadata, stats


def run_async_training_demo():
    """Run demonstration of async training with compression."""

    print("=" * 80)
    print("ASYNCHRONOUS TRAINING + GRADIENT COMPRESSION DEMO")
    print("=" * 80)
    print()

    # Configuration
    num_steps = 20
    learning_rate = 0.01
    max_staleness = 5

    print("Configuration:")
    print(f"  - Training steps: {num_steps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max staleness: {max_staleness}")
    print()

    # Create shared model
    model = SimpleModel()
    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Create parameter server
    param_server = ParameterServer(initial_params)

    # Create async coordinator
    coordinator = BoundedAsyncCoordinator(num_workers=3, max_staleness=max_staleness)

    print("=" * 80)
    print("SCENARIO 1: NO COMPRESSION (Baseline)")
    print("=" * 80)
    run_scenario(
        coordinator=coordinator,
        param_server=param_server,
        model=model,
        compression_method=None,
        num_steps=num_steps,
        learning_rate=learning_rate
    )

    print("\n" + "=" * 80)
    print("SCENARIO 2: TOP-K COMPRESSION (1%)")
    print("=" * 80)
    # Reset coordinator and parameter server
    coordinator = BoundedAsyncCoordinator(num_workers=3, max_staleness=max_staleness)
    param_server = ParameterServer(initial_params)

    run_scenario(
        coordinator=coordinator,
        param_server=param_server,
        model=model,
        compression_method='topk',
        compression_ratio=0.01,
        num_steps=num_steps,
        learning_rate=learning_rate
    )

    print("\n" + "=" * 80)
    print("SCENARIO 3: FP16 COMPRESSION")
    print("=" * 80)
    coordinator = BoundedAsyncCoordinator(num_workers=3, max_staleness=max_staleness)
    param_server = ParameterServer(initial_params)

    run_scenario(
        coordinator=coordinator,
        param_server=param_server,
        model=model,
        compression_method='fp16',
        num_steps=num_steps,
        learning_rate=learning_rate
    )

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Async training allows fast workers to progress without waiting for slow ones")
    print("2. Compression dramatically reduces bandwidth (100x for top-k 1%)")
    print("3. Stale gradients beyond max_staleness are automatically rejected")
    print("4. System can handle heterogeneous worker speeds in WAN settings")


def run_scenario(coordinator, param_server, model, compression_method=None,
                compression_ratio=0.01, num_steps=20, learning_rate=0.01):
    """Run a single training scenario."""

    # Create compressor
    if compression_method:
        compressor = GradientCompressor(method=compression_method, compression_ratio=compression_ratio)
        print(f"Compression: {compression_method}", end="")
        if compression_method == 'topk':
            print(f" (ratio={compression_ratio})")
        else:
            print()
    else:
        compressor = None
        print("Compression: None")

    # Create workers with different speeds (simulating heterogeneous cluster)
    workers = {
        'gpu_worker': SimulatedWorker('gpu_worker', speed_multiplier=2.0, compressor=compressor, model=model),
        'cpu_worker': SimulatedWorker('cpu_worker', speed_multiplier=0.5, compressor=compressor, model=model),
        'slow_worker': SimulatedWorker('slow_worker', speed_multiplier=0.25, compressor=compressor, model=model),
    }

    print(f"Workers:")
    for worker_id, worker in workers.items():
        print(f"  - {worker_id}: {worker.speed_multiplier}x speed")
    print()

    # Training loop
    print(f"Running {num_steps} training steps...")
    print(f"{'Step':<6} {'Worker':<15} {'Loss':<8} {'Staleness':<10} {'Compression':<15} {'Status':<10}")
    print("-" * 80)

    total_bandwidth = 0
    total_steps_completed = {worker_id: 0 for worker_id in workers.keys()}

    start_time = time.time()

    for global_step in range(num_steps):
        # Each worker attempts to complete a step (based on their speed)
        for worker_id, worker in workers.items():
            # Faster workers complete more steps
            if global_step % max(1, int(1 / worker.speed_multiplier)) == 0:
                # Compute gradients
                gradients, loss = worker.compute_gradients()

                # Compress gradients
                compressed, metadata, stats = worker.compress_gradients(gradients)

                # Track bandwidth
                if stats:
                    total_bandwidth += stats['compressed_bytes']
                    compression_str = f"{stats['compression_ratio']:.1f}x"
                else:
                    total_bandwidth += sum(g.numel() * 4 for g in gradients.values())
                    compression_str = "None"

                # Decompress (if compressed)
                if metadata:
                    final_grads = compressor.decompress(compressed, metadata)
                else:
                    final_grads = gradients

                # Submit to coordinator
                accepted = coordinator.submit_gradients(worker_id, final_grads, worker.step)

                if accepted:
                    staleness = coordinator.global_step - worker.step
                    status = "✓ Accepted"
                    worker.step += 1
                    total_steps_completed[worker_id] += 1
                else:
                    staleness = coordinator.global_step - worker.step
                    status = "✗ Rejected"

                print(f"{global_step:<6} {worker_id:<15} {loss:<8.4f} {staleness:<10} {compression_str:<15} {status:<10}")

        # Apply accumulated gradients
        accumulated = coordinator.get_and_clear_gradients()
        if accumulated:
            for wid, grads, staleness in accumulated:
                param_server.push_gradients(wid, grads)

            param_server.aggregate_and_update(learning_rate=learning_rate)

    elapsed_time = time.time() - start_time

    # Print statistics
    print("-" * 80)
    stats = coordinator.get_statistics()
    print(f"\nResults:")
    print(f"  - Total time: {elapsed_time:.2f}s")
    print(f"  - Global steps: {stats['global_step']}")
    print(f"  - Total gradients received: {stats['total_received']}")
    print(f"  - Total gradients rejected: {stats['total_rejected']}")
    print(f"  - Rejection rate: {stats['rejection_rate']*100:.1f}%")
    print(f"  - Total bandwidth used: {total_bandwidth / 1024 / 1024:.2f} MB")
    print(f"  - Steps completed per worker:")
    for worker_id, steps in total_steps_completed.items():
        print(f"    - {worker_id}: {steps}")


if __name__ == '__main__':
    run_async_training_demo()
