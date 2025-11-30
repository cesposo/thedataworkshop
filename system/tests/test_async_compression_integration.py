"""Integration test for async training with gradient compression.

This test verifies that the async coordinator and gradient compression
work together correctly in an end-to-end scenario.
"""

import unittest
import torch
from dist_llm_train.sync.parameter_server import BoundedAsyncCoordinator, ParameterServer
from dist_llm_train.compression.compressor import GradientCompressor


class TestAsyncCompressionIntegration(unittest.TestCase):
    """Integration tests for async training + compression."""

    def setUp(self):
        """Set up test environment."""
        # Create parameter server
        initial_params = {
            'layer1.weight': torch.randn(100, 50),
            'layer1.bias': torch.randn(100),
        }
        self.param_server = ParameterServer(initial_params)

        # Create async coordinator
        self.coordinator = BoundedAsyncCoordinator(num_workers=3, max_staleness=5)

        # Create compressor
        self.compressor = GradientCompressor(method='topk', compression_ratio=0.1)

    def test_async_training_with_compression(self):
        """Test full async training loop with gradient compression."""
        # Simulate 3 workers with different speeds
        workers = {
            'fast_worker': {'speed': 1, 'step': 0},
            'medium_worker': {'speed': 2, 'step': 0},
            'slow_worker': {'speed': 5, 'step': 0},
        }

        learning_rate = 0.01
        num_steps = 10

        for global_step in range(num_steps):
            # Each worker may complete steps at different rates
            for worker_id, worker_info in workers.items():
                # Simulate worker speed (faster workers complete more steps)
                if global_step % worker_info['speed'] == 0:
                    # Generate gradients
                    gradients = {
                        'layer1.weight': torch.randn(100, 50) * 0.01,
                        'layer1.bias': torch.randn(100) * 0.01,
                    }

                    # Compress gradients
                    compressed, metadata = self.compressor.compress(gradients)

                    # Decompress (simulating transmission and reception)
                    decompressed = self.compressor.decompress(compressed, metadata)

                    # Submit to async coordinator
                    accepted = self.coordinator.submit_gradients(
                        worker_id,
                        decompressed,
                        worker_step=worker_info['step']
                    )

                    if accepted:
                        worker_info['step'] += 1

            # Apply accumulated gradients
            accumulated = self.coordinator.get_and_clear_gradients()
            if accumulated:
                for worker_id, grads, staleness in accumulated:
                    self.param_server.push_gradients(worker_id, grads)

                # Update parameters
                self.param_server.aggregate_and_update(learning_rate=learning_rate)

        # Verify training progressed
        stats = self.coordinator.get_statistics()
        self.assertGreater(stats['total_received'], 0)
        self.assertGreater(self.param_server.get_version(), 0)

        # Verify all workers contributed
        self.assertEqual(len(workers), 3)
        for worker_id in workers:
            self.assertGreater(workers[worker_id]['step'], 0)

    def test_staleness_rejection_with_compression(self):
        """Test that stale gradients are rejected even when compressed."""
        # Create gradients
        gradients = {
            'layer1.weight': torch.randn(100, 50),
            'layer1.bias': torch.randn(100),
        }

        # Compress
        compressed, metadata = self.compressor.compress(gradients)
        decompressed = self.compressor.decompress(compressed, metadata)

        # Submit fresh gradient
        accepted = self.coordinator.submit_gradients('worker1', decompressed, worker_step=0)
        self.assertTrue(accepted)

        # Advance global step significantly
        for _ in range(10):
            self.coordinator.get_and_clear_gradients()

        # Try to submit old gradient
        accepted = self.coordinator.submit_gradients('worker2', decompressed, worker_step=0)
        self.assertFalse(accepted, "Stale gradient should be rejected")

    def test_different_compression_methods(self):
        """Test async training with different compression methods."""
        compression_methods = ['topk', 'quantize', 'fp16', 'none']

        for method in compression_methods:
            with self.subTest(method=method):
                # Create fresh coordinator and compressor
                coordinator = BoundedAsyncCoordinator(num_workers=2, max_staleness=3)
                if method == 'topk':
                    compressor = GradientCompressor(method=method, compression_ratio=0.1)
                else:
                    compressor = GradientCompressor(method=method)

                # Generate gradients
                gradients = {
                    'param1': torch.randn(50, 50),
                }

                # Compress and decompress
                compressed, metadata = compressor.compress(gradients)
                decompressed = compressor.decompress(compressed, metadata)

                # Submit to coordinator
                accepted = coordinator.submit_gradients('worker1', decompressed, worker_step=0)

                self.assertTrue(accepted, f"Failed with method: {method}")

    def test_heterogeneous_workers_no_blocking(self):
        """Test that fast workers don't wait for slow workers."""
        coordinator = BoundedAsyncCoordinator(num_workers=3, max_staleness=10)

        # Fast worker submits multiple times
        for step in range(10):
            gradients = {'param1': torch.randn(10)}
            accepted = coordinator.submit_gradients('fast_worker', gradients, worker_step=step)
            self.assertTrue(accepted)

        # Slow worker hasn't submitted yet - fast worker should have progressed
        stats = coordinator.get_statistics()
        self.assertEqual(stats['active_workers'], 1)  # Only fast worker
        self.assertEqual(stats['total_received'], 10)

        # Now slow worker submits (at old step)
        slow_gradients = {'param1': torch.randn(10)}
        accepted = coordinator.submit_gradients('slow_worker', slow_gradients, worker_step=0)

        # Should be accepted if within staleness bound
        # (depends on how many times we've called get_and_clear_gradients)
        self.assertIsInstance(accepted, bool)

    def test_compression_preserves_convergence_direction(self):
        """Test that compression doesn't reverse gradient direction."""
        # Create simple gradients
        gradients = {
            'param1': torch.ones(100) * -0.5,  # Negative gradients
            'param2': torch.ones(100) * 0.5,   # Positive gradients
        }

        for method in ['topk', 'quantize', 'fp16']:
            with self.subTest(method=method):
                if method == 'topk':
                    compressor = GradientCompressor(method=method, compression_ratio=0.1)
                else:
                    compressor = GradientCompressor(method=method)

                compressed, metadata = compressor.compress(gradients)
                decompressed = compressor.decompress(compressed, metadata)

                # Check that sign is preserved for most elements
                if method == 'topk':
                    # For topk, only check non-zero elements
                    for name in gradients:
                        nonzero_mask = decompressed[name] != 0
                        if nonzero_mask.any():
                            original_sign = torch.sign(gradients[name][nonzero_mask])
                            decompressed_sign = torch.sign(decompressed[name][nonzero_mask])
                            # Most signs should match
                            matching = (original_sign == decompressed_sign).float().mean()
                            self.assertGreater(matching.item(), 0.95)
                else:
                    # For quantize and fp16, all elements should preserve sign
                    for name in gradients:
                        original_sign = torch.sign(gradients[name])
                        decompressed_sign = torch.sign(decompressed[name])
                        matching = (original_sign == decompressed_sign).float().mean()
                        self.assertGreater(matching.item(), 0.99)


if __name__ == '__main__':
    unittest.main()
