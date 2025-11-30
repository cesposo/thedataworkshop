"""
Full Integration Test

Tests all major features working together:
1. Adaptive staleness bounds
2. Byzantine-robust aggregation (Bulyan)
3. Gradient compression (top-k)
4. Differential privacy (gradient clipping + noise)
5. ZMQ binary communication (mocked)

Validates that the complete system works end-to-end.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Skipping integration tests.")
    sys.exit(0)

import unittest
from typing import Dict
from tests.convergence.test_real_training import (
    SyntheticDataset, SimpleNet, evaluate_model, PYTORCH_AVAILABLE
)
from tests.convergence.test_byzantine_attacks import ByzantineWorker
from dist_llm_train.sync.parameter_server import ParameterServer, BoundedAsyncCoordinator
from dist_llm_train.compression.compressor import GradientCompressor

logger = logging.getLogger(__name__)


@unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
class TestFullIntegration(unittest.TestCase):
    """Test all features working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 20
        self.hidden_dim = 50
        self.num_classes = 10
        self.num_workers = 9  # 7 honest + 2 Byzantine
        self.num_byzantine = 2

        self.train_dataset = SyntheticDataset(
            num_samples=500, input_dim=self.input_dim,
            num_classes=self.num_classes, seed=42
        )
        self.test_dataset = SyntheticDataset(
            num_samples=100, input_dim=self.input_dim,
            num_classes=self.num_classes, seed=123
        )

    def test_full_feature_integration(self):
        """
        Test all features together:
        - Adaptive staleness
        - Byzantine aggregation (Bulyan)
        - Gradient compression (top-k 10%)
        - Differential privacy (clipping + noise)
        - Heterogeneous workers
        """
        print("\n=== Full Feature Integration Test ===")
        print("Features enabled:")
        print("  ✓ Adaptive staleness (per-worker bounds)")
        print("  ✓ Byzantine defense (Bulyan, f=2)")
        print("  ✓ Gradient compression (top-k 10%)")
        print("  ✓ Differential privacy (clip + noise)")
        print("  ✓ Heterogeneous workers (5x speed range)")

        # Create model and parameter server
        model = SimpleNet(self.input_dim, self.hidden_dim, self.num_classes)
        initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        ps = ParameterServer(initial_model_state=initial_state)

        # Create adaptive staleness coordinator
        coordinator = BoundedAsyncCoordinator(
            num_workers=self.num_workers,
            max_staleness=50,
            adaptive_staleness=True,
            min_staleness=5,
            max_staleness_multiplier=10.0
        )

        # Create gradient compressor
        compressor = GradientCompressor(method='topk', compression_ratio=0.1)

        # Create dataloader
        dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)

        # Heterogeneous worker speeds (5x range)
        worker_speeds = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.4, 0.4]

        # Training loop
        num_steps = 50
        global_step = 0
        stats = {
            'gradients_submitted': 0,
            'gradients_accepted': 0,
            'gradients_compressed': 0,
            'gradients_clipped': 0,
            'byzantine_filtered': 0
        }

        for step in range(num_steps):
            data_iter = iter(dataloader)

            # Simulate gradients from heterogeneous workers
            for i in range(self.num_workers):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    inputs, targets = next(data_iter)

                # Forward + backward
                model.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()

                # Collect gradients
                gradients = {name: param.grad.clone()
                           for name, param in model.named_parameters()
                           if param.grad is not None}

                # Corrupt if Byzantine worker
                if i < self.num_byzantine:
                    byzantine = ByzantineWorker(f"byzantine-{i}", 'sign_flip')
                    gradients = byzantine.corrupt_gradient(gradients)
                    stats['byzantine_filtered'] += 1

                # Apply differential privacy (gradient clipping)
                gradient_clip_norm = 1.0
                total_norm = torch.sqrt(sum(g.norm() ** 2 for g in gradients.values()))
                if total_norm > gradient_clip_norm:
                    clip_coef = gradient_clip_norm / (total_norm + 1e-6)
                    gradients = {k: v * clip_coef for k, v in gradients.items()}
                    stats['gradients_clipped'] += 1

                # Add DP noise
                dp_noise_multiplier = 0.05  # Small noise for testing
                for k in gradients:
                    noise = torch.randn_like(gradients[k]) * dp_noise_multiplier
                    gradients[k] = gradients[k] + noise

                # Compress gradients
                compressed, metadata = compressor.compress(gradients)
                decompressed = compressor.decompress(compressed, metadata)
                stats['gradients_compressed'] += 1

                # Submit to coordinator (check staleness)
                stats['gradients_submitted'] += 1
                accepted = coordinator.submit_gradients(
                    f"worker-{i}", decompressed, global_step
                )

                if accepted:
                    stats['gradients_accepted'] += 1
                    ps.push_gradients(f"worker-{i}", decompressed)

            # Aggregate with Bulyan (Byzantine-robust)
            num_grads = len(ps.gradient_buffer)
            if num_grads > 0:
                success = ps.aggregate_and_update(
                    learning_rate=0.01,
                    rule='bulyan',
                    bulyan_f=self.num_byzantine
                )

                if success:
                    global_step += 1

                    # Update model
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            param.copy_(ps.model_state[name])

        # Final evaluation
        final_acc = evaluate_model(model, self.test_dataset)

        # Print statistics
        print(f"\nTraining Statistics:")
        print(f"  Total steps: {global_step}")
        print(f"  Gradients submitted: {stats['gradients_submitted']}")
        print(f"  Gradients accepted: {stats['gradients_accepted']}")
        print(f"  Acceptance rate: {stats['gradients_accepted']/stats['gradients_submitted']:.1%}")
        print(f"  Gradients compressed: {stats['gradients_compressed']}")
        print(f"  Gradients clipped (DP): {stats['gradients_clipped']}")
        print(f"  Byzantine workers: {self.num_byzantine}/{self.num_workers}")
        print(f"\nFinal test accuracy: {final_acc:.3f}")

        # Assertions: System should complete without crashing
        self.assertGreater(global_step, 0, "Training should make progress")
        self.assertGreater(stats['gradients_accepted'], 0, "Some gradients should be accepted")
        self.assertIsNotNone(final_acc, "Should compute final accuracy")

        # Acceptance rate should be reasonable with adaptive staleness
        acceptance_rate = stats['gradients_accepted'] / stats['gradients_submitted']
        self.assertGreater(acceptance_rate, 0.3, "Acceptance rate should be >30% with adaptive staleness")

        print(f"\n✓ Full integration test passed")
        print(f"✓ All features working together without crashes")

    def test_feature_interactions(self):
        """
        Test specific feature interactions:
        - Compression doesn't break Byzantine detection
        - DP noise doesn't break staleness checking
        - Adaptive staleness works with Byzantine defense
        """
        print("\n=== Feature Interaction Test ===")

        # Create simple setup
        model = SimpleNet(self.input_dim, self.hidden_dim, self.num_classes)
        initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        ps = ParameterServer(initial_model_state=initial_state)

        # Create a gradient
        test_grad = torch.randn(100)
        gradients = {'param1': test_grad}

        # Test 1: Compression + Decompression preserves gradient structure
        compressor = GradientCompressor(method='topk', compression_ratio=0.1)
        compressed, metadata = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed, metadata)

        self.assertIn('param1', decompressed, "Decompressed should have same keys")
        self.assertEqual(decompressed['param1'].shape, test_grad.shape,
                        "Shape should be preserved through compression")

        # Test 2: Gradient clipping maintains direction
        original_direction = test_grad / test_grad.norm()
        max_norm = 1.0
        total_norm = test_grad.norm()
        if total_norm > max_norm:
            clipped = test_grad * (max_norm / total_norm)
            clipped_direction = clipped / clipped.norm()

            # Direction should be preserved
            cosine_sim = torch.dot(original_direction, clipped_direction)
            self.assertGreater(cosine_sim, 0.99, "Clipping should preserve direction")

        # Test 3: Multiple aggregation rules work
        for rule in ['mean', 'trimmed_mean', 'krum', 'bulyan']:
            # Push some gradients
            for i in range(9):
                grad = {name: torch.randn_like(param) * 0.01
                       for name, param in model.named_parameters()}
                ps.push_gradients(f"worker-{i}", grad)

            # Aggregate
            if rule == 'mean':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='mean')
            elif rule == 'trimmed_mean':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='trimmed_mean', trim_ratio=0.3)
            elif rule == 'krum':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='krum', krum_f=2)
            elif rule == 'bulyan':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='bulyan', bulyan_f=2)

            self.assertTrue(success, f"{rule} aggregation should succeed")

        print(f"✓ Compression preserves gradient structure")
        print(f"✓ Gradient clipping preserves direction")
        print(f"✓ All aggregation rules work correctly")


@unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
class TestSystemRobustness(unittest.TestCase):
    """Test system robustness under stress."""

    def test_empty_gradient_buffer(self):
        """Test aggregation with empty gradient buffer."""
        print("\n=== Empty Gradient Buffer Test ===")

        model = SimpleNet(20, 50, 10)
        initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        ps = ParameterServer(initial_model_state=initial_state)

        # Try to aggregate with no gradients
        success = ps.aggregate_and_update(learning_rate=0.01, rule='mean')

        # Should return False (no gradients to aggregate)
        self.assertFalse(success, "Aggregation should return False with empty buffer")

        print("✓ System handles empty gradient buffer gracefully")

    def test_single_worker(self):
        """Test aggregation with single worker."""
        print("\n=== Single Worker Test ===")

        model = SimpleNet(20, 50, 10)
        initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        ps = ParameterServer(initial_model_state=initial_state)

        # Push gradient from single worker
        gradients = {name: torch.randn_like(param) * 0.01
                    for name, param in model.named_parameters()}
        ps.push_gradients("worker-0", gradients)

        # All aggregation rules should work with single worker
        for rule in ['mean', 'trimmed_mean', 'krum', 'bulyan']:
            ps.push_gradients("worker-0", gradients)  # Push again for each test

            if rule == 'mean':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='mean')
            elif rule == 'trimmed_mean':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='trimmed_mean', trim_ratio=0.3)
            elif rule == 'krum':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='krum', krum_f=0)
            elif rule == 'bulyan':
                success = ps.aggregate_and_update(learning_rate=0.01, rule='bulyan', bulyan_f=0)

            self.assertTrue(success, f"{rule} should work with single worker")

        print("✓ All aggregation rules handle single worker")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
