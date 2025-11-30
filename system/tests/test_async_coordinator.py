"""Tests for BoundedAsyncCoordinator."""

import unittest
import torch
from dist_llm_train.sync.parameter_server import BoundedAsyncCoordinator


class TestBoundedAsyncCoordinator(unittest.TestCase):
    """Test cases for asynchronous training coordinator."""

    def setUp(self):
        """Set up test fixtures."""
        self.coordinator = BoundedAsyncCoordinator(num_workers=3, max_staleness=5)

    def test_initialization(self):
        """Test coordinator initialization."""
        self.assertEqual(self.coordinator.num_workers, 3)
        self.assertEqual(self.coordinator.max_staleness, 5)
        self.assertEqual(self.coordinator.global_step, 0)
        self.assertEqual(len(self.coordinator.worker_steps), 0)
        self.assertEqual(len(self.coordinator.gradient_buffer), 0)

    def test_submit_fresh_gradients(self):
        """Test submitting fresh gradients (no staleness)."""
        gradients = {'param1': torch.randn(10), 'param2': torch.randn(20)}

        # Submit gradients at step 0
        accepted = self.coordinator.submit_gradients('worker1', gradients, worker_step=0)

        self.assertTrue(accepted)
        self.assertEqual(len(self.coordinator.gradient_buffer), 1)
        self.assertEqual(self.coordinator.worker_steps['worker1'], 0)

    def test_submit_stale_gradients_accepted(self):
        """Test that gradients within staleness bound are accepted."""
        gradients = {'param1': torch.randn(10)}

        # Advance global step
        self.coordinator.global_step = 4

        # Submit gradients from step 0 (staleness = 4, within bound of 5)
        accepted = self.coordinator.submit_gradients('worker1', gradients, worker_step=0)

        self.assertTrue(accepted)
        self.assertEqual(len(self.coordinator.gradient_buffer), 1)

    def test_submit_stale_gradients_rejected(self):
        """Test that gradients exceeding staleness bound are rejected."""
        gradients = {'param1': torch.randn(10)}

        # Advance global step
        self.coordinator.global_step = 10

        # Submit gradients from step 0 (staleness = 10, exceeds bound of 5)
        accepted = self.coordinator.submit_gradients('worker1', gradients, worker_step=0)

        self.assertFalse(accepted)
        self.assertEqual(len(self.coordinator.gradient_buffer), 0)
        self.assertEqual(self.coordinator.total_gradients_rejected, 1)

    def test_multiple_workers_no_blocking(self):
        """Test that multiple workers can submit without blocking."""
        grad1 = {'param1': torch.randn(10)}
        grad2 = {'param1': torch.randn(10)}
        grad3 = {'param1': torch.randn(10)}

        # Submit from different workers at different steps
        accepted1 = self.coordinator.submit_gradients('worker1', grad1, worker_step=0)
        accepted2 = self.coordinator.submit_gradients('worker2', grad2, worker_step=0)
        accepted3 = self.coordinator.submit_gradients('worker3', grad3, worker_step=1)

        self.assertTrue(accepted1)
        self.assertTrue(accepted2)
        self.assertTrue(accepted3)
        self.assertEqual(len(self.coordinator.gradient_buffer), 3)

    def test_get_and_clear_gradients(self):
        """Test retrieving and clearing accumulated gradients."""
        grad1 = {'param1': torch.randn(10)}
        grad2 = {'param1': torch.randn(10)}

        self.coordinator.submit_gradients('worker1', grad1, worker_step=0)
        self.coordinator.submit_gradients('worker2', grad2, worker_step=0)

        # Get gradients
        accumulated = self.coordinator.get_and_clear_gradients()

        self.assertEqual(len(accumulated), 2)
        self.assertEqual(accumulated[0][0], 'worker1')  # worker_id
        self.assertEqual(accumulated[0][2], 0)  # staleness
        self.assertEqual(accumulated[1][0], 'worker2')

        # Buffer should be cleared
        self.assertEqual(len(self.coordinator.gradient_buffer), 0)

        # Global step should increment
        self.assertEqual(self.coordinator.global_step, 1)

    def test_wait_for_all_returns_immediately(self):
        """Test that wait_for_all returns immediately (no blocking)."""
        # In async mode, this should always return True immediately
        result = self.coordinator.wait_for_all('worker1')
        self.assertTrue(result)

    def test_statistics_tracking(self):
        """Test statistics collection."""
        grad = {'param1': torch.randn(10)}

        # Submit some gradients
        self.coordinator.submit_gradients('worker1', grad, worker_step=0)
        self.coordinator.submit_gradients('worker2', grad, worker_step=0)

        # Advance and submit stale gradient
        self.coordinator.global_step = 10
        self.coordinator.submit_gradients('worker3', grad, worker_step=0)

        stats = self.coordinator.get_statistics()

        self.assertEqual(stats['total_received'], 3)
        self.assertEqual(stats['total_rejected'], 1)
        self.assertAlmostEqual(stats['rejection_rate'], 1/3, places=2)
        self.assertEqual(stats['active_workers'], 3)

    def test_staleness_calculation(self):
        """Test that staleness is correctly calculated."""
        grad = {'param1': torch.randn(10)}

        # Set global step to 5
        self.coordinator.global_step = 5

        # Submit gradient from step 2 (staleness = 3)
        self.coordinator.submit_gradients('worker1', grad, worker_step=2)

        accumulated = self.coordinator.get_and_clear_gradients()

        self.assertEqual(accumulated[0][2], 3)  # staleness

    def test_default_worker_step(self):
        """Test that worker_step defaults to global_step if not provided."""
        grad = {'param1': torch.randn(10)}

        # Submit without specifying worker_step
        accepted = self.coordinator.submit_gradients('worker1', grad)

        self.assertTrue(accepted)
        self.assertEqual(self.coordinator.worker_steps['worker1'], 0)

    def test_barrier_count_compatibility(self):
        """Test that barrier_count increments for compatibility."""
        grad = {'param1': torch.randn(10)}

        initial_count = self.coordinator.barrier_count

        self.coordinator.submit_gradients('worker1', grad, worker_step=0)
        self.coordinator.get_and_clear_gradients()

        self.assertEqual(self.coordinator.barrier_count, initial_count + 1)


if __name__ == '__main__':
    unittest.main()
