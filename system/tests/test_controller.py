import unittest
import time
from unittest.mock import MagicMock, patch

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from tests.utils import DummyCommunicator

class TestMainController(unittest.TestCase):

    def setUp(self):
        """Set up a controller for testing."""
        self.controller = MainController(host='localhost', port=0, communicator=DummyCommunicator())

    def tearDown(self):
        """Clean up the controller's RPC server."""
        self.controller.communicator.stop_server()

    def test_worker_registration(self):
        """Test that a worker can register with the controller."""
        worker_info = {
            'id': 'worker-1',
            'address': f'http://localhost:{self.controller.communicator.port + 1}',
            'info': MagicMock()
        }
        self.controller.register_worker(worker_info['id'], worker_info['address'], worker_info['info'])
        self.assertIn(worker_info['id'], self.controller.workers)
        self.assertEqual(self.controller.workers[worker_info['id']]['address'], worker_info['address'])

    def test_worker_health_check(self):
        """Test that the controller correctly identifies healthy and failed workers."""
        worker_info = {
            'id': 'worker-1',
            'address': f'http://localhost:{self.controller.communicator.port + 1}',
            'info': MagicMock()
        }
        self.controller.register_worker(worker_info['id'], worker_info['address'], worker_info['info'])

        # Test healthy worker
        self.controller.workers[worker_info['id']]['last_heartbeat'] = time.time()
        self.controller.check_worker_health()
        self.assertEqual(self.controller.workers[worker_info['id']]['status'], 'available')

        # Test failed worker
        self.controller.workers[worker_info['id']]['last_heartbeat'] = time.time() - 1000
        self.controller.check_worker_health()
        self.assertEqual(self.controller.workers[worker_info['id']]['status'], 'offline')

    @patch('dist_llm_train.communication.rpc.RPCCommunicator.send')
    @patch('dist_llm_train.controller.main_controller.GaleShapleyScheduler')
    def test_task_management(self, mock_scheduler, mock_send):
        """Test that the controller can add, assign, and complete tasks."""
        # Setup
        task = TrainingTask(task_id='task-1', model_name='test-model', model_layer=0, model_shard_size=1, data_size=1, required_flops=1)
        worker_id = 'worker-1'
        self.controller.add_task(task)
        self.controller.register_worker(worker_id, f'http://localhost:{self.controller.communicator.port + 1}', MagicMock())

        # Mock scheduler to return a specific assignment
        mock_scheduler.return_value.schedule.return_value = {task.id: worker_id}

        # Run scheduling cycle
        self.controller.run_scheduling_cycle()

        # Assert task is assigned
        self.assertNotIn(task.id, self.controller.pending_tasks)
        self.assertEqual(self.controller.assignments[worker_id], task.id)

        # Mark task as complete
        self.controller.task_completed(worker_id, task.id)
        self.assertNotIn(worker_id, self.controller.assignments)
        self.assertIn(task, self.controller.completed_tasks)

    @patch('dist_llm_train.communication.rpc.RPCCommunicator.send')
    @patch('dist_llm_train.controller.main_controller.GaleShapleyScheduler')
    def test_fault_tolerance(self, mock_scheduler, mock_send):
        """Test that the controller requeues tasks from failed workers."""
        # Setup
        task = TrainingTask(task_id='task-1', model_name='test-model', model_layer=0, model_shard_size=1, data_size=1, required_flops=1)
        worker_id = 'worker-1'
        self.controller.add_task(task)
        self.controller.register_worker(worker_id, f'http://localhost:{self.controller.communicator.port + 1}', MagicMock())

        # Mock scheduler to return a specific assignment
        mock_scheduler.return_value.schedule.return_value = {task.id: worker_id}

        # Run scheduling cycle to assign the task
        self.controller.run_scheduling_cycle()
        self.assertEqual(self.controller.assignments.get(worker_id), task.id)

        # Mark the worker as failed
        self.controller.workers[worker_id]['last_heartbeat'] = time.time() - 1000
        self.controller.check_worker_health()

        # Check that the task is back in the pending queue
        self.assertNotIn(worker_id, self.controller.assignments)
        self.assertIn(task.id, self.controller.pending_tasks)

if __name__ == '__main__':
    unittest.main()
