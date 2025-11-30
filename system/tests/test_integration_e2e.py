import time
import unittest
from unittest.mock import patch

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from tests.utils import DummyCommunicator


class TestIntegrationE2E(unittest.TestCase):
    @patch('dist_llm_train.worker.task_executor.TaskExecutor.execute_task')
    def test_controller_worker_roundtrip(self, mock_exec):
        # Patch execute_task to immediately notify completion directly on controller
        # Note: This avoids heavy training while still exercising controller->worker RPC

        controller = MainController(host='localhost', port=0, scheduler_name='priority', communicator=DummyCommunicator())

        try:
            # Create two workers on auto ports and register with controller
            controller_addr = f"http://{controller.communicator.host}:{controller.communicator.port}"
            w1 = WorkerNode('w1', 16, 100, 1000, 'localhost', 0, controller_addr, communicator=DummyCommunicator())
            w2 = WorkerNode('w2', 24, 150, 1000, 'localhost', 0, controller_addr, communicator=DummyCommunicator())

            controller.register_worker(w1.id, f"http://{w1.communicator.host}:{w1.communicator.port}", w1)
            controller.register_worker(w2.id, f"http://{w2.communicator.host}:{w2.communicator.port}", w2)

            # Add a simple task
            t = TrainingTask('t0', 'm', 0, 0.5, 0.1, 10)
            controller.add_task(t)

            # Define fast execute that marks completion after scheduling decides assignment
            def fast(task):
                worker_id = next((wid for wid, tid in controller.assignments.items() if tid == task.id), None)
                if worker_id:
                    controller.task_completed(worker_id, task.id)
            mock_exec.side_effect = fast

            # Run scheduling cycle (controller->worker RPC), then wait briefly
            controller.run_scheduling_cycle()
            time.sleep(0.1)

            # Assert the task completed and assignments cleared
            self.assertIn(t, controller.completed_tasks)
            self.assertEqual(len(controller.assignments), 0)
            self.assertEqual(len(controller.pending_tasks), 0)
        finally:
            try: controller.communicator.stop_server()
            except Exception: pass
            try: w1.communicator.stop_server(); w2.communicator.stop_server()
            except Exception: pass


if __name__ == '__main__':
    unittest.main()
