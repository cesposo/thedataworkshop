import time
import unittest
from unittest.mock import patch

from dist_llm_train.worker.node import WorkerNode
from tests.utils import DummyCommunicator


class TestWorkerNode(unittest.TestCase):
    @patch('dist_llm_train.worker.node.TaskExecutor')
    def test_receive_task_sets_state_and_spawns(self, mock_executor_cls):
        # Stub executor to no-op on execute_task
        instance = mock_executor_cls.return_value
        instance.execute_task.side_effect = lambda task: None

        controller_addr = "http://localhost:0"
        w = WorkerNode('w', 16, 100, 1000, 'localhost', 0, controller_addr, communicator=DummyCommunicator())
        try:
            tdict = {
                'id': 'tX', 'model_name': 'm', 'model_layer': 0,
                'model_shard_size': 0.5, 'data_size': 0.1, 'required_flops': 1,
            }
            w.receive_task(tdict)

            # Immediately after receive, worker reflects assigned state
            self.assertEqual(w.assigned_task_id, 'tX')
            self.assertEqual(w.status, 'busy')

            # Give the thread a brief moment to call execute_task
            time.sleep(0.05)
            instance.execute_task.assert_called()
        finally:
            try:
                w.communicator.stop_server()
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
