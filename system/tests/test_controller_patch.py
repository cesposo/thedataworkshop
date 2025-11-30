import unittest
from unittest.mock import MagicMock, patch

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.task.training_task import TrainingTask
from tests.utils import DummyCommunicator


class TestControllerSchedulerPatching(unittest.TestCase):
    @patch('dist_llm_train.communication.rpc.RPCCommunicator.send')
    @patch('dist_llm_train.controller.main_controller.GaleShapleyScheduler')
    def test_patch_affects_controller(self, mock_scheduler, _mock_send):
        controller = MainController(host='localhost', port=0, scheduler_name='gale-shapley', communicator=DummyCommunicator())
        try:
            # Prepare task and worker
            task = TrainingTask(task_id='t1', model_name='m', model_layer=0, model_shard_size=1, data_size=1, required_flops=1)
            controller.add_task(task)
            worker_id = 'w1'
            controller.register_worker(worker_id, 'http://localhost:0', MagicMock())

            # Mock scheduler to return desired matching
            mock_scheduler.return_value.schedule.return_value = {task.id: worker_id}

            # Run a scheduling cycle and assert assignment used mock
            controller.run_scheduling_cycle()
            self.assertEqual(controller.assignments.get(worker_id), task.id)
        finally:
            controller.communicator.stop_server()


if __name__ == '__main__':
    unittest.main()
