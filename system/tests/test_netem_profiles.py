import time
import unittest
from unittest.mock import patch

from dist_llm_train.communication.chaos import ChaoticCommunicator, load_profile
from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from tests.utils import DummyCommunicator


class TestChaoticCommunicator(unittest.TestCase):
    def test_latency_and_stats(self):
        server = DummyCommunicator(host="localhost", port=8300)
        client = DummyCommunicator(host="localhost", port=8301)
        chaos = ChaoticCommunicator(client, load_profile({"base_rtt_ms": 10, "jitter_ms": 0, "loss_pct": 0}), seed=42)

        server.register_function(lambda x: x, "echo")
        res = chaos.send(server.address, {"method": "echo", "params": ["hi"]})
        self.assertEqual(res, "hi")
        self.assertEqual(chaos.stats["sent"], 1)
        self.assertEqual(chaos.stats["dropped"], 0)
        self.assertGreaterEqual(len(chaos.stats["latency_ms"]), 1)


class TestWANIntegration(unittest.TestCase):
    @patch("dist_llm_train.worker.task_executor.TaskExecutor.execute_task")
    def test_tasks_complete_with_latency_and_brownout(self, mock_exec):
        # Fast completion: mark task completed on controller once scheduled
        def fast(task):
            # find controller via closure
            wid = next((w_id for w_id, t_id in controller.assignments.items() if t_id == task.id), None)
            if wid:
                controller.task_completed(wid, task.id)
        mock_exec.side_effect = fast

        controller_base = DummyCommunicator(host="localhost", port=8400)
        controller_comm = ChaoticCommunicator(controller_base, load_profile({"base_rtt_ms": 40, "jitter_ms": 10, "loss_pct": 0.0}), seed=1)
        controller = MainController(host="localhost", port=0, scheduler_name="priority", communicator=controller_comm)

        try:
            ctrl_addr = f"http://{controller.communicator.host}:{controller.communicator.port}"
            w1_comm = ChaoticCommunicator(DummyCommunicator(host="localhost", port=8401), load_profile("good"), seed=2)
            w2_comm = ChaoticCommunicator(DummyCommunicator(host="localhost", port=8402), load_profile("good"), seed=3)
            w1 = WorkerNode("wan-w1", 8, 50, 500, "localhost", 0, ctrl_addr, communicator=w1_comm)
            w2 = WorkerNode("wan-w2", 8, 50, 500, "localhost", 0, ctrl_addr, communicator=w2_comm)

            controller.register_worker(w1.id, f"http://{w1.communicator.host}:{w1.communicator.port}", w1)
            controller.register_worker(w2.id, f"http://{w2.communicator.host}:{w2.communicator.port}", w2)

            task = TrainingTask("wan-task-0", "m", 0, 0.5, 0.1, 10)
            controller.add_task(task)

            controller.run_scheduling_cycle()
            # allow worker thread to execute
            time.sleep(0.05)

            self.assertIn(task, controller.completed_tasks)
            self.assertEqual(len(controller.assignments), 0)
            self.assertEqual(len(controller.pending_tasks), 0)
            # chaos stats should have recorded latency
            self.assertGreater(len(controller_comm.stats["latency_ms"]), 0)
        finally:
            try:
                controller.communicator.stop_server()
            except Exception:
                pass
            try:
                w1.communicator.stop_server()
                w2.communicator.stop_server()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
