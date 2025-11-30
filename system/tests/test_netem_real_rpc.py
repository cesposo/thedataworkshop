import os
import time
from unittest.mock import patch

import pytest

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask


@pytest.mark.skipif(os.getenv("RUN_REAL_RPC") not in ("1", "true", "True"), reason="Set RUN_REAL_RPC=1 to enable real RPC netem test")
def test_real_rpc_netem_good_profile():
    controller = None
    worker = None
    with patch("dist_llm_train.worker.task_executor.TaskExecutor.execute_task") as mock_exec:
        def fast(task):
            wid = next((wid for wid, tid in controller.assignments.items() if tid == task.id), None)
            if wid:
                controller.task_completed(wid, task.id)
        mock_exec.side_effect = fast

        try:
            controller = MainController(host="localhost", port=0, scheduler_name="priority", netem_profile="good")
        except PermissionError:
            pytest.skip("Socket permission denied for real RPC test")

        try:
            ctrl_addr = f"http://{controller.communicator.host}:{controller.communicator.port}"
            worker = WorkerNode("net-w1", 4, 50, 500, "localhost", 0, ctrl_addr, netem_profile="good")
            controller.register_worker(worker.id, f"http://{worker.communicator.host}:{worker.communicator.port}", worker)
            task = TrainingTask("net-task", "m", 0, 0.1, 0.1, 5)
            controller.add_task(task)

            controller.run_scheduling_cycle()
            time.sleep(0.1)

            assert task in controller.completed_tasks
            assert len(controller.assignments) == 0
        finally:
            try:
                controller.communicator.stop_server()
            except Exception:
                pass
            if worker:
                try:
                    worker.communicator.stop_server()
                except Exception:
                    pass
