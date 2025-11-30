import unittest

from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.capability import CapabilityScheduler
from tests.utils import DummyCommunicator


class TestCapabilityScheduler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.worker1 = WorkerNode(
            name="W1-Fast",
            memory=16,
            flops_per_second=200,
            network_bandwidth=1000,
            host="localhost",
            port=0,
            controller_address="http://localhost:0",
            communicator=DummyCommunicator()
        )
        cls.worker2 = WorkerNode(
            name="W2-Medium",
            memory=12,
            flops_per_second=100,
            network_bandwidth=500,
            host="localhost",
            port=0,
            controller_address="http://localhost:0",
            communicator=DummyCommunicator()
        )
        cls.worker3 = WorkerNode(
            name="W3-BigMemSlow",
            memory=24,
            flops_per_second=50,
            network_bandwidth=200,
            host="localhost",
            port=0,
            controller_address="http://localhost:0",
            communicator=DummyCommunicator()
        )

        cls.workers = [cls.worker1, cls.worker2, cls.worker3]
        cls.taskA = TrainingTask("TaskA-Heavy", "m", 0, 10, 4, 150)
        cls.taskB = TrainingTask("TaskB-Medium", "m", 1, 8, 2, 100)
        cls.taskC = TrainingTask("TaskC-Light", "m", 2, 4, 1, 50)
        cls.tasks = [cls.taskA, cls.taskB, cls.taskC]

    @classmethod
    def tearDownClass(cls):
        for w in cls.workers:
            try:
                w.communicator.stop_server()
            except Exception:
                pass

    def _wrap(self, workers):
        return [
            {'info': w, 'status': 'available', 'address': f'http://localhost:{w.communicator.port}'}
            for w in workers
        ]

    def test_matches_non_empty(self):
        s = CapabilityScheduler(self.__class__.tasks, self._wrap(self.__class__.workers))
        m = s.schedule()
        self.assertGreaterEqual(len(m), 2)

    def test_no_workers(self):
        s = CapabilityScheduler(self.__class__.tasks, [])
        self.assertEqual(s.schedule(), {})

    def test_no_tasks(self):
        s = CapabilityScheduler([], self._wrap(self.__class__.workers))
        self.assertEqual(s.schedule(), {})

    def test_more_tasks_than_workers(self):
        s = CapabilityScheduler(self.__class__.tasks, self._wrap([self.__class__.worker1, self.__class__.worker2]))
        m = s.schedule()
        self.assertEqual(len(m), 2)

    def test_more_workers_than_tasks(self):
        s = CapabilityScheduler([self.__class__.taskA, self.__class__.taskB], self._wrap(self.__class__.workers))
        m = s.schedule()
        self.assertEqual(len(m), 2)
