import unittest

from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.gale_shapley import GaleShapleyScheduler
from tests.utils import DummyCommunicator

class TestGaleShapleyScheduler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a scenario with workers and tasks for testing."""
        # Create workers with different capabilities
        cls.worker1 = WorkerNode(
            name="W1-Fast",
            memory=16,
            flops_per_second=200,
            network_bandwidth=1000,
            host="localhost",
            port=9001,
            controller_address="http://localhost:9000",
            communicator=DummyCommunicator(host="localhost", port=9001)
        )
        cls.worker2 = WorkerNode(
            name="W2-Medium",
            memory=12,
            flops_per_second=100,
            network_bandwidth=500,
            host="localhost",
            port=9002,
            controller_address="http://localhost:9000",
            communicator=DummyCommunicator(host="localhost", port=9002)
        )
        cls.worker3 = WorkerNode(
            name="W3-Slow",
            memory=24,
            flops_per_second=50,
            network_bandwidth=200,
            host="localhost",
            port=9003,
            controller_address="http://localhost:9000",
            communicator=DummyCommunicator(host="localhost", port=9003)
        )

        # Create tasks with different requirements
        cls.taskA = TrainingTask(
            task_id="TaskA-Heavy",
            model_name="test-model",
            model_layer=0,
            model_shard_size=10,
            data_size=4,
            required_flops=150
        )
        cls.taskB = TrainingTask(
            task_id="TaskB-Medium",
            model_name="test-model",
            model_layer=1,
            model_shard_size=8,
            data_size=2,
            required_flops=100
        )
        cls.taskC = TrainingTask(
            task_id="TaskC-Light",
            model_name="test-model",
            model_layer=2,
            model_shard_size=4,
            data_size=1,
            required_flops=50
        )

        cls.workers = [cls.worker1, cls.worker2, cls.worker3]
        cls.tasks = [cls.taskA, cls.taskB, cls.taskC]

    @classmethod
    def tearDownClass(cls):
        """Clean up worker RPC servers."""
        for worker in cls.workers:
            try:
                worker.communicator.stop_server()
            except:
                pass  # Server might not have started

    def test_schedule_correctness(self):
        """
        Tests that the scheduler produces a correct matching, where each worker is assigned at most one task.
        """
        workers_wrapped = [
            {'info': worker, 'status': 'available', 'address': f'http://localhost:{9001+i}'}
            for i, worker in enumerate(self.__class__.workers)
        ]

        scheduler = GaleShapleyScheduler(self.__class__.tasks, workers_wrapped)
        matches = scheduler.schedule()

        worker_assignments = {}
        for task_id, worker_id in matches.items():
            if worker_id in worker_assignments:
                self.fail(f"Worker {worker_id} is assigned multiple tasks: {worker_assignments[worker_id]} and {task_id}")
            worker_assignments[worker_id] = task_id

    def test_schedule_completeness(self):
        """
        Tests that the scheduler assigns as many tasks as possible.
        """
        workers_wrapped = [
            {'info': worker, 'status': 'available', 'address': f'http://localhost:{9001+i}'}
            for i, worker in enumerate(self.__class__.workers)
        ]

        scheduler = GaleShapleyScheduler(self.__class__.tasks, workers_wrapped)
        matches = scheduler.schedule()

        # In this setup, we have 3 workers and 3 tasks, so all tasks should be matched.
        self.assertEqual(len(matches), len(self.__class__.tasks))

    def test_no_workers(self):
        """
        Tests that the scheduler returns an empty matching when there are no workers.
        """
        scheduler = GaleShapleyScheduler(self.__class__.tasks, [])
        matches = scheduler.schedule()
        self.assertEqual(len(matches), 0)

    def test_no_tasks(self):
        """
        Tests that the scheduler returns an empty matching when there are no tasks.
        """
        workers_wrapped = [
            {'info': worker, 'status': 'available', 'address': f'http://localhost:{9001+i}'}
            for i, worker in enumerate(self.__class__.workers)
        ]
        scheduler = GaleShapleyScheduler([], workers_wrapped)
        matches = scheduler.schedule()
        self.assertEqual(len(matches), 0)

    def test_more_tasks_than_workers(self):
        """
        Tests that the scheduler assigns tasks to all workers when there are more tasks than workers.
        """
        workers_wrapped = [
            {'info': self.__class__.worker1, 'status': 'available', 'address': 'http://localhost:9001'},
            {'info': self.__class__.worker2, 'status': 'available', 'address': 'http://localhost:9002'}
        ]
        scheduler = GaleShapleyScheduler(self.__class__.tasks, workers_wrapped)
        matches = scheduler.schedule()
        self.assertEqual(len(matches), len(workers_wrapped))

    def test_more_workers_than_tasks(self):
        """
        Tests that the scheduler assigns all tasks when there are more workers than tasks.
        """
        tasks = [self.__class__.taskA, self.__class__.taskB]
        workers_wrapped = [
            {'info': worker, 'status': 'available', 'address': f'http://localhost:{9001+i}'}
            for i, worker in enumerate(self.__class__.workers)
        ]
        scheduler = GaleShapleyScheduler(tasks, workers_wrapped)
        matches = scheduler.schedule()
        self.assertEqual(len(matches), len(tasks))


    def test_matching_is_stable(self):
        """
        Tests that the Gale-Shapley algorithm produces a stable matching.
        """
        # Wrap workers in the format expected by the scheduler
        workers_wrapped = [
            {'info': worker, 'status': 'available', 'address': f'http://localhost:{9001+i}'}
            for i, worker in enumerate(self.__class__.workers)
        ]

        scheduler = GaleShapleyScheduler(self.__class__.tasks, workers_wrapped)
        matches = scheduler.schedule()

        # In a stable matching, there should be no "blocking pair".
        # A blocking pair (task, worker) is one where both the task and the worker
        # would prefer each other over their current assignments.

        unmatched_tasks = [t for t in self.__class__.tasks if t.assigned_worker_id is None]

        for task in self.__class__.tasks:
            current_worker_id = task.assigned_worker_id

            # Find the rank of the current worker in the task's preferences.
            # If the task is unassigned, its rank is effectively infinite.
            current_worker_rank = float('inf')
            if current_worker_id:
                current_worker_rank = task.preferences.index(current_worker_id)

            # Now, iterate through all workers this task prefers over its current assignment.
            for i in range(int(current_worker_rank)):
                preferred_worker_id = task.preferences[i]
                preferred_worker = scheduler.workers[preferred_worker_id]['info']

                # This is a worker the task prefers. Now, does this worker also prefer this task?
                other_task_id = preferred_worker.assigned_task_id

                if other_task_id is None:
                    # The preferred worker is free. This is an unstable pair.
                    self.fail(f"Unstable pair found: Task {task.id} is matched with {current_worker_id} but prefers free worker {preferred_worker.id}.")

                other_task_rank = preferred_worker.preferences.index(other_task_id)
                current_task_rank = preferred_worker.preferences.index(task.id)

                if current_task_rank < other_task_rank:
                    # The preferred worker also prefers this task. This is an unstable pair.
                    self.fail(f"Unstable pair found: Task {task.id} and Worker {preferred_worker.id} prefer each other over their current matches.")

        # If we get here, no unstable pairs were found.

if __name__ == '__main__':
    unittest.main()
