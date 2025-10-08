import unittest

from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.gale_shapley import Scheduler

class TestGaleShapleyScheduler(unittest.TestCase):

    def setUp(self):
        """Set up a scenario with workers and tasks for testing."""
        # Create workers with different capabilities
        self.worker1 = WorkerNode(name="W1-Fast", memory=16, flops_per_second=200, network_bandwidth=1000)
        self.worker2 = WorkerNode(name="W2-Medium", memory=12, flops_per_second=100, network_bandwidth=500)
        self.worker3 = WorkerNode(name="W3-Slow", memory=24, flops_per_second=50, network_bandwidth=200)

        # Create tasks with different requirements
        self.taskA = TrainingTask(task_id="TaskA-Heavy", model_shard_size=10, data_size=4, required_flops=150)
        self.taskB = TrainingTask(task_id="TaskB-Medium", model_shard_size=8, data_size=2, required_flops=100)
        self.taskC = TrainingTask(task_id="TaskC-Light", model_shard_size=4, data_size=1, required_flops=50)

        self.workers = [self.worker1, self.worker2, self.worker3]
        self.tasks = [self.taskA, self.taskB, self.taskC]

    def test_matching_is_stable(self):
        """
        Tests that the Gale-Shapley algorithm produces a stable matching.
        """
        scheduler = Scheduler(self.tasks, self.workers)
        matches = scheduler.match()

        # In a stable matching, there should be no "blocking pair".
        # A blocking pair (task, worker) is one where both the task and the worker
        # would prefer each other over their current assignments.

        unmatched_tasks = [t for t in self.tasks if t.assigned_worker_id is None]

        for task in self.tasks:
            current_worker_id = task.assigned_worker_id

            # Find the rank of the current worker in the task's preferences.
            # If the task is unassigned, its rank is effectively infinite.
            current_worker_rank = float('inf')
            if current_worker_id:
                current_worker_rank = task.preferences.index(current_worker_id)

            # Now, iterate through all workers this task prefers over its current assignment.
            for i in range(current_worker_rank):
                preferred_worker_id = task.preferences[i]
                preferred_worker = scheduler.workers[preferred_worker_id]

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
        print("\nScheduler test passed: The resulting matching is stable.")
        print(f"Final matches: {matches}")

if __name__ == '__main__':
    unittest.main()