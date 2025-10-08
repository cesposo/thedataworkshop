import time
from typing import List, Dict

from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.gale_shapley import Scheduler

class MainController:
    """
    Manages the overall state of the distributed training system, including
    fault tolerance through a heartbeat mechanism.
    """
    def __init__(self, heartbeat_timeout: int = 30):
        self.workers: Dict[str, WorkerNode] = {}
        self.tasks: Dict[str, TrainingTask] = {} # Master list of all tasks
        self.pending_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        self.assignments: Dict[str, str] = {} # {worker_id: task_id}
        self.heartbeat_timeout = heartbeat_timeout

    def register_worker(self, worker: WorkerNode):
        """Adds a new worker to the system."""
        print(f"[Controller] Registering worker: {worker.id}")
        self.workers[worker.id] = worker

    def add_task(self, task: TrainingTask):
        """Adds a new training task to the system."""
        print(f"[Controller] Adding task: {task.id}")
        self.tasks[task.id] = task
        self.pending_tasks[task.id] = task

    def check_worker_health(self):
        """
        Checks all workers for heartbeats. If a worker has missed its
        heartbeat, it is marked as offline and its task is requeued.
        """
        print("\n[Controller] Checking worker health...")
        current_time = time.time()
        failed_workers = []

        for worker_id, worker in self.workers.items():
            if worker.status != 'offline' and (current_time - worker.last_heartbeat) > self.heartbeat_timeout:
                print(f"[Controller] Worker {worker_id} timed out. Marking as offline.")
                worker.status = 'offline'
                failed_workers.append(worker_id)

        # Requeue tasks from failed workers
        for worker_id in failed_workers:
            if worker_id in self.assignments:
                task_id = self.assignments.pop(worker_id)
                task = self.tasks.get(task_id)
                if task:
                    print(f"[Controller] Requeuing task {task_id} from failed worker {worker_id}.")
                    self.pending_tasks[task_id] = task
                    task.assigned_worker_id = None
                    worker = self.workers[worker_id]
                    worker.assigned_task_id = None


    def run_scheduling_cycle(self):
        """
        Runs one full cycle of scheduling and assignment.
        """
        print("\n[Controller] Starting new scheduling cycle...")

        available_workers = [w for w in self.workers.values() if w.status == 'available']
        tasks_to_schedule = list(self.pending_tasks.values())

        if not available_workers or not tasks_to_schedule:
            print("[Controller] No available workers or pending tasks. Skipping cycle.")
            return

        scheduler = Scheduler(tasks_to_schedule, available_workers)
        new_assignments = scheduler.match()
        self.assignments.update(new_assignments)

        print(f"[Controller] Scheduling complete. New assignments: {new_assignments}")

        # Update system state based on new assignments
        for worker_id, task_id in new_assignments.items():
            if worker_id in self.workers and task_id in self.pending_tasks:
                self.workers[worker_id].status = 'busy'
                self.workers[worker_id].assigned_task_id = task_id

                self.tasks[task_id].assigned_worker_id = worker_id
                del self.pending_tasks[task_id]

    def simulate_task_completion(self, worker_id: str):
        """
        Simulates a worker completing its assigned task.
        """
        if worker_id in self.assignments:
            task_id = self.assignments.pop(worker_id)
            worker = self.workers[worker_id]
            task = self.tasks.get(task_id)

            if task:
                print(f"[Controller] Worker {worker_id} completed Task {task_id}.")
                worker.status = 'available'
                worker.assigned_task_id = None
                self.completed_tasks.append(task)
            else:
                print(f"[Controller] Error: Task {task_id} not found in master list.")
        else:
            print(f"[Controller] Warning: Tried to complete task on worker {worker_id}, but it has no assignment.")

    def get_system_status(self):
        """Prints the current status of all workers and tasks."""
        print("\n--- System Status ---")
        print("Workers:")
        for worker in self.workers.values():
            print(f"  - {worker.id}: Status='{worker.status}', Assigned Task='{worker.assigned_task_id}'")
        print("Pending Tasks:")
        for task in self.pending_tasks.values():
            print(f"  - {task.id}")
        print(f"Completed Tasks: {len(self.completed_tasks)}")
        print("---------------------\n")