import time
from typing import List, Dict, Type

from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.base import BaseScheduler
from dist_llm_train.scheduler.gale_shapley import GaleShapleyScheduler
from dist_llm_train.communication.rpc import RPCCommunicator

class MainController:
    """
    Manages the overall state of the distributed training system, including
    fault tolerance through a heartbeat mechanism.
    """
    def __init__(self, host: str, port: int, scheduler_class: Type[BaseScheduler] = GaleShapleyScheduler, heartbeat_timeout: int = 30):
        self.workers: Dict[str, Dict] = {}
        self.tasks: Dict[str, TrainingTask] = {} # Master list of all tasks
        self.pending_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        self.assignments: Dict[str, str] = {} # {worker_id: task_id}
        self.scheduler_class = scheduler_class
        self.heartbeat_timeout = heartbeat_timeout

        self.communicator = RPCCommunicator(host, port)
        self.communicator.register_function(self.heartbeat, 'heartbeat')
        self.communicator.register_function(self.task_completed, 'task_completed')
        self.communicator.start_server()

    def register_worker(self, worker_id: str, worker_address: str, worker_info: Dict):
        """Adds a new worker to the system."""
        print(f"[Controller] Registering worker: {worker_id}")
        self.workers[worker_id] = {
            'address': worker_address,
            'info': worker_info,
            'last_heartbeat': time.time(),
            'status': 'available'
        }

    def heartbeat(self, worker_id: str, status: str):
        """Receives a heartbeat from a worker."""
        if worker_id in self.workers:
            self.workers[worker_id]['last_heartbeat'] = time.time()
            self.workers[worker_id]['status'] = status
        else:
            print(f"[Controller] Received heartbeat from unknown worker: {worker_id}")

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
            if worker['status'] != 'offline' and (current_time - worker['last_heartbeat']) > self.heartbeat_timeout:
                print(f"[Controller] Worker {worker_id} timed out. Marking as offline.")
                worker['status'] = 'offline'
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

    def run_scheduling_cycle(self):
        """
        Runs one full cycle of scheduling and assignment.
        """
        print("\n[Controller] Starting new scheduling cycle...")

        available_workers = [w for w in self.workers.values() if w['status'] == 'available']
        tasks_to_schedule = list(self.pending_tasks.values())

        if not available_workers or not tasks_to_schedule:
            print("[Controller] No available workers or pending tasks. Skipping cycle.")
            return

        scheduler = self.scheduler_class(tasks_to_schedule, available_workers)
        new_assignments = scheduler.schedule()
        self.assignments.update(new_assignments)

        print(f"[Controller] Scheduling complete. New assignments: {new_assignments}")

        # Update system state based on new assignments
        for worker_id, task_id in new_assignments.items():
            if worker_id in self.workers and task_id in self.pending_tasks:
                worker = self.workers[worker_id]
                task = self.tasks[task_id]
                try:
                    # Convert task to dict for RPC serialization
                    task_dict = {
                        'id': task.id,
                        'model_name': task.model_name,
                        'model_layer': task.model_layer,
                        'model_shard_size': task.model_shard_size,
                        'data_size': task.data_size,
                        'required_flops': task.required_flops,
                        'optimizer_state_size': task.optimizer_state_size,
                        'gradient_size': task.gradient_size,
                        'activation_size': task.activation_size,
                        'model_config': task.model_config,
                        'training_config': task.training_config
                    }
                    self.communicator.send(worker['address'], {
                        'method': 'receive_task',
                        'params': [task_dict]
                    })
                    worker['status'] = 'busy'
                    task.assigned_worker_id = worker_id
                    del self.pending_tasks[task_id]
                except Exception as e:
                    print(f"[Controller] Failed to assign task to worker {worker_id}: {e}")

    def task_completed(self, worker_id: str, task_id: str):
        """Receives a notification from a worker that a task is completed."""
        print(f"[Controller] Worker {worker_id} completed Task {task_id}.")
        if worker_id in self.assignments and self.assignments[worker_id] == task_id:
            del self.assignments[worker_id]
            task = self.tasks.get(task_id)
            if task:
                self.completed_tasks.append(task)
            if worker_id in self.workers:
                self.workers[worker_id]['status'] = 'available'
        else:
            print(f"[Controller] Warning: Received completion for unassigned task {task_id} from worker {worker_id}.")

    def get_system_status(self):
        """Prints the current status of all workers and tasks."""
        print("\n--- System Status ---")
        print("Workers:")
        for worker_id, worker in self.workers.items():
            print(f"  - {worker_id}: Status='{worker['status']}', Address='{worker['address']}'")
        print("Pending Tasks:")
        for task in self.pending_tasks.values():
            print(f"  - {task.id}")
        print(f"Completed Tasks: {len(self.completed_tasks)}")
        print("---------------------\n")