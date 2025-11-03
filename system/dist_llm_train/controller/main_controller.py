import time
import threading
import os
import logging
from typing import List, Dict, Type

import torch

from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.base import BaseScheduler
from dist_llm_train.scheduler.gale_shapley import GaleShapleyScheduler
from dist_llm_train.scheduler.priority import PriorityScheduler
from dist_llm_train.scheduler.capability import CapabilityScheduler
from dist_llm_train.communication.rpc import RPCCommunicator
from dist_llm_train.sync.parameter_server import ParameterServer, SimpleSyncCoordinator

SCHEDULERS = {
    'gale-shapley': GaleShapleyScheduler,
    'priority': PriorityScheduler,
    'capability': CapabilityScheduler,
}

logger = logging.getLogger("dist_llm_train.controller")


class MainController:
    """
    Manages the overall state of the distributed training system, including
    fault tolerance through a heartbeat mechanism.
    """
    def __init__(self, host: str, port: int, scheduler_name: str = 'gale-shapley', heartbeat_timeout: int = 30, state_db_path: str = None, telemetry_alpha: float = None):
        self.workers: Dict[str, Dict] = {}
        self.tasks: Dict[str, TrainingTask] = {} # Master list of all tasks
        self.pending_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        self.assignments: Dict[str, str] = {} # {worker_id: task_id}
        # Store scheduler name and resolve class at runtime to honor test patching
        self.scheduler_name = scheduler_name
        if self.scheduler_name not in SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        self.heartbeat_timeout = heartbeat_timeout

        self.communicator = RPCCommunicator(host, port)
        # Core RPCs
        self.communicator.register_function(self.heartbeat, 'heartbeat')
        self.communicator.register_function(self.task_completed, 'task_completed')
        self.communicator.register_function(self.task_failed, 'task_failed')
        self.communicator.register_function(self.report_metrics, 'report_metrics')
        self.communicator.register_function(self.get_status, 'get_status')
        # Telemetry
        self.worker_telemetry: Dict[str, Dict] = {}
        self._telemetry_rollups: Dict[str, Dict[str, float]] = {}
        # EWMA smoothing parameter (can be overridden by env TELEMETRY_ALPHA)
        if telemetry_alpha is None:
            try:
                telemetry_alpha = float(os.getenv('TELEMETRY_ALPHA', '0.2'))
            except Exception:
                telemetry_alpha = 0.2
        self._telemetry_alpha = max(0.0, min(1.0, telemetry_alpha))
        self.communicator.register_function(self.report_telemetry, 'report_telemetry')
        # Sync/PS RPCs
        self.param_server = ParameterServer()
        self.sync_coordinator = SimpleSyncCoordinator(num_workers=0)
        self._barrier_lock = threading.Lock()
        self._last_update_barrier = 0
        self._training_configured = False
        self.communicator.register_function(self.configure_training_sync, 'configure_training_sync')
        self.communicator.register_function(self.ps_initialize_if_empty, 'ps_initialize_if_empty')
        self.communicator.register_function(self.ps_get_parameters, 'ps_get_parameters')
        self.communicator.register_function(self.ps_sync_step, 'ps_sync_step')
        self.communicator.register_function(self.ps_save, 'ps_save')
        self.communicator.register_function(self.ps_load, 'ps_load')
        self.communicator.register_function(self.ps_set_aggregation, 'ps_set_aggregation')
        self.communicator.start_server()

        # Metrics store
        self.task_metrics: Dict[str, Dict] = {}
        # Optional persistence
        self._state_store = None
        if state_db_path:
            try:
                from dist_llm_train.persistence.state_store import StateStore
                self._state_store = StateStore(state_db_path)
            except Exception as e:
                logger.error(f"Failed to initialize state store: {e}")

    def _persist_state_if_configured(self):
        if self._state_store:
            try:
                self._state_store.save_system_status(self.get_system_status())
            except Exception as e:
                logger.error(f"Failed to persist controller state: {e}")

    def register_worker(self, worker_id: str, worker_address: str, worker_info: Dict):
        """Adds a new worker to the system."""
        logger.info(f"Registering worker: {worker_id}")
        self.workers[worker_id] = {
            'address': worker_address,
            'info': worker_info,
            'last_heartbeat': time.time(),
            'status': 'available'
        }
        self._persist_state_if_configured()

    def heartbeat(self, worker_id: str, status: str):
        """Receives a heartbeat from a worker."""
        if worker_id in self.workers:
            self.workers[worker_id]['last_heartbeat'] = time.time()
            self.workers[worker_id]['status'] = status
        else:
            logger.warning(f"Received heartbeat from unknown worker: {worker_id}")

    def add_task(self, task: TrainingTask):
        """Adds a new training task to the system."""
        logger.info(f"Adding task: {task.id}")
        self.tasks[task.id] = task
        self.pending_tasks[task.id] = task
        self._persist_state_if_configured()

    def check_worker_health(self):
        """
        Checks all workers for heartbeats. If a worker has missed its
        heartbeat, it is marked as offline and its task is requeued.
        """
        logger.info("Checking worker health…")
        current_time = time.time()
        failed_workers = []

        for worker_id, worker in self.workers.items():
            if worker['status'] != 'offline' and (current_time - worker['last_heartbeat']) > self.heartbeat_timeout:
                logger.warning(f"Worker {worker_id} timed out. Marking as offline.")
                worker['status'] = 'offline'
                failed_workers.append(worker_id)

        # Requeue tasks from failed workers
        for worker_id in failed_workers:
            if worker_id in self.assignments:
                task_id = self.assignments.pop(worker_id)
                task = self.tasks.get(task_id)
                if task:
                    logger.info(f"Requeuing task {task_id} from failed worker {worker_id}.")
                    self.pending_tasks[task_id] = task
                    task.assigned_worker_id = None
        self._persist_state_if_configured()

    def run_scheduling_cycle(self):
        """
        Runs one full cycle of scheduling and assignment.
        """
        logger.debug("Starting new scheduling cycle…")

        # Proactive health check each cycle
        try:
            self.check_worker_health()
        except Exception as e:
            logger.error(f"Health check failed: {e}")

        available_workers = [w for w in self.workers.values() if w['status'] == 'available']
        tasks_to_schedule = list(self.pending_tasks.values())

        if not available_workers or not tasks_to_schedule:
            logger.info("No available workers or pending tasks. Skipping cycle.")
            return

        # Resolve scheduler class at runtime (see _resolve_scheduler_class) so unit tests can patch the symbol
        scheduler_class = self._resolve_scheduler_class()

        scheduler = scheduler_class(tasks_to_schedule, available_workers)
        # Prefer structured scheduler input if supported
        try:
            from dist_llm_train.scheduler.interfaces import (
                TelemetrySnapshot,
                SchedulerWorkerInfo,
                SchedulerTaskInfo,
                SchedulerInput,
            )
            if getattr(scheduler_class, 'SUPPORTS_STRUCTURED_INPUT', False) is True:
                # Build structured input
                workers_struct = []
                for wid, w in self.workers.items():
                    if w['status'] != 'available':
                        continue
                    tel = self.worker_telemetry.get(wid)
                    tel_obj = None
                    if tel:
                        # include EWMA averages if available
                        roll = self._telemetry_rollups.get(wid, {})
                        avg_tps = roll.get('ewma_tps')
                        avg_step = roll.get('ewma_step')
                        tel_obj = TelemetrySnapshot(
                            worker_id=wid,
                            tokens_per_sec=tel.get('tokens_per_sec'),
                            step_time_s=tel.get('step_time_s'),
                            avg_tokens_per_sec=avg_tps,
                            avg_step_time_s=avg_step,
                            epoch=tel.get('epoch'),
                            batch=tel.get('batch'),
                            mode=tel.get('mode'),
                            ts=tel.get('ts'),
                        )
                    info = w['info']
                    workers_struct.append(SchedulerWorkerInfo(
                        id=getattr(info, 'id', wid),
                        memory=float(getattr(info, 'memory', 0.0)),
                        flops_per_second=int(getattr(info, 'flops_per_second', 0)),
                        network_bandwidth=int(getattr(info, 'network_bandwidth', 0)),
                        status=w['status'],
                        address=w['address'],
                        telemetry=tel_obj,
                    ))
                tasks_struct = [
                    SchedulerTaskInfo(
                        id=t.id,
                        total_memory_req=float(t.get_total_memory_req()),
                        required_flops=int(t.required_flops),
                        priority=int(getattr(t, 'priority', 0)),
                    )
                    for t in tasks_to_schedule
                ]
                sched_input = SchedulerInput(tasks=tasks_struct, workers=workers_struct)
                output = scheduler.schedule_from_input(sched_input)
                task_to_worker = output.assignments
            else:
                # Fallback to legacy API
                task_to_worker = scheduler.schedule()
        except Exception as e:
            logger.error(f"Structured scheduling failed, falling back to legacy: {e}")
            task_to_worker = scheduler.schedule()
        # Convert to controller's internal mapping {worker_id: task_id}
        for t_id, w_id in task_to_worker.items():
            self.assignments[w_id] = t_id

        logger.info(f"Scheduling complete. New assignments: {task_to_worker}")

        # Update system state based on new assignments
        for task_id, worker_id in task_to_worker.items():
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
                        'training_config': task.training_config,
                        'priority': task.priority
                    }
                    self.communicator.send(worker['address'], {
                        'method': 'receive_task',
                        'params': [task_dict]
                    })
                    worker['status'] = 'busy'
                    task.assigned_worker_id = worker_id
                    del self.pending_tasks[task_id]
                except Exception as e:
                    logger.error(f"Failed to assign task to worker {worker_id}: {e}")
        self._persist_state_if_configured()

    def _resolve_scheduler_class(self):
        """Returns the scheduler class for the configured scheduler name.

        Deliberately resolves at call time to ensure test patches on
        GaleShapleyScheduler/PriorityScheduler take effect.
        """
        if self.scheduler_name == 'gale-shapley':
            return GaleShapleyScheduler
        if self.scheduler_name == 'priority':
            return PriorityScheduler
        if self.scheduler_name == 'capability':
            return CapabilityScheduler
        raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    def task_completed(self, worker_id: str, task_id: str):
        """Receives a notification from a worker that a task is completed."""
        logger.info(f"Worker {worker_id} completed Task {task_id}.")
        if worker_id in self.assignments and self.assignments[worker_id] == task_id:
            del self.assignments[worker_id]
            task = self.tasks.get(task_id)
            if task:
                self.completed_tasks.append(task)
            if worker_id in self.workers:
                self.workers[worker_id]['status'] = 'available'
        else:
            logger.warning(f"Received completion for unassigned task {task_id} from worker {worker_id}.")
        self._persist_state_if_configured()

    def task_failed(self, worker_id: str, task_id: str, error_msg: str = ""):
        """Called by workers to indicate task failure; requeues the task."""
        logger.error(f"Worker {worker_id} reported failure for Task {task_id}: {error_msg}")
        if worker_id in self.assignments and self.assignments[worker_id] == task_id:
            del self.assignments[worker_id]
        task = self.tasks.get(task_id)
        if task:
            self.pending_tasks[task_id] = task
            task.assigned_worker_id = None
        if worker_id in self.workers:
            self.workers[worker_id]['status'] = 'available'
        self._persist_state_if_configured()

    def get_system_status(self) -> Dict:
        """Returns the current status of all workers and tasks (machine-readable)."""
        status = {
            'workers': {
                worker_id: {
                    'status': w['status'],
                    'address': w['address'],
                    'last_heartbeat': w['last_heartbeat']
                }
                for worker_id, w in self.workers.items()
            },
            'workers_detail': {
                worker_id: {
                    'id': getattr(w.get('info'), 'id', worker_id) if isinstance(w, dict) else worker_id,
                    'memory': float(getattr(w.get('info'), 'memory', 0.0)) if isinstance(w, dict) else 0.0,
                    'flops_per_second': int(getattr(w.get('info'), 'flops_per_second', 0)) if isinstance(w, dict) else 0,
                    'network_bandwidth': int(getattr(w.get('info'), 'network_bandwidth', 0)) if isinstance(w, dict) else 0,
                    'status': w['status'],
                    'address': w['address'],
                }
                for worker_id, w in self.workers.items()
            },
            'pending_tasks': list(self.pending_tasks.keys()),
            'completed_tasks': [t.id for t in self.completed_tasks],
            'metrics': self.task_metrics,
            'telemetry': self.worker_telemetry,
            'telemetry_rollups': self._telemetry_rollups,
            'assignments': self.assignments,
            'tasks_detail': {
                t.id: {
                    'id': t.id,
                    'model_name': t.model_name,
                    'model_layer': t.model_layer,
                    'model_shard_size': t.model_shard_size,
                    'data_size': t.data_size,
                    'required_flops': t.required_flops,
                    'optimizer_state_size': t.optimizer_state_size,
                    'gradient_size': t.gradient_size,
                    'activation_size': t.activation_size,
                    'model_config': t.model_config,
                    'training_config': t.training_config,
                    'priority': getattr(t, 'priority', 0)
                }
                for t in self.tasks.values()
            }
        }
        logger.debug(
            "System status | workers=%d pending=%d completed=%d",
            len(status['workers']), len(status['pending_tasks']), len(status['completed_tasks'])
        )
        return status

    # RPC-friendly alias
    def get_status(self) -> Dict:
        return self.get_system_status()

    # Metrics reporting from workers
    def report_metrics(self, worker_id: str, task_id: str, metrics: Dict):
        self.task_metrics[task_id] = {
            'worker_id': worker_id,
            **(metrics or {})
        }
        logger.info(f"Metrics reported for task {task_id}: {metrics}")
        self._persist_state_if_configured()

    # Worker runtime telemetry (e.g., throughput)
    def report_telemetry(self, worker_id: str, telemetry: Dict):
        t = telemetry or {}
        now = time.time()
        self.worker_telemetry[worker_id] = {**t, 'ts': now}
        # EWMA aggregates
        roll = self._telemetry_rollups.get(worker_id, {'ewma_tps': None, 'ewma_step': None, 'alpha': self._telemetry_alpha})
        alpha = roll.get('alpha', self._telemetry_alpha)
        if 'tokens_per_sec' in t and t['tokens_per_sec'] is not None:
            v = float(t['tokens_per_sec'])
            prev = roll.get('ewma_tps')
            roll['ewma_tps'] = v if prev is None else (alpha * v + (1 - alpha) * prev)
        if 'step_time_s' in t and t['step_time_s'] is not None:
            v = float(t['step_time_s'])
            prevs = roll.get('ewma_step')
            roll['ewma_step'] = v if prevs is None else (alpha * v + (1 - alpha) * prevs)
        roll['alpha'] = alpha
        roll['ts'] = now
        self._telemetry_rollups[worker_id] = roll
        logger.debug(f"Telemetry from {worker_id}: {telemetry}")
        self._persist_state_if_configured()

    # ---- Parameter server + sync RPCs ----
    def configure_training_sync(self, num_workers: int, window_size: int = None, max_wait_s: float = None) -> bool:
        """Configure the number of workers participating and window for sync steps."""
        self.sync_coordinator = SimpleSyncCoordinator(num_workers=num_workers, window_size=window_size, max_wait_s=max_wait_s)
        self._training_configured = True
        logger.info(f"Training sync configured for {num_workers} workers")
        return True

    def ps_initialize_if_empty(self, parameters_serializable: Dict[str, list]) -> bool:
        """Initialize PS parameters only if not yet set."""
        if self.param_server.get_parameters().get('parameters'):
            return False
        params = {k: torch.tensor(v) for k, v in parameters_serializable.items()}
        self.param_server.set_parameters(params)
        logger.info("Parameter server initialized from worker parameters")
        return True

    def ps_get_parameters(self) -> Dict:
        data = self.param_server.get_parameters()
        # Serialize tensors to lists for RPC
        serial = {k: v.cpu().tolist() for k, v in data['parameters'].items()}
        return {'parameters': serial, 'version': data['version']}

    def ps_sync_step(self, worker_id: str, gradients: Dict[str, list], learning_rate: float) -> Dict:
        """Submit gradients and wait for all workers, then return updated params."""
        if not self._training_configured:
            # Default to all known workers
            self.configure_training_sync(num_workers=max(1, len(self.workers)))

        # Deserialize gradients
        grad_tensors = {k: torch.tensor(v) for k, v in gradients.items()}
        self.param_server.push_gradients(worker_id, grad_tensors)

        # Barrier wait
        self.sync_coordinator.wait_for_all(worker_id)

        # Single-update per barrier round
        with self._barrier_lock:
            barrier_round = self.sync_coordinator.barrier_count
            if barrier_round > self._last_update_barrier:
                _ = self.param_server.aggregate_and_update(learning_rate=learning_rate, rule=getattr(self, '_aggr_rule', 'mean'), trim_ratio=getattr(self, '_trim_ratio', 0.0))
                self._last_update_barrier = barrier_round

        # Return updated parameters
        data = self.param_server.get_parameters()
        serial = {k: v.cpu().tolist() for k, v in data['parameters'].items()}
        return {'parameters': serial, 'version': data['version']}

    # Aggregation control
    def ps_set_aggregation(self, rule: str = 'mean', trim_ratio: float = 0.0, compression: str = None) -> bool:
        self._aggr_rule = rule
        self._trim_ratio = float(trim_ratio)
        self._compression = compression
        logger.info(f"Aggregation set: rule={rule} trim_ratio={trim_ratio} compression={compression}")
        return True

    def ps_save(self, path: str) -> bool:
        """Checkpoint parameter server state to a file."""
        return self.param_server.save_to_file(path)

    def ps_load(self, path: str) -> bool:
        """Restore parameter server state from a file."""
        return self.param_server.load_from_file(path)
"""Controller module

Defines the MainController responsible for worker registration, task lifecycle,
and scheduling. It supports both legacy schedulers (`schedule()`) and structured
schedulers via a normalized `SchedulerInput` carrying tasks, worker features,
and EWMA telemetry.

Mapping conventions:
- Schedulers return `{task_id: worker_id}`.
- Controller stores assignments internally as `{worker_id: task_id}`.
"""
