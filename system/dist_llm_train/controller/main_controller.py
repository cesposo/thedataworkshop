import time
import threading
import os
import logging
from typing import List, Dict, Type, Optional, Set, Any

import torch

from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.scheduler.base import BaseScheduler
from dist_llm_train.scheduler.gale_shapley import GaleShapleyScheduler
from dist_llm_train.scheduler.priority import PriorityScheduler
from dist_llm_train.scheduler.capability import CapabilityScheduler
from dist_llm_train.scheduler.lattice import StableMatchingLattice
from dist_llm_train.scheduler.preference_learning import PerformancePredictor
from dist_llm_train.scheduler.learned_preferences import LearnedPreferenceBuilder
from dist_llm_train.communication.rpc import RPCCommunicator
try:
    from dist_llm_train.communication.zmq_comm import HybridCommunicator  # type: ignore
except Exception:
    HybridCommunicator = None  # Optional dependency; fallback to RPC/chaos when missing
from dist_llm_train.sync.parameter_server import ParameterServer, SimpleSyncCoordinator, BoundedAsyncCoordinator
from dist_llm_train.compression.compressor import GradientCompressor
from dist_llm_train.communication.chaos import build_communicator
from dist_llm_train.communication.communicator import Communicator

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
    def __init__(self, host: str, port: int, scheduler_name: str = 'gale-shapley',
                 heartbeat_timeout: int = 30, state_db_path: str = None,
                 telemetry_alpha: float = None, use_lattice: bool = False,
                 max_lattice_size: int = 100, lattice_rebuild_interval: float = 300.0,
                 preference_model_path: Optional[str] = None, use_async_training: bool = True,
                 max_staleness: int = 50, use_zmq: bool = True,
                 gradient_accumulation_size: int = 3,
                 gradient_clip_norm: float = 1.0,
                 enable_differential_privacy: bool = False,
                 dp_noise_multiplier: float = 0.1,
                 adaptive_staleness: bool = True,
                 min_staleness: int = 5,
                 max_staleness_multiplier: float = 10.0,
                 communicator: Optional[Communicator] = None,
                 netem_profile: Optional[dict] = None,
                 netem_seed: Optional[int] = None):
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

        # Lattice-aware scheduling support
        self.use_lattice = use_lattice
        self.max_lattice_size = max_lattice_size
        self.lattice_rebuild_interval = lattice_rebuild_interval
        self.current_lattice: Optional[StableMatchingLattice] = None
        self.current_matching_id: Optional[int] = None
        self.lattice_built_time: float = 0.0

        # Learned preference support
        self.preference_predictor: Optional[PerformancePredictor] = None
        self.preference_builder: Optional[LearnedPreferenceBuilder] = None

        if preference_model_path and os.path.exists(preference_model_path):
            try:
                self.preference_predictor = PerformancePredictor.load(preference_model_path)
                self.preference_builder = LearnedPreferenceBuilder(
                    predictor=self.preference_predictor,
                    objective='throughput'
                )
                logger.info(f"Loaded preference model from {preference_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load preference model: {e}, using heuristics")

        # Initialize communicator with ZMQ support for efficient binary gradient transfer
        if communicator is not None:
            self.communicator = communicator
        else:
            if use_zmq:
                try:
                    self.communicator = HybridCommunicator(host, port, mode='router')
                    logger.info("Using HybridCommunicator (ZMQ + MessagePack for binary efficiency)")
                except Exception as e:
                    logger.warning(f"Failed to initialize ZMQ communicator: {e}, falling back to RPC")
                    self.communicator = build_communicator(host, port, netem_profile=netem_profile, seed=netem_seed)
            else:
                self.communicator = build_communicator(host, port, netem_profile=netem_profile, seed=netem_seed)
                logger.info("Using legacy RPCCommunicator (XML-RPC)")
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
        self.use_async_training = use_async_training
        self.max_staleness = max_staleness
        self.adaptive_staleness = adaptive_staleness
        if use_async_training:
            self.sync_coordinator = BoundedAsyncCoordinator(
                num_workers=0,
                max_staleness=max_staleness,
                adaptive_staleness=adaptive_staleness,
                min_staleness=min_staleness,
                max_staleness_multiplier=max_staleness_multiplier
            )
            if adaptive_staleness:
                logger.info(f"Using asynchronous training with ADAPTIVE staleness "
                           f"(base={max_staleness}, min={min_staleness}, max={max_staleness * max_staleness_multiplier:.0f})")
            else:
                logger.info(f"Using asynchronous training with FIXED staleness={max_staleness}")
        else:
            self.sync_coordinator = SimpleSyncCoordinator(num_workers=0)
            logger.info("Using synchronous training (DEPRECATED for heterogeneous environments)")
        self._barrier_lock = threading.Lock()
        self._last_update_barrier = 0
        self._training_configured = False

        # Gradient accumulation buffer (prevents micro-aggregations)
        self.gradient_accumulation_size = gradient_accumulation_size
        self._gradient_buffer = []  # Buffer for mini-batch aggregation
        self._buffer_lock = threading.Lock()
        logger.info(f"Gradient accumulation: will aggregate every {gradient_accumulation_size} gradients")

        # Differential privacy and security
        self.gradient_clip_norm = gradient_clip_norm
        self.enable_differential_privacy = enable_differential_privacy
        self.dp_noise_multiplier = dp_noise_multiplier
        if gradient_clip_norm > 0:
            logger.info(f"Gradient clipping enabled: max_norm={gradient_clip_norm}")
        if enable_differential_privacy:
            logger.info(f"Differential privacy enabled: noise_multiplier={dp_noise_multiplier}")

        # Compression support - decompressor initialized as needed
        self._decompressor = None
        self.communicator.register_function(self.configure_training_sync, 'configure_training_sync')
        self.communicator.register_function(self.ps_initialize_if_empty, 'ps_initialize_if_empty')
        self.communicator.register_function(self.ps_get_parameters, 'ps_get_parameters')
        self.communicator.register_function(self.ps_sync_step, 'ps_sync_step')
        self.communicator.register_function(self.ps_save, 'ps_save')
        self.communicator.register_function(self.ps_load, 'ps_load')
        self.communicator.register_function(self.ps_set_aggregation, 'ps_set_aggregation')
        if hasattr(self.communicator, "start_server"):
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
        Enhanced with lattice-aware scheduling and rotation-based reconfiguration.
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

        # Check if lattice mode is enabled
        if self.use_lattice and self.scheduler_name == 'gale-shapley':
            task_to_worker = self._run_lattice_aware_scheduling(available_workers, tasks_to_schedule)
        else:
            task_to_worker = self._run_standard_scheduling(available_workers, tasks_to_schedule)

        # Convert to controller's internal mapping {worker_id: task_id}
        for t_id, w_id in task_to_worker.items():
            self.assignments[w_id] = t_id

        logger.info(f"Scheduling complete. New assignments: {task_to_worker}")

        # Update system state based on new assignments
        self._dispatch_tasks(task_to_worker)
        self._persist_state_if_configured()

    def _run_standard_scheduling(self, available_workers, tasks_to_schedule) -> Dict[str, str]:
        """Run standard (non-lattice) scheduling."""
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

        return task_to_worker

    def _run_lattice_aware_scheduling(self, available_workers, tasks_to_schedule) -> Dict[str, str]:
        """
        Run lattice-aware scheduling with rotation-based reconfiguration.

        This implements the paper's concept of navigating the lattice of stable matchings
        via rotation operations for fast reconfiguration.
        """
        current_time = time.time()

        # Check if we need to rebuild the lattice
        needs_rebuild = (
            self.current_lattice is None or
            (current_time - self.lattice_built_time) > self.lattice_rebuild_interval
        )

        if needs_rebuild:
            logger.info("Building/rebuilding stable matching lattice...")
            task_to_worker = self._build_lattice_and_schedule(available_workers, tasks_to_schedule)
            self.lattice_built_time = current_time
            return task_to_worker

        # Lattice exists - try rotation-based reconfiguration
        failed_workers = self._detect_failed_workers()

        if failed_workers:
            logger.info(f"Detected failed workers: {failed_workers}, attempting rotation reconfiguration")
            task_to_worker = self._reconfigure_via_rotations(failed_workers, available_workers, tasks_to_schedule)

            if task_to_worker is not None:
                logger.info("Successfully reconfigured via rotations")
                return task_to_worker
            else:
                logger.warning("Rotation reconfiguration failed, rebuilding lattice")
                task_to_worker = self._build_lattice_and_schedule(available_workers, tasks_to_schedule)
                self.lattice_built_time = current_time
                return task_to_worker
        else:
            # No failures - use current matching from lattice
            if self.current_matching_id is not None and self.current_lattice:
                current_matching = self.current_lattice.find_matching_by_id(self.current_matching_id)
                if current_matching:
                    logger.debug("Using existing lattice matching")
                    return current_matching

            # Fallback: rebuild
            logger.debug("No current matching available, rebuilding lattice")
            task_to_worker = self._build_lattice_and_schedule(available_workers, tasks_to_schedule)
            self.lattice_built_time = current_time
            return task_to_worker

    def _build_lattice_and_schedule(self, available_workers, tasks_to_schedule) -> Dict[str, str]:
        """Build lattice and select a matching."""
        # Apply learned preferences if available
        if self.preference_builder:
            try:
                worker_nodes = [w['info'] for w in available_workers]
                self.preference_builder.apply_preferences_to_tasks_and_workers(
                    tasks_to_schedule, worker_nodes
                )
                logger.info("Applied learned preferences to tasks and workers")
            except Exception as e:
                logger.warning(f"Failed to apply learned preferences: {e}")

        # Create scheduler with lattice mode
        scheduler = GaleShapleyScheduler(
            tasks_to_schedule,
            available_workers,
            compute_lattice=True,
            max_lattice_size=self.max_lattice_size
        )

        # Build lattice
        self.current_lattice = scheduler.build_lattice()
        logger.info(f"Built lattice with {self.current_lattice.size()} stable matchings")

        # Select task-optimal matching
        self.current_matching_id = self.current_lattice.task_optimal_id
        matching = self.current_lattice.find_matching_by_id(self.current_matching_id)

        return matching

    def _detect_failed_workers(self) -> Set[str]:
        """Detect workers that have failed since last cycle."""
        failed = set()
        for worker_id, worker in self.workers.items():
            if worker['status'] == 'offline':
                failed.add(worker_id)
        return failed

    def _reconfigure_via_rotations(self, failed_workers: Set[str],
                                   available_workers, tasks_to_schedule) -> Optional[Dict[str, str]]:
        """
        Attempt fast reconfiguration via rotation operations.

        This is the key innovation: instead of full re-scheduling, we navigate
        the lattice to find a nearby stable matching that avoids failed workers.
        """
        if not self.current_lattice:
            return None

        # Find a matching that avoids failed workers
        result = self.current_lattice.get_matching_avoiding_workers(failed_workers)

        if result is None:
            return None

        matching_id, new_matching = result
        old_id = self.current_matching_id

        # Calculate rotation path length
        if old_id is not None:
            old_rotations = self.current_lattice.matching_to_rotations.get(old_id, set())
            new_rotations = self.current_lattice.matching_to_rotations.get(matching_id, set())
            path_length = len(new_rotations.symmetric_difference(old_rotations))
            logger.info(f"Reconfiguration path: {path_length} rotation steps from matching {old_id} to {matching_id}")

        self.current_matching_id = matching_id
        return new_matching

    def _dispatch_tasks(self, task_to_worker: Dict[str, str]):
        """Dispatch tasks to workers via RPC."""
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
                    logger.debug(f"Dispatched task {task_id} to worker {worker_id}")
                except Exception as e:
                    logger.error(f"Failed to assign task to worker {worker_id}: {e}")

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
        def _summarize_netem(obj):
            stats = getattr(obj, 'stats', None)
            if not stats:
                return None
            lats = stats.get('latency_ms') or []
            if lats:
                slats = sorted(lats)
                def _p(v):
                    if not slats:
                        return None
                    k = max(0, min(len(slats) - 1, int((v/100.0) * (len(slats) - 1))))
                    return slats[k]
                p50 = _p(50)
                p95 = _p(95)
            else:
                p50 = p95 = None
            return {
                'sent': stats.get('sent', 0),
                'dropped': stats.get('dropped', 0),
                'duplicated': stats.get('duplicated', 0),
                'bytes': stats.get('bytes', 0),
                'p50_latency_ms': p50,
                'p95_latency_ms': p95,
            }

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
        # Netem summaries (controller + workers) if available
        netem_workers = {}
        for worker_id, w in self.workers.items():
            info = w.get('info')
            comm = getattr(info, 'communicator', None)
            summary = _summarize_netem(comm) if comm else None
            if summary:
                netem_workers[worker_id] = summary
        ctrl_netem = _summarize_netem(self.communicator)
        if ctrl_netem or netem_workers:
            status['netem_stats'] = {
                'controller': ctrl_netem,
                'workers': netem_workers,
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
    def configure_training_sync(self, num_workers: int, window_size: int = None, max_wait_s: float = None, use_async: bool = None) -> bool:
        """Configure the number of workers participating and window for sync steps.

        Args:
            num_workers: Number of workers
            window_size: Window size for sync mode (ignored in async mode)
            max_wait_s: Max wait time for sync mode (ignored in async mode)
            use_async: Override async setting (defaults to controller's use_async_training)
        """
        if use_async is None:
            use_async = self.use_async_training

        if use_async:
            self.sync_coordinator = BoundedAsyncCoordinator(
                num_workers=num_workers,
                max_staleness=self.max_staleness,
                window_size=window_size,
                max_wait_s=max_wait_s
            )
            logger.info(f"Training sync configured for {num_workers} workers (async mode, max_staleness={self.max_staleness})")
        else:
            self.sync_coordinator = SimpleSyncCoordinator(
                num_workers=num_workers,
                window_size=window_size,
                max_wait_s=max_wait_s
            )
            logger.info(f"Training sync configured for {num_workers} workers (sync mode)")
        self._training_configured = True
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

    def ps_sync_step(self, worker_id: str, gradients: Dict[str, Any], learning_rate: float, worker_step: int = None, metadata: Dict[str, Any] = None) -> Dict:
        """Submit gradients and coordinate parameter updates.

        In async mode: gradients are applied immediately without barriers.
        In sync mode: workers wait at barriers before aggregation.

        Args:
            worker_id: Worker identifier
            gradients: Serialized (possibly compressed) gradients
            learning_rate: Learning rate for update
            worker_step: Worker's local step number (for async staleness tracking)
            metadata: Optional compression metadata for decompression

        Returns:
            Updated parameters and version
        """
        if not self._training_configured:
            # Default to all known workers
            self.configure_training_sync(num_workers=max(1, len(self.workers)))

        # Decompress gradients if metadata present
        if metadata is not None:
            # Initialize decompressor if needed
            if self._decompressor is None:
                method = metadata.get('method', 'none')
                self._decompressor = GradientCompressor(method=method)
                logger.info(f"Initialized gradient decompressor with method={method}")

            # Decompress
            grad_tensors = self._decompressor.decompress(gradients, metadata)
        else:
            # Deserialize uncompressed gradients
            grad_tensors = {k: torch.tensor(v) for k, v in gradients.items()}

        if self.use_async_training:
            # Async mode: submit gradients with staleness check
            accepted = self.sync_coordinator.submit_gradients(worker_id, grad_tensors, worker_step)
            if not accepted:
                logger.warning(f"Rejected stale gradients from worker {worker_id} at step {worker_step}")
                # Return current parameters without update
                data = self.param_server.get_parameters()
                serial = {k: v.cpu().tolist() for k, v in data['parameters'].items()}
                return {'parameters': serial, 'version': data['version']}

            # Get newly accepted gradients from coordinator
            accumulated_grads = self.sync_coordinator.get_and_clear_gradients()

            # Add to accumulation buffer (prevents micro-aggregations)
            with self._buffer_lock:
                self._gradient_buffer.extend(accumulated_grads)

                # Only aggregate when buffer reaches threshold
                if len(self._gradient_buffer) >= self.gradient_accumulation_size:
                    logger.debug(f"Buffer full ({len(self._gradient_buffer)} gradients), triggering aggregation")

                    # Apply gradient clipping and differential privacy
                    for wid, grads, staleness in self._gradient_buffer:
                        processed_grads = grads

                        # Clip gradients to prevent extreme values
                        if self.gradient_clip_norm > 0:
                            processed_grads = self._apply_gradient_clipping(
                                processed_grads,
                                max_norm=self.gradient_clip_norm
                            )

                        # Add differential privacy noise
                        if self.enable_differential_privacy:
                            processed_grads = self._apply_differential_privacy(
                                processed_grads,
                                noise_multiplier=self.dp_noise_multiplier,
                                sensitivity=self.gradient_clip_norm
                            )

                        # Push to parameter server
                        self.param_server.push_gradients(wid, processed_grads)

                    # Aggregate and update
                    self.param_server.aggregate_and_update(
                        learning_rate=learning_rate,
                        rule=getattr(self, '_aggr_rule', 'mean'),
                        trim_ratio=getattr(self, '_trim_ratio', 0.3),
                        krum_f=getattr(self, '_krum_f', None),
                        bulyan_f=getattr(self, '_bulyan_f', None)
                    )

                    # Clear buffer
                    self._gradient_buffer.clear()
                else:
                    logger.debug(f"Buffer size: {len(self._gradient_buffer)}/{self.gradient_accumulation_size}")
        else:
            # Sync mode: traditional barrier-based synchronization
            self.param_server.push_gradients(worker_id, grad_tensors)

            # Barrier wait
            self.sync_coordinator.wait_for_all(worker_id)

            # Single-update per barrier round
            with self._barrier_lock:
                barrier_round = self.sync_coordinator.barrier_count
                if barrier_round > self._last_update_barrier:
                    _ = self.param_server.aggregate_and_update(
                        learning_rate=learning_rate,
                        rule=getattr(self, '_aggr_rule', 'mean'),
                        trim_ratio=getattr(self, '_trim_ratio', 0.0)
                    )
                    self._last_update_barrier = barrier_round

        # Return updated parameters
        data = self.param_server.get_parameters()
        serial = {k: v.cpu().tolist() for k, v in data['parameters'].items()}
        return {'parameters': serial, 'version': data['version']}

    def _apply_gradient_clipping(self, gradients: Dict[str, torch.Tensor], max_norm: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply gradient clipping for stability and basic differential privacy.

        Args:
            gradients: Dictionary of gradient tensors
            max_norm: Maximum L2 norm for gradients

        Returns:
            Clipped gradients
        """
        # Compute total L2 norm across all parameters
        total_norm = torch.sqrt(sum(g.norm() ** 2 for g in gradients.values()))

        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            clipped_grads = {k: v * clip_coef for k, v in gradients.items()}
            logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {max_norm}")
            return clipped_grads
        else:
            return gradients

    def _apply_differential_privacy(self, gradients: Dict[str, torch.Tensor],
                                    noise_multiplier: float = 0.1,
                                    sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise for differential privacy.

        Args:
            gradients: Dictionary of gradient tensors
            noise_multiplier: Multiplier for noise scale (higher = more privacy, less accuracy)
            sensitivity: Sensitivity of the gradient computation (max gradient norm)

        Returns:
            Noisy gradients
        """
        noisy_grads = {}
        for k, v in gradients.items():
            # Gaussian mechanism: noise ~ N(0, (sensitivity * noise_multiplier)^2)
            noise_scale = sensitivity * noise_multiplier
            noise = torch.randn_like(v) * noise_scale
            noisy_grads[k] = v + noise

        logger.debug(f"Added DP noise with multiplier={noise_multiplier}, sensitivity={sensitivity}")
        return noisy_grads

    # Aggregation control
    def ps_set_aggregation(self, rule: str = 'mean', trim_ratio: float = 0.3,
                          krum_f: int = None, bulyan_f: int = None, compression: str = None) -> bool:
        """Set gradient aggregation configuration.

        Args:
            rule: Aggregation rule ('mean', 'trimmed_mean', 'krum', 'bulyan')
            trim_ratio: Fraction to trim for trimmed_mean (default: 0.3, was 0.0)
            krum_f: Number of Byzantine workers to tolerate for Krum (default: n//4)
            bulyan_f: Number of Byzantine workers to tolerate for Bulyan (default: n//4)
            compression: Compression method (for future use)

        Returns:
            True if successful
        """
        self._aggr_rule = rule
        self._trim_ratio = float(trim_ratio)
        self._krum_f = krum_f
        self._bulyan_f = bulyan_f
        self._compression = compression
        logger.info(f"Aggregation set: rule={rule} trim_ratio={trim_ratio} "
                   f"krum_f={krum_f} bulyan_f={bulyan_f} compression={compression}")
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
