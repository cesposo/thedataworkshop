from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class TelemetrySnapshot:
    worker_id: str
    tokens_per_sec: Optional[float] = None
    step_time_s: Optional[float] = None
    avg_tokens_per_sec: Optional[float] = None
    avg_step_time_s: Optional[float] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None
    mode: Optional[str] = None
    ts: Optional[float] = None


@dataclass
class SchedulerWorkerInfo:
    id: str
    memory: float
    flops_per_second: int
    network_bandwidth: int
    status: str
    address: str
    telemetry: Optional[TelemetrySnapshot] = None


@dataclass
class SchedulerTaskInfo:
    id: str
    total_memory_req: float
    required_flops: int
    priority: int
    job_id: Optional[str] = None
    job_weight: Optional[float] = None


@dataclass
class SchedulerInput:
    tasks: List[SchedulerTaskInfo]
    workers: List[SchedulerWorkerInfo]


@dataclass
class SchedulerOutput:
    assignments: Dict[str, str]  # {task_id: worker_id}
