import unittest

from dist_llm_train.scheduler.capability import CapabilityScheduler
from dist_llm_train.scheduler.interfaces import (
    SchedulerInput, SchedulerWorkerInfo, SchedulerTaskInfo, TelemetrySnapshot
)


class TestStructuredScheduler(unittest.TestCase):
    def test_telemetry_influences_assignment(self):
        # Two workers: same memory, flops/bw, different telemetry (EWMA tps)
        w1 = SchedulerWorkerInfo(
            id='w1', memory=16, flops_per_second=100, network_bandwidth=1000, status='available', address='http://x',
            telemetry=TelemetrySnapshot(worker_id='w1', avg_tokens_per_sec=100.0)
        )
        w2 = SchedulerWorkerInfo(
            id='w2', memory=16, flops_per_second=100, network_bandwidth=1000, status='available', address='http://x',
            telemetry=TelemetrySnapshot(worker_id='w2', avg_tokens_per_sec=200.0)
        )
        t = SchedulerTaskInfo(id='t1', total_memory_req=2.0, required_flops=10, priority=1)
        sched_input = SchedulerInput(tasks=[t], workers=[w1, w2])
        out = CapabilityScheduler([], []).schedule_from_input(sched_input)
        self.assertEqual(out.assignments.get('t1'), 'w2')


if __name__ == '__main__':
    unittest.main()

