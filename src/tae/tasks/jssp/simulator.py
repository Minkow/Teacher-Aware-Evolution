from __future__ import annotations

from bisect import insort
from dataclasses import dataclass

from tae.tasks.jssp.core import CandidateFeature, JSSPInstance, ScheduledOperation


@dataclass(order=True, frozen=True)
class _MachineSlot:
    start: int
    end: int
    op_id: int


class ScheduleState:
    """Incremental serial scheduler with earliest feasible insertion."""

    def __init__(self, instance: JSSPInstance):
        self.instance = instance.normalized()
        self.next_op = [0 for _ in range(self.instance.n_jobs)]
        self.job_ready = [0 for _ in range(self.instance.n_jobs)]
        self.machine_slots: list[list[_MachineSlot]] = [
            [] for _ in range(self.instance.n_machines)
        ]
        self.scheduled_by_id: dict[int, ScheduledOperation] = {}
        self.actions: list[int] = []

    def clone(self) -> "ScheduleState":
        other = ScheduleState(self.instance)
        other.next_op = self.next_op[:]
        other.job_ready = self.job_ready[:]
        other.machine_slots = [slots[:] for slots in self.machine_slots]
        other.scheduled_by_id = self.scheduled_by_id.copy()
        other.actions = self.actions[:]
        return other

    @property
    def done(self) -> bool:
        return len(self.actions) == self.instance.n_operations

    @property
    def makespan(self) -> int:
        return max(self.job_ready, default=0)

    def candidate_ops(self) -> list[int]:
        return [
            self.op_id(job_id, op_idx)
            for job_id, op_idx in enumerate(self.next_op)
            if op_idx < self.instance.n_machines
        ]

    def op_id(self, job_id: int, op_index: int) -> int:
        return job_id * self.instance.n_machines + op_index

    def decode(self, op_id: int) -> tuple[int, int]:
        return divmod(op_id, self.instance.n_machines)

    def earliest_insertion(self, op_id: int) -> tuple[int, int]:
        job_id, op_index = self.decode(op_id)
        if self.next_op[job_id] != op_index:
            raise ValueError(f"operation {op_id} is not currently feasible")
        processing_time = self.instance.durations[job_id][op_index]
        machine_id = self.instance.machines[job_id][op_index]
        ready = self.job_ready[job_id]
        slots = self.machine_slots[machine_id]

        time = ready
        insert_at = len(slots)
        for idx, slot in enumerate(slots):
            if time + processing_time <= slot.start:
                insert_at = idx
                return time, insert_at
            if slot.end > time:
                time = slot.end
        return time, insert_at

    def dispatch(self, op_id: int) -> ScheduledOperation:
        job_id, op_index = self.decode(op_id)
        if self.next_op[job_id] != op_index:
            raise ValueError(f"operation {op_id} is not currently feasible")

        processing_time = self.instance.durations[job_id][op_index]
        machine_id = self.instance.machines[job_id][op_index]
        start, _ = self.earliest_insertion(op_id)
        end = start + processing_time
        scheduled = ScheduledOperation(
            op_id=op_id,
            job_id=job_id,
            op_index=op_index,
            machine_id=machine_id,
            start=start,
            end=end,
            processing_time=processing_time,
        )

        insort(self.machine_slots[machine_id], _MachineSlot(start, end, op_id))
        self.scheduled_by_id[op_id] = scheduled
        self.next_op[job_id] += 1
        self.job_ready[job_id] = end
        self.actions.append(op_id)
        return scheduled

    def candidate_features(self) -> list[CandidateFeature]:
        return [self.feature_for(op_id) for op_id in self.candidate_ops()]

    def feature_for(self, op_id: int) -> CandidateFeature:
        job_id, op_index = self.decode(op_id)
        processing_time = self.instance.durations[job_id][op_index]
        machine_id = self.instance.machines[job_id][op_index]
        start, _ = self.earliest_insertion(op_id)
        remaining = self.instance.durations[job_id][op_index:]
        machine_slots = self.machine_slots[machine_id]
        machine_ready = max((slot.end for slot in machine_slots), default=0)
        machine_total_work = sum(
            self.instance.durations[j][k]
            for j in range(self.instance.n_jobs)
            for k in range(self.next_op[j], self.instance.n_machines)
            if self.instance.machines[j][k] == machine_id
        )
        lower_bound_after = max(
            start + sum(remaining),
            max(self.job_ready[j] + sum(self.instance.durations[j][self.next_op[j] :])
                for j in range(self.instance.n_jobs)),
        )
        return CandidateFeature(
            op_id=op_id,
            job_id=job_id,
            op_index=op_index,
            machine_id=machine_id,
            processing_time=processing_time,
            remaining_work=sum(remaining),
            remaining_ops=len(remaining),
            job_ready_time=self.job_ready[job_id],
            machine_ready_time=machine_ready,
            earliest_start=start,
            earliest_finish=start + processing_time,
            machine_queue_len=len(machine_slots),
            machine_total_work=machine_total_work,
            job_progress=op_index / self.instance.n_machines,
            lower_bound_after=lower_bound_after,
        )

    def schedule(self) -> tuple[ScheduledOperation, ...]:
        return tuple(self.scheduled_by_id[op_id] for op_id in self.actions)

