from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JSSPInstance:
    """A job-shop instance.

    `durations[j][k]` is the processing time of operation k in job j.
    `machines[j][k]` is the machine id of that operation. Machine ids may be
    0-based or 1-based at construction time; `normalized()` converts to 0-based.
    """

    durations: tuple[tuple[int, ...], ...]
    machines: tuple[tuple[int, ...], ...]
    name: str = ""

    @classmethod
    def from_lists(
        cls,
        durations: list[list[int]] | tuple[tuple[int, ...], ...],
        machines: list[list[int]] | tuple[tuple[int, ...], ...],
        name: str = "",
    ) -> "JSSPInstance":
        return cls(
            durations=tuple(tuple(int(x) for x in row) for row in durations),
            machines=tuple(tuple(int(x) for x in row) for row in machines),
            name=name,
        ).normalized()

    @property
    def n_jobs(self) -> int:
        return len(self.durations)

    @property
    def n_machines(self) -> int:
        return len(self.durations[0]) if self.durations else 0

    @property
    def n_operations(self) -> int:
        return self.n_jobs * self.n_machines

    def validate(self) -> None:
        if not self.durations:
            raise ValueError("instance must contain at least one job")
        width = len(self.durations[0])
        if width == 0:
            raise ValueError("each job must contain at least one operation")
        if len(self.machines) != len(self.durations):
            raise ValueError("duration and machine matrices must have the same number of jobs")
        for j, (dur_row, mach_row) in enumerate(zip(self.durations, self.machines)):
            if len(dur_row) != width or len(mach_row) != width:
                raise ValueError(f"job {j} has inconsistent operation count")
            if any(p <= 0 for p in dur_row):
                raise ValueError(f"job {j} contains non-positive processing time")
            if sorted(mach_row) != list(range(width)):
                raise ValueError(
                    f"job {j} must visit each machine exactly once after normalization"
                )

    def normalized(self) -> "JSSPInstance":
        flat = [m for row in self.machines for m in row]
        offset = 1 if min(flat) == 1 and max(flat) == self.n_machines else 0
        normalized_machines = tuple(tuple(m - offset for m in row) for row in self.machines)
        instance = JSSPInstance(self.durations, normalized_machines, self.name)
        instance.validate()
        return instance


@dataclass(frozen=True)
class ScheduledOperation:
    op_id: int
    job_id: int
    op_index: int
    machine_id: int
    start: int
    end: int
    processing_time: int


@dataclass(frozen=True)
class CandidateFeature:
    op_id: int
    job_id: int
    op_index: int
    machine_id: int
    processing_time: int
    remaining_work: int
    remaining_ops: int
    job_ready_time: int
    machine_ready_time: int
    earliest_start: int
    earliest_finish: int
    machine_queue_len: int
    machine_total_work: int
    job_progress: float
    lower_bound_after: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RolloutResult:
    makespan: int
    actions: tuple[int, ...]
    schedule: tuple[ScheduledOperation, ...]

