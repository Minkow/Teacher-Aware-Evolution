from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from tae.tasks.jssp.core import CandidateFeature
from tae.tasks.jssp.simulator import ScheduleState


class PriorityRule(Protocol):
    name: str

    def score(self, feature: CandidateFeature, state: ScheduleState) -> float:
        ...


@dataclass(frozen=True)
class FunctionRule:
    name: str
    fn: Callable[[CandidateFeature, ScheduleState], float]

    def score(self, feature: CandidateFeature, state: ScheduleState) -> float:
        return self.fn(feature, state)


@dataclass(frozen=True)
class LinearPriorityRule:
    name: str
    weights: Mapping[str, float]
    intercept: float = 0.0

    def score(self, feature: CandidateFeature, state: ScheduleState) -> float:
        values = feature.as_dict()
        return self.intercept + sum(
            float(weight) * float(values[key])
            for key, weight in self.weights.items()
            if key in values
        )

    def describe(self) -> str:
        terms = [f"{self.intercept:.4g}"]
        for key, value in sorted(self.weights.items()):
            terms.append(f"{value:+.4g}*{key}")
        return " ".join(terms)


def _fifo(f: CandidateFeature, state: ScheduleState) -> float:
    return -f.op_id


def _spt(f: CandidateFeature, state: ScheduleState) -> float:
    return -f.processing_time


def _lpt(f: CandidateFeature, state: ScheduleState) -> float:
    return f.processing_time


def _mwkr(f: CandidateFeature, state: ScheduleState) -> float:
    return f.remaining_work


def _mor(f: CandidateFeature, state: ScheduleState) -> float:
    return f.remaining_ops


def _fdd_mwkr(f: CandidateFeature, state: ScheduleState) -> float:
    flow_due_date = sum(state.instance.durations[f.job_id][: f.op_index + 1])
    return -flow_due_date / max(1, f.remaining_work)


def _earliest_finish(f: CandidateFeature, state: ScheduleState) -> float:
    return -f.earliest_finish


def _critical_lb(f: CandidateFeature, state: ScheduleState) -> float:
    return f.lower_bound_after


RULE_REGISTRY: dict[str, PriorityRule] = {
    "fifo": FunctionRule("fifo", _fifo),
    "spt": FunctionRule("spt", _spt),
    "lpt": FunctionRule("lpt", _lpt),
    "mwkr": FunctionRule("mwkr", _mwkr),
    "mor": FunctionRule("mor", _mor),
    "fdd_mwkr": FunctionRule("fdd_mwkr", _fdd_mwkr),
    "eft": FunctionRule("eft", _earliest_finish),
    "critical_lb": FunctionRule("critical_lb", _critical_lb),
}


DEFAULT_LINEAR_FEATURES = (
    "processing_time",
    "remaining_work",
    "remaining_ops",
    "job_ready_time",
    "machine_ready_time",
    "earliest_start",
    "earliest_finish",
    "machine_queue_len",
    "machine_total_work",
    "job_progress",
    "lower_bound_after",
)
