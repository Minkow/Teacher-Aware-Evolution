from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from tae.tasks.jssp.core import JSSPInstance, RolloutResult
from tae.tasks.jssp.rules import PriorityRule
from tae.tasks.jssp.simulator import ScheduleState


@dataclass(frozen=True)
class EvaluationResult:
    rule_name: str
    count: int
    mean_makespan: float
    best_makespan: int
    worst_makespan: int
    elapsed_s: float
    makespans: tuple[int, ...]

    @property
    def throughput(self) -> float:
        return self.count / self.elapsed_s if self.elapsed_s > 0 else float("inf")


def choose_action(rule: PriorityRule, state: ScheduleState) -> int:
    features = state.candidate_features()
    if not features:
        raise ValueError("cannot choose action from a terminal state")
    scored = [(rule.score(feature, state), -feature.op_id, feature.op_id) for feature in features]
    return max(scored)[2]


def rollout_rule(instance: JSSPInstance, rule: PriorityRule) -> RolloutResult:
    state = ScheduleState(instance)
    while not state.done:
        state.dispatch(choose_action(rule, state))
    return RolloutResult(
        makespan=state.makespan,
        actions=tuple(state.actions),
        schedule=state.schedule(),
    )


def evaluate_rule(instances: list[JSSPInstance], rule: PriorityRule) -> EvaluationResult:
    start = perf_counter()
    makespans = tuple(rollout_rule(instance, rule).makespan for instance in instances)
    elapsed = perf_counter() - start
    return EvaluationResult(
        rule_name=rule.name,
        count=len(instances),
        mean_makespan=mean(makespans) if makespans else 0.0,
        best_makespan=min(makespans) if makespans else 0,
        worst_makespan=max(makespans) if makespans else 0,
        elapsed_s=elapsed,
        makespans=makespans,
    )


def compare_rules(
    instances: list[JSSPInstance],
    rules: list[PriorityRule],
) -> list[EvaluationResult]:
    results = [evaluate_rule(instances, rule) for rule in rules]
    return sorted(results, key=lambda result: result.mean_makespan)

