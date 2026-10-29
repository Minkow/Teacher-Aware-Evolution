from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

import numpy as np

from tae.tasks.maxcut.core import MaxCutInstance, MaxCutState
from tae.tasks.maxcut.program import ProgramActionScorer


@dataclass(frozen=True)
class RolloutResult:
    best_cut: float
    final_cut: float
    actions: tuple[int, ...]
    best_spins: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationResult:
    rule_name: str
    count: int
    mean_cut: float
    best_cut: float
    worst_cut: float
    elapsed_s: float
    cuts: tuple[float, ...]


def rollout_program(
    instance: MaxCutInstance,
    rule: ProgramActionScorer,
    *,
    max_steps: int | None = None,
    init_spins=None,
) -> RolloutResult:
    state = MaxCutState(instance, max_steps=max_steps, init_spins=init_spins)
    while not state.done:
        action = rule.select_action(state.decision_state())
        state.dispatch(action)
    return RolloutResult(
        best_cut=float(state.best_cut),
        final_cut=float(state.current_cut),
        actions=tuple(state.actions),
        best_spins=tuple(int(x) for x in state.best_spins),
    )


def evaluate_program(
    instances: list[MaxCutInstance],
    rule: ProgramActionScorer,
    *,
    max_steps: int | None = None,
    attempts: int = 1,
    seed: int = 0,
) -> EvaluationResult:
    start = perf_counter()
    rng = np.random.default_rng(seed)
    cuts: list[float] = []
    for inst in instances:
        best = float("-inf")
        for attempt in range(max(1, attempts)):
            init = None if attempt == 0 else rng.integers(0, 2, size=inst.n_nodes)
            best = max(best, rollout_program(inst, rule, max_steps=max_steps, init_spins=init).best_cut)
        cuts.append(best)
    elapsed = perf_counter() - start
    return EvaluationResult(
        rule_name=rule.name,
        count=len(instances),
        mean_cut=mean(cuts) if cuts else 0.0,
        best_cut=max(cuts) if cuts else 0.0,
        worst_cut=min(cuts) if cuts else 0.0,
        elapsed_s=elapsed,
        cuts=tuple(cuts),
    )
