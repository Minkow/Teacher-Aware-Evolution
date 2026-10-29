from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from tae.tasks.maxcut.core import MaxCutInstance, MaxCutState
from tae.tasks.maxcut.program import ProgramActionScorer


@dataclass(frozen=True)
class TeacherPreference:
    candidate_actions: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    teacher_action: int


@dataclass(frozen=True)
class AlignmentResult:
    mean_rank_correlation: float
    top1_agreement: float
    mean_teacher_action_percentile: float
    states: int


def policy_alignment(
    rule: ProgramActionScorer,
    instances: list[MaxCutInstance],
    teacher,
    *,
    max_steps: int | None = None,
    sample_stride: int = 1,
    max_states_per_instance: int | None = None,
) -> AlignmentResult:
    correlations: list[float] = []
    agreements: list[float] = []
    percentiles: list[float] = []
    for inst in instances:
        state = MaxCutState(inst, max_steps=max_steps)
        sampled = 0
        while not state.done:
            decision_state = state.decision_state()
            action = rule.select_action(decision_state)
            should_sample = sample_stride <= 1 or state.step % sample_stride == 0
            if should_sample and (max_states_per_instance is None or sampled < max_states_per_instance):
                preference: TeacherPreference = teacher.preference(state)
                rule_scores = [1.0 if a == action else 0.0 for a in preference.candidate_actions]
                teacher_scores = list(preference.teacher_scores)
                correlations.append(spearman_correlation(rule_scores, teacher_scores))
                teacher_rank = _rank_of_action(preference.candidate_actions, teacher_scores, action)
                agreements.append(float(action == preference.teacher_action))
                percentiles.append(_rank_percentile(teacher_rank, len(preference.candidate_actions)))
                sampled += 1
            state.dispatch(action)
    return AlignmentResult(
        mean_rank_correlation=mean(correlations) if correlations else 0.0,
        top1_agreement=mean(agreements) if agreements else 0.0,
        mean_teacher_action_percentile=mean(percentiles) if percentiles else 0.0,
        states=len(correlations),
    )


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = mean(rx)
    my = mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = sum((x - mx) ** 2 for x in rx) ** 0.5
    deny = sum((y - my) ** 2 for y in ry) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return float(num / (denx * deny))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        j = idx
        while j + 1 < len(order) and values[order[j + 1]] == values[order[idx]]:
            j += 1
        rank = (idx + j) / 2.0
        for k in range(idx, j + 1):
            ranks[order[k]] = rank
        idx = j + 1
    return ranks


def _rank_of_action(candidate_actions: tuple[int, ...], scores: list[float], action: int) -> int:
    ordered = sorted(zip(candidate_actions, scores), key=lambda item: (item[1], -item[0]), reverse=True)
    for rank, (candidate, _score) in enumerate(ordered):
        if candidate == action:
            return rank
    return len(candidate_actions) - 1


def _rank_percentile(rank: int, candidate_count: int) -> float:
    if candidate_count <= 1:
        return 1.0
    return 1.0 - rank / (candidate_count - 1)
