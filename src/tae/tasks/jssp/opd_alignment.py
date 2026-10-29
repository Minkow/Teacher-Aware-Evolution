from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from tae.tasks.jssp.alignment import AlignmentResult, spearman_correlation
from tae.tasks.jssp.core import CandidateFeature, JSSPInstance
from tae.tasks.jssp.evaluator import choose_action
from tae.tasks.jssp.offline_ld_teacher import TeacherPreference
from tae.tasks.jssp.rules import PriorityRule
from tae.tasks.jssp.simulator import ScheduleState


class OnPolicyTeacher(Protocol):
    def reset_instance(self, instance: JSSPInstance): ...

    def preference(self, state) -> TeacherPreference: ...

    def step(self, action: int): ...


def opd_policy_alignment(
    rule: PriorityRule,
    instances: list[JSSPInstance],
    teacher: OnPolicyTeacher,
    *,
    sample_stride: int = 1,
    max_states_per_instance: int | None = None,
) -> AlignmentResult:
    """Compare a program with teacher preferences on the program's own states.

    The candidate program chooses actions. The teacher is only queried to rank
    the feasible actions at the states induced by that program, which is the
    OPD-like part of the design.
    """

    correlations: list[float] = []
    agreements: list[float] = []
    percentiles: list[float] = []
    stage_hits: dict[str, list[float]] = {"early": [], "mid": [], "late": []}
    feature_deltas: dict[str, list[float]] = defaultdict(list)

    for instance in instances:
        state = teacher.reset_instance(instance)
        mirror = ScheduleState(instance)
        sampled = 0
        done = False
        while not done and not mirror.done:
            features = mirror.candidate_features()
            feature_by_op = {feature.op_id: feature for feature in features}
            action = choose_action(rule, mirror)

            should_sample = sample_stride <= 1 or len(mirror.actions) % sample_stride == 0
            if should_sample and (max_states_per_instance is None or sampled < max_states_per_instance):
                preference = teacher.preference(state)
                aligned = _aligned_scores(rule, preference, feature_by_op, mirror)
                if aligned is not None:
                    candidate_ops, rule_scores, teacher_scores = aligned
                    correlations.append(spearman_correlation(rule_scores, teacher_scores))
                    teacher_rank = _rank_of_action(candidate_ops, teacher_scores, action)
                    top1 = float(action == preference.teacher_action)
                    agreements.append(top1)
                    percentiles.append(_rank_percentile(teacher_rank, len(candidate_ops)))
                    stage_hits[_stage_name(len(mirror.actions), instance.n_operations)].append(top1)
                    _record_feature_delta(
                        feature_deltas,
                        feature_by_op.get(action),
                        feature_by_op.get(preference.teacher_action),
                    )
                    sampled += 1

            mirror.dispatch(action)
            state, _reward, terminated, truncated, _info = teacher.step(action)
            done = bool(terminated or truncated)

    return AlignmentResult(
        mean_rank_correlation=sum(correlations) / len(correlations) if correlations else 0.0,
        top1_agreement=sum(agreements) / len(agreements) if agreements else 0.0,
        states=len(correlations),
        mean_teacher_action_percentile=sum(percentiles) / len(percentiles) if percentiles else 0.0,
        diagnostic=_diagnostic(stage_hits, feature_deltas),
    )


def _aligned_scores(
    rule: PriorityRule,
    preference: TeacherPreference,
    feature_by_op: dict[int, CandidateFeature],
    mirror: ScheduleState,
) -> tuple[tuple[int, ...], list[float], list[float]] | None:
    candidate_ops = tuple(op for op in preference.candidate_ops if op in feature_by_op)
    if len(candidate_ops) < 2:
        return None
    teacher_by_op = dict(zip(preference.candidate_ops, preference.teacher_scores))
    try:
        rule_scores = [rule.score(feature_by_op[op], mirror) for op in candidate_ops]
    except Exception:
        return candidate_ops, [-1.0 for _ in candidate_ops], [teacher_by_op[op] for op in candidate_ops]
    teacher_scores = [teacher_by_op[op] for op in candidate_ops]
    return candidate_ops, rule_scores, teacher_scores


def _rank_of_action(candidate_ops: tuple[int, ...], scores: list[float], action: int) -> int:
    ordered = sorted(zip(candidate_ops, scores), key=lambda item: (item[1], -item[0]), reverse=True)
    for rank, (op_id, _score) in enumerate(ordered):
        if op_id == action:
            return rank
    return len(candidate_ops) - 1


def _rank_percentile(rank: int, candidate_count: int) -> float:
    if candidate_count <= 1:
        return 1.0
    return 1.0 - rank / (candidate_count - 1)


def _stage_name(step: int, total_steps: int) -> str:
    frac = step / max(1, total_steps)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def _record_feature_delta(
    deltas: dict[str, list[float]],
    chosen: CandidateFeature | None,
    teacher_best: CandidateFeature | None,
) -> None:
    if chosen is None or teacher_best is None:
        return
    for name in (
        "processing_time",
        "remaining_work",
        "remaining_ops",
        "earliest_finish",
        "machine_queue_len",
        "machine_total_work",
        "job_progress",
        "lower_bound_after",
    ):
        deltas[name].append(float(getattr(chosen, name)) - float(getattr(teacher_best, name)))


def _diagnostic(stage_hits: dict[str, list[float]], feature_deltas: dict[str, list[float]]) -> str:
    stage_parts = []
    for name in ("early", "mid", "late"):
        values = stage_hits[name]
        if values:
            stage_parts.append(f"{name}_top1={sum(values) / len(values):.3f}")
    delta_rows = []
    for name, values in feature_deltas.items():
        if not values:
            continue
        avg = sum(values) / len(values)
        delta_rows.append((abs(avg), name, avg))
    delta_rows.sort(reverse=True)
    delta_text = ", ".join(f"{name}={avg:+.3f}" for _abs, name, avg in delta_rows[:5])
    if delta_text:
        stage_parts.append(f"chosen_minus_teacher_best: {delta_text}")
    return "; ".join(stage_parts)
