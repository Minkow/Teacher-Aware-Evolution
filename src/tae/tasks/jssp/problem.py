from __future__ import annotations

import copy
import os
import threading
import time
import types
import warnings
from pathlib import Path

import numpy as np
from tae.engine.evaluation_pool import RolloutPool, bounded_evaluation

from tae.tasks.jssp.alignment import spearman_correlation
from tae.tasks.jssp.core import CandidateFeature, JSSPInstance
from tae.tasks.jssp.data import load_instances
from tae.tasks.jssp.evaluator import evaluate_rule
from tae.tasks.jssp.llm_loop import BASELINE_GUIDE
from tae.tasks.jssp.offline_ld_teacher import OfflineLDTeacher
from tae.tasks.jssp.rules import RULE_REGISTRY
from tae.tasks.jssp.simulator import ScheduleState

_TRACE_FEATURES = (
    "processing_time",
    "remaining_work",
    "remaining_ops",
    "earliest_start",
    "earliest_finish",
    "machine_queue_len",
    "machine_total_work",
    "job_progress",
    "lower_bound_after",
    "wait_for_machine",
)


from .prompts import GetPrompts


class JSSPCONST:
    def __init__(self) -> None:
        if not (os.environ.get("EOH_JSSP_TRAIN_NPY") or os.environ.get("EOH_JSSP_TRAIN_JSONL")):
            raise ValueError("JSSP training requires a fixed dataset: set EOH_JSSP_TRAIN_NPY or EOH_JSSP_TRAIN_JSONL")
        self.ndelay = 1
        self.n_jobs = int(os.environ.get("EOH_JSSP_N_JOBS", "10"))
        self.n_machines = int(os.environ.get("EOH_JSSP_N_MACHINES", "10"))
        self.n_instance = int(os.environ.get("EOH_JSSP_N_INSTANCE", "20"))
        self.seed = int(os.environ.get("EOH_JSSP_SEED", "0"))
        self.running_time = 10
        self.prompts = GetPrompts()
        self.teacher_aware = os.environ.get("EOH_TEACHER_AWARE", "0") == "1"
        self.teacher_stride = max(1, int(os.environ.get("EOH_TA_STRIDE", "4")))
        self.teacher_max_states = max(1, int(os.environ.get("EOH_TA_MAX_STATES", "18")))
        self.teacher_sample_instances = max(1, int(os.environ.get("EOH_TA_SAMPLE_INSTANCES", "4")))
        self.teacher_stage_stratified = os.environ.get("EOH_TA_STAGE_STRATIFIED", "0") == "1"
        self.teacher_lock = threading.Lock()
        self.rollout_workers = max(1, int(os.environ.get("EOH_INSTANCE_ROLLOUT_WORKERS", "1")))
        self.rollout_pool = None
        self.rollout_pool_lock = threading.Lock()
        self.teacher = self._build_teacher() if self.teacher_aware else None
        if self.teacher is not None:
            self.teacher._load()
        self.instance_data = load_instances(
            npy=os.environ.get("EOH_JSSP_TRAIN_NPY") or None,
            jsonl=os.environ.get("EOH_JSSP_TRAIN_JSONL") or None,
            n_jobs=self.n_jobs,
            n_machines=self.n_machines,
            count=self.n_instance,
            seed=self.seed,
        )
        self.baseline_guide = BASELINE_GUIDE.strip()
        self.baseline_summary = _baseline_summary(self.instance_data)


    def _get_rollout_pool(self):
        with self.rollout_pool_lock:
            if self.rollout_pool is None:
                self.rollout_pool = RolloutPool(self.rollout_workers)
            return self.rollout_pool

    def _greedy_metadata_parallel(self, code_string):
        pool = self._get_rollout_pool()
        results = pool.map(_jssp_rollout_worker, [
            (code_string, instance)
            for idx, instance in enumerate(self.instance_data[: self.n_instance])
        ])
        if any(result is None for result in results):
            return None

        makespans = [int(result["makespan"]) for result in results]
        runtime_stats = _empty_runtime_stats()
        for result in results:
            for key in runtime_stats:
                runtime_stats[key] += int(result["runtime_stats"].get(key, 0))

        score_requests = []
        value_scores = []
        top1_hits = []
        percentiles = []
        case_records = []
        chosen_trace_stats = _empty_trace_stats()
        teacher_trace_stats = _empty_trace_stats()
        if self.teacher is not None:
            with self.teacher_lock:
                for instance_idx, (instance, result) in enumerate(
                    zip(self.instance_data[: self.n_instance], results)
                ):
                    if instance_idx >= self.teacher_sample_instances:
                        break
                    analyzed = self._analyze_action_trace(instance, result["actions"])
                    if analyzed is None:
                        return None
                    records, rank_corrs, values, top1, pct, chosen_stats, teacher_stats = analyzed
                    case_records.extend(records)
                    score_requests.extend(rank_corrs)
                    value_scores.extend(values)
                    top1_hits.extend(top1)
                    percentiles.extend(pct)
                    _merge_trace_stats(chosen_trace_stats, chosen_stats)
                    _merge_trace_stats(teacher_trace_stats, teacher_stats)

        rank_correlations = (
            pool.map(_jssp_trace_scores_worker, [(code_string, score_requests)])[0]
            if score_requests else []
        )
        return _metadata(
            float(np.mean(makespans)),
            rank_correlations,
            value_scores,
            top1_hits,
            percentiles,
            case_records,
            chosen_trace_stats,
            teacher_trace_stats,
            runtime_stats=runtime_stats,
            baseline_guide=self.baseline_guide,
            baseline_summary=self.baseline_summary,
        )

    def _analyze_action_trace(self, instance, actions):
        state = ScheduleState(instance)
        teacher_state = self.teacher.reset_instance(instance)
        records = []
        rank_correlations = []
        value_scores = []
        top1_hits = []
        percentiles = []
        chosen_trace_stats = _empty_trace_stats()
        teacher_trace_stats = _empty_trace_stats()
        sample_steps = (
            _stratified_sample_steps(instance.n_operations, self.teacher_max_states)
            if self.teacher_stage_stratified
            else set(range(0, instance.n_operations, self.teacher_stride))
        )
        sampled = 0
        for action in actions:
            features = state.candidate_features()
            feature_by_op = {feature.op_id: feature for feature in features}
            action = int(action)
            if action not in feature_by_op:
                return None
            if sampled < self.teacher_max_states and len(state.actions) in sample_steps:
                pref = self.teacher.preference(teacher_state)
                candidate_ops = tuple(op for op in pref.candidate_ops if op in feature_by_op)
                if len(candidate_ops) >= 2:
                    scores_by_op = {
                        int(op): float(score)
                        for op, score in zip(pref.candidate_ops, pref.teacher_scores)
                    }
                    teacher_action = (
                        int(pref.teacher_action)
                        if int(pref.teacher_action) in candidate_ops
                        else max(candidate_ops, key=lambda op: (scores_by_op[int(op)], -int(op)))
                    )
                    rank_correlations.append((candidate_ops, feature_by_op, copy.deepcopy(state),
                                              [scores_by_op[int(op)] for op in candidate_ops]))
                    value_scores.append(_normalize_value(scores_by_op, action))
                    top1_hits.append(float(action == teacher_action))
                    percentiles.append(_rank_percentile(candidate_ops, scores_by_op, action))
                    stage = _stage_name(len(state.actions), instance.n_operations)
                    _record_trace_offset(chosen_trace_stats, stage, feature_by_op[action], features)
                    _record_trace_offset(
                        teacher_trace_stats,
                        stage,
                        feature_by_op[teacher_action],
                        features,
                    )
                    if action != teacher_action:
                        records.append(
                            _case_record(
                                state,
                                feature_by_op[action],
                                feature_by_op[teacher_action],
                                scores_by_op,
                            )
                        )
                    sampled += 1
            state.dispatch(action)
            teacher_state, _reward, terminated, truncated, _info = self.teacher.step(action)
            if bool(terminated or truncated) and not state.done:
                return None
        if not state.done:
            return None
        chosen_trace_stats["makespans"].append(float(state.makespan))
        return (
            records,
            rank_correlations,
            value_scores,
            top1_hits,
            percentiles,
            chosen_trace_stats,
            teacher_trace_stats,
        )

    def evaluate(self, code_string):
        result = self.evaluate_metadata(code_string)
        return None if result is None else result["objective"]

    @bounded_evaluation
    def evaluate_metadata(self, code_string):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._greedy_metadata_parallel(code_string)


    def _teacher_greedy_trace(self, instance: JSSPInstance) -> dict[str, object]:
        trace = _empty_trace_stats()
        state = self.teacher.reset_instance(instance)
        mirror = ScheduleState(instance)
        sampled = 0
        sample_steps = (
            _stratified_sample_steps(instance.n_operations, self.teacher_max_states)
            if self.teacher_stage_stratified
            else None
        )
        done = False
        while not done and not mirror.done:
            features = mirror.candidate_features()
            feature_by_op = {feature.op_id: feature for feature in features}
            if not features:
                break
            preference = self.teacher.preference(state) if len(features) >= 2 else None
            teacher_action = (
                int(preference.teacher_action)
                if preference is not None and int(preference.teacher_action) in feature_by_op
                else int(features[0].op_id)
            )
            teacher_feature = feature_by_op.get(teacher_action)
            if teacher_feature is None:
                break
            should_sample = (
                len(features) >= 2
                and sampled < self.teacher_max_states
                and (
                    len(mirror.actions) in sample_steps
                    if sample_steps is not None
                    else True
                )
            )
            if should_sample:
                _record_trace_offset(
                    trace,
                    _stage_name(len(mirror.actions), instance.n_operations),
                    teacher_feature,
                    features,
                )
                sampled += 1
            mirror.dispatch(teacher_action)
            state, _reward, terminated, truncated, _info = self.teacher.step(teacher_action)
            done = bool(terminated or truncated)
        trace["makespans"].append(float(mirror.makespan))
        return trace

    def _build_teacher(self):
        checkpoint_dir = os.environ.get("EOH_JSSP_TEACHER_CHECKPOINT_DIR", "").strip()
        offline_root = os.environ.get("EOH_JSSP_OFFLINE_LD_ROOT", "third_party/Offline-LD")
        checkpoint_step = _parse_step(os.environ.get("EOH_JSSP_TEACHER_STEP", "best"))
        device = os.environ.get("EOH_TA_DEVICE", "cpu")
        if checkpoint_dir:
            return OfflineLDTeacher(
                offline_root,
                checkpoint_dir,
                checkpoint_step=checkpoint_step,
                device=device,
            )
        return OfflineLDTeacher.from_checkpoint_root(
            offline_root,
            os.environ.get("EOH_JSSP_TEACHER_CHECKPOINT_ROOT", "results/offline_ld_full/checkpoints_100"),
            self.n_jobs,
            self.n_machines,
            checkpoint_step=checkpoint_step,
            device=device,
        )


def _jssp_trace_scores_worker(code_string, requests):
    module = types.ModuleType("jssp_diagnostic_worker")
    exec(code_string, module.__dict__)
    return [spearman_correlation(_rule_scores(module, ops, features, state), teacher_scores)
            for ops, features, state, teacher_scores in requests]


def _jssp_rollout_worker(code_string, instance):
    module = types.ModuleType(f"jssp_heuristic_worker_{os.getpid()}")
    exec(code_string, module.__dict__)
    if not hasattr(module, "score") and not hasattr(module, "select_operation"):
        return None
    state = ScheduleState(instance)
    runtime_stats = _empty_runtime_stats()
    while not state.done:
        features = state.candidate_features()
        action = _call_select(module, features, state, runtime_stats=runtime_stats)
        if action is None or int(action) not in {feature.op_id for feature in features}:
            return None
        state.dispatch(int(action))
    return {
        "makespan": int(state.makespan),
        "actions": tuple(int(action) for action in state.actions),
        "runtime_stats": runtime_stats,
    }


def _call_select(
    eva,
    features: list[CandidateFeature],
    state: ScheduleState,
    *,
    runtime_stats=None,
) -> int | None:
    start_ns = time.thread_time_ns()
    if hasattr(eva, "score"):
        try:
            scored = [(float(eva.score(feature, state)), -feature.op_id, feature.op_id) for feature in features]
            action = int(max(scored)[2])
        except Exception:
            return None
    else:
        try:
            action = int(eva.select_operation(features, state))
        except Exception:
            return None
    elapsed_ns = time.thread_time_ns() - start_ns
    if runtime_stats is not None:
        runtime_stats["cpu_ns"] += int(elapsed_ns)
        runtime_stats["decisions"] += 1
        runtime_stats["score_calls"] += len(features)
    return action


def _rule_scores(
    eva,
    candidate_ops: tuple[int, ...],
    feature_by_op: dict[int, CandidateFeature],
    state: ScheduleState,
) -> list[float]:
    try:
        return [float(eva.score(feature_by_op[int(op)], state)) for op in candidate_ops]
    except Exception:
        return [-1.0 for _op in candidate_ops]


def _metadata(
    objective: float,
    rank_correlations,
    value_scores,
    top1_hits,
    percentiles,
    case_records,
    chosen_trace_stats: dict[str, object],
    teacher_trace_stats: dict[str, object],
    *,
    runtime_stats: dict[str, int] | None = None,
    baseline_guide: str = "",
    baseline_summary: str = "",
) -> dict[str, object]:
    top1 = float(np.mean(top1_hits)) if top1_hits else 0.0
    mean_rank_corr = float(np.mean(rank_correlations)) if rank_correlations else 0.0
    teacher_value = float(np.mean(value_scores)) if value_scores else 0.0
    teacher_pct = float(np.mean(percentiles)) if percentiles else 0.0
    alignment_score, alignment_metric = _select_alignment_score(
        top1=top1,
        teacher_value=teacher_value,
        teacher_percentile=teacher_pct,
    )
    views = _teacher_views(case_records)
    chosen_trace_offsets = _finalize_trace_offsets(chosen_trace_stats)
    teacher_trace_offsets = _finalize_trace_offsets(teacher_trace_stats)
    trace_summary = _format_trace_summary("candidate", chosen_trace_stats)
    teacher_trace_summary = _format_trace_summary("teacher_on_candidate_states", teacher_trace_stats)
    trace_delta_summary = _format_trace_delta(
        "candidate - teacher on paired candidate states",
        chosen_trace_offsets,
        teacher_trace_offsets,
    )
    runtime_stats = runtime_stats or _empty_runtime_stats()
    runtime_per_decision_us = (
        float(runtime_stats["cpu_ns"]) / max(1, int(runtime_stats["decisions"])) / 1_000.0
    )
    runtime_per_score_call_us = (
        float(runtime_stats["cpu_ns"]) / max(1, int(runtime_stats["score_calls"])) / 1_000.0
    )
    return {
        "objective": objective,
        "alignment_score": alignment_score,
        "rank_corr": mean_rank_corr,
        "mean_rank_correlation": mean_rank_corr,
        "teacher_value_score": teacher_value,
        "top1_agreement": top1,
        "teacher_percentile": teacher_pct,
        "alignment_metric": alignment_metric,
        "alignment_states": len(top1_hits),
        "runtime_per_decision_us": runtime_per_decision_us,
        "runtime_per_score_call_us": runtime_per_score_call_us,
        "runtime_decisions": int(runtime_stats["decisions"]),
        "runtime_score_calls": int(runtime_stats["score_calls"]),
        "case_summary": views["structure_cases"],
        "stage_summary": views["stage_summary"],
        "structure_cases": views["structure_cases"],
        "parameter_cases": views["parameter_cases"],
        "structure_guidance": views["structure_guidance"],
        "parameter_guidance": views["parameter_guidance"],
        "feature_preference_summary": views["feature_preference_summary"],
        "baseline_guide": baseline_guide.strip(),
        "baseline_summary": baseline_summary.strip(),
        "trace_summary": trace_summary,
        "teacher_trace_summary": teacher_trace_summary,
        "trace_delta_summary": trace_delta_summary,
        "trace_stage_offsets": chosen_trace_offsets,
        "teacher_trace_stage_offsets": teacher_trace_offsets,
        "teacher_trace_semantics": "paired_candidate_on_policy_states",
        "diagnostic": (
            f"objective={objective:.3f}; align_metric={alignment_metric}; align={alignment_score:.3f}; "
            f"rank_corr={mean_rank_corr:.3f}; "
            f"teacher_value={teacher_value:.3f}; top1={top1:.3f}; "
            f"sampled_states={len(top1_hits)}; "
            f"runtime_per_decision_us={runtime_per_decision_us:.3f}"
        ),
    }


def _empty_runtime_stats() -> dict[str, int]:
    return {"cpu_ns": 0, "decisions": 0, "score_calls": 0}


def _select_alignment_score(*, top1: float, teacher_value: float, teacher_percentile: float) -> tuple[float, str]:
    metric = os.environ.get("EOH_ALIGN_METRIC", "top1").strip().lower()
    aliases = {
        "top1": "top1",
        "top-1": "top1",
        "agreement": "top1",
        "value": "value",
        "teacher_value": "value",
        "normalized_value": "value",
        "percentile": "percentile",
        "rank_percentile": "percentile",
    }
    metric = aliases.get(metric, "top1")
    if metric == "value":
        return teacher_value, metric
    if metric == "percentile":
        return teacher_percentile, metric
    return top1, "top1"


def _baseline_summary(instances: list[JSSPInstance]) -> str:
    rows = []
    for name, rule in RULE_REGISTRY.items():
        result = evaluate_rule(instances, rule)
        rows.append((name, float(result.mean_makespan), int(result.best_makespan), int(result.worst_makespan)))
    rows.sort(key=lambda item: item[1])
    return "\n".join(
        f"- {name}: mean={mean_makespan:.3f}, best={best}, worst={worst}"
        for name, mean_makespan, best, worst in rows
    )


def _case_record(
    state: ScheduleState,
    chosen: CandidateFeature,
    teacher_best: CandidateFeature,
    scores_by_op: dict[int, float],
) -> dict[str, object]:
    best_score = scores_by_op.get(int(teacher_best.op_id), 0.0)
    chosen_score = scores_by_op.get(int(chosen.op_id), min(scores_by_op.values()) if scores_by_op else 0.0)
    return {
        "stage": _stage_name(len(state.actions), state.instance.n_operations),
        "remaining_fraction": 1.0 - len(state.actions) / max(1, state.instance.n_operations),
        "feasible_count": len(state.candidate_ops()),
        "current": len(state.actions),
        "teacher_score_gap": float(best_score - chosen_score),
        "chosen_action": int(chosen.op_id),
        "teacher_action": int(teacher_best.op_id),
        "chosen_profile": _profile(chosen),
        "teacher_profile": _profile(teacher_best),
        "chosen_pct": _profile_percentiles(chosen, state.candidate_features()),
        "teacher_pct": _profile_percentiles(teacher_best, state.candidate_features()),
    }


def _profile(feature: CandidateFeature) -> dict[str, float]:
    wait_for_machine = max(0.0, float(feature.machine_ready_time - feature.job_ready_time))
    return {
        "processing_time": float(feature.processing_time),
        "remaining_work": float(feature.remaining_work),
        "remaining_ops": float(feature.remaining_ops),
        "earliest_start": float(feature.earliest_start),
        "earliest_finish": float(feature.earliest_finish),
        "machine_queue_len": float(feature.machine_queue_len),
        "machine_total_work": float(feature.machine_total_work),
        "job_progress": float(feature.job_progress),
        "lower_bound_after": float(feature.lower_bound_after),
        "wait_for_machine": wait_for_machine,
    }


def _profile_percentiles(feature: CandidateFeature, features: list[CandidateFeature]) -> dict[str, float]:
    prof = _profile(feature)
    all_profiles = [_profile(item) for item in features]
    return {key: _percentile([row[key] for row in all_profiles], value) for key, value in prof.items()}


def _teacher_views(records: list[dict[str, object]]) -> dict[str, str]:
    if not records:
        empty = "No representative teacher disagreements were sampled."
        return {
            "stage_summary": "No stage-level teacher failures were summarized.",
            "structure_cases": empty,
            "parameter_cases": empty,
            "structure_guidance": "No structural recommendation was derived.",
            "parameter_guidance": "No parameter recommendation was derived.",
            "feature_preference_summary": "No teacher-feature preference mismatch was summarized.",
        }
    return {
        "stage_summary": _stage_summary(records),
        "structure_cases": _case_summary(records, max_cases=3),
        "parameter_cases": _case_summary(records, max_cases=2),
        "structure_guidance": _structure_guidance(records),
        "parameter_guidance": _parameter_guidance(records),
        "feature_preference_summary": _feature_preference_summary(records),
    }


def _case_summary(records: list[dict[str, object]], *, max_cases: int) -> str:
    rows = sorted(records, key=lambda item: float(item["teacher_score_gap"]), reverse=True)[:max_cases]
    lines = []
    for row in rows:
        lines.append(
            "stage={stage}, remaining_fraction={remaining_fraction:.2f}, feasible_count={feasible_count}, "
            "step={current}, teacher_score_gap={teacher_score_gap:+.3f}; delta={delta}; "
            "chosen[{chosen}] vs teacher[{teacher}]".format(
                stage=row["stage"],
                remaining_fraction=float(row["remaining_fraction"]),
                feasible_count=int(row["feasible_count"]),
                current=int(row["current"]),
                teacher_score_gap=float(row["teacher_score_gap"]),
                delta=_delta_text(row),
                chosen=_compact_profile(int(row["chosen_action"]), row["chosen_profile"], row["chosen_pct"]),
                teacher=_compact_profile(int(row["teacher_action"]), row["teacher_profile"], row["teacher_pct"]),
            )
        )
    return "\n".join(lines)


def _stage_summary(records: list[dict[str, object]]) -> str:
    lines = []
    for stage in ("early", "mid", "late"):
        rows = [row for row in records if row["stage"] == stage]
        if not rows:
            continue
        counts: dict[str, int] = {}
        for row in rows:
            for tag in _case_tags(row):
                counts[tag] = counts.get(tag, 0) + 1
        dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        lines.append(f"- {stage}: {_tag_sentence(dominant)}")
    return "\n".join(lines) if lines else "No stage-level teacher failures were summarized."


def _structure_guidance(records: list[dict[str, object]]) -> str:
    scores: dict[str, float] = {}
    for row in records:
        for tag in _case_tags(row):
            scores[tag] = scores.get(tag, 0.0) + float(row["teacher_score_gap"])
    dominant = max(scores.items(), key=lambda item: (item[1], item[0]))[0] if scores else "misc"
    return {
        "short_job_bias": "Replace a pure short-processing-time bias with a two-stage rule: shortlist quick feasible operations, then rerank by downstream remaining work and machine load.",
        "long_job_bias": "Add a gate that prevents long operations from monopolizing bottleneck machines unless they reduce lower-bound pressure.",
        "machine_congestion": "Introduce an explicit bottleneck-machine relief branch using machine_total_work and machine_queue_len before local processing-time tie-breaks.",
        "late_lb": "Add a late-stage lower-bound control branch that prioritizes operations reducing lower_bound_after over local earliest finish.",
        "wait_mismatch": "Change the ready-time gate so operations that cause large machine/job waiting are penalized before final tie-breaks.",
        "misc": "Preserve the best dispatch backbone, but replace one local tie-break with a bottleneck-aware reranking step.",
    }[dominant]


def _parameter_guidance(records: list[dict[str, object]]) -> str:
    lines = []
    for stage in ("early", "mid", "late"):
        rows = [row for row in records if row["stage"] == stage]
        if not rows:
            continue
        counts: dict[str, int] = {}
        for row in rows:
            for tag in _case_tags(row):
                counts[tag] = counts.get(tag, 0) + 1
        dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        lines.append(f"- {stage}: retune weights for {_tag_sentence(dominant)}")
    return "\n".join(lines) if lines else "No parameter recommendation was derived."


def _feature_preference_summary(records: list[dict[str, object]]) -> str:
    grouped = {"early": {}, "mid": {}, "late": {}}
    for row in records:
        stage = str(row["stage"])
        chosen_pct = row.get("chosen_pct") or {}
        teacher_pct = row.get("teacher_pct") or {}
        for feature in set(chosen_pct).intersection(teacher_pct):
            grouped.setdefault(stage, {}).setdefault(feature, []).append(
                float(chosen_pct[feature]) - float(teacher_pct[feature])
            )
    lines = []
    for stage in ("early", "mid", "late"):
        deltas = grouped.get(stage, {})
        if not deltas:
            lines.append(f"- {stage}: no teacher-feature preference mismatch was summarized.")
            continue
        ranked = sorted(
            ((feature, float(np.mean(values))) for feature, values in deltas.items() if values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        phrases = []
        for feature, delta in ranked:
            direction = "larger" if delta > 0 else "smaller"
            phrases.append(f"{direction} {feature} ({delta:+.2f} pct)")
        lines.append(
            f"- {stage}: relative to teacher, the current parent prefers " + ", ".join(phrases) + "."
        )
    return "\n".join(lines)


def _compact_profile(action_id: int, profile: dict[str, float], pct: dict[str, float]) -> str:
    keys = (
        "processing_time",
        "remaining_work",
        "earliest_finish",
        "machine_total_work",
        "job_progress",
        "lower_bound_after",
        "wait_for_machine",
    )
    body = ", ".join(f"{key}={profile[key]:.3f}|pct={pct[key]:.2f}" for key in keys)
    return f"op_id={action_id}, {body}"


def _case_tags(row: dict[str, object]) -> list[str]:
    chosen = row["chosen_profile"]
    teacher = row["teacher_profile"]
    tags = []
    if chosen["processing_time"] + 1e-9 < teacher["processing_time"] and chosen["remaining_work"] + 1e-9 < teacher["remaining_work"]:
        tags.append("short_job_bias")
    if chosen["processing_time"] > teacher["processing_time"] + 1e-9 and chosen["lower_bound_after"] > teacher["lower_bound_after"] + 1e-9:
        tags.append("long_job_bias")
    if chosen["machine_total_work"] + 1e-9 < teacher["machine_total_work"]:
        tags.append("machine_congestion")
    if chosen["lower_bound_after"] > teacher["lower_bound_after"] + 1e-9:
        tags.append("late_lb")
    if chosen["wait_for_machine"] > teacher["wait_for_machine"] + 1e-9:
        tags.append("wait_mismatch")
    return tags or ["misc"]


def _delta_text(row: dict[str, object]) -> str:
    return "; ".join(_tag_sentence(tag) for tag in _case_tags(row))


def _tag_sentence(tag: str) -> str:
    return {
        "short_job_bias": "chosen is too biased toward short/low-remaining-work jobs",
        "long_job_bias": "chosen accepts a long operation without enough lower-bound benefit",
        "machine_congestion": "chosen under-reacts to bottleneck machine congestion",
        "late_lb": "chosen leaves a worse lower-bound-after state",
        "wait_mismatch": "chosen creates more waiting or readiness mismatch",
        "misc": "chosen differs from teacher without one dominant simple pattern",
    }[tag]


def _normalize_value(scores_by_op: dict[int, float], chosen_op: int) -> float:
    if not scores_by_op or chosen_op not in scores_by_op:
        return 0.0
    lo = min(scores_by_op.values())
    hi = max(scores_by_op.values())
    if hi - lo <= 1e-12:
        return 1.0
    return float((scores_by_op[chosen_op] - lo) / (hi - lo))


def _rank_percentile(candidate_ops: tuple[int, ...], scores_by_op: dict[int, float], chosen_op: int) -> float:
    ordered = sorted(candidate_ops, key=lambda op: (scores_by_op.get(op, -np.inf), -op), reverse=True)
    for rank, op in enumerate(ordered):
        if int(op) == int(chosen_op):
            return 1.0 if len(ordered) <= 1 else 1.0 - rank / (len(ordered) - 1)
    return 0.0


def _percentile(values: list[float], value: float) -> float:
    if len(values) <= 1:
        return 1.0
    ordered = sorted(values)
    pos = 0
    for idx, item in enumerate(ordered):
        if item <= value:
            pos = idx
    return pos / (len(ordered) - 1)


def _stage_name(step: int, total_steps: int) -> str:
    frac = step / max(1, total_steps)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def _stratified_sample_steps(total_steps: int, max_states: int) -> set[int]:
    """Place a fixed query budget uniformly across normalized trajectory time."""
    total = max(1, int(total_steps))
    count = min(total, max(1, int(max_states)))
    return {((2 * index + 1) * total) // (2 * count) for index in range(count)}


def _parse_step(value: str) -> int | str:
    return int(value) if str(value).isdigit() else value


def _empty_trace_stats() -> dict[str, object]:
    return {
        "counts": {"early": 0, "mid": 0, "late": 0},
        "offsets": {
            "early": {name: [] for name in _TRACE_FEATURES},
            "mid": {name: [] for name in _TRACE_FEATURES},
            "late": {name: [] for name in _TRACE_FEATURES},
        },
        "makespans": [],
    }


def _record_trace_offset(stats: dict[str, object], stage: str, chosen: CandidateFeature, feasible: list[CandidateFeature]) -> None:
    chosen_profile = _profile(chosen)
    feasible_profiles = [_profile(item) for item in feasible]
    counts = stats["counts"]
    offsets = stats["offsets"]
    counts[stage] += 1
    for feature in _TRACE_FEATURES:
        avg = float(np.mean([row[feature] for row in feasible_profiles]))
        offsets[stage][feature].append(float(chosen_profile[feature] - avg))


def _merge_trace_stats(dst: dict[str, object], src: dict[str, object]) -> None:
    for stage in ("early", "mid", "late"):
        dst["counts"][stage] += int(src["counts"][stage])
        for feature in _TRACE_FEATURES:
            dst["offsets"][stage][feature].extend(src["offsets"][stage][feature])
    dst["makespans"].extend(src["makespans"])


def _finalize_trace_offsets(stats: dict[str, object]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for stage in ("early", "mid", "late"):
        stage_map: dict[str, float] = {}
        for feature in _TRACE_FEATURES:
            values = stats["offsets"][stage][feature]
            if values:
                stage_map[feature] = float(np.mean(values))
        result[stage] = stage_map
    return result


def _format_trace_summary(label: str, stats: dict[str, object]) -> str:
    counts = stats["counts"]
    offsets = _finalize_trace_offsets(stats)
    if not any(int(counts[stage]) for stage in ("early", "mid", "late")):
        return f"{label}: no sampled non-terminal decision states."
    sampled_states = sum(int(counts[stage]) for stage in ("early", "mid", "late"))
    if stats["makespans"]:
        mean_makespan = float(np.mean(stats["makespans"]))
        header = (
            f"{label}: sampled_instances={len(stats['makespans'])}, "
            f"sampled_states={sampled_states}, mean_trace_makespan={mean_makespan:.3f}"
        )
    else:
        header = f"{label}: sampled_states={sampled_states}"
    lines = [header]
    for stage in ("early", "mid", "late"):
        rows = sorted(offsets[stage].items(), key=lambda item: abs(item[1]), reverse=True)[:6]
        if not rows:
            lines.append(f"- {stage}: no sampled states")
            continue
        text = ", ".join(f"{name}={value:+.3f}" for name, value in rows)
        lines.append(f"- {stage} ({counts[stage]} states): {text}")
    return "\n".join(lines)


def _format_trace_delta(label: str, left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> str:
    lines = [label + ":"]
    wrote = False
    for stage in ("early", "mid", "late"):
        rows = []
        for feature in _TRACE_FEATURES:
            left_avg = left.get(stage, {}).get(feature)
            right_avg = right.get(stage, {}).get(feature)
            if left_avg is None or right_avg is None:
                continue
            delta = float(left_avg - right_avg)
            rows.append((abs(delta), feature, delta))
        rows.sort(reverse=True)
        if rows:
            lines.append("- {stage}: {text}".format(
                stage=stage,
                text=", ".join(f"{feature}={delta:+.3f}" for _abs, feature, delta in rows[:5]),
            ))
            wrote = True
    if not wrote:
        lines.append("- no overlapping sampled stages")
    return "\n".join(lines)
