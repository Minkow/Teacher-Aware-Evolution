from __future__ import annotations

import os
import signal
import sys
import threading
import time
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tae.engine.evaluation_pool import RolloutPool, bounded_evaluation

from .prompts import GetPrompts
from tae.analysis.population import (
    build_teacher_views,
    normalize_teacher_value,
    percentile_profile,
    rank_percentile,
    summarize_alignment_metrics,
    summarize_case_records,
)
from tae.tasks.maxcut.core import MaxCutInstance, MaxCutState
from tae.tasks.maxcut.data import (
    load_eco_graph_set,
    load_maxcut_bench_directory,
    load_maxcut_bench_mixed_split,
    load_maxcut_bench_split,
)
from tae.tasks.maxcut.eco_teacher import EcoDQNTeacher


class MAXCUTFLIP:
    def __init__(self) -> None:
        self.prompts = GetPrompts()
        self.n_instance = int(os.environ.get("EOH_MAXCUT_N_INSTANCE", "8"))
        self.n_nodes = int(os.environ.get("EOH_MAXCUT_N_NODES", "40"))
        self.max_steps = int(os.environ.get("EOH_MAXCUT_MAX_STEPS", str(2 * self.n_nodes)))
        self.attempts = max(1, int(os.environ.get("EOH_MAXCUT_ATTEMPTS", "1")))
        self.seed = int(os.environ.get("EOH_MAXCUT_SEED", "0"))
        self.distribution = os.environ.get("EOH_MAXCUT_DISTRIBUTION", "er").strip().lower()
        self.teacher_aware = os.environ.get("EOH_TEACHER_AWARE", "0") == "1"
        self.teacher_stride = max(1, int(os.environ.get("EOH_TA_STRIDE", "4")))
        self.teacher_max_states = max(1, int(os.environ.get("EOH_TA_MAX_STATES", "18")))
        self.teacher_sample_instances = max(1, int(os.environ.get("EOH_TA_SAMPLE_INSTANCES", "4")))
        self.teacher_device = os.environ.get("EOH_TA_DEVICE", "cpu")
        self.interface = os.environ.get("EOH_MAXCUT_INTERFACE", "action").strip().lower()
        self.bench_distribution = os.environ.get("EOH_MAXCUT_BENCH_DISTRIBUTION", "").strip()
        self.bench_distributions = [
            item.strip()
            for item in os.environ.get("EOH_MAXCUT_BENCH_DISTRIBUTIONS", "").split(",")
            if item.strip()
        ]
        self.eco_root = os.environ.get("EOH_ECO_ROOT", "").strip()
        self.eco_ckpt = os.environ.get("EOH_ECO_CKPT", "").strip()
        self.teacher_cache: dict[str, EcoDQNTeacher] = {}
        self.teacher_lock = threading.RLock()
        self.rollout_workers = max(1, int(os.environ.get("EOH_MAXCUT_ROLLOUT_WORKERS", "1")))
        self.rollout_timeout_s = max(
            1,
            int(os.environ.get("EOH_MAXCUT_ROLLOUT_TIMEOUT_S", "120")),
        )
        self.rollout_pool = None
        self.rollout_pool_lock = threading.Lock()
        self.instance_data = self._load_instances()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["teacher_cache"] = {}
        state["rollout_pool"] = None
        state["rollout_pool_lock"] = None
        state["teacher_lock"] = None
        return state

    def _load_instances(self) -> list[MaxCutInstance]:
        bench_dir = os.environ.get("EOH_MAXCUT_BENCH_DIR", "").strip()
        if bench_dir:
            return load_maxcut_bench_directory(bench_dir, limit=self.n_instance)
        bench_root = os.environ.get("EOH_MAXCUT_BENCH_ROOT", "").strip()
        if bench_root and self.bench_distributions:
            bench_split = os.environ.get("EOH_MAXCUT_BENCH_SPLIT", "training").strip()
            return load_maxcut_bench_mixed_split(
                bench_root,
                split=bench_split,
                distributions=self.bench_distributions,
                limit=self.n_instance,
            )
        if bench_root and self.bench_distribution:
            bench_split = os.environ.get("EOH_MAXCUT_BENCH_SPLIT", "training").strip()
            return load_maxcut_bench_split(
                bench_root,
                split=bench_split,
                distribution=self.bench_distribution,
                limit=self.n_instance,
            )
        graph_set = os.environ.get("EOH_MAXCUT_GRAPH_SET", "").strip()
        if graph_set:
            return load_eco_graph_set(graph_set)[: self.n_instance]
        raise ValueError("Max-Cut training requires a fixed graph dataset; set EOH_MAXCUT_BENCH_DIR")

    def _teacher_for_instance(self, instance: MaxCutInstance):
        if not self.teacher_aware:
            return None
        dist_name = self._distribution_for_instance(instance)
        with self.teacher_lock:
            if dist_name in self.teacher_cache:
                return self.teacher_cache[dist_name]
            if not self.eco_root:
                raise ValueError("ECO-DQN teacher requires EOH_ECO_ROOT")
            if self.bench_distributions:
                checkpoint = str(Path(self.eco_root) / "pretrained agents" / dist_name / "network_best.pth")
            else:
                checkpoint = self.eco_ckpt.strip()
                if not checkpoint and dist_name:
                    checkpoint = str(Path(self.eco_root) / "pretrained agents" / dist_name / "network_best.pth")
            if not checkpoint:
                raise ValueError("ECO-DQN teacher requires EOH_ECO_CKPT or mixed-distribution auto checkpoints")
            teacher = EcoDQNTeacher(eco_root=self.eco_root, checkpoint=checkpoint, device=self.teacher_device)
            self.teacher_cache[dist_name] = teacher
            return teacher

    def _distribution_for_instance(self, instance: MaxCutInstance) -> str:
        name = getattr(instance, "name", "")
        if "::" in name:
            return name.split("::", 1)[0].strip()
        return self.bench_distribution

    def evaluate(self, code_string):
        result = self.evaluate_metadata(code_string)
        if result is None:
            return None
        return result["objective"]

    @bounded_evaluation
    def evaluate_metadata(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._evaluate_code_parallel(code_string)
        except TimeoutError:
            raise
        except Exception:
            return None

    def _get_rollout_pool(self):
        with self.rollout_pool_lock:
            if self.rollout_pool is None:
                self.rollout_pool = RolloutPool(self.rollout_workers)
            return self.rollout_pool

    def _evaluate_code_parallel(self, code_string):
        pool = self._get_rollout_pool()
        results = pool.map(_evaluate_instance_worker, [
            (
                code_string,
                instance,
                self.max_steps,
                self.attempts,
                self.seed,
                instance_idx,
                bool(self.teacher_aware and instance_idx < self.teacher_sample_instances),
                self.teacher_max_states,
                self.teacher_stride,
                os.environ.get("EOH_TA_STAGE_STRATIFIED", "0") == "1",
                self.rollout_timeout_s,
            )
            for instance_idx, instance in enumerate(self.instance_data)
        ], task_timeout=self.rollout_timeout_s)
        best_cuts = [float("-inf")] * len(self.instance_data)
        records: list[dict[str, object]] = []
        runtime_cpu_ns = 0
        runtime_decisions = 0
        # Teacher requests and summaries follow input order, not completion order.
        for instance_idx, (instance, result) in enumerate(zip(self.instance_data, results)):
            best_cuts[instance_idx] = float(result["best_cut"])
            runtime_cpu_ns += int(result["runtime_cpu_ns"])
            runtime_decisions += int(result["runtime_decisions"])
            samples = result.get("teacher_samples") or []
            if samples:
                teacher = self._teacher_for_instance(instance)
                records.extend(self._analyze_teacher_samples(instance, samples, teacher))
        if any(not np.isfinite(value) for value in best_cuts):
            raise ValueError("parallel Max-Cut rollout returned a non-finite result")
        return self._summarize(
            best_cuts,
            records,
            runtime_cpu_ns=runtime_cpu_ns,
            runtime_decisions=runtime_decisions,
        )

    def _analyze_teacher_samples(self, instance, samples, teacher):
        states = [_restore_state(instance, sample["state"], self.max_steps) for sample in samples]
        with self.teacher_lock:
            preferences = teacher.preference_batch(states)
        return [
            _teacher_record(
                state,
                int(sample["action"]),
                sample.get("student_scores"),
                preference,
            )
            for state, sample, preference in zip(states, samples, preferences)
        ]

    def _summarize(self, best_cuts, records, *, runtime_cpu_ns, runtime_decisions):
        value_scores = [float(record["teacher_value"]) for record in records]
        top1_hits = [float(record["top1"]) for record in records]
        percentiles = [float(record["teacher_percentile"]) for record in records]
        rank_correlations = [
            float(record["rank_corr"])
            for record in records
            if record.get("rank_corr") is not None
        ]
        case_records = [record["case"] for record in records if record.get("case") is not None]
        mean_cut = float(np.mean(best_cuts)) if best_cuts else 0.0
        objective = -mean_cut
        runtime_per_decision_us = (
            float(runtime_cpu_ns) / max(1, runtime_decisions) / 1_000.0
        )
        metrics = summarize_alignment_metrics(value_scores, top1_hits, percentiles)
        metrics["rank_corr"] = float(np.mean(rank_correlations)) if rank_correlations else 0.0
        teacher_views = build_teacher_views(case_records)
        diagnostic = (
            f"mean_cut={mean_cut:.6f}; objective={objective:.6f}; "
            f"teacher_value={metrics['teacher_value_score']:.3f}; "
            f"top1={metrics['top1_agreement']:.3f}; sampled_cases={len(case_records)}; "
            f"runtime_per_decision_us={runtime_per_decision_us:.3f}"
        )
        return {
            "objective": objective,
            "runtime_per_decision_us": runtime_per_decision_us,
            "runtime_decisions": runtime_decisions,
            "alignment_score": metrics["alignment_score"],
            "teacher_value_score": metrics["teacher_value_score"],
            "top1_agreement": metrics["top1_agreement"],
            "teacher_percentile": metrics["teacher_percentile"],
            "rank_corr": metrics.get("rank_corr", 0.0),
            "mean_rank_correlation": metrics.get("rank_corr", 0.0),
            "case_summary": summarize_case_records(case_records),
            "stage_summary": teacher_views["stage_summary"],
            "structure_cases": teacher_views["structure_cases"],
            "parameter_cases": teacher_views["parameter_cases"],
            "structure_guidance": teacher_views["structure_guidance"],
            "parameter_guidance": teacher_views["parameter_guidance"],
            "feature_preference_summary": teacher_views["feature_preference_summary"],
            "diagnostic": diagnostic,
        }


def _evaluate_instance_worker(
    code_string,
    instance,
    max_steps,
    attempts,
    seed,
    instance_idx,
    collect_teacher,
    teacher_max_states,
    teacher_stride,
    stage_stratified,
    timeout_s,
):
    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"Max-Cut instance rollout exceeded {timeout_s}s")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_s))
    try:
        heuristic_module = types.ModuleType(
            f"maxcut_heuristic_worker_{os.getpid()}_{instance_idx}"
        )
        exec(code_string, heuristic_module.__dict__)
        if not hasattr(heuristic_module, "select_flip_node") and not hasattr(heuristic_module, "score"):
            raise ValueError("heuristic is missing select_flip_node/score")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(instance_idx)]))
        best_cut = float("-inf")
        teacher_samples = []
        runtime_cpu_ns = 0
        runtime_decisions = 0
        sample_steps = _teacher_sample_steps(
            max_steps,
            teacher_max_states,
            teacher_stride,
            stage_stratified,
        )
        for attempt in range(attempts):
            init_spins = None if attempt == 0 else rng.integers(0, 2, size=instance.n_nodes)
            state = MaxCutState(instance, max_steps=max_steps, init_spins=init_spins)
            while not state.done:
                start_ns = time.thread_time_ns()
                action, student_scores = select_action_from_module(heuristic_module, state)
                runtime_cpu_ns += time.thread_time_ns() - start_ns
                runtime_decisions += 1
                if action < 0 or action >= state.n_nodes:
                    raise ValueError(f"invalid flip action {action}")
                if collect_teacher and attempt == 0 and state.step in sample_steps:
                    teacher_samples.append(
                        {
                            "state": _snapshot_state(state),
                            "action": int(action),
                            "student_scores": student_scores,
                        }
                    )
                state.dispatch(action)
            best_cut = max(best_cut, float(state.best_cut))
        return {
            "best_cut": best_cut,
            "teacher_samples": teacher_samples,
            "runtime_cpu_ns": runtime_cpu_ns,
            "runtime_decisions": runtime_decisions,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _teacher_sample_steps(max_steps, max_states, stride, stage_stratified):
    if max_steps <= 0 or max_states <= 0:
        return set()
    if stage_stratified:
        count = min(int(max_states), int(max_steps))
        return set(int(value) for value in np.linspace(0, max_steps - 1, count))
    return set(range(0, min(max_steps, max_states * stride), max(1, stride)))


def _snapshot_state(state):
    return {
        "step": int(state.step),
        "spins": tuple(int(value) for value in state.spins),
        "actions": tuple(int(value) for value in state.actions),
        "current_cut": float(state.current_cut),
        "best_cut": float(state.best_cut),
        "best_spins": tuple(int(value) for value in state.best_spins),
    }


def _restore_state(instance, snapshot, max_steps):
    state = MaxCutState(instance, max_steps=max_steps, init_spins=snapshot["spins"])
    state.step = int(snapshot["step"])
    state.actions = [int(value) for value in snapshot["actions"]]
    state.current_cut = float(snapshot["current_cut"])
    state.best_cut = float(snapshot["best_cut"])
    state.best_spins = np.asarray(snapshot["best_spins"], dtype=np.int8)
    state._last_flip_steps.fill(-1)
    for step, action in enumerate(state.actions):
        state._last_flip_steps[int(action)] = step
    state._flip_gains = state._compute_flip_gains()
    return state


def _teacher_record(state, action, student_scores, pref):
    value = normalize_teacher_value(pref.teacher_scores, pref.candidate_actions, action)
    top1 = float(action == pref.teacher_action)
    pct = rank_percentile(pref.teacher_scores, pref.candidate_actions, action)
    rank_corr = None
    if student_scores is not None:
        rank_corr = spearman_correlation(
            [float(student_scores[int(node)]) for node in pref.candidate_actions],
            [float(score) for score in pref.teacher_scores],
        )
    case = None
    if action != pref.teacher_action:
        profiles = maxcut_candidate_profiles(state)
        percentile_maps = {
            node: {
                key: percentile_profile([candidate[key] for candidate in profiles.values()], item)
                for key, item in profile.items()
            }
            for node, profile in profiles.items()
        }
        score_by_node = {
            int(node): float(score)
            for node, score in zip(pref.candidate_actions, pref.teacher_scores)
        }
        teacher_ranked = sorted(
            score_by_node.items(),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        chosen_teacher_rank = next(
            idx + 1
            for idx, (node, _score) in enumerate(teacher_ranked)
            if int(node) == int(action)
        )
        top1_score = float(teacher_ranked[0][1])
        top2_score = float(teacher_ranked[1][1]) if len(teacher_ranked) > 1 else top1_score
        score_values = list(score_by_node.values())
        case = {
            "stage": _stage_from_step(state.step, state.max_steps),
            "remaining_fraction": float(1.0 - state.step / max(1, state.max_steps)),
            "feasible_count": int(len(pref.candidate_actions)),
            "current": int(action),
            "teacher_score_gap": float(
                score_by_node[int(pref.teacher_action)]
                - score_by_node.get(action, min(score_by_node.values()))
            ),
            "teacher_top1_top2_margin": float(top1_score - top2_score),
            "teacher_score_spread": float(max(score_values) - min(score_values)) if score_values else 0.0,
            "chosen_teacher_rank": int(chosen_teacher_rank),
            "teacher_top_actions": [
                {
                    "action": int(node),
                    "score_delta_from_top": float(top1_score - score),
                    "flip_gain": float(profiles[int(node)]["flip_gain"]),
                    "weighted_degree": float(profiles[int(node)]["weighted_degree"]),
                    "after_top5_gain_mean": float(profiles[int(node)]["after_top5_gain_mean"]),
                }
                for node, score in teacher_ranked[:5]
            ],
            "chosen_action": int(action),
            "teacher_action": int(pref.teacher_action),
            "chosen_profile": profiles[int(action)],
            "teacher_profile": profiles[int(pref.teacher_action)],
            "chosen_pct": percentile_maps[int(action)],
            "teacher_pct": percentile_maps[int(pref.teacher_action)],
            "chosen_semantics": maxcut_candidate_semantics(
                state, int(action), profiles[int(action)], percentile_maps[int(action)]
            ),
            "teacher_semantics": maxcut_candidate_semantics(
                state,
                int(pref.teacher_action),
                profiles[int(pref.teacher_action)],
                percentile_maps[int(pref.teacher_action)],
            ),
        }
    return {
        "teacher_value": value,
        "top1": top1,
        "teacher_percentile": pct,
        "rank_corr": rank_corr,
        "case": case,
    }


def select_action_from_module(eva, state: MaxCutState) -> tuple[int, dict[int, float] | None]:
    if hasattr(eva, "score"):
        decision_state = maxcut_decision_state(state)
        scores: dict[int, float] = {}
        for feature in decision_state.candidate_features:
            value = float(eva.score(feature, decision_state))
            scores[int(feature.node_id)] = value
        action = max(scores, key=lambda node: (scores[node], -node))
        return int(action), scores
    action = int(
        eva.select_flip_node(
            state.spins.copy(),
            state.adjacency.copy(),
            float(state.current_cut),
            float(state.best_cut),
            int(state.step),
            int(state.max_steps),
            list(state.actions),
        )
    )
    return action, None


def maxcut_decision_state(state: MaxCutState):
    raw_features = state.candidate_features()
    profiles = maxcut_candidate_profiles(state, raw_features)
    features = tuple(maxcut_score_feature(state, feature, profiles[int(feature.action)]) for feature in raw_features)
    return SimpleNamespace(
        n_nodes=int(state.n_nodes),
        step=int(state.step),
        max_steps=int(state.max_steps),
        current_cut=float(state.current_cut),
        best_cut=float(state.best_cut),
        spins=tuple(int(x) for x in state.spins),
        adjacency=tuple(tuple(float(x) for x in row) for row in state.adjacency),
        flip_history=tuple(int(x) for x in state.actions),
        candidate_features=features,
        features_by_action={int(feature.node_id): feature for feature in features},
    )


def maxcut_score_feature(state: MaxCutState, feature, profile: dict[str, float] | None = None):
    n = max(1, int(state.n_nodes))
    side1 = float(feature.side1_fraction)
    spin = int(feature.spin)
    side1_after = side1 + (1.0 / n if spin == 0 else -1.0 / n)
    profile = profile or {}
    return SimpleNamespace(
        node_id=int(feature.action),
        spin=spin,
        flip_gain=float(feature.flip_gain),
        normalized_gain=float(feature.normalized_gain),
        same_weight=float(feature.same_weight),
        opposite_weight=float(feature.opposite_weight),
        weighted_degree=float(feature.weighted_degree),
        degree=int(feature.degree),
        time_since_flip=float(feature.time_since_flip),
        recent_flip=bool(feature.time_since_flip <= 2),
        side0_fraction=float(feature.side0_fraction),
        side1_fraction=side1,
        step_fraction=float(feature.step_fraction),
        aspiration_gain=max(0.0, float(state.current_cut) + float(feature.flip_gain) - float(state.best_cut)),
        balance_after=abs(side1_after - 0.5),
        same_weight_share=float(profile.get("same_weight_share", 0.0)),
        opposite_weight_share=float(profile.get("opposite_weight_share", 0.0)),
        boundary_mix=float(profile.get("boundary_mix", 0.0)),
        neighbor_gain_mean=float(profile.get("neighbor_gain_mean", 0.0)),
        neighbor_gain_max=float(profile.get("neighbor_gain_max", 0.0)),
        positive_neighbor_gain_count=float(profile.get("positive_neighbor_gain_count", 0.0)),
        positive_neighbor_gain_sum=float(profile.get("positive_neighbor_gain_sum", 0.0)),
        neighbor_weighted_degree_mean=float(profile.get("neighbor_weighted_degree_mean", 0.0)),
        neighbor_weighted_degree_max=float(profile.get("neighbor_weighted_degree_max", 0.0)),
        two_hop_degree_proxy=float(profile.get("two_hop_degree_proxy", 0.0)),
        after_positive_gain_count=float(profile.get("after_positive_gain_count", 0.0)),
        after_positive_gain_count_delta=float(profile.get("after_positive_gain_count_delta", 0.0)),
        after_max_gain=float(profile.get("after_max_gain", 0.0)),
        after_top5_gain_mean=float(profile.get("after_top5_gain_mean", 0.0)),
        after_positive_gain_sum=float(profile.get("after_positive_gain_sum", 0.0)),
        current_cut=float(state.current_cut),
        best_cut=float(state.best_cut),
    )


def maxcut_candidate_profiles(state: MaxCutState, features=None) -> dict[int, dict[str, float]]:
    features = features or state.candidate_features()
    gains = np.asarray([float(feature.flip_gain) for feature in features], dtype=np.float64)
    degrees = np.asarray([float(feature.weighted_degree) for feature in features], dtype=np.float64)
    positive_before = int(np.sum(gains > 1e-9))
    profiles: dict[int, dict[str, float]] = {}
    for feature in features:
        node = int(feature.action)
        row = np.asarray(state.adjacency[node], dtype=np.float64)
        neighbors = np.flatnonzero(np.abs(row) > 1e-12)
        age = float(feature.time_since_flip)
        next_gains = gains.copy()
        spin_node = int(state.spins[node])
        next_gains[node] = -next_gains[node]
        for u in neighbors:
            if int(u) == node:
                continue
            same_sign = 1.0 if int(state.spins[int(u)]) == spin_node else -1.0
            next_gains[int(u)] -= 2.0 * float(row[int(u)]) * same_sign
        top_count = min(5, len(next_gains))
        top_after = np.sort(next_gains)[-top_count:] if top_count > 0 else np.asarray([0.0])
        neighbor_gains = gains[neighbors] if len(neighbors) else np.asarray([], dtype=np.float64)
        neighbor_degrees = degrees[neighbors] if len(neighbors) else np.asarray([], dtype=np.float64)
        same_share = float(feature.same_weight / max(1e-9, feature.weighted_degree))
        opposite_share = float(feature.opposite_weight / max(1e-9, feature.weighted_degree))
        profiles[node] = {
            "spin": float(feature.spin),
            "flip_gain": float(feature.flip_gain),
            "normalized_gain": float(feature.normalized_gain),
            "same_weight": float(feature.same_weight),
            "opposite_weight": float(feature.opposite_weight),
            "weighted_degree": float(feature.weighted_degree),
            "degree": float(feature.degree),
            "tabu_age": age,
            "recent_flip": float(age <= 2),
            "side0_fraction": float(feature.side0_fraction),
            "side1_fraction": float(feature.side1_fraction),
            "step_fraction": float(feature.step_fraction),
            "same_weight_share": same_share,
            "opposite_weight_share": opposite_share,
            "boundary_mix": float(min(same_share, opposite_share)),
            "neighbor_gain_mean": float(np.mean(neighbor_gains)) if len(neighbor_gains) else 0.0,
            "neighbor_gain_max": float(np.max(neighbor_gains)) if len(neighbor_gains) else 0.0,
            "positive_neighbor_gain_count": float(np.sum(neighbor_gains > 1e-9)) if len(neighbor_gains) else 0.0,
            "positive_neighbor_gain_sum": float(np.sum(np.maximum(neighbor_gains, 0.0))) if len(neighbor_gains) else 0.0,
            "neighbor_weighted_degree_mean": float(np.mean(neighbor_degrees)) if len(neighbor_degrees) else 0.0,
            "neighbor_weighted_degree_max": float(np.max(neighbor_degrees)) if len(neighbor_degrees) else 0.0,
            "two_hop_degree_proxy": float(np.sum(neighbor_degrees)) if len(neighbor_degrees) else 0.0,
            "after_positive_gain_count": float(np.sum(next_gains > 1e-9)),
            "after_positive_gain_count_delta": float(np.sum(next_gains > 1e-9) - positive_before),
            "after_max_gain": float(np.max(next_gains)) if len(next_gains) else 0.0,
            "after_top5_gain_mean": float(np.mean(top_after)),
            "after_positive_gain_sum": float(np.sum(np.maximum(next_gains, 0.0))),
        }
    return profiles


def maxcut_candidate_profile(state: MaxCutState, node: int) -> dict[str, float]:
    return maxcut_candidate_profiles(state)[int(node)]


def maxcut_candidate_semantics(
    state: MaxCutState,
    node: int,
    profile: dict[str, float],
    percentiles: dict[str, float],
) -> str:
    gain = float(profile["flip_gain"])
    norm_gain = float(profile["normalized_gain"])
    future_cut = float(state.current_cut) + gain
    aspiration = future_cut > float(state.best_cut) + 1e-9
    gain_pct = float(percentiles.get("flip_gain", 0.0))
    degree_pct = float(percentiles.get("weighted_degree", 0.0))
    same_pct = float(percentiles.get("same_weight", 0.0))
    after_top5_pct = float(percentiles.get("after_top5_gain_mean", 0.0))
    boundary_pct = float(percentiles.get("boundary_mix", 0.0))
    neighbor_degree_pct = float(percentiles.get("neighbor_weighted_degree_mean", 0.0))
    recent = bool(profile["recent_flip"] > 0.5)
    side1 = float(profile["side1_fraction"])
    spin = int(profile["spin"])
    n = max(1, int(state.n_nodes))
    side1_after = side1 + (1.0 / n if spin == 0 else -1.0 / n)
    before_imbalance = abs(side1 - 0.5)
    after_imbalance = abs(side1_after - 0.5)

    if aspiration:
        gain_role = "aspiration-improving"
    elif gain > 1e-9:
        gain_role = "positive-gain"
    elif abs(gain) <= 1e-9:
        gain_role = "neutral-escape"
    else:
        gain_role = "negative-escape"

    if gain_pct >= 0.80:
        gain_rank = "top-gain"
    elif gain_pct >= 0.50:
        gain_rank = "mid-gain"
    else:
        gain_rank = "low-gain"

    if degree_pct >= 0.80:
        node_role = "hub"
    elif degree_pct <= 0.20:
        node_role = "peripheral"
    else:
        node_role = "medium-degree"

    if same_pct >= 0.75:
        pressure_role = "heavy-internal-pressure"
    elif norm_gain > 0.15:
        pressure_role = "efficient-normalized-pressure"
    elif norm_gain < -0.10:
        pressure_role = "likely-cut-damaging"
    else:
        pressure_role = "weak-pressure"

    if recent:
        memory_role = "recent-reversal-risk"
    elif float(profile["tabu_age"]) <= 5.0:
        memory_role = "cooldown"
    else:
        memory_role = "fresh-or-old"

    if after_imbalance + 1e-12 < before_imbalance:
        balance_role = "balance-repair"
    elif after_imbalance > before_imbalance + 1e-12:
        balance_role = "balance-worsening"
    else:
        balance_role = "balance-neutral"

    if after_top5_pct >= 0.80:
        future_role = "opens-strong-followup"
    elif float(profile.get("after_positive_gain_count_delta", 0.0)) < 0:
        future_role = "shrinks-positive-frontier"
    else:
        future_role = "neutral-followup"

    if boundary_pct >= 0.70:
        structure_role = "boundary-mixed"
    elif neighbor_degree_pct >= 0.80:
        structure_role = "hub-neighborhood"
    else:
        structure_role = "local-neighborhood"

    return (
        f"{_stage_from_step(state.step, state.max_steps)} {node_role} {gain_role} "
        f"{gain_rank} {pressure_role} {memory_role} {balance_role} {future_role} {structure_role}"
    )


def _stage_from_step(step: int, max_steps: int) -> str:
    frac = step / max(1, max_steps)
    if frac < 1.0 / 3.0:
        return "early"
    if frac < 2.0 / 3.0:
        return "mid"
    return "late"


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = float(np.mean(rx))
    my = float(np.mean(ry))
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
