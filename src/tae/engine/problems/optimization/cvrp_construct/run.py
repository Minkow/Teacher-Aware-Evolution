import os
import threading
import pickle
import sys
import types
import time
import warnings
from pathlib import Path

import numpy as np
from tae.engine.evaluation_pool import RolloutPool, bounded_evaluation

from .get_instance import GetData
from .prompts import GetPrompts

from tae.analysis.population import (
    build_teacher_views,
    cvrp_candidate_profile,
    cvrp_teacher_preference,
    cvrp_teacher_preference_batch,
    normalize_teacher_value,
    percentile_profile,
    rank_percentile,
    summarize_alignment_metrics,
    summarize_case_records,
)


class CVRPCONST():
    def __init__(self) -> None:
        self.ndelay = 1
        self.problem_size = int(__import__("os").environ.get("EOH_CVRP_PROBLEM_SIZE", "50"))
        self.n_instance = int(__import__("os").environ.get("EOH_CVRP_N_INSTANCE", "64"))
        capacity_text = os.environ.get("EOH_CVRP_CAPACITY", "").strip()
        self.capacity_override = float(capacity_text) if capacity_text else None
        self.capacity = self.capacity_override if self.capacity_override is not None else 50.0
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("CVRP capacity must be finite and positive")
        self.running_time = 10
        self.prompts = GetPrompts()
        self.teacher_aware = __import__("os").environ.get("EOH_TEACHER_AWARE", "0") == "1"
        self.teacher_device = __import__("os").environ.get("EOH_TA_DEVICE", "cpu")
        self.teacher_stride = max(1, int(__import__("os").environ.get("EOH_TA_STRIDE", "4")))
        self.teacher_max_states = max(1, int(__import__("os").environ.get("EOH_TA_MAX_STATES", "18")))
        self.teacher_sample_instances = max(1, int(__import__("os").environ.get("EOH_TA_SAMPLE_INSTANCES", "4")))
        self.teacher_lock = threading.Lock()
        self.rollout_workers = max(1, int(os.environ.get("EOH_INSTANCE_ROLLOUT_WORKERS", "1")))
        self.rollout_pool = None
        self.rollout_pool_lock = threading.Lock()
        self.teacher = None
        if self.teacher_aware:
            from tae.teachers.lehd.routing import LEHDCVRPTeacher

            self.teacher = LEHDCVRPTeacher(device=self.teacher_device)

        train_pkl = __import__("os").environ.get("EOH_CVRP_TRAIN_PKL", "").strip()
        if train_pkl:
            self.instance_data = self._load_from_pickle(Path(train_pkl), self.n_instance, self.problem_size)
        else:
            get_data = GetData(self.n_instance, self.problem_size, self.capacity)
            self.instance_data = get_data.generate_instances()
        self.n_instance = len(self.instance_data)
        if not self.n_instance:
            raise ValueError("CVRP requires at least one training instance")


    def _get_rollout_pool(self):
        with self.rollout_pool_lock:
            if self.rollout_pool is None:
                self.rollout_pool = RolloutPool(self.rollout_workers)
            return self.rollout_pool

    def close(self):
        with self.rollout_pool_lock:
            pool, self.rollout_pool = self.rollout_pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def greedy_metadata_parallel(self, code_string):
        pool = self._get_rollout_pool()
        results = pool.map(_cvrp_rollout_worker, [
            (
                code_string,
                item,
                self.problem_size,
            )
            for index, item in enumerate(self.instance_data[: self.n_instance])
        ])
        if any(result is None for result in results):
            return None

        value_scores = []
        top1_hits = []
        percentiles = []
        case_records = []
        if self.teacher is not None:
            with self.teacher_lock:
                contexts = []
                for instance_idx, (item, result) in enumerate(zip(self.instance_data, results)):
                    if instance_idx >= self.teacher_sample_instances:
                        break
                    item_contexts = self._collect_action_contexts(item, result["actions"])
                    if item_contexts is None:
                        return None
                    contexts.extend(item_contexts)
                records = self._score_action_contexts(contexts)
                for record in records:
                    value_scores.append(record["teacher_value"])
                    top1_hits.append(record["top1"])
                    percentiles.append(record["teacher_percentile"])
                    if record.get("case") is not None:
                        case_records.append(record["case"])

        metrics = summarize_alignment_metrics(value_scores, top1_hits, percentiles)
        teacher_views = build_teacher_views(case_records)
        objective = float(np.mean([result["cost"] for result in results]))
        runtime_cpu_ns = sum(result["runtime_cpu_ns"] for result in results)
        runtime_decisions = sum(result["runtime_decisions"] for result in results)
        diagnostic = (
            f"objective={objective:.6f}; teacher_value={metrics['teacher_value_score']:.3f}; "
            f"top1={metrics['top1_agreement']:.3f}; sampled_cases={len(case_records)}"
        )
        return {
            "objective": objective,
            "runtime_per_decision_us": runtime_cpu_ns / max(1, runtime_decisions) / 1000.0,
            "runtime_decisions": runtime_decisions,
            "alignment_score": metrics["alignment_score"],
            "teacher_value_score": metrics["teacher_value_score"],
            "top1_agreement": metrics["top1_agreement"],
            "teacher_percentile": metrics["teacher_percentile"],
            "case_summary": summarize_case_records(case_records),
            "stage_summary": teacher_views["stage_summary"],
            "structure_cases": teacher_views["structure_cases"],
            "parameter_cases": teacher_views["parameter_cases"],
            "structure_guidance": teacher_views["structure_guidance"],
            "parameter_guidance": teacher_views["parameter_guidance"],
            "feature_preference_summary": teacher_views["feature_preference_summary"],
            "diagnostic": diagnostic,
        }

    def _collect_action_contexts(self, item, actions):
        coords = np.asarray(item["coords"], dtype=np.float64)
        demands = np.asarray(item["demands"], dtype=np.float64)
        capacity = float(item["capacity"])
        visited = set()
        selected_customers = []
        current = 0
        rest_capacity = capacity
        contexts = []

        for step, raw_action in enumerate(actions, start=1):
            action = int(raw_action)
            unvisited = tuple(
                node for node in range(1, self.problem_size + 1) if node not in visited
            )
            candidate_nodes = tuple(
                [node for node in unvisited if float(demands[node]) <= rest_capacity + 1e-9]
                + [0]
            )
            if (
                len(contexts) < self.teacher_max_states
                and (step - 1) % self.teacher_stride == 0
                and candidate_nodes
            ):
                contexts.append(
                    {
                        "coords": coords,
                        "demands": demands,
                        "capacity": capacity,
                        "selected": tuple(selected_customers),
                        "rest_capacity": rest_capacity,
                        "candidate_nodes": candidate_nodes,
                        "chosen_action": action,
                        "current": current,
                        "remaining_nodes": list(unvisited),
                    }
                )

            if action == 0:
                if current == 0:
                    return None
                current = 0
                rest_capacity = capacity
                continue
            if action not in unvisited or float(demands[action]) > rest_capacity + 1e-9:
                return None
            visited.add(action)
            selected_customers.append(action)
            current = action
            rest_capacity -= float(demands[action])
        if len(visited) != self.problem_size:
            return None
        return contexts

    def _score_action_contexts(self, contexts):
        preferences = cvrp_teacher_preference_batch(
            self.teacher,
            [
                (
                    context["coords"],
                    context["demands"],
                    context["capacity"],
                    context["selected"],
                    context["rest_capacity"],
                    context["candidate_nodes"],
                )
                for context in contexts
            ],
        )
        records = []
        for context, pref in zip(contexts, preferences):
            coords = context["coords"]
            demands = context["demands"]
            capacity = context["capacity"]
            action = context["chosen_action"]
            value = normalize_teacher_value(pref.teacher_scores, pref.candidate_nodes, action)
            chosen_pct = rank_percentile(pref.teacher_scores, pref.candidate_nodes, action)
            case = None
            if action != pref.teacher_action:
                profiles = {
                    int(node): cvrp_candidate_profile(
                        coords,
                        demands,
                        capacity,
                        context["current"],
                        int(node),
                        context["rest_capacity"],
                        context["remaining_nodes"],
                    )
                    for node in pref.candidate_nodes
                }
                percentile_maps = {
                    node: {
                        key: percentile_profile(
                            [candidate[key] for candidate in profiles.values()], item
                        )
                        for key, item in profile.items()
                    }
                    for node, profile in profiles.items()
                }
                scores_by_node = {
                    int(node): float(score)
                    for node, score in zip(pref.candidate_nodes, pref.teacher_scores)
                }
                remaining_fraction = len(context["remaining_nodes"]) / max(1, self.problem_size)
                case = {
                    "stage": (
                        "early" if remaining_fraction > 2.0 / 3.0
                        else "mid" if remaining_fraction > 1.0 / 3.0
                        else "late"
                    ),
                    "remaining_fraction": float(remaining_fraction),
                    "feasible_count": int(len(pref.candidate_nodes)),
                    "current": int(context["current"]),
                    "teacher_score_gap": float(
                        scores_by_node[int(pref.teacher_action)]
                        - scores_by_node.get(action, min(scores_by_node.values()))
                    ),
                    "chosen_action": action,
                    "teacher_action": int(pref.teacher_action),
                    "chosen_profile": profiles.get(action, profiles[int(pref.teacher_action)]),
                    "teacher_profile": profiles[int(pref.teacher_action)],
                    "chosen_pct": percentile_maps.get(action, percentile_maps[int(pref.teacher_action)]),
                    "teacher_pct": percentile_maps[int(pref.teacher_action)],
                }
            records.append(
                {
                    "teacher_value": value,
                    "top1": float(action == pref.teacher_action),
                    "teacher_percentile": chosen_pct,
                    "case": case,
                }
            )
        return records


    def evaluate(self, code_string):
        result = self.evaluate_metadata(code_string)
        return None if result is None else result["objective"]

    @bounded_evaluation
    def evaluate_metadata(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.greedy_metadata_parallel(code_string)
        except TimeoutError:
            raise
        except Exception:
            return None

    def _load_from_pickle(self, path: Path, limit: int, problem_size: int):
        with path.open("rb") as fh:
            raw = pickle.load(fh)
        instance_data = []
        for item in raw:
            coords = np.asarray(item["coords"], dtype=np.float32)
            demands = np.asarray(item["demands"], dtype=np.float32)
            capacity = float(item["capacity"])
            if coords.shape[0] != problem_size + 1:
                continue
            if self.capacity_override is not None:
                capacity = self.capacity_override
            if not np.isfinite(capacity) or capacity <= 0:
                raise ValueError("CVRP instance capacity must be finite and positive")
            if demands.shape != (problem_size + 1,) or not np.all(np.isfinite(demands)):
                raise ValueError("CVRP demands must be a finite vector matching the coordinates")
            if np.any(demands < 0) or np.any(demands[1:] > capacity):
                raise ValueError("CVRP demands must be nonnegative and no customer may exceed capacity")
            instance_data.append(
                {
                    "name": str(item.get("name", "")),
                    "coords": coords,
                    "demands": demands,
                    "capacity": capacity,
                }
            )
            if len(instance_data) >= limit:
                break
        if not instance_data:
            raise ValueError(f"no CVRP instances of size {problem_size} loaded from {path}")
        return instance_data


def _cvrp_rollout_worker(code_string, item, problem_size):
    module = types.ModuleType(f"cvrp_heuristic_worker_{os.getpid()}")
    exec(code_string, module.__dict__)
    if not hasattr(module, "select_next_node"):
        return None
    coords = np.asarray(item["coords"], dtype=np.float64)
    demands = np.asarray(item["demands"], dtype=np.float64)
    capacity = float(item["capacity"])
    distance_matrix = np.linalg.norm(
        coords[:, np.newaxis, :] - coords[np.newaxis, :, :],
        axis=2,
    )
    visited = set()
    current = 0
    rest_capacity = capacity
    actions = []
    runtime_cpu_ns = 0
    runtime_decisions = 0
    max_steps = max(20, problem_size * 4 + 8)
    for _step in range(max_steps):
        if len(visited) >= problem_size:
            break
        unvisited = tuple(node for node in range(1, problem_size + 1) if node not in visited)
        input_demands, input_matrix = demands.copy(), distance_matrix.copy()
        start_ns = time.thread_time_ns()
        next_node = (
            module.select_next_node(
                current,
                0,
                unvisited,
                rest_capacity,
                input_demands,
                input_matrix,
            )
        )
        runtime_cpu_ns += time.thread_time_ns() - start_ns
        runtime_decisions += 1
        if isinstance(next_node, (bool, np.bool_)) or not isinstance(next_node, (int, np.integer)):
            return None
        if next_node == 0:
            if current == 0:
                return None
            actions.append(0)
            current = 0
            rest_capacity = capacity
            continue
        if next_node not in unvisited or float(demands[next_node]) > rest_capacity + 1e-9:
            return None
        visited.add(next_node)
        actions.append(next_node)
        current = next_node
        rest_capacity -= float(demands[next_node])
    if len(visited) != problem_size:
        return None
    total = 0.0
    current = 0
    for action in actions:
        total += float(np.linalg.norm(coords[current] - coords[int(action)]))
        current = int(action)
    total += float(np.linalg.norm(coords[current] - coords[0]))
    return {"actions": tuple(actions), "cost": total,
            "runtime_cpu_ns": runtime_cpu_ns, "runtime_decisions": runtime_decisions}
