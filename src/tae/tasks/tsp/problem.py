import numpy as np
from tae.engine.evaluation_pool import RolloutPool, bounded_evaluation
import os
import pickle
import sys
import time
import types
import threading
import warnings
from pathlib import Path
from .prompts import GetPrompts

from tae.analysis.population import (
    build_teacher_views,
    normalize_teacher_value,
    rank_percentile,
    summarize_alignment_metrics,
    summarize_case_records,
    tsp_candidate_profile,
    tsp_teacher_preference,
    tsp_teacher_preference_batch,
    percentile_profile,
)


def _uniform_sample_steps(first_step, last_step, max_states):
    """Place query targets uniformly across all constructive decisions."""
    first = int(first_step)
    last = int(last_step)
    total = max(0, last - first + 1)
    count = min(total, max(1, int(max_states)))
    if count == 0:
        return frozenset()
    return frozenset(
        first + ((2 * index + 1) * total) // (2 * count)
        for index in range(count)
    )

class TSPCONST():
    def __init__(self) -> None:
        # ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        # sys.path.append(ABS_PATH)  # This is for finding all the modules
        # Construct the absolute path to the pickle file
        #pickle_file_path = os.path.join(ABS_PATH, 'instances.pkl')

        # with open("./instances.pkl" , 'rb') as f:
        #     self.instance_data = pickle.load(f)
        self.ndelay = 1
        self.problem_size = int(os.environ.get("EOH_TSP_PROBLEM_SIZE", "50"))
        self.neighbor_size = np.minimum(int(os.environ.get("EOH_TSP_NEIGHBOR_SIZE", "50")), self.problem_size)
        self.n_instance = int(os.environ.get("EOH_TSP_N_INSTANCE", "8"))
        self.running_time = 10


        self.prompts = GetPrompts()
        self.teacher_aware = os.environ.get("EOH_TEACHER_AWARE", "0") == "1"
        self.teacher_device = os.environ.get("EOH_TA_DEVICE", "cpu")
        self.teacher_stride = max(1, int(os.environ.get("EOH_TA_STRIDE", "4")))
        self.teacher_max_states = max(1, int(os.environ.get("EOH_TA_MAX_STATES", "18")))
        self.teacher_sample_instances = max(1, int(os.environ.get("EOH_TA_SAMPLE_INSTANCES", "4")))
        self.teacher_stage_stratified = os.environ.get("EOH_TA_STAGE_STRATIFIED", "0") == "1"
        self.teacher_lock = threading.Lock()
        self.rollout_workers = max(1, int(os.environ.get("EOH_INSTANCE_ROLLOUT_WORKERS", "1")))
        self.rollout_pool = None
        self.rollout_pool_lock = threading.Lock()
        self.teacher = None
        if self.teacher_aware:
            from tae.teachers.lehd.routing import LEHDTSPTeacher

            self.teacher = LEHDTSPTeacher(device=self.teacher_device)

        train_pkl = os.environ.get("EOH_TSP_TRAIN_PKL", "").strip()
        if train_pkl:
            self.instance_data = self._load_from_pickle(Path(train_pkl), self.n_instance, self.problem_size)
        else:
            raise ValueError("TSP training requires a fixed dataset: set EOH_TSP_TRAIN_PKL")
        self.n_instance = len(self.instance_data)
        if not self.n_instance:
            raise ValueError("TSP requires at least one training instance")
        


    #@func_set_timeout(5)


    def _get_rollout_pool(self):
        with self.rollout_pool_lock:
            if self.rollout_pool is None:
                self.rollout_pool = RolloutPool(self.rollout_workers)
            return self.rollout_pool

    def greedy_metadata_parallel(self, code_string):
        pool = self._get_rollout_pool()
        results = pool.map(_tsp_rollout_worker, [
            (
                code_string,
                np.asarray(instance, dtype=float),
                np.asarray(distance_matrix, dtype=float),
                self.problem_size,
                self.neighbor_size,
            )
            for idx, (instance, distance_matrix) in enumerate(self.instance_data[: self.n_instance])
        ])
        if any(result is None for result in results):
            return None

        value_scores = []
        top1_hits = []
        percentiles = []
        case_records = []
        if self.teacher is not None:
            with self.teacher_lock:
                analysis_inputs = [
                    (
                        np.asarray(instance, dtype=np.float32),
                        np.asarray(distance_matrix, dtype=float),
                        result["route"],
                    )
                    for (instance, distance_matrix), result in list(
                        zip(self.instance_data, results)
                    )[: self.teacher_sample_instances]
                ]
                records = self._analyze_routes(analysis_inputs)
                if records is None:
                    return None
                for record in records:
                    value_scores.append(record["teacher_value"])
                    top1_hits.append(record["top1"])
                    percentiles.append(record["teacher_percentile"])
                    if record.get("case") is not None:
                        case_records.append(record["case"])

        metrics = summarize_alignment_metrics(value_scores, top1_hits, percentiles)
        teacher_views = build_teacher_views(case_records)
        objective = float(np.mean([result["cost"] for result in results]))
        runtime_cpu_ns = sum(int(result["runtime_cpu_ns"]) for result in results)
        runtime_decisions = sum(int(result["runtime_decisions"]) for result in results)
        runtime_per_decision_us = (
            float(runtime_cpu_ns) / max(1, runtime_decisions) / 1_000.0
        )
        diagnostic = (
            f"objective={objective:.6f}; teacher_value={metrics['teacher_value_score']:.3f}; "
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
            "case_summary": summarize_case_records(case_records),
            "stage_summary": teacher_views["stage_summary"],
            "structure_cases": teacher_views["structure_cases"],
            "parameter_cases": teacher_views["parameter_cases"],
            "structure_guidance": teacher_views["structure_guidance"],
            "parameter_guidance": teacher_views["parameter_guidance"],
            "feature_preference_summary": teacher_views["feature_preference_summary"],
            "diagnostic": diagnostic,
        }

    def _analyze_routes(self, route_inputs):
        contexts = []
        for instance, _distance_matrix, route in route_inputs:
            neighbor_matrix = np.argsort(
                np.linalg.norm(instance[:, None, :] - instance[None, :, :], axis=2),
                axis=1,
            )
            sample_steps = (
                _uniform_sample_steps(
                    1,
                    self.problem_size - 2,
                    self.teacher_max_states,
                )
                if self.teacher_stage_stratified
                else set(range(1, self.problem_size - 1, self.teacher_stride))
            )
            sampled_for_route = 0
            for i in range(1, self.problem_size - 1):
                current_node = int(route[i - 1])
                next_node = int(route[i])
                near_nodes = neighbor_matrix[current_node][1:]
                mask = ~np.isin(near_nodes, route[:i])
                candidates = near_nodes[mask][: self.neighbor_size]
                if next_node not in candidates:
                    return None
                if (
                    sampled_for_route >= self.teacher_max_states
                    or i not in sample_steps
                    or candidates.size == 0
                ):
                    continue
                contexts.append(
                    {
                        "instance": instance,
                        "current_node": current_node,
                        "next_node": next_node,
                        "previous_node": int(route[i - 2]) if i >= 2 else None,
                        "selected_prefix": [int(node) for node in route[:i]],
                        "candidates": [int(node) for node in candidates.tolist()],
                    }
                )
                sampled_for_route += 1

        preferences = tsp_teacher_preference_batch(
            self.teacher,
            [
                (context["instance"], context["selected_prefix"], context["candidates"])
                for context in contexts
            ],
        )
        records = []
        for context, pref in zip(contexts, preferences):
            current_node = context["current_node"]
            next_node = context["next_node"]
            instance = context["instance"]
            value = normalize_teacher_value(pref.teacher_scores, pref.candidate_nodes, next_node)
            chosen_pct = rank_percentile(pref.teacher_scores, pref.candidate_nodes, next_node)
            case = None
            if next_node != pref.teacher_action:
                remaining_nodes = [
                    int(node)
                    for node in range(self.problem_size)
                    if node not in context["selected_prefix"]
                ]
                profiles = {
                    int(node): tsp_candidate_profile(
                        instance,
                        current_node,
                        int(node),
                        remaining_nodes,
                        context["previous_node"],
                    )
                    for node in context["candidates"]
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
                case = {
                    "stage": (
                        "early" if len(remaining_nodes) / max(1, self.problem_size - 1) > 2.0 / 3.0
                        else "mid" if len(remaining_nodes) / max(1, self.problem_size - 1) > 1.0 / 3.0
                        else "late"
                    ),
                    "remaining_fraction": float(len(remaining_nodes) / max(1, self.problem_size - 1)),
                    "feasible_count": int(len(pref.candidate_nodes)),
                    "current": current_node,
                    "teacher_score_gap": float(
                        scores_by_node[int(pref.teacher_action)] - scores_by_node[next_node]
                    ),
                    "chosen_action": next_node,
                    "teacher_action": int(pref.teacher_action),
                    "chosen_profile": profiles[next_node],
                    "teacher_profile": profiles[int(pref.teacher_action)],
                    "chosen_pct": percentile_maps[next_node],
                    "teacher_pct": percentile_maps[int(pref.teacher_action)],
                }
            records.append(
                {
                    "teacher_value": value,
                    "top1": float(next_node == pref.teacher_action),
                    "teacher_percentile": chosen_pct,
                    "case": case,
                }
            )
        return records

    def _analyze_route(self, instance, distance_matrix, route):
        return self._analyze_routes([(instance, distance_matrix, route)])


    def evaluate(self, code_string):
        result = self.evaluate_metadata(code_string)
        return None if result is None else result["objective"]
    
    def _load_from_pickle(self, path: Path, limit: int, problem_size: int):
        with path.open("rb") as fh:
            raw = pickle.load(fh)
        instance_data = []
        for item in raw:
            coords = np.asarray(item[0], dtype=float)
            if coords.shape[0] != problem_size:
                continue
            dist = np.asarray(item[1], dtype=float)
            instance_data.append((coords, dist))
            if len(instance_data) >= limit:
                break
        if not instance_data:
            raise ValueError(f"no TSP instances of size {problem_size} loaded from {path}")
        return instance_data

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


def _valid_node(node, candidates):
    return (not isinstance(node, (bool, np.bool_))
            and isinstance(node, (int, np.integer)) and node in candidates)


def _tsp_rollout_worker(code_string, instance, distance_matrix, problem_size, neighbor_size):
    module = types.ModuleType(f"tsp_heuristic_worker_{os.getpid()}")
    exec(code_string, module.__dict__)
    if not hasattr(module, "select_next_node"):
        return None
    neighbor_matrix = np.argsort(
        np.linalg.norm(instance[:, None, :] - instance[None, :, :], axis=2),
        axis=1,
    )
    route = np.zeros(problem_size, dtype=int)
    current_node = 0
    destination_node = 0
    runtime_cpu_ns = 0
    runtime_decisions = 0
    for i in range(1, problem_size - 1):
        near_nodes = neighbor_matrix[current_node]
        mask = ~np.isin(near_nodes, route[:i])
        candidates = near_nodes[mask][:neighbor_size]
        start_ns = time.thread_time_ns()
        next_node = (
            module.select_next_node(
                int(current_node),
                int(destination_node),
                candidates.copy(),
                distance_matrix,
            )
        )
        runtime_cpu_ns += time.thread_time_ns() - start_ns
        runtime_decisions += 1
        if not _valid_node(next_node, candidates):
            return None
        route[i] = next_node
        current_node = next_node
    remaining = np.arange(problem_size)[~np.isin(np.arange(problem_size), route[: problem_size - 1])]
    if len(remaining) != 1:
        return None
    route[problem_size - 1] = int(remaining[0])
    edges = instance[route] - instance[np.roll(route, -1)]
    cost = float(np.linalg.norm(edges, axis=1).sum())
    return {
        "route": tuple(int(node) for node in route),
        "cost": cost,
        "runtime_cpu_ns": runtime_cpu_ns,
        "runtime_decisions": runtime_decisions,
    }
