from __future__ import annotations

import ast
import os
import json
import re
import urllib.request
import urllib.error
import http.client
import socket
import time
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

import numpy as np

from tae.teachers.lehd.routing import LEHDCVRPTeacher, LEHDTSPTeacher
from tae.analysis.memory import transition_evidence_text


_LAST_VALID_ANALYZER_PAYLOAD: dict[str, object] | None = None


def _is_analyzer_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {408, 504}:
            return True
        if exc.code in {502, 503}:
            try:
                body = exc.read(4096).decode("utf-8", errors="replace").lower()
            except Exception:
                body = ""
            return any(
                marker in body
                for marker in ("timeout", "timed out", "deadline exceeded")
            )
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (TimeoutError, socket.timeout))
    return False


@dataclass(frozen=True)
class TSPTeacherPreference:
    candidate_nodes: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    teacher_action: int


@dataclass(frozen=True)
class CVRPTeacherPreference:
    candidate_nodes: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    teacher_action: int


def tsp_teacher_preference(
    teacher: LEHDTSPTeacher,
    coords: np.ndarray,
    selected_prefix: Iterable[int],
    candidate_nodes: Iterable[int],
) -> TSPTeacherPreference:
    return tsp_teacher_preference_batch(
        teacher,
        [(coords, selected_prefix, candidate_nodes)],
    )[0]


def tsp_teacher_preference_batch(
    teacher: LEHDTSPTeacher,
    requests: Iterable[tuple[np.ndarray, Iterable[int], Iterable[int]]],
) -> list[TSPTeacherPreference]:
    """Score routing states in shape-compatible batches without changing semantics."""
    rows = []
    for coords, selected_prefix, candidate_nodes in requests:
        coords_np = np.asarray(coords, dtype=np.float32)
        selected = tuple(int(node) for node in selected_prefix)
        nodes = tuple(int(node) for node in candidate_nodes)
        if not nodes:
            raise ValueError("candidate_nodes must be non-empty")
        rows.append((coords_np, selected, nodes))
    if not rows:
        return []

    import torch

    outputs: list[TSPTeacherPreference | None] = [None] * len(rows)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (coords_np, selected, _nodes) in enumerate(rows):
        groups.setdefault((len(coords_np), len(selected)), []).append(index)

    for indices in groups.values():
        coords_t = torch.as_tensor(
            np.stack([rows[index][0] for index in indices]),
            dtype=torch.float32,
            device=teacher.device,
        )
        selected_t = torch.as_tensor(
            np.asarray([rows[index][1] for index in indices], dtype=np.int64),
            dtype=torch.long,
            device=teacher.device,
        )
        with torch.no_grad():
            encoded = teacher.model.encoder(coords_t)
            probabilities = teacher.model.decoder(encoded, selected_t).detach().cpu().numpy()
        for batch_index, row_index in enumerate(indices):
            nodes = rows[row_index][2]
            scores = tuple(float(probabilities[batch_index, node]) for node in nodes)
            best_idx = max(range(len(nodes)), key=lambda idx: (scores[idx], -nodes[idx]))
            outputs[row_index] = TSPTeacherPreference(nodes, scores, nodes[best_idx])

    if any(output is None for output in outputs):
        raise RuntimeError("incomplete TSP teacher batch")
    return [output for output in outputs if output is not None]


def cvrp_teacher_preference(
    teacher: LEHDCVRPTeacher,
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    selected_customers: Iterable[int],
    rest_capacity: float,
    candidate_nodes: Iterable[int],
) -> CVRPTeacherPreference:
    return cvrp_teacher_preference_batch(
        teacher,
        [(coords, demands, capacity, selected_customers, rest_capacity, candidate_nodes)],
    )[0]


def cvrp_teacher_preference_batch(
    teacher: LEHDCVRPTeacher,
    requests: Iterable[
        tuple[np.ndarray, np.ndarray, float, Iterable[int], float, Iterable[int]]
    ],
) -> list[CVRPTeacherPreference]:
    rows = []
    for coords, demands, capacity, selected_customers, rest_capacity, candidate_nodes in requests:
        coords_np = np.asarray(coords, dtype=np.float32)
        demands_np = np.asarray(demands, dtype=np.float32)
        selected = tuple(int(node) for node in selected_customers)
        nodes = tuple(int(node) for node in candidate_nodes)
        if not nodes:
            raise ValueError("candidate_nodes must be non-empty")
        cap_col = np.full((len(coords_np), 1), float(rest_capacity), dtype=np.float32)
        data = np.concatenate([coords_np, demands_np[:, None], cap_col], axis=1)
        data[0, 3] = float(capacity)
        rows.append((data, float(capacity), selected, float(rest_capacity), nodes))
    if not rows:
        return []

    import torch

    outputs: list[CVRPTeacherPreference | None] = [None] * len(rows)
    groups: dict[tuple[int, int, float], list[int]] = {}
    for index, (data, capacity, selected, _rest, _nodes) in enumerate(rows):
        groups.setdefault((len(data), len(selected), capacity), []).append(index)

    for (_size, _selected_len, capacity), indices in groups.items():
        problem = torch.as_tensor(
            np.stack([rows[index][0] for index in indices]),
            dtype=torch.float32,
            device=teacher.device,
        )
        selected_t = torch.as_tensor(
            np.asarray([rows[index][2] for index in indices], dtype=np.int64),
            dtype=torch.long,
            device=teacher.device,
        )
        remaining = torch.as_tensor(
            [rows[index][3] for index in indices],
            dtype=torch.float32,
            device=teacher.device,
        )
        with torch.no_grad():
            encoded = teacher.model.encoder(problem, capacity)
            probabilities = teacher.model.decoder(
                encoded,
                selected_t,
                capacity,
                remaining,
            ).detach().cpu().numpy()
        for batch_index, row_index in enumerate(indices):
            data, _capacity, selected, _rest, nodes = rows[row_index]
            selected_set = set(selected)
            n_customers = len(data) - 1
            depot_score = max(
                (
                    float(probabilities[batch_index, n_customers + customer - 1])
                    for customer in range(1, n_customers + 1)
                    if customer not in selected_set
                ),
                default=0.0,
            )
            scores = tuple(
                depot_score if node == 0 else float(probabilities[batch_index, node - 1])
                for node in nodes
            )
            best_idx = max(range(len(nodes)), key=lambda idx: (scores[idx], -nodes[idx]))
            outputs[row_index] = CVRPTeacherPreference(nodes, scores, nodes[best_idx])

    if any(output is None for output in outputs):
        raise RuntimeError("incomplete CVRP teacher batch")
    return [output for output in outputs if output is not None]


def tsp_candidate_profile(
    coords: np.ndarray,
    current_node: int,
    node: int,
    remaining_nodes: list[int],
    previous_node: int | None = None,
) -> dict[str, float]:
    coords_np = np.asarray(coords, dtype=np.float64)
    current = coords_np[int(current_node)]
    target = coords_np[int(node)]
    depot = coords_np[0]
    others = [idx for idx in remaining_nodes if idx != int(node)]
    if others:
        nearest = min(_dist(target, coords_np[idx]) for idx in others)
        centroid = np.mean(coords_np[remaining_nodes], axis=0)
    else:
        nearest = 0.0
        centroid = target
    heading = 0.0
    if previous_node is not None and int(previous_node) != int(current_node):
        prev_vec = current - coords_np[int(previous_node)]
        cur_vec = target - current
        prev_norm = np.linalg.norm(prev_vec)
        cur_norm = np.linalg.norm(cur_vec)
        if prev_norm > 1e-12 and cur_norm > 1e-12:
            heading = float(np.dot(prev_vec, cur_vec) / (prev_norm * cur_norm))
    return {
        "travel_from_current": _dist(current, target),
        "nearest_remaining_distance": float(nearest),
        "distance_to_remaining_centroid_sq": float(np.sum((target - centroid) ** 2)),
        "saving_proxy": _dist(current, depot) + _dist(depot, target) - _dist(current, target),
        "heading_alignment": float(heading),
    }


def cvrp_candidate_profile(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    current_node: int,
    node: int,
    rest_capacity: float,
    remaining_nodes: list[int],
) -> dict[str, float]:
    coords_np = np.asarray(coords, dtype=np.float64)
    demands_np = np.asarray(demands, dtype=np.float64)
    depot = coords_np[0]
    current = coords_np[int(current_node)]
    if int(node) == 0:
        nearest = min((_dist(depot, coords_np[idx]) for idx in remaining_nodes), default=0.0)
        centroid = np.mean(coords_np[remaining_nodes], axis=0) if remaining_nodes else depot
        return {
            "via_depot": 1.0,
            "travel_from_current": _dist(current, depot),
            "distance_to_depot": 0.0,
            "nearest_remaining_distance": float(nearest),
            "distance_to_remaining_centroid_sq": float(np.sum((depot - centroid) ** 2)),
            "demand_ratio": 0.0,
            "remaining_capacity_ratio": float(rest_capacity / max(1e-9, capacity)),
            "capacity_after_ratio": 1.0,
            "is_capacity_tight": 0.0,
            "saving_proxy": 0.0,
        }
    target = coords_np[int(node)]
    others = [idx for idx in remaining_nodes if idx != int(node)]
    nearest = min((_dist(target, coords_np[idx]) for idx in others), default=0.0)
    centroid = np.mean(coords_np[remaining_nodes], axis=0) if remaining_nodes else target
    demand = float(demands_np[int(node)])
    capacity_after = float(rest_capacity - demand)
    return {
        "via_depot": 0.0,
        "travel_from_current": _dist(current, target),
        "distance_to_depot": _dist(target, depot),
        "nearest_remaining_distance": float(nearest),
        "distance_to_remaining_centroid_sq": float(np.sum((target - centroid) ** 2)),
        "demand_ratio": float(demand / max(1e-9, capacity)),
        "remaining_capacity_ratio": float(rest_capacity / max(1e-9, capacity)),
        "capacity_after_ratio": float(capacity_after / max(1e-9, capacity)),
        "is_capacity_tight": float(capacity_after < 0.15 * capacity),
        "saving_proxy": _dist(current, depot) + _dist(depot, target) - _dist(current, target),
    }


def percentile_profile(values: list[float], value: float) -> float:
    if len(values) <= 1:
        return 1.0
    ordered = sorted(values)
    pos = 0
    for idx, item in enumerate(ordered):
        if item <= value:
            pos = idx
    return pos / (len(ordered) - 1)


def rank_percentile(scores: tuple[float, ...], candidate_nodes: tuple[int, ...], chosen_node: int) -> float:
    ordered = sorted(zip(candidate_nodes, scores), key=lambda item: (item[1], -item[0]), reverse=True)
    for rank, (node, _score) in enumerate(ordered):
        if int(node) == int(chosen_node):
            if len(ordered) <= 1:
                return 1.0
            return 1.0 - rank / (len(ordered) - 1)
    return 0.0


def summarize_case_records(records: list[dict[str, object]], *, max_cases: int = 4) -> str:
    if not records:
        return "No representative teacher disagreements were sampled."
    buckets: dict[str, list[dict[str, object]]] = {"early": [], "mid": [], "late": []}
    for row in records:
        stage = str(row.get("stage") or _stage_from_remaining_fraction(float(row.get("remaining_fraction", 0.5))))
        buckets.setdefault(stage, []).append(row)

    def sort_key(item):
        return float(item.get("teacher_score_gap", 0.0))

    rows = []
    per_stage_cap = max(1, max_cases // 3)
    for stage in ("early", "mid", "late"):
        stage_rows = sorted(buckets.get(stage, []), key=sort_key, reverse=True)
        rows.extend(stage_rows[:per_stage_cap])
    if len(rows) < max_cases:
        used = {id(item) for item in rows}
        fallback = sorted(records, key=sort_key, reverse=True)
        for row in fallback:
            if id(row) in used:
                continue
            rows.append(row)
            used.add(id(row))
            if len(rows) >= max_cases:
                break

    lines = []
    for row in rows:
        chosen = row["chosen_profile"]
        teacher = row["teacher_profile"]
        semantic_text = _case_semantics_text(row)
        confidence_text = _teacher_confidence_text(row)
        lines.append(
            "stage={stage}, remaining_fraction={remaining_fraction:.2f}, feasible_count={feasible_count}, current={current}, "
            "teacher_score_gap={teacher_score_gap:+.3f}{confidence}; delta={delta}; chosen[{chosen}] vs teacher[{teacher}]{semantic}".format(
                stage=str(row.get("stage") or _stage_from_remaining_fraction(float(row.get("remaining_fraction", 0.5)))),
                remaining_fraction=float(row["remaining_fraction"]),
                feasible_count=int(row["feasible_count"]),
                current=int(row["current"]),
                teacher_score_gap=float(row["teacher_score_gap"]),
                confidence=confidence_text,
                delta=_case_delta_text(row),
                chosen=_compact_profile(int(row["chosen_action"]), chosen, row["chosen_pct"]),
                teacher=_compact_profile(int(row["teacher_action"]), teacher, row["teacher_pct"]),
                semantic=semantic_text,
            )
        )
    return "\n".join(lines)


def normalize_teacher_value(scores: tuple[float, ...], candidate_nodes: tuple[int, ...], chosen_node: int) -> float:
    if not scores:
        return 0.0
    score_map = {int(node): float(score) for node, score in zip(candidate_nodes, scores)}
    chosen = score_map.get(int(chosen_node))
    if chosen is None:
        return 0.0
    lo = min(score_map.values())
    hi = max(score_map.values())
    if hi - lo <= 1e-12:
        return 1.0
    return float((chosen - lo) / (hi - lo))


def summarize_alignment_metrics(value_scores: list[float], top1_hits: list[float], percentiles: list[float]) -> dict[str, float]:
    top1 = float(mean(top1_hits)) if top1_hits else 0.0
    teacher_value = float(mean(value_scores)) if value_scores else 0.0
    teacher_percentile = float(mean(percentiles)) if percentiles else 0.0
    alignment_score, alignment_metric = _select_alignment_score(
        top1=top1,
        teacher_value=teacher_value,
        teacher_percentile=teacher_percentile,
    )
    return {
        "top1_agreement": top1,
        "teacher_percentile": teacher_percentile,
        "teacher_value_score": teacher_value,
        "alignment_score": alignment_score,
        "alignment_metric": alignment_metric,
    }


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


def pareto_summary(population: list[dict[str, object]], *, limit: int = 6) -> str:
    if not population:
        return "Pareto pool is empty."
    rows = []
    for idx, item in enumerate(population[:limit], start=1):
        desc = str(item.get("algorithm") or "").strip().splitlines()[0]
        if len(desc) > 160:
            desc = desc[:157] + "..."
        rows.append(
            f"{idx}. objective={float(item['objective']):.6f}, "
            f"alignment={float(item.get('alignment_score', 0.0)):.3f}, "
            f"teacher_pct={float(item.get('teacher_percentile', 0.0)):.3f}, "
            f"top1={float(item.get('top1_agreement', 0.0)):.3f}, "
            f"algorithm={desc}"
        )
    return "\n".join(rows)


def operator_guidance(
    population: list[dict[str, object]],
    structural_transitions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if not population:
        return _empty_operator_guidance()
    llm_guidance = _llm_operator_guidance(
        population,
        structural_transitions=structural_transitions,
    )
    if llm_guidance is not None:
        return llm_guidance
    by_obj = min(population, key=lambda item: float(item["objective"]))
    with_align = [item for item in population if item.get("alignment_score") is not None]
    by_align = max(with_align, key=lambda item: float(item.get("alignment_score", 0.0))) if with_align else by_obj
    pareto = pareto_summary(population)
    obj_stage = str(by_obj.get("stage_summary") or "No stage-level teacher failures were summarized.")
    align_stage = str(by_align.get("stage_summary") or "No stage-level teacher failures were summarized.")
    obj_pref = str(by_obj.get("feature_preference_summary") or "No teacher-feature preference mismatch was summarized.")
    align_pref = str(by_align.get("feature_preference_summary") or "No teacher-feature preference mismatch was summarized.")
    obj_structure_cases = str(by_obj.get("structure_cases") or "No sampled disagreement cases.")
    align_structure_cases = str(by_align.get("structure_cases") or "No sampled disagreement cases.")
    jssp_context = _jssp_baseline_context(by_obj, by_align)
    jssp_raw_trace = _jssp_raw_trace_evidence(by_obj, by_align)
    family = _problem_family()
    domain_s_context = _domain_operator_context(family, "s")
    domain_p_context = _domain_operator_context(family, "p")
    domain_e2_context = _domain_operator_context(family, "m")
    e2_pro = (
        "Teacher-aware mechanism restructuring brief:\n"
        f"{domain_e2_context}"
        "Current Pareto pool:\n"
        f"{pareto}\n"
        f"{jssp_context}"
        f"High-performance anchor stage summary:\n{obj_stage}\n"
        f"High-performance anchor teacher-feature mismatch:\n{obj_pref}\n"
        f"High-alignment anchor stage summary:\n{align_stage}\n"
        f"High-alignment anchor teacher-feature mismatch:\n{align_pref}\n"
        f"High-performance anchor representative disagreements:\n{obj_structure_cases}\n"
        f"High-alignment anchor representative disagreements:\n{align_structure_cases}\n"
        f"{jssp_raw_trace}"
        "Synthesis direction:\n"
        "Treat the parents as Pareto siblings and attribute their behavioral differences using objective and teacher-trace evidence. "
        "Create one concise, coherent decision process, but do not assume either sibling supplies the backbone. Decide from the evidence "
        "what to retain, discard, reorganize, or transform, and choose the transformation without an implementation recipe. The child "
        "must not increase asymptotic per-decision complexity relative to the more efficient parent.\n"
        "Negative guidance:\n"
        "Do not default to Parent 1 plus a fallback, a reranker, a small patch, or coefficient averaging. "
        "Do not copy teacher-like behavior that only improves local agreement while worsening the objective; prefer performance-backed mechanisms over imitation-only patches."
    )
    s = (
        "Teacher-conditioned structural exploration brief:\n"
        f"{domain_s_context}"
        "Current Pareto pool:\n"
        f"{pareto}\n"
        f"{jssp_context}"
        f"{jssp_raw_trace}"
        f"High-performance parent stage summary:\n{obj_stage}\n"
        f"High-performance parent teacher-feature mismatch:\n{obj_pref}\n"
        f"High-performance parent representative disagreements:\n{obj_structure_cases}\n"
        "Primary proposal:\n"
        "Treat the selected parent as behavioral evidence rather than a mandatory template. Use its objective and teacher-trace diagnosis to decide "
        "what should be retained, removed, reorganized, or replaced. Choose one coherent structural experiment without an implementation recipe, "
        "keep its asymptotic per-decision complexity no worse than the parent, and avoid imitation-only behavior that improves agreement without improving the objective."
    )
    p = (
        "Trace-proposer-guided parameter calibration brief:\n"
        f"{domain_p_context}"
        "Current Pareto pool:\n"
        f"{pareto}\n"
        f"{jssp_context}"
        f"{jssp_raw_trace}"
        f"High-performance parent stage summary:\n{obj_stage}\n"
        f"High-performance parent teacher-feature mismatch:\n{obj_pref}\n"
        "Primary proposal:\n"
        "Write the next version from the high-performance parent by keeping its scoring formula recognizable and retuning existing numeric tradeoffs "
        "suggested by its teacher-trace diagnosis. "
        "Do not add a new decision module; make the existing tradeoff sharper."
    )
    guidance = _empty_operator_guidance()
    # Fusion diagnoses its actual pair inline; population guidance is consumed
    # by structural rewrite and parameter calibration.
    guidance.update(
        {
            "e2": e2_pro,
            "m": e2_pro,
            "e3": e2_pro,
            "s": s,
            "p": p,
        }
    )
    return guidance


def _llm_operator_guidance(
    population: list[dict[str, object]],
    *,
    structural_transitions: list[dict[str, object]] | None = None,
) -> dict[str, object] | None:
    if os.environ.get("EOH_TA_USE_LLM_ANALYZER", "0") != "1":
        return None
    strict_analyzer = os.environ.get("EOH_TA_STRICT_ANALYZER", "0") == "1"
    family = _problem_family()
    if family not in {"maxcut", "jssp", "tsp", "cvrp"}:
        return None
    by_obj = min(population, key=lambda item: float(item["objective"]))
    with_align = [item for item in population if item.get("alignment_score") is not None]
    by_align = max(with_align, key=lambda item: float(item.get("alignment_score", 0.0))) if with_align else by_obj
    if family == "maxcut":
        prompt = _build_maxcut_analyzer_prompt(population, by_obj, by_align)
    elif family == "jssp":
        prompt = _build_jssp_analyzer_prompt(population, by_obj, by_align)
    else:
        prompt = _build_routing_analyzer_prompt(population, by_obj, by_align, family)
    structural_evidence = _structural_transition_prompt(structural_transitions or [])
    if structural_evidence:
        prompt += "\n\n" + structural_evidence
    response, payload = _request_analyzer_payload(prompt)
    if payload is None:
        failure_policy = os.environ.get(
            "EOH_TA_ANALYZER_FAILURE_POLICY",
            "raise" if strict_analyzer else "skip",
        ).strip().lower()
        if failure_policy == "reuse_last" and _LAST_VALID_ANALYZER_PAYLOAD is not None:
            payload = dict(_LAST_VALID_ANALYZER_PAYLOAD)
            payload["_analyzer_fallback"] = "reused_last_valid_payload"
            response = "[reused last valid analyzer payload after format failure]"
        elif failure_policy == "empty":
            payload = {
                "s": {},
                "p": {},
                "m": {},
                "_analyzer_fallback": "empty_evidence_only_payload",
            }
        if strict_analyzer:
            if payload is None:
                raise RuntimeError(
                    "LLM analyzer returned no valid JSON payload after all format retries"
                )
        elif payload is None:
            return None
    _log_analyzer_exchange(prompt, response, payload, population)
    s_sections = payload.get("s") or {}
    p_sections = payload.get("p") or {}
    e2_sections = payload.get("m") or {}
    s_hypotheses = (
        _json_hypotheses(
            s_sections,
            fallback_keys=("rewrite_direction", "fusion_direction", "mutation_direction"),
        )
        if structural_evidence
        else []
    )
    e2_hypotheses = (
        _json_hypotheses(
            e2_sections,
            fallback_keys=("mechanism_fusion_direction", "fusion_direction"),
        )
        if structural_evidence
        else []
    )
    domain_s_context = _domain_operator_context(family, "s")
    domain_p_context = _domain_operator_context(family, "p")
    domain_e2_context = _domain_operator_context(family, "m")
    e2_pro = (
        "Population-level mechanism-fusion guidance:\n"
        f"{domain_e2_context}"
        "Current Pareto pool:\n"
        f"{pareto_summary(population)}\n"
        f"Performance representative observation:\n{_json_text_any(e2_sections, ['performance_parent_diagnosis', 'performance_parent_mechanisms'])}\n"
        f"Alignment representative observation:\n{_json_text_any(e2_sections, ['alignment_parent_diagnosis', 'alignment_parent_mechanisms'])}\n"
        f"Teacher-conditioned attribution:\n{_json_text_any(e2_sections, ['teacher_conditioned_attribution', 'branch_attribution'])}\n"
        f"Mechanisms to retain:\n{_json_text_any(e2_sections, ['mechanisms_to_retain', 'mechanisms_to_preserve'])}\n"
        f"Mechanisms to discard or use as negative evidence:\n{_json_text_any(e2_sections, ['mechanisms_to_discard', 'do_not_copy'])}\n"
        f"Fusion direction:\n{_json_text_any(e2_sections, ['mechanism_fusion_direction', 'fusion_direction'])}\n"
        f"Counterevidence and warning:\n{_json_text_any(e2_sections, ['do_not_do', 'do_not_copy'])}\n"
        "Apply this as population-level evidence, not as a mandatory recipe. The actual E request will compare its selected pair directly."
    )
    s_prefix = (
        "Population-level structural hypothesis:\n"
        f"{domain_s_context}"
        "Current Pareto pool:\n"
        f"{pareto_summary(population)}\n"
    )
    s = (
        s_prefix
        +
        f"Performance representative observation:\n{_json_text(s_sections, 'performance_parent_diagnosis')}\n"
        f"Teacher-conditioned branch attribution:\n{_json_text_any(s_sections, ['teacher_conditioned_attribution', 'branch_attribution'])}\n"
        f"Backbone decision:\n{_json_text_any(s_sections, ['backbone_decision', 'dominant_signal_decision'])}\n"
        f"Mechanisms to retain:\n{_json_text_any(s_sections, ['mechanisms_to_retain', 'mechanisms_to_preserve'])}\n"
        f"Mechanisms to discard:\n{_json_text_any(s_sections, ['mechanisms_to_discard', 'rejectable_mechanisms', 'do_not_copy'])}\n"
        f"Representation change to consider:\n{_json_text_any(s_sections, ['representation_change', 'structural_representation'])}\n"
        f"Candidate rewrite hypothesis:\n{_json_text_any(s_sections, ['rewrite_direction', 'fusion_direction', 'mutation_direction'])}\n"
        f"Behavior to question:\n{_json_text_any(s_sections, ['do_not_copy', 'do_not_do'])}\n"
        "Validate this hypothesis against the actual selected parent supplied by the operator. Objective evidence decides whether the old "
        "backbone survives; teacher evidence diagnoses the behavioral consequence of changing it."
    )
    p_prefix = (
        "Population-level calibration hypothesis:\n"
        f"{domain_p_context}"
        "Current Pareto pool:\n"
        f"{pareto_summary(population)}\n"
    )
    p = (
        p_prefix
        +
        f"Calibration diagnosis:\n{_json_text_any(p_sections, ['parent_diagnosis', 'performance_parent_diagnosis'])}\n"
        f"Parameter axes:\n{_json_list_text(p_sections.get('parameter_axes'))}\n"
        f"Calibration direction:\n{_json_text_any(p_sections, ['calibration_direction', 'rewrite_direction', 'fusion_direction'])}\n"
        f"Calibration warning:\n{_json_text_any(p_sections, ['do_not_do', 'do_not_copy'])}\n"
        "Validate this hypothesis against the actual selected parent supplied by the operator."
    )
    guidance = _empty_operator_guidance()
    guidance.update(
        {
            "e2": e2_pro,
            "m": e2_pro,
            "e3": e2_pro,
            "s": s,
            "p": p,
            "s_hypotheses": s_hypotheses,
            "m_hypotheses": e2_hypotheses,
        }
    )
    return guidance


def _request_analyzer_payload(
    prompt: str,
) -> tuple[str, dict[str, object] | None]:
    global _LAST_VALID_ANALYZER_PAYLOAD
    attempts = max(
        1,
        int(os.environ.get("EOH_TA_ANALYZER_JSON_RETRIES", "3")),
    )
    last_response = ""
    for attempt in range(attempts):
        if attempt and last_response:
            repair_limit = max(
                2000,
                int(os.environ.get("EOH_TA_ANALYZER_REPAIR_CHARS", "24000")),
            )
            request_prompt = (
                "Repair the following analyzer response into exactly one valid JSON object. "
                "Preserve its substantive content; do not redo the analysis. The top-level "
                "object must contain s, p, and m objects. Output JSON only, without "
                "Markdown fences or comments.\n\nMALFORMED RESPONSE:\n"
                + last_response[:repair_limit]
            )
        else:
            request_prompt = prompt
        response = _llm_text_response(request_prompt)
        if not response:
            continue
        last_response = response
        payload = _extract_json_payload(response)
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("s"), dict)
            and isinstance(payload.get("p"), dict)
            and isinstance(payload.get("m"), dict)
        ):
            _LAST_VALID_ANALYZER_PAYLOAD = dict(payload)
            return response, payload
    failure_policy = os.environ.get(
        "EOH_TA_ANALYZER_FAILURE_POLICY",
        "raise",
    ).strip().lower()
    if failure_policy in {"reuse_last", "reuse_last_or_empty"}:
        if _LAST_VALID_ANALYZER_PAYLOAD is not None:
            payload = dict(_LAST_VALID_ANALYZER_PAYLOAD)
            payload["_analyzer_fallback"] = "reused_last_valid_payload"
            return "[reused last valid analyzer payload after format failure]", payload
        if failure_policy == "reuse_last_or_empty":
            return (
                "[empty evidence-only analyzer payload after format failure]",
                {
                    "s": {},
                    "p": {},
                    "m": {},
                    "_analyzer_fallback": "empty_evidence_only_payload",
                },
            )
    return last_response, None


def _empty_operator_guidance() -> dict[str, object]:
    return {
        "e1": "",
        "e2": "",
        "m": "",
        "e3": "",
        "m1": "",
        "s": "",
        "p": "",
        "s_hypotheses": [],
        "m_hypotheses": [],
    }


def _structural_transition_prompt(transitions: list[dict[str, object]]) -> str:
    if os.environ.get("EOH_STRUCTURAL_MEMORY", "0") != "1":
        return ""
    case_limit = max(
        1,
        int(os.environ.get("EOH_STRUCTURAL_MEMORY_CASES", "3")),
    )
    evidence = transition_evidence_text(
        transitions,
        success_limit=case_limit,
        failure_limit=case_limit,
    )
    if not evidence:
        return ""
    hypothesis_limit = max(
        2,
        int(os.environ.get("EOH_STRUCTURAL_HYPOTHESES", "4")),
    )
    return (
        evidence
        + "\n\nUse these evaluated transitions for structural credit assignment. "
        "Do not infer that a whole child is reusable merely because it improved; identify the "
        "program transformation that plausibly caused the change, and use failed transitions "
        "to distinguish useful changes from repeated dead ends.\n"
        f"Add a structural_hypotheses array with {hypothesis_limit} distinct entries under both "
        "s and m in the JSON response. Each entry must have exactly three string fields: "
        "hypothesis (a task-independent program transformation to test), evidence (which successful "
        "or failed transitions support it), and avoid (the nearby failure mode). The hypotheses must "
        "cover materially different program transformations inferred from the evaluated evidence rather than from "
        "a predefined menu. Each hypothesis must preserve the operator's interface and must not increase asymptotic "
        "per-decision complexity. They are search proposals, not instructions to copy a complete program."
    )


def _json_hypotheses(
    section: dict[str, object],
    *,
    fallback_keys: tuple[str, ...],
) -> list[dict[str, str]]:
    limit = max(1, int(os.environ.get("EOH_STRUCTURAL_HYPOTHESES", "4")))
    value = section.get("structural_hypotheses")
    hypotheses: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                hypothesis = item.strip()
                evidence = ""
                avoid = ""
            elif isinstance(item, dict):
                hypothesis = str(item.get("hypothesis") or "").strip()
                evidence = str(item.get("evidence") or "").strip()
                avoid = str(item.get("avoid") or "").strip()
            else:
                continue
            if not hypothesis:
                continue
            hypotheses.append(
                {
                    "hypothesis": hypothesis,
                    "evidence": evidence,
                    "avoid": avoid,
                }
            )
            if len(hypotheses) >= limit:
                break
    if hypotheses:
        return hypotheses
    for key in fallback_keys:
        fallback = section.get(key)
        if isinstance(fallback, str) and fallback.strip():
            return [
                {
                    "hypothesis": fallback.strip(),
                    "evidence": "population-level analyzer fallback",
                    "avoid": "Do not copy a complete parent or make a cosmetic-only edit.",
                }
            ]
    return []


def build_teacher_views(records: list[dict[str, object]]) -> dict[str, str]:
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
        "stage_summary": _stage_summary_text(records),
        "structure_cases": summarize_case_records(records, max_cases=3),
        "parameter_cases": _top_case_records(records, max_cases=2),
        "structure_guidance": _structure_guidance_text(records),
        "parameter_guidance": _parameter_guidance_text(records),
        "feature_preference_summary": _feature_preference_summary_text(records),
    }


def _feature_preference_summary_text(records: list[dict[str, object]]) -> str:
    grouped = _stage_feature_percentile_deltas(records)
    lines: list[str] = []
    for stage in ("early", "mid", "late"):
        deltas = grouped.get(stage, {})
        if not deltas:
            lines.append(f"- {stage}: no teacher-feature preference mismatch was summarized.")
            continue
        ranked = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)[:4]
        phrases = []
        for feature, delta in ranked:
            direction = "larger" if delta > 0 else "smaller"
            phrases.append(f"{direction} {_feature_label(feature)} ({delta:+.2f} pct)")
        lines.append(
            f"- {stage}: relative to teacher, the current parent prefers "
            + ", ".join(phrases)
            + "."
        )
    return "\n".join(lines)


def _stage_feature_percentile_deltas(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {"early": {}, "mid": {}, "late": {}}
    for row in records:
        stage = str(row.get("stage") or _stage_from_remaining_fraction(float(row.get("remaining_fraction", 0.5))))
        chosen_pct = row.get("chosen_pct") or {}
        teacher_pct = row.get("teacher_pct") or {}
        shared = set(chosen_pct).intersection(teacher_pct)
        for feature in shared:
            delta = float(chosen_pct[feature]) - float(teacher_pct[feature])
            grouped.setdefault(stage, {}).setdefault(feature, []).append(delta)
    return {
        stage: {feature: float(mean(values)) for feature, values in feature_map.items() if values}
        for stage, feature_map in grouped.items()
    }


def _feature_label(feature: str) -> str:
    labels = {
        "flip_gain": "immediate flip gain",
        "normalized_gain": "degree-normalized gain",
        "same_weight": "same-side edge pressure",
        "opposite_weight": "opposite-side edge weight",
        "weighted_degree": "weighted degree",
        "tabu_age": "tabu age",
        "recent_flip": "recent-flip indicator",
        "degree": "degree",
        "side0_fraction": "side-0 balance",
        "side1_fraction": "side-1 balance",
        "step_fraction": "late-phase preference",
        "same_weight_share": "same-side weight share",
        "opposite_weight_share": "opposite-side weight share",
        "boundary_mix": "boundary mixedness",
        "neighbor_gain_mean": "neighbor gain mean",
        "neighbor_gain_max": "neighbor gain max",
        "positive_neighbor_gain_count": "positive-neighbor count",
        "positive_neighbor_gain_sum": "positive-neighbor gain sum",
        "neighbor_weighted_degree_mean": "neighbor weighted-degree mean",
        "neighbor_weighted_degree_max": "neighbor weighted-degree max",
        "two_hop_degree_proxy": "two-hop degree proxy",
        "after_positive_gain_count": "post-flip positive-gain count",
        "after_positive_gain_count_delta": "post-flip positive-gain count delta",
        "after_max_gain": "post-flip max gain",
        "after_top5_gain_mean": "post-flip top-5 gain mean",
        "after_positive_gain_sum": "post-flip positive-gain sum",
        "travel_from_current": "travel distance from current",
        "nearest_remaining_distance": "nearest remaining distance",
        "distance_to_remaining_centroid_sq": "distance to remaining centroid",
        "saving_proxy": "saving proxy",
        "heading_alignment": "heading alignment",
        "via_depot": "restart-via-depot choice",
        "distance_to_depot": "distance to depot",
        "demand_ratio": "demand ratio",
        "remaining_capacity_ratio": "remaining capacity ratio",
        "capacity_after_ratio": "capacity after ratio",
        "is_capacity_tight": "capacity-tight indicator",
    }
    return labels.get(feature, feature.replace("_", " "))


def _build_maxcut_analyzer_prompt(
    population: list[dict[str, object]],
    by_obj: dict[str, object],
    by_align: dict[str, object],
) -> str:
    adapter = _domain_adapter("maxcut")
    return (
        "You are analyzing a policy behavioral traces for Max-Cut node-flipping local-search heuristics.\n"
        "Your job is to summarize behavior patterns from traces, not to write code.\n"
        f"Task objective:\n{adapter['objective']}\n"
        f"Domain feature semantics:\n{adapter['features']}\n"
        f"Valid structural modules for s:\n{adapter['s_scope']}\n"
        f"Valid calibration axes for p:\n{adapter['p_scope']}\n"
        f"Known domain failure modes:\n{adapter['failure_modes']}\n"
        "Use the unified proposer framework: s proposes one structural module rewrite from the high-performance parent; "
        "p preserves that parent's backbone and calibrates existing numeric tradeoffs.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "s": {\n'
        '    "performance_parent_diagnosis": "...",\n'
        '    "alignment_parent_diagnosis": "...",\n'
        '    "teacher_conditioned_attribution": "...",\n'
        '    "backbone_decision": "...",\n'
        '    "mechanisms_to_retain": "...",\n'
        '    "mechanisms_to_discard": "...",\n'
        '    "representation_change": "...",\n'
        '    "rewrite_direction": "...",\n'
        '    "do_not_copy": "...",\n'
        '    "do_not_do": "..."\n'
        "  },\n"
        '  "p": {\n'
        '    "parent_diagnosis": "...",\n'
        '    "parameter_axes": ["...", "..."],\n'
        '    "calibration_direction": "...",\n'
        '    "do_not_do": "..."\n'
        "  },\n"
        '  "m": {\n'
        '    "performance_parent_mechanisms": "...",\n'
        '    "alignment_parent_mechanisms": "...",\n'
        '    "teacher_conditioned_attribution": "...",\n'
        '    "mechanisms_to_retain": "...",\n'
        '    "mechanisms_to_discard": "...",\n'
        '    "mechanism_fusion_direction": "...",\n'
        '    "do_not_copy": "...",\n'
        '    "do_not_do": "..."\n'
        "  }\n"
        "}\n\n"
        "Focus on:\n"
        "- what makes the high-performance parent strong\n"
        "- what teacher-captured preference the high-alignment parent has that the high-performance parent misses\n"
        "- whether teacher disagreement is high-confidence or low-margin noise; avoid overreacting to low teacher margins\n"
        "- whether the teacher preference is actually objective-useful; do not treat ECO-DQN top-1 agreement as a direct target\n"
        "- whether each proposed change is implementable from the action interface inputs: spins, adjacency, current_cut, best_cut, step, max_steps, flip_history\n"
        "- whether the proposal preserves the correct gain sign: same-side weight minus opposite-side weight is the immediate cut gain\n"
        "- graph-local evidence such as boundary mixedness, neighbor gains, hub-neighborhood context, and one-step after-flip follow-up potential\n"
        "- what teacher-conditioned structural experiment should be attempted in s, including whether the dominant backbone should be retained or replaced\n"
        "- what proposer-guided parameter calibration should be attempted in p\n"
        "- what population-level mechanism contrast may help E fuse a selected pair\n"
        "- do not recommend fallback-only, trivial reranking-only, or patch-layer edits unless the evidence really supports that\n\n"
        f"Pareto pool:\n{pareto_summary(population)}\n\n"
        f"High-performance parent algorithm:\n{str(by_obj.get('algorithm') or '').strip()}\n"
        f"High-performance parent feature-preference summary:\n{str(by_obj.get('feature_preference_summary') or 'No teacher-feature preference mismatch was summarized.')}\n"
        f"High-performance parent representative disagreements:\n{str(by_obj.get('structure_cases') or 'No sampled disagreement cases.')}\n\n"
        f"High-alignment parent algorithm:\n{str(by_align.get('algorithm') or '').strip()}\n"
        f"High-alignment parent feature-preference summary:\n{str(by_align.get('feature_preference_summary') or 'No teacher-feature preference mismatch was summarized.')}\n"
        f"High-alignment parent representative disagreements:\n{str(by_align.get('structure_cases') or 'No sampled disagreement cases.')}\n"
    )


def _build_jssp_analyzer_prompt(
    population: list[dict[str, object]],
    by_obj: dict[str, object],
    by_align: dict[str, object],
) -> str:
    adapter = _domain_adapter("jssp")
    baseline_guide = str(by_obj.get("baseline_guide") or by_align.get("baseline_guide") or "").strip()
    baseline_summary = str(by_obj.get("baseline_summary") or by_align.get("baseline_summary") or "").strip()
    perf_teacher_trace = str(
        by_obj.get("teacher_trace_summary")
        or "teacher_on_candidate_states: no sampled non-terminal decision states."
    ).strip()
    align_teacher_trace = str(
        by_align.get("teacher_trace_summary")
        or "teacher_on_candidate_states: no sampled non-terminal decision states."
    ).strip()
    perf_trace = str(by_obj.get("trace_summary") or "candidate: no sampled non-terminal decision states.").strip()
    align_trace = str(by_align.get("trace_summary") or "candidate: no sampled non-terminal decision states.").strip()
    perf_teacher_delta = _trace_delta_text_from_items("best_makespan - teacher", by_obj, teacher_source=by_obj)
    align_teacher_delta = _trace_delta_text_from_items("best_alignment - teacher", by_align, teacher_source=by_align)
    return (
        "You are a search proposer for LLM-based heuristic evolution on Job Shop Scheduling.\n"
        "Do not write code.\n"
        f"Task objective:\n{adapter['objective']}\n"
        "The executable program scores a feasible operation using only the declared feature/state interface.\n"
        f"Observable feature semantics:\n{adapter['features']}\n"
        "Do not assume a predefined heuristic module. Infer useful structural or calibration hypotheses from "
        "the objective results, code, and paired behavioral evidence below.\n"
        "Produce strict JSON only with this schema:\n"
        "{\n"
        '  "s": {\n'
        '    "performance_parent_diagnosis": "...",\n'
        '    "alignment_parent_diagnosis": "...",\n'
        '    "teacher_conditioned_attribution": "...",\n'
        '    "backbone_decision": "...",\n'
        '    "mechanisms_to_retain": "...",\n'
        '    "mechanisms_to_discard": "...",\n'
        '    "representation_change": "...",\n'
        '    "rewrite_direction": "...",\n'
        '    "do_not_copy": "...",\n'
        '    "do_not_do": "..."\n'
        "  },\n"
        '  "p": {\n'
        '    "parent_diagnosis": "...",\n'
        '    "parameter_axes": ["...", "..."],\n'
        '    "calibration_direction": "...",\n'
        '    "do_not_do": "..."\n'
        "  },\n"
        '  "m": {\n'
        '    "performance_parent_mechanisms": "...",\n'
        '    "alignment_parent_mechanisms": "...",\n'
        '    "teacher_conditioned_attribution": "...",\n'
        '    "mechanisms_to_retain": "...",\n'
        '    "mechanisms_to_discard": "...",\n'
        '    "mechanism_fusion_direction": "...",\n'
        '    "do_not_copy": "...",\n'
        '    "do_not_do": "..."\n'
        "  }\n"
        "}\n\n"
        "Each representative program is compared with the Offline-LD teacher on states induced by that same "
        "program. Candidate and teacher offsets within a pair use exactly the same feasible action set. "
        "The two representatives may visit different states, so do not interpret their raw offsets as a "
        "controlled pairwise comparison.\n"
        "Trace summaries report feature(chosen_action) - average feature(feasible_actions). Positive means the policy prefers larger values than its local feasible average.\n"
        "Use the comparison to distinguish makespan-improving behavior from teacher-imitation-only behavior.\n"
        "Do not recommend copying the teacher blindly.\n"
        "Explicitly call out teacher-like behaviors that should not be copied when they appear to improve agreement but not makespan.\n\n"
        f"Pareto pool:\n{pareto_summary(population)}\n\n"
        f"Baseline heuristic meanings:\n{baseline_guide or 'No baseline guide was provided.'}\n\n"
        f"Baseline metrics on the current training instances:\n{baseline_summary or 'No baseline metrics were provided.'}\n\n"
        f"Best makespan parent objective={float(by_obj.get('objective', 0.0)):.6f}, alignment={float(by_obj.get('alignment_score', 0.0)):.3f}, top1={float(by_obj.get('top1_agreement', 0.0)):.3f}\n"
        f"Best makespan parent algorithm:\n{str(by_obj.get('algorithm') or '').strip()}\n"
        f"Best makespan parent trace summary:\n{perf_trace}\n"
        f"Teacher actions on the best makespan parent's sampled states:\n{perf_teacher_trace}\n"
        f"Best makespan parent teacher-feature mismatch:\n{str(by_obj.get('feature_preference_summary') or 'No teacher-feature preference mismatch was summarized.')}\n"
        f"Best makespan parent representative disagreements:\n{str(by_obj.get('structure_cases') or 'No sampled disagreement cases.')}\n\n"
        f"Best teacher-aligned parent objective={float(by_align.get('objective', 0.0)):.6f}, alignment={float(by_align.get('alignment_score', 0.0)):.3f}, top1={float(by_align.get('top1_agreement', 0.0)):.3f}\n"
        f"Best teacher-aligned parent algorithm:\n{str(by_align.get('algorithm') or '').strip()}\n"
        f"Best teacher-aligned parent trace summary:\n{align_trace}\n"
        f"Teacher actions on the aligned parent's sampled states:\n{align_teacher_trace}\n"
        f"Best teacher-aligned parent teacher-feature mismatch:\n{str(by_align.get('feature_preference_summary') or 'No teacher-feature preference mismatch was summarized.')}\n"
        f"Best teacher-aligned parent representative disagreements:\n{str(by_align.get('structure_cases') or 'No sampled disagreement cases.')}\n\n"
        "Paired trace deltas. Positive means the candidate chooses actions with larger feature offsets than "
        "the teacher on the same candidate-induced states.\n"
        f"{perf_teacher_delta}\n"
        f"{align_teacher_delta}\n\n"
        "For s, use the performance representative and paired teacher evidence to attribute the current branches, then decide whether "
        "the dominant backbone should survive. Propose one testable structural experiment inferred from the evidence, without prescribing "
        "implementation devices. The actual operator may use a different sampled parent and will validate this hypothesis against it.\n"
        "For p, form one focused calibration hypothesis rather than a long list of knobs.\n"
        "For m, summarize population-level mechanisms and counterevidence; the operator will compare its actual selected pair itself.\n"
        "Reject proposals that increase asymptotic per-decision complexity; a per-candidate score must not scan the full instance or feasible set.\n"
    )


def _domain_adapter(family: str) -> dict[str, str]:
    if family == "maxcut":
        return {
            "task": "Max-Cut node-flip local-search heuristics",
            "objective_short": "final cut value after a fixed local-search budget",
            "objective": (
                "Repeatedly choose one node to flip from the current 0/1 partition. The evaluator records the best cut value reached during a fixed "
                "local-search budget, so a useful rule must both intensify around improving flips and escape when the current partition is locally stuck. "
                "Training objective values are negated cut values; lower objective means better cut quality. Teacher agreement is only behavioral evidence, "
                "not an objective surrogate."
            ),
            "features": (
                "Runtime action interface inputs are spins, adjacency, current_cut, best_cut, step, max_steps, and flip_history. Generated code must compute "
                "candidate statistics from these inputs; feature.* fields exist only in score-interface diagnostics.\n"
                "- immediate_gain for node i must be same_side_weight - opposite_side_weight. This sign is critical: positive gain immediately increases the cut.\n"
                "- normalized_gain = immediate_gain / weighted_degree separates true improving moves from high-degree scale effects.\n"
                "- same_side_weight, opposite_side_weight, same/opposite shares, and boundary_mix describe whether the node is embedded in one side, already on the cut boundary, or a repair opportunity.\n"
                "- weighted_degree/degree identify hubs. In BA weighted graphs, hubs matter, but pure degree chasing is harmful unless gain or follow-up evidence supports it.\n"
                "- time_since_flip from flip_history, aspiration_gain = max(0, current_cut + gain - best_cut), and recent-flip indicators control cycling.\n"
                "- side balance after a flip is available from spins, but balance is only a weak regularizer; do not optimize balance for its own sake.\n"
                "- one-step lookahead can be approximated by updating neighbor gains after a hypothetical flip and measuring after_max_gain, after_top_k_gain, or positive-gain count.\n"
                "- step_fraction separates early construction/intensification, mid search, and late escape/recovery behavior."
            ),
            "s_scope": (
                "Choose exactly one structural module to rewrite, while keeping the action interface deterministic: "
                "candidate admissibility/cycling control, aspiration override, phase scheduler, candidate shortlist, immediate-gain versus normalized-gain backbone, "
                "one-step neighbor-lookahead module, hub/boundary modifier for BA weighted graphs, plateau/negative-gain escape rule, or final tie-break hierarchy. "
                "A structural rewrite should change which candidates are considered or which decision layer dominates, not merely rescale constants."
            ),
            "p_scope": (
                "Retune weights, exponents, clipping, normalizers, phase boundaries, aspiration thresholds, recency tenure, recent-flip penalty shape, "
                "gain versus normalized-gain balance, hub bonus/penalty, boundary pressure, balance penalty, neighbor-lookahead coefficients, escape thresholds, "
                "and deterministic tie-break priorities. Keep the same admissibility and scoring backbone unless s is being used."
            ),
            "failure_modes": (
                "Avoid copying ECO-DQN teacher agreement when it weakens cut value; high top-1 agreement can be imitation noise. Avoid reversing the gain sign, "
                "over-penalizing recent flips so the search cannot revisit useful hubs, pure degree chasing, ignoring positive immediate_gain, "
                "optimizing cosmetic balance, too many nested fallback layers, randomization, expensive all-pairs simulation inside every step, "
                "and rerank patches that leave the main decision unchanged."
            ),
        }
    if family == "jssp":
        return {
            "task": "Job Shop Scheduling dispatching-rule heuristics",
            "objective_short": "makespan of a constructed schedule",
            "objective": (
                "At each construction step, score the feasible next operation of each unfinished job and schedule the highest-scoring operation. "
                "Minimize final makespan while respecting job precedence and single-machine capacity. Teacher traces are evidence, not the objective."
            ),
            "features": (
                "- processing_time: local machine occupancy; short operations reduce waiting, long operations may clear critical work.\n"
                "- remaining_work/remaining_ops: unfinished tail of the job; high values indicate critical-job pressure.\n"
                "- job_ready_time/machine_ready_time/earliest_start/earliest_finish: immediate timing and waiting pressure.\n"
                "- machine_queue_len/machine_total_work: bottleneck-machine pressure.\n"
                "- job_progress: whether the job is early or near its tail.\n"
                "- lower_bound_after: simple future makespan risk after choosing the operation.\n"
                "- state exposes actions, job_ready, next_op, makespan, and instance. The declared nested instance fields are "
                "durations, machines, n_jobs, n_machines, and n_operations. These are valid observable context and may be "
                "aggregated into new scalar statistics; no undeclared attributes may be invented."
            ),
            "s_scope": (
                "Choose exactly one structural module to rewrite: critical-job priority, bottleneck-machine priority, earliest-finish/start gate, "
                "tail-completion behavior, slack/lower-bound risk handling, phase schedule by progress, or tie-break hierarchy among feasible operations."
            ),
            "p_scope": (
                "Retune weights, nonlinear transforms, clipping, phase boundaries, slack/lower-bound pressure, bottleneck coefficients, "
                "remaining-work/remaining-ops balance, processing-time preference, and tie-break priorities. Keep the same dispatching backbone."
            ),
            "failure_modes": (
                "Avoid blindly imitating Offline-LD when it hurts makespan, pure SPT or pure MWKR collapse, starving short operations, "
                "overweighting machine congestion without job criticality, and state-field inventions outside score(feature, state)."
            ),
        }
    if family == "cvrp":
        return {
            "task": "Capacitated Vehicle Routing constructive heuristics",
            "objective_short": "total route length with capacity-feasible depot-separated routes",
            "objective": (
                "Construct capacity-feasible vehicle routes. Each decision either serves one feasible customer on the current route "
                "or starts/continues from the depot when route closure is appropriate. Minimize total travel distance, including depot returns; "
                "do not trade route feasibility for teacher agreement."
            ),
            "features": (
                "Runtime inputs are current_node, depot, unvisited_nodes, rest_capacity, demands, and distance_matrix.\n"
                "- depot/restart choice: returning depot is legal except when already at depot; it closes the current route.\n"
                "- distance_matrix[current_node][candidate]: added local travel if served next; smaller is compact continuation.\n"
                "- distance_matrix[candidate][depot]: customer remoteness and future route-return cost.\n"
                "- candidate-to-unvisited min/mean distances from distance_matrix: local cluster support or isolation.\n"
                "- demands[candidate], demand/rest_capacity, and rest_capacity - demand: route fill and residual capacity pressure.\n"
                "- saving proxy distance(current,depot)+distance(depot,candidate)-distance(current,candidate): in-route saving versus serving separately.\n"
                "Teacher traces may additionally report capacity_after_ratio/is_capacity_tight using the true capacity; generated code should approximate them from rest_capacity."
            ),
            "s_scope": (
                "Choose exactly one structural module to rewrite while preserving capacity feasibility: route seeding/opening policy, "
                "customer shortlist construction, continuation score composition, route close/restart gate, phase schedule between expansion and compaction, "
                "or late tail-packing policy. Strong CVRP structures often combine remote/depot-aware seeding, savings continuation, cluster support, "
                "capacity fill, and late compact tail packing."
            ),
            "p_scope": (
                "Retune weights, exponents, gates, normalizers, phase boundaries, shortlist width, route-fill thresholds, depot-close thresholds, "
                "saving-vs-travel balance, demand/fill pressure, cluster support, and tie-break priorities. Keep the same route-construction backbone."
            ),
            "failure_modes": (
                "Avoid over-extreme farthest seeding, underfilled routes, premature depot restart, pure saving moves that create long detours, "
                "ignoring demand/fill pressure, and adding a second decision branch that violates the current route feasibility logic."
            ),
        }
    if family == "tsp":
        return {
            "task": "Traveling Salesman constructive heuristics",
            "objective_short": "tour length over all cities",
            "objective": (
                "Construct one Hamiltonian tour by repeatedly selecting the next unvisited city. Minimize closed-tour length. "
                "There is no capacity or depot restart decision; every step should balance local travel, future insertion difficulty, and global coverage."
            ),
            "features": (
                "Runtime inputs are current_node, destination_node, unvisited_nodes, and distance_matrix. Coordinates, previous_node, and visited order are not available.\n"
                "- distance_matrix[current_node][candidate]: immediate edge length from current city; smaller is local compactness.\n"
                "- distance_matrix[candidate][destination_node]: final return pressure to the start/depot node.\n"
                "- candidate-to-unvisited min/mean distances from distance_matrix: cluster support, isolation, and future insertion difficulty.\n"
                "- candidate rank/order inside unvisited_nodes: the evaluator passes a nearest-current shortlist, so early ranks are locally close to current.\n"
                "- remaining count and distance dispersion over unvisited_nodes: rough phase/global-coverage signals.\n"
                "- saving proxy distance(current,destination)+distance(destination,candidate)-distance(current,candidate): return-aware move preference.\n"
                "Teacher traces may report centroid or heading diagnostics computed from coordinates; generated code should translate them into distance-matrix proxies rather than using unavailable fields."
            ),
            "s_scope": (
                "Choose exactly one structural module to rewrite: candidate shortlist, local-vs-global phase schedule, outlier/centroid handling, "
                "nearest-cluster continuation, directional sweep behavior, or tie-break hierarchy. Do not introduce depot/capacity concepts."
            ),
            "p_scope": (
                "Retune weights, exponents, normalization, phase boundaries, shortlist width, outlier pressure, centroid pressure, local-neighbor pressure, "
                "heading continuity, saving-vs-distance balance, and tie-break priorities. Keep the same next-city scoring backbone."
            ),
            "failure_modes": (
                "Avoid greedy nearest-neighbor collapse, excessive outlier jumps, late isolated clusters, zig-zag directional changes, "
                "and any CVRP-style route restart or capacity logic."
            ),
        }
    return {"task": "heuristic search", "objective_short": "task objective", "objective": "", "features": "", "s_scope": "", "p_scope": "", "failure_modes": ""}


def _domain_operator_context(family: str, operator: str) -> str:
    if family not in {"maxcut", "jssp", "tsp", "cvrp"}:
        return ""
    adapter = _domain_adapter(family)
    if operator == "s":
        return (
            "S is a structural rewrite operator. The domain adapter supplies only the optimization objective "
            "and observable interface; it does not prescribe a heuristic or rank actions.\n"
            f"Objective: {adapter['objective_short']}.\n"
        )
    if operator == "p":
        return (
            "P is a parameter calibration operator. The domain adapter supplies only the optimization "
            "objective and observable interface; it does not prescribe which parameters should change.\n"
            f"Objective: {adapter['objective_short']}.\n"
        )
    if operator in {"e2", "m", "e3"}:
        return (
            "E2-lite is a mechanism comparison and synthesis operator. The domain adapter supplies only the "
            "optimization objective and observable interface; the evidence decides what transfers.\n"
            f"Objective: {adapter['objective_short']}.\n"
        )
    return ""


def _build_routing_analyzer_prompt(
    population: list[dict[str, object]],
    by_obj: dict[str, object],
    by_align: dict[str, object],
    family: str,
) -> str:
    adapter = _domain_adapter(family)
    return (
        f"You are analyzing teacher-student behavioral traces for {adapter['task']}.\n"
        "Your job is to summarize behavior patterns from traces, not to write code.\n"
        "Lower objective is better. Parent 1 is the high-performance anchor. Parent 2 is the high-alignment anchor.\n"
        f"Task objective:\n{adapter['objective']}\n"
        f"Domain feature semantics:\n{adapter['features']}\n"
        f"Valid structural modules for s:\n{adapter['s_scope']}\n"
        f"Valid calibration axes for p:\n{adapter['p_scope']}\n"
        f"Known domain failure modes:\n{adapter['failure_modes']}\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "s": {\n'
        '    "performance_parent_diagnosis": "...",\n'
        '    "alignment_parent_diagnosis": "...",\n'
        '    "teacher_conditioned_attribution": "...",\n'
        '    "backbone_decision": "...",\n'
        '    "mechanisms_to_retain": "...",\n'
        '    "mechanisms_to_discard": "...",\n'
        '    "representation_change": "...",\n'
        '    "rewrite_direction": "...",\n'
        '    "do_not_copy": "...",\n'
        '    "do_not_do": "..."\n'
        "  },\n"
        '  "p": {\n'
        '    "parent_diagnosis": "...",\n'
        '    "parameter_axes": ["...", "..."],\n'
        '    "calibration_direction": "...",\n'
        '    "do_not_do": "..."\n'
        "  },\n"
        '  "m": {\n'
        '    "performance_parent_mechanisms": "...",\n'
        '    "alignment_parent_mechanisms": "...",\n'
        '    "teacher_conditioned_attribution": "...",\n'
        '    "mechanisms_to_retain": "...",\n'
        '    "mechanisms_to_discard": "...",\n'
        '    "mechanism_fusion_direction": "...",\n'
        '    "do_not_copy": "...",\n'
        '    "do_not_do": "..."\n'
        "  }\n"
        "}\n\n"
        "Focus on:\n"
        f"- what makes the high-performance parent strong on {adapter['objective_short']}\n"
        "- what teacher-captured preference the high-alignment parent has that the high-performance parent misses\n"
        "- whether teacher disagreement is high-confidence or low-margin noise; avoid overreacting to low teacher margins\n"
        "- route-local evidence from the feature-preference summaries and disagreement cases\n"
        "- s as one domain-valid structural experiment whose backbone decision is justified by objective and teacher-trace evidence\n"
        "- p as parameter calibration of the high-performance parent's existing domain modules\n\n"
        "- m as population-level mechanism evidence for a pair-local fusion decision\n\n"
        f"Pareto pool:\n{pareto_summary(population)}\n\n"
        f"Parent 1 algorithm:\n{str(by_obj.get('algorithm') or '').strip()}\n"
        f"Parent 1 objective={float(by_obj.get('objective', 0.0)):.6f}, alignment={float(by_obj.get('alignment_score', 0.0)):.3f}, top1={float(by_obj.get('top1_agreement', 0.0)):.3f}\n"
        f"Parent 1 feature-preference summary:\n{str(by_obj.get('feature_preference_summary') or 'No teacher-feature preference mismatch was summarized.')}\n"
        f"Parent 1 representative disagreements:\n{str(by_obj.get('structure_cases') or 'No sampled disagreement cases.')}\n"
        f"Parent 1 parameter cases:\n{str(by_obj.get('parameter_cases') or 'No sampled disagreement cases.')}\n\n"
        f"Parent 2 algorithm:\n{str(by_align.get('algorithm') or '').strip()}\n"
        f"Parent 2 objective={float(by_align.get('objective', 0.0)):.6f}, alignment={float(by_align.get('alignment_score', 0.0)):.3f}, top1={float(by_align.get('top1_agreement', 0.0)):.3f}\n"
        f"Parent 2 feature-preference summary:\n{str(by_align.get('feature_preference_summary') or 'No teacher-feature preference mismatch was summarized.')}\n"
        f"Parent 2 representative disagreements:\n{str(by_align.get('structure_cases') or 'No sampled disagreement cases.')}\n"
        f"Parent 2 parameter cases:\n{str(by_align.get('parameter_cases') or 'No sampled disagreement cases.')}\n"
    )


def _log_analyzer_exchange(
    prompt: str,
    response: str,
    payload: dict[str, object],
    population: list[dict[str, object]],
) -> None:
    path = os.environ.get("EOH_TA_ANALYZER_LOG_PATH", "").strip()
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    representatives = []
    for item in population:
        code = str(item.get("code") or "")
        representatives.append(
            {
                "objective": item.get("objective"),
                "alignment_score": item.get("alignment_score"),
                "top1_agreement": item.get("top1_agreement"),
                "fe_index": item.get("fe_index"),
                "fe_phase": item.get("fe_phase"),
                "code_prefix": code[:120],
            }
        )
    record = {
        "timestamp": time.time(),
        "family": _problem_family(),
        "prompt": prompt,
        "response": response,
        "payload": payload,
        "population": representatives,
    }
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _llm_text_response(prompt: str) -> str | None:
    endpoint = (
        os.environ.get("EOH_TA_ANALYZER_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    api_key = (
        os.environ.get("EOH_TA_ANALYZER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    model = (
        os.environ.get("EOH_TA_ANALYZER_MODEL")
        or os.environ.get("MODEL")
        or "gpt-5.4"
    ).strip()
    if not endpoint or not api_key:
        return None
    url = endpoint.rstrip("/")
    if url.endswith("/v1"):
        url = url + "/chat/completions"
    elif not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    reasoning_effort = (
        os.environ.get("EOH_TA_ANALYZER_REASONING_EFFORT")
        or os.environ.get("LLM_REASONING_EFFORT")
        or ""
    ).strip()
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-api2d-no-cache": "1",
        },
        method="POST",
    )
    timeout_s = int(os.environ.get("EOH_TA_ANALYZER_TIMEOUT_S", "180"))
    request_deadline = time.monotonic() + timeout_s
    retries = int(os.environ.get("EOH_TA_ANALYZER_RETRIES", "2"))
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                from tae.engine.llm.api_general import InterfaceAPI
                reader = InterfaceAPI(endpoint, api_key, model, False)
                reader._read_streaming_content = _read_streaming_chat_content
                content = reader._read_with_deadline(response, request_deadline)
            if content:
                return content
            raise RuntimeError("analyzer returned an empty streaming response")
        except (
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            RuntimeError,
            http.client.HTTPException,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if _is_analyzer_timeout_error(exc):
                raise TimeoutError(f"analyzer request timed out: {exc}") from exc
            print(
                f"analyzer API retry {attempt}/{retries} after "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt == retries:
                break
            time.sleep(min(10, 2 ** attempt))
    return None


def _read_streaming_chat_content(response) -> str:
    content_parts: list[str] = []
    buffered_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if not line.startswith("data:"):
            buffered_lines.append(line)
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("error"):
            raise RuntimeError(f"analyzer stream error: {chunk['error']}")
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        piece = delta.get("content")
        if not isinstance(piece, str):
            piece = message.get("content")
        if isinstance(piece, str) and piece:
            content_parts.append(piece)
    if content_parts:
        return "".join(content_parts).strip()
    if buffered_lines:
        raw = json.loads("\n".join(buffered_lines))
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def _extract_json_payload(text: str) -> dict[str, object] | None:
    text = (text or "").strip()
    if not text:
        return None
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [text]
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        normalized = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            data = json.loads(normalized)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(normalized)
            except (SyntaxError, ValueError):
                continue
        if isinstance(data, dict):
            return data
    return None


def _json_text(section: dict[str, object], key: str) -> str:
    value = section.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "No LLM analyzer summary was produced."


def _json_text_any(section: dict[str, object], keys: list[str]) -> str:
    for key in keys:
        value = section.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Do not over-copy teacher-local preferences when they are not supported by objective evidence."


def _json_list_text(value: object) -> str:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return "\n".join(f"- {item}" for item in items)
    return "- No LLM analyzer calibration axes were produced."


def _problem_family() -> str:
    return os.environ.get("EOH_PROBLEM_FAMILY", "tsp").strip().lower()


def _trace_delta_between_items(label: str, left: dict[str, object], right: dict[str, object]) -> str:
    left_offsets = left.get("trace_stage_offsets")
    right_offsets = right.get("trace_stage_offsets")
    return _trace_delta_text(label, left_offsets, right_offsets)


def _trace_delta_text_from_items(label: str, left: dict[str, object], *, teacher_source: dict[str, object]) -> str:
    left_offsets = left.get("trace_stage_offsets")
    teacher_offsets = teacher_source.get("teacher_trace_stage_offsets")
    return _trace_delta_text(label, left_offsets, teacher_offsets)


def _trace_delta_text(label: str, left_offsets: object, right_offsets: object) -> str:
    if not isinstance(left_offsets, dict) or not isinstance(right_offsets, dict):
        return f"{label}:\n- no overlapping sampled stages"
    lines = [f"{label}:"]
    wrote = False
    for stage in ("early", "mid", "late"):
        left_stage = left_offsets.get(stage)
        right_stage = right_offsets.get(stage)
        if not isinstance(left_stage, dict) or not isinstance(right_stage, dict):
            continue
        rows = []
        for name in set(left_stage).intersection(right_stage):
            try:
                delta = float(left_stage[name]) - float(right_stage[name])
            except (TypeError, ValueError):
                continue
            rows.append((abs(delta), str(name), delta))
        rows.sort(reverse=True)
        if rows:
            lines.append(f"- {stage}: " + ", ".join(f"{name}={delta:+.3f}" for _abs, name, delta in rows[:5]))
            wrote = True
    if not wrote:
        lines.append("- no overlapping sampled stages")
    return "\n".join(lines)


def _jssp_baseline_context(by_obj: dict[str, object], by_align: dict[str, object]) -> str:
    if _problem_family() != "jssp":
        return ""
    guide = str(by_obj.get("baseline_guide") or by_align.get("baseline_guide") or "").strip()
    summary = str(by_obj.get("baseline_summary") or by_align.get("baseline_summary") or "").strip()
    if not guide and not summary:
        return ""
    parts = []
    if guide:
        parts.append("Baseline heuristic meanings:\n" + guide)
    if summary:
        parts.append("Baseline metrics on current training instances:\n" + summary)
    return "\n".join(parts) + "\n"


def _jssp_raw_trace_evidence(by_obj: dict[str, object], by_align: dict[str, object]) -> str:
    if _problem_family() != "jssp":
        return ""
    perf_teacher_summary = str(by_obj.get("teacher_trace_summary") or "").strip()
    align_teacher_summary = str(by_align.get("teacher_trace_summary") or "").strip()
    perf_trace = str(by_obj.get("trace_summary") or "").strip()
    align_trace = str(by_align.get("trace_summary") or "").strip()
    perf_teacher_delta = _trace_delta_text_from_items("best_makespan - teacher", by_obj, teacher_source=by_obj)
    align_teacher_delta = _trace_delta_text_from_items("best_alignment - teacher", by_align, teacher_source=by_align)
    blocks = [
        "Paired on-policy trace evidence:",
        "Best makespan parent trace summary:\n" + (perf_trace or "candidate: no sampled non-terminal decision states."),
        "Teacher actions on best makespan parent states:\n"
        + (perf_teacher_summary or "teacher_on_candidate_states: no sampled states."),
        "Best teacher-aligned parent trace summary:\n" + (align_trace or "candidate: no sampled non-terminal decision states."),
        "Teacher actions on aligned parent states:\n"
        + (align_teacher_summary or "teacher_on_candidate_states: no sampled states."),
        "Within-pair deltas. Positive means the candidate chooses actions with larger feature offsets than "
        "the teacher on the same candidate-induced states.",
        perf_teacher_delta,
        align_teacher_delta,
    ]
    return "\n".join(blocks) + "\n"


def _compact_profile(action_id: int, profile: dict[str, float], percentiles: dict[str, float]) -> str:
    if _is_maxcut_profile(profile):
        return (
            f"action_id={action_id}, "
            f"flip_gain={profile['flip_gain']:.3f}|pct={percentiles['flip_gain']:.2f}, "
            f"normalized_gain={profile['normalized_gain']:.3f}|pct={percentiles['normalized_gain']:.2f}, "
            f"same_weight={profile['same_weight']:.3f}|pct={percentiles['same_weight']:.2f}, "
            f"opposite_weight={profile['opposite_weight']:.3f}|pct={percentiles['opposite_weight']:.2f}, "
            f"weighted_degree={profile['weighted_degree']:.3f}|pct={percentiles['weighted_degree']:.2f}, "
            f"boundary_mix={profile.get('boundary_mix', 0.0):.3f}|pct={percentiles.get('boundary_mix', 0.0):.2f}, "
            f"neighbor_gain_max={profile.get('neighbor_gain_max', 0.0):.3f}|pct={percentiles.get('neighbor_gain_max', 0.0):.2f}, "
            f"after_top5_gain_mean={profile.get('after_top5_gain_mean', 0.0):.3f}|pct={percentiles.get('after_top5_gain_mean', 0.0):.2f}, "
            f"after_pos_delta={profile.get('after_positive_gain_count_delta', 0.0):+.0f}|pct={percentiles.get('after_positive_gain_count_delta', 0.0):.2f}, "
            f"tabu_age={profile['tabu_age']:.1f}|pct={percentiles['tabu_age']:.2f}, "
            f"recent_flip={profile['recent_flip']:.0f}|pct={percentiles['recent_flip']:.2f}"
        )
    if _is_cvrp_profile(profile):
        return (
            f"action_id={action_id}, "
            f"via_depot={profile['via_depot']:.0f}|pct={percentiles['via_depot']:.2f}, "
            f"travel_from_current={profile['travel_from_current']:.3f}|pct={percentiles['travel_from_current']:.2f}, "
            f"distance_to_depot={profile['distance_to_depot']:.3f}|pct={percentiles['distance_to_depot']:.2f}, "
            f"nearest_remaining_distance={profile['nearest_remaining_distance']:.3f}|pct={percentiles['nearest_remaining_distance']:.2f}, "
            f"demand_ratio={profile['demand_ratio']:.3f}|pct={percentiles['demand_ratio']:.2f}, "
            f"capacity_after_ratio={profile['capacity_after_ratio']:.3f}|pct={percentiles['capacity_after_ratio']:.2f}, "
            f"saving_proxy={profile['saving_proxy']:.3f}|pct={percentiles['saving_proxy']:.2f}"
        )
    return (
        f"action_id={action_id}, "
        f"travel_from_current={profile['travel_from_current']:.3f}|pct={percentiles['travel_from_current']:.2f}, "
        f"nearest_remaining_distance={profile['nearest_remaining_distance']:.3f}|pct={percentiles['nearest_remaining_distance']:.2f}, "
        f"distance_to_remaining_centroid_sq={profile['distance_to_remaining_centroid_sq']:.3f}|pct={percentiles['distance_to_remaining_centroid_sq']:.2f}, "
        f"saving_proxy={profile['saving_proxy']:.3f}|pct={percentiles['saving_proxy']:.2f}, "
        f"heading_alignment={profile['heading_alignment']:.3f}|pct={percentiles['heading_alignment']:.2f}"
    )


def _case_semantics_text(row: dict[str, object]) -> str:
    chosen = str(row.get("chosen_semantics") or "").strip()
    teacher = str(row.get("teacher_semantics") or "").strip()
    if not chosen and not teacher:
        return ""
    return f"; semantic_roles: chosen={chosen or 'n/a'}; teacher={teacher or 'n/a'}"


def _teacher_confidence_text(row: dict[str, object]) -> str:
    parts = []
    if row.get("chosen_teacher_rank") is not None:
        parts.append(f"chosen_teacher_rank={int(row['chosen_teacher_rank'])}")
    if row.get("teacher_top1_top2_margin") is not None:
        parts.append(f"top1_top2_margin={float(row['teacher_top1_top2_margin']):+.3f}")
    if row.get("teacher_score_spread") is not None:
        parts.append(f"score_spread={float(row['teacher_score_spread']):.3f}")
    top_actions = row.get("teacher_top_actions")
    if isinstance(top_actions, list) and top_actions:
        compact = []
        for item in top_actions[:3]:
            if not isinstance(item, dict):
                continue
            compact.append(
                "a{action}/d{delta:+.3f}/g{gain:.1f}/deg{degree:.1f}/aft{after:.1f}".format(
                    action=int(item.get("action", -1)),
                    delta=float(item.get("score_delta_from_top", 0.0)),
                    gain=float(item.get("flip_gain", 0.0)),
                    degree=float(item.get("weighted_degree", 0.0)),
                    after=float(item.get("after_top5_gain_mean", 0.0)),
                )
            )
        if compact:
            parts.append("teacher_top3=" + ",".join(compact))
    return "; " + "; ".join(parts) if parts else ""


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _stage_from_remaining_fraction(remaining_fraction: float) -> str:
    if remaining_fraction > (2.0 / 3.0):
        return "early"
    if remaining_fraction > (1.0 / 3.0):
        return "mid"
    return "late"


def _is_cvrp_profile(profile: dict[str, float]) -> bool:
    return "via_depot" in profile


def _is_maxcut_profile(profile: dict[str, float]) -> bool:
    return "flip_gain" in profile


def _case_tags(row: dict[str, object]) -> list[str]:
    chosen = row["chosen_profile"]
    teacher = row["teacher_profile"]
    if _is_maxcut_profile(chosen):
        return _case_tags_maxcut(chosen, teacher)
    if _is_cvrp_profile(chosen):
        return _case_tags_cvrp(chosen, teacher)
    return _case_tags_tsp(chosen, teacher)


def _case_tags_maxcut(chosen: dict[str, float], teacher: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if chosen["flip_gain"] + 1e-6 < teacher["flip_gain"]:
        tags.append("low_flip_gain")
    if chosen["normalized_gain"] + 1e-6 < teacher["normalized_gain"]:
        tags.append("low_normalized_gain")
    if chosen["recent_flip"] > teacher["recent_flip"] + 1e-6:
        tags.append("tabu_violation")
    if chosen["weighted_degree"] + 1e-6 < teacher["weighted_degree"] and chosen["flip_gain"] + 1e-6 < teacher["flip_gain"]:
        tags.append("underuse_high_degree")
    if chosen["same_weight"] + 1e-6 < teacher["same_weight"]:
        tags.append("weak_same_side_pressure")
    if abs(chosen["side1_fraction"] - 0.5) > abs(teacher["side1_fraction"] - 0.5) + 1e-6:
        tags.append("worse_balance")
    if chosen.get("after_top5_gain_mean", 0.0) + 1e-6 < teacher.get("after_top5_gain_mean", 0.0):
        tags.append("weaker_followup_potential")
    if chosen.get("after_positive_gain_count_delta", 0.0) + 1e-6 < teacher.get("after_positive_gain_count_delta", 0.0):
        tags.append("shrinks_positive_frontier")
    if chosen.get("boundary_mix", 0.0) + 1e-6 < teacher.get("boundary_mix", 0.0):
        tags.append("underuses_boundary_node")
    if chosen.get("neighbor_weighted_degree_mean", 0.0) + 1e-6 < teacher.get("neighbor_weighted_degree_mean", 0.0):
        tags.append("weaker_hub_neighborhood")
    if not tags:
        tags.append("misc_misalignment")
    return tags


def _case_tags_tsp(chosen: dict[str, float], teacher: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if chosen["travel_from_current"] + 1e-6 < teacher["travel_from_current"] and chosen["nearest_remaining_distance"] > teacher["nearest_remaining_distance"] + 1e-6:
        tags.append("over_local_isolated")
    if chosen["distance_to_remaining_centroid_sq"] > teacher["distance_to_remaining_centroid_sq"] + 1e-6:
        tags.append("high_centroid_drift")
    if chosen["heading_alignment"] + 1e-6 < teacher["heading_alignment"]:
        tags.append("weak_heading")
    if chosen["saving_proxy"] > teacher["saving_proxy"] + 1e-6:
        tags.append("high_saving")
    elif chosen["saving_proxy"] + 1e-6 < teacher["saving_proxy"]:
        tags.append("low_saving")
    if not tags:
        tags.append("misc_misalignment")
    return tags


def _case_tags_cvrp(chosen: dict[str, float], teacher: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if chosen["via_depot"] > teacher["via_depot"] + 1e-6:
        tags.append("over_restart")
    if chosen["via_depot"] + 1e-6 < teacher["via_depot"] and chosen["capacity_after_ratio"] < teacher["capacity_after_ratio"] - 1e-6:
        tags.append("delayed_restart")
    if chosen["demand_ratio"] + 1e-6 < teacher["demand_ratio"] and chosen["capacity_after_ratio"] > teacher["capacity_after_ratio"] + 1e-6:
        tags.append("underfill_route")
    if chosen["distance_to_depot"] > teacher["distance_to_depot"] + 1e-6 and chosen["nearest_remaining_distance"] > teacher["nearest_remaining_distance"] + 1e-6:
        tags.append("expensive_tail")
    if chosen["saving_proxy"] > teacher["saving_proxy"] + 1e-6:
        tags.append("high_saving")
    elif chosen["saving_proxy"] + 1e-6 < teacher["saving_proxy"]:
        tags.append("low_saving")
    if not tags:
        tags.append("misc_misalignment")
    return tags


def _case_delta_text(row: dict[str, object]) -> str:
    if _is_maxcut_profile(row["chosen_profile"]):
        tag_text = {
            "low_flip_gain": "chosen flip has lower immediate cut gain than teacher",
            "low_normalized_gain": "chosen flip has weaker degree-normalized gain",
            "tabu_violation": "chosen revisits a recently flipped node more eagerly than teacher",
            "underuse_high_degree": "chosen underuses a higher-degree/high-impact node",
            "weak_same_side_pressure": "chosen ignores stronger same-side edge pressure that teacher would cut",
            "worse_balance": "chosen worsens partition balance more than teacher",
            "weaker_followup_potential": "chosen opens weaker next-step gain potential than teacher",
            "shrinks_positive_frontier": "chosen leaves fewer positive-gain follow-up moves than teacher",
            "underuses_boundary_node": "chosen underuses a more mixed boundary node",
            "weaker_hub_neighborhood": "chosen has weaker high-degree neighborhood context than teacher",
            "misc_misalignment": "chosen differs from teacher without one dominant flip-level pattern",
        }
    elif _is_cvrp_profile(row["chosen_profile"]):
        tag_text = {
            "over_restart": "chosen restarts at depot too eagerly",
            "delayed_restart": "chosen delays depot restart despite tighter capacity",
            "underfill_route": "chosen underfills the route and leaves too much capacity unused",
            "expensive_tail": "chosen creates a more expensive route tail",
            "high_saving": "chosen leans too hard on saving-like jumps",
            "low_saving": "chosen underuses productive savings-aware continuation",
            "misc_misalignment": "chosen differs from teacher without a dominant simple pattern",
        }
    else:
        tag_text = {
            "over_local_isolated": "chosen is more local but much more isolated",
            "high_centroid_drift": "chosen drifts farther from the remaining-set center",
            "weak_heading": "chosen breaks heading continuity",
            "high_saving": "chosen leans too hard on saving-like jumps",
            "low_saving": "chosen underuses productive bridge-like moves",
            "misc_misalignment": "chosen differs from teacher without a dominant simple pattern",
        }
    ordered = []
    seen = set()
    for tag in _case_tags(row):
        if tag in seen:
            continue
        seen.add(tag)
        ordered.append(tag_text[tag])
    return "; ".join(ordered)


def _top_case_records(records: list[dict[str, object]], *, max_cases: int = 2) -> str:
    rows = sorted(records, key=lambda item: float(item.get("teacher_score_gap", 0.0)), reverse=True)[:max_cases]
    if not rows:
        return "No representative teacher disagreements were sampled."
    lines = []
    for row in rows:
        chosen = row["chosen_profile"]
        teacher = row["teacher_profile"]
        semantic_text = _case_semantics_text(row)
        confidence_text = _teacher_confidence_text(row)
        lines.append(
            "stage={stage}, remaining_fraction={remaining_fraction:.2f}, feasible_count={feasible_count}, current={current}, "
            "teacher_score_gap={teacher_score_gap:+.3f}{confidence}; delta={delta}; chosen[{chosen}] vs teacher[{teacher}]{semantic}".format(
                stage=str(row.get("stage") or _stage_from_remaining_fraction(float(row.get("remaining_fraction", 0.5)))),
                remaining_fraction=float(row["remaining_fraction"]),
                feasible_count=int(row["feasible_count"]),
                current=int(row["current"]),
                teacher_score_gap=float(row["teacher_score_gap"]),
                confidence=confidence_text,
                delta=_case_delta_text(row),
                chosen=_compact_profile(int(row["chosen_action"]), chosen, row["chosen_pct"]),
                teacher=_compact_profile(int(row["teacher_action"]), teacher, row["teacher_pct"]),
                semantic=semantic_text,
            )
        )
    return "\n".join(lines)


def _stage_summary_text(records: list[dict[str, object]]) -> str:
    stage_lines = []
    for stage in ("early", "mid", "late"):
        stage_records = [row for row in records if str(row.get("stage")) == stage]
        if not stage_records:
            continue
        counts: dict[str, int] = {}
        for row in stage_records:
            for tag in _case_tags(row):
                counts[tag] = counts.get(tag, 0) + 1
        dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        stage_lines.append(f"- {stage}: {_stage_dominant_text(stage, dominant)}")
    return "\n".join(stage_lines) if stage_lines else "No stage-level teacher failures were summarized."


def _stage_dominant_text(stage: str, tag: str) -> str:
    if _problem_family() == "maxcut":
        mapping = {
            ("early", "low_flip_gain"): "early flips leave immediate cut gains unused; teacher takes larger positive-gain moves first.",
            ("early", "tabu_violation"): "early search revisits nodes too soon; teacher diversifies before cycling back.",
            ("mid", "low_normalized_gain"): "mid-search chooses lower normalized-gain flips; teacher favors gain per degree to avoid wasting steps.",
            ("mid", "underuse_high_degree"): "mid-search underuses high-impact nodes; teacher flips higher-degree nodes when gain supports it.",
            ("late", "low_flip_gain"): "late search spends moves on weak gains; teacher preserves stronger local improvements.",
            ("late", "tabu_violation"): "late search cycles through recent flips; teacher uses stronger anti-cycle control.",
        }
        fallback = {
            "weak_same_side_pressure": "misses same-side pressure that teacher converts into cut edges.",
            "worse_balance": "over-tilts partition balance compared with teacher.",
            "weaker_followup_potential": "misses flips that preserve stronger follow-up gains.",
            "shrinks_positive_frontier": "shrinks the positive-gain frontier faster than teacher.",
            "underuses_boundary_node": "underuses mixed-boundary nodes that teacher treats as useful pivots.",
            "weaker_hub_neighborhood": "chooses nodes with weaker high-degree neighborhood context than teacher.",
            "misc_misalignment": "repeatedly disagrees with teacher without a single dominant simple pattern.",
        }
    elif _problem_family() == "cvrp":
        mapping = {
            ("early", "over_restart"): "restarts routes too early; teacher keeps serving feasible direct customers before closing.",
            ("early", "underfill_route"): "underfills routes; teacher serves larger feasible customers before considering a restart.",
            ("mid", "delayed_restart"): "keeps the route open too long under tighter capacity; teacher closes/restarts earlier.",
            ("mid", "expensive_tail"): "mid-route choices create costly future tails; teacher reduces tail risk before continuing.",
            ("late", "delayed_restart"): "late closure is too delayed; teacher restarts to cleanly finish remaining customers.",
            ("late", "expensive_tail"): "late-stage route tail is too expensive; teacher uses tighter closure control.",
        }
        fallback = {
            "high_saving": "leans too hard on saving-like moves without enough route control.",
            "low_saving": "misses productive continuation or closure moves that teacher accepts.",
            "misc_misalignment": "repeatedly disagrees with teacher on route continuation versus closure.",
        }
    else:
        mapping = {
            ("early", "over_local_isolated"): "over-selects isolated local nodes; teacher prefers slightly farther but better-connected nodes.",
            ("early", "high_centroid_drift"): "jumps outward too early; teacher keeps lower centroid drift before bridging.",
            ("early", "weak_heading"): "breaks heading continuity too early; teacher follows smoother local continuation.",
            ("mid", "over_local_isolated"): "stays too local while creating isolated leftovers; teacher accepts broader but better-connected moves.",
            ("mid", "high_centroid_drift"): "exits dense regions too early; teacher delays bridge moves until centroid drift is lower.",
            ("mid", "weak_heading"): "mid-route direction changes are too abrupt; teacher preserves smoother bridge geometry.",
            ("late", "over_local_isolated"): "cleanup leaves isolated tail nodes; teacher resolves leftovers earlier and more compactly.",
            ("late", "high_centroid_drift"): "late-stage cleanup is too outward; teacher closes with tighter geometry.",
            ("late", "weak_heading"): "closure is poorly aligned; teacher prefers short, smooth finishing moves.",
        }
        fallback = {
            "high_saving": "relies too much on saving-like jumps; teacher uses more balanced geometry.",
            "low_saving": "misses productive bridge-like moves that teacher accepts.",
            "misc_misalignment": "repeatedly disagrees with teacher without a single dominant simple pattern.",
        }
    if (stage, tag) in mapping:
        return mapping[(stage, tag)]
    return fallback.get(tag, "repeatedly disagrees with teacher on local action choice.")


def _structure_guidance_text(records: list[dict[str, object]]) -> str:
    scores: dict[str, float] = {}
    for row in records:
        gap = float(row.get("teacher_score_gap", 0.0))
        for tag in _case_tags(row):
            scores[tag] = scores.get(tag, 0.0) + gap
    dominant = max(scores.items(), key=lambda item: (item[1], item[0]))[0] if scores else "misc_misalignment"
    if _problem_family() == "maxcut":
        recommendations = {
            "low_flip_gain": "Replace pure tie-breaking with a gain-first shortlist, then rerank by tabu age and normalized gain.",
            "low_normalized_gain": "Add a degree-normalized gain branch so high-degree nodes are not chosen only because of raw degree.",
            "tabu_violation": "Add explicit anti-cycle structure: avoid recently flipped nodes unless their gain clears a strong aspiration threshold.",
            "underuse_high_degree": "Add a high-impact node shortlist that considers weighted degree only when flip gain is competitive.",
            "weak_same_side_pressure": "Use same-side edge pressure as a structural signal: prioritize flips that convert heavy same-side edges into cut edges.",
            "worse_balance": "Add a light balance gate to avoid repeatedly moving the partition toward one side when gain is comparable.",
            "weaker_followup_potential": "Add a one-step lookahead signal: after a candidate flip, prefer states with stronger top-k follow-up gains.",
            "shrinks_positive_frontier": "Add a frontier-preservation gate so a flip is penalized when it sharply reduces the number of positive-gain follow-up moves.",
            "underuses_boundary_node": "Add a boundary-mixedness branch that favors nodes connecting both sides when their gain is competitive.",
            "weaker_hub_neighborhood": "Add a neighborhood-impact branch using neighbor weighted degree and neighbor gain, not only the candidate node degree.",
            "misc_misalignment": "Preserve the local-search backbone, but replace the flip choice with gain-shortlist plus teacher-like anti-cycle reranking.",
        }
    elif _problem_family() == "cvrp":
        recommendations = {
            "over_restart": "Replace an over-eager depot-return tendency with an explicit direct-vs-depot gate that continues the route when feasible customers remain well-connected.",
            "delayed_restart": "Add a closure gate that restarts from depot earlier when remaining capacity becomes tight and tail risk rises.",
            "underfill_route": "Introduce a route-fill shortlist that prefers feasible higher-demand customers before restarting.",
            "expensive_tail": "Add a late tail-control branch that closes or restarts routes before distant leftovers become expensive.",
            "high_saving": "Reduce pure savings-driven continuation by adding route-fill or tail-risk gating before committing to a customer.",
            "low_saving": "Add a controlled continuation shortlist so the heuristic can accept slightly longer but structurally better route-extension moves.",
            "misc_misalignment": "Preserve the current route-control backbone, but replace the continuation-versus-restart decision with a two-step shortlist-plus-gate structure.",
        }
    else:
        recommendations = {
            "over_local_isolated": "Replace a single-pool nearest-first decision with shortlist(local candidates) plus rerank(connectivity / isolation risk).",
            "high_centroid_drift": "Add an explicit bridge gate: keep the local backbone, but require lower centroid drift before cross-cluster moves.",
            "weak_heading": "Add a dedicated route-continuity branch that reranks candidates by smoother heading before accepting disruptive turns.",
            "high_saving": "Reduce direct saving-driven jumps by inserting a local-consistency gate before bridge decisions.",
            "low_saving": "Add a controlled bridge shortlist so the heuristic can accept slightly longer travel when it improves remaining-set structure.",
            "misc_misalignment": "Preserve the best local backbone, but replace the current rerank rule with a two-step shortlist-plus-rerank structure.",
        }
    return recommendations.get(dominant, recommendations["misc_misalignment"])


def _parameter_guidance_text(records: list[dict[str, object]]) -> str:
    recommendations = []
    by_stage = {stage: [row for row in records if str(row.get("stage")) == stage] for stage in ("early", "mid", "late")}
    for stage, rows in by_stage.items():
        if not rows:
            continue
        counts: dict[str, int] = {}
        for row in rows:
            for tag in _case_tags(row):
                counts[tag] = counts.get(tag, 0) + 1
        dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        if _problem_family() == "maxcut":
            if dominant == "low_flip_gain":
                recommendations.append(f"- {stage}: increase immediate flip-gain priority or tighten the positive-gain shortlist.")
            elif dominant == "low_normalized_gain":
                recommendations.append(f"- {stage}: raise the normalized-gain weight relative to raw degree or balance terms.")
            elif dominant == "tabu_violation":
                recommendations.append(f"- {stage}: increase recent-flip penalty and only override it with a higher aspiration gain.")
            elif dominant == "underuse_high_degree":
                recommendations.append(f"- {stage}: raise high-degree impact only after gain passes a competitive threshold.")
            elif dominant == "weak_same_side_pressure":
                recommendations.append(f"- {stage}: increase same-side pressure weight to favor flips that cut heavier internal edges.")
            elif dominant == "worse_balance":
                recommendations.append(f"- {stage}: strengthen balance penalty when candidate gains are close.")
            elif dominant == "weaker_followup_potential":
                recommendations.append(f"- {stage}: increase one-step follow-up potential, especially top-k after-flip gain.")
            elif dominant == "shrinks_positive_frontier":
                recommendations.append(f"- {stage}: penalize flips that reduce the positive-gain frontier unless immediate gain is clearly superior.")
            elif dominant == "underuses_boundary_node":
                recommendations.append(f"- {stage}: raise boundary-mixedness only inside a competitive gain shortlist.")
            elif dominant == "weaker_hub_neighborhood":
                recommendations.append(f"- {stage}: add neighbor-degree or neighbor-gain context without blindly preferring hubs.")
            else:
                recommendations.append(f"- {stage}: retune gain, tabu-age, and normalized-degree priorities without changing the backbone.")
        elif _problem_family() == "cvrp":
            if stage == "early" and dominant == "over_restart":
                recommendations.append("- early: make depot-restart less eager and widen the direct-customer shortlist.")
            elif stage == "early" and dominant == "underfill_route":
                recommendations.append("- early: favor slightly higher route fill before allowing a restart.")
            elif stage == "mid" and dominant == "delayed_restart":
                recommendations.append("- mid: close/restart routes earlier when remaining capacity gets tight.")
            elif stage == "late" and dominant == "expensive_tail":
                recommendations.append("- late: strengthen tail-control and depot-closure thresholds.")
            elif dominant == "high_saving":
                recommendations.append(f"- {stage}: reduce the weight or priority of pure saving-like moves.")
            elif dominant == "low_saving":
                recommendations.append(f"- {stage}: allow slightly broader continuation moves when they improve route structure.")
            else:
                recommendations.append(f"- {stage}: retune route continuation versus depot-restart thresholds without changing structure.")
        else:
            if stage == "early" and dominant == "over_local_isolated":
                recommendations.append("- early: narrow shortlist width or strengthen isolation penalty.")
            elif stage == "mid" and dominant == "high_centroid_drift":
                recommendations.append("- mid: delay bridge threshold or strengthen centroid-drift control before outward moves.")
            elif stage == "late" and dominant in {"weak_heading", "high_centroid_drift"}:
                recommendations.append("- late: strengthen heading/closure tie-breaks and penalize outward cleanup moves.")
            elif dominant == "high_saving":
                recommendations.append(f"- {stage}: reduce the weight or priority of saving-like jumps.")
            elif dominant == "low_saving":
                recommendations.append(f"- {stage}: allow slightly broader bridge moves when connectivity improves.")
            else:
                recommendations.append(f"- {stage}: retune local tie-break priorities without changing structure.")
    return "\n".join(recommendations) if recommendations else "No parameter recommendation was derived."
