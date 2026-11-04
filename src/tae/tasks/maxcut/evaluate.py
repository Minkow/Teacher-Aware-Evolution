from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tae.tasks.maxcut.core import MaxCutState
from tae.tasks.maxcut.data import load_maxcut_bench_directory, load_maxcut_bench_optimal

_HEURISTIC_MODULE = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native EOH Max-Cut output on MaxCut-Bench datasets.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--bench-root", required=True, help="Path to MaxCut-Bench repo root.")
    parser.add_argument("--pop-id", type=int, default=5)
    parser.add_argument("--selection", choices=("objective", "align"), default="objective")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names under data/testing. Empty means all.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--method-name", default="eoh_native_best")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    best_path, payload = _load_selected_payload(result_dir, args.pop_id, args.selection)
    code = str(payload["code"])

    run_config_path = result_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else {}
    max_steps = int(args.max_steps or run_config.get("max_steps") or 40)
    attempts = int(args.attempts or run_config.get("attempts") or 1)
    bench_root = Path(args.bench_root).resolve()
    testing_root = bench_root / "data" / "testing"
    if args.datasets.strip():
        dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    else:
        dataset_names = sorted((path.name for path in testing_root.iterdir() if path.is_dir()), key=_dataset_sort_key)

    rows: list[dict[str, object]] = []
    overall_cuts: list[float] = []
    overall_ratios: list[float] = []
    overall_gaps: list[float] = []

    for dataset_name in dataset_names:
        dataset_dir = testing_root / dataset_name
        instances = load_maxcut_bench_directory(dataset_dir)
        if not instances:
            print(f"dataset={dataset_name}, skipped=empty")
            continue
        optimal = load_maxcut_bench_optimal(dataset_dir)
        started = time.perf_counter()
        mean_cut, best_cut, worst_cut, raw = _evaluate_code(
            code,
            instances,
            max_steps=max_steps,
            attempts=attempts,
            seed=args.seed,
            workers=max(1, args.workers),
        )
        wall_clock_sec = time.perf_counter() - started
        ref_values = [float(optimal[item["name"]]) for item in raw if item["name"] in optimal]
        ratios = [float(item["cut"]) / float(optimal[item["name"]]) for item in raw if item["name"] in optimal and optimal[item["name"]] != 0]
        gaps = [float(optimal[item["name"]]) - float(item["cut"]) for item in raw if item["name"] in optimal]
        overall_cuts.extend(float(item["cut"]) for item in raw)
        overall_ratios.extend(ratios)
        overall_gaps.extend(gaps)
        rows.append(
            {
                "method": args.method_name,
                "dataset": dataset_name,
                "count": len(raw),
                "mean_cut": mean_cut,
                "best_cut": best_cut,
                "worst_cut": worst_cut,
                "reference_mean": float(np.mean(ref_values)) if ref_values else None,
                "mean_ratio": float(np.mean(ratios)) if ratios else None,
                "mean_gap": float(np.mean(gaps)) if gaps else None,
                "wall_clock_sec": wall_clock_sec,
            }
        )
        _write_outputs(
            out_csv=Path(args.out_csv),
            out_json=Path(args.out_json),
            rows=rows,
            best_path=best_path,
            payload=payload,
            bench_root=bench_root,
            max_steps=max_steps,
            attempts=attempts,
            workers=max(1, args.workers),
            selection=args.selection,
            overall_cuts=overall_cuts,
            overall_ratios=overall_ratios,
            overall_gaps=overall_gaps,
        )
        print(
            f"dataset={dataset_name}, mean_cut={float(rows[-1]['mean_cut']):.3f}, "
            f"ratio={rows[-1]['mean_ratio'] if rows[-1]['mean_ratio'] is not None else 'NA'}, "
            f"gap={rows[-1]['mean_gap'] if rows[-1]['mean_gap'] is not None else 'NA'}",
            flush=True,
        )

    if not rows:
        raise RuntimeError(f"no non-empty MaxCut-Bench testing datasets found under {testing_root}")


def _evaluate_code(
    code: str,
    instances,
    *,
    max_steps: int,
    attempts: int,
    seed: int,
    workers: int,
) -> tuple[float, float, float, list[dict[str, object]]]:
    tasks = [(instance, max_steps, attempts, seed + idx) for idx, instance in enumerate(instances)]
    if workers <= 1 or len(tasks) <= 1:
        _init_worker(code)
        results = [_evaluate_instance_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_init_worker,
            initargs=(code,),
        ) as pool:
            results = list(pool.map(_evaluate_instance_task, tasks, chunksize=1))
    cuts = np.asarray([float(item["cut"]) for item in results], dtype=float)
    return float(cuts.mean()), float(cuts.max()), float(cuts.min()), results


def _write_outputs(
    *,
    out_csv: Path,
    out_json: Path,
    rows: list[dict[str, object]],
    best_path: Path,
    payload: dict[str, object],
    bench_root: Path,
    max_steps: int,
    attempts: int,
    workers: int,
    selection: str,
    overall_cuts: list[float],
    overall_ratios: list[float],
    overall_gaps: list[float],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "best_population_file": str(best_path),
        "selection": selection,
        "algorithm": payload.get("algorithm"),
        "objective_train": payload.get("objective"),
        "bench_root": str(bench_root),
        "max_steps": max_steps,
        "attempts": attempts,
        "workers": workers,
        "rows": rows,
        "overall": {
            "dataset_count": len(rows),
            "mean_cut": float(np.mean(overall_cuts)) if overall_cuts else None,
            "mean_ratio": float(np.mean(overall_ratios)) if overall_ratios else None,
            "mean_gap": float(np.mean(overall_gaps)) if overall_gaps else None,
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _load_selected_payload(result_dir: Path, pop_id: int, selection: str) -> tuple[Path, dict[str, object]]:
    if selection == "objective":
        best_path = result_dir / "results" / "pops_best" / f"population_generation_{pop_id}.json"
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        return best_path, payload
    pop_path = result_dir / "results" / "pops" / f"population_generation_{pop_id}.json"
    population = json.loads(pop_path.read_text(encoding="utf-8"))
    if not isinstance(population, list) or not population:
        raise ValueError(f"population file is not a non-empty list: {pop_path}")
    candidates = [
        item for item in population
        if isinstance(item, dict) and item.get("code") and item.get("alignment_score") is not None
    ]
    if not candidates:
        raise ValueError(f"no alignment_score candidates found in {pop_path}")
    payload = max(
        candidates,
        key=lambda item: (
            float(item.get("alignment_score", item.get("top1_agreement", 0.0))),
            -float(item.get("objective", 1e100)),
        ),
    )
    return pop_path, payload


def _dataset_sort_key(name: str):
    if name.startswith("."):
        return (10**9, name)
    match = re.search(r"(^|_)(\d+)(vertices|spin)?($|_)", name)
    if match:
        return (int(match.group(2)), name)
    return (10**9 - 1, name)


def _init_worker(code: str) -> None:
    global _HEURISTIC_MODULE
    module = types.ModuleType("maxcut_eval_module")
    exec(code, module.__dict__)
    if not hasattr(module, "select_flip_node") and hasattr(module, "select_flip_node_v2"):
        module.select_flip_node = module.select_flip_node_v2
    _HEURISTIC_MODULE = module


def _evaluate_instance_task(task) -> dict[str, object]:
    instance, max_steps, attempts, seed = task
    if _HEURISTIC_MODULE is None:
        raise RuntimeError("heuristic module was not initialized")
    rng = np.random.default_rng(seed)
    best_cut = float("-inf")
    for attempt in range(max(1, int(attempts))):
        init_spins = None if attempt == 0 else rng.integers(0, 2, size=instance.n_nodes)
        state = MaxCutState(instance, max_steps=max_steps, init_spins=init_spins)
        while not state.done:
            action = _select_action(_HEURISTIC_MODULE, state)
            if action < 0 or action >= state.n_nodes:
                raise ValueError(f"invalid flip action {action}")
            state.dispatch(action)
        best_cut = max(best_cut, float(state.best_cut))
    return {"name": instance.name, "cut": best_cut}


def _select_action(module, state: MaxCutState) -> int:
    if hasattr(module, "score"):
        decision_state = _decision_state(state)
        scores = {
            int(feature.node_id): float(module.score(feature, decision_state))
            for feature in decision_state.candidate_features
        }
        return int(max(scores, key=lambda node: (scores[node], -node)))
    return int(
        module.select_flip_node(
            state.spins.copy(),
            state.adjacency.copy(),
            float(state.current_cut),
            float(state.best_cut),
            int(state.step),
            int(state.max_steps),
            list(state.actions),
        )
    )


def _decision_state(state: MaxCutState):
    raw_features = state.candidate_features()
    profiles = _candidate_profiles(state, raw_features)
    features = tuple(_score_feature(state, feature, profiles[int(feature.action)]) for feature in raw_features)
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


def _score_feature(state: MaxCutState, feature, profile: dict[str, float] | None = None):
    n = max(1, int(state.n_nodes))
    side1 = float(feature.side1_fraction)
    side1_after = side1 + (1.0 / n if int(feature.spin) == 0 else -1.0 / n)
    profile = profile or {}
    return SimpleNamespace(
        node_id=int(feature.action),
        spin=int(feature.spin),
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


def _candidate_profiles(state: MaxCutState, features) -> dict[int, dict[str, float]]:
    gains = np.asarray([float(feature.flip_gain) for feature in features], dtype=np.float64)
    degrees = np.asarray([float(feature.weighted_degree) for feature in features], dtype=np.float64)
    positive_before = int(np.sum(gains > 1e-9))
    profiles: dict[int, dict[str, float]] = {}
    for feature in features:
        node = int(feature.action)
        row = np.asarray(state.adjacency[node], dtype=np.float64)
        neighbors = np.flatnonzero(np.abs(row) > 1e-12)
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


if __name__ == "__main__":
    main()
