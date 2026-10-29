from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from tae.runners.tsp_dataset_io import load_lehd_tsp_txt, load_tsp_pickle

ROOT = Path(__file__).resolve().parent
_HEURISTIC_MODULE = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native EOH tsp_construct output on TSP test sets.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--pop-id", type=int, default=10)
    parser.add_argument("--selection", choices=("objective", "align"), default="objective")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--n-proc", type=int, default=int(os.environ.get("TSP_EVAL_N_PROC", "1")))
    parser.add_argument("--method-name", default="eoh_native_best")
    parser.add_argument("--sizes", default="20,50,100", help="Comma-separated EoH TSP test sizes to evaluate.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of test instances per size.")
    parser.add_argument("--test-root", default="data/tsp/testingdata")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-lehd-txt")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    best_path, payload = _load_selected_payload(result_dir, args.pop_id, args.selection)
    code = str(payload["code"])

    rows: list[dict[str, object]] = []
    dataset_source = None
    if args.test_lehd_txt:
        raw = load_lehd_tsp_txt(args.test_lehd_txt)[: max(0, int(args.limit))]
        size = int(len(raw[0]["coords"])) if raw else 0
        started = time.perf_counter()
        mean_dis, best_dis, worst_dis = _evaluate_code_lehd(code, raw, args.n_proc)
        wall_clock_sec = time.perf_counter() - started
        rows.append(
            {
                "method": args.method_name,
                "size": size,
                "mean_distance": mean_dis,
                "best_distance": best_dis,
                "worst_distance": worst_dis,
                "reference_mean": float(np.mean([float(item["reference_cost"]) for item in raw])) if raw else None,
                "count": len(raw),
                "wall_clock_sec": wall_clock_sec,
            }
        )
        dataset_source = str(Path(args.test_lehd_txt).resolve())
    else:
        dataset_source = str(Path(args.test_root).resolve())
        sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
        for size in sizes:
            pkl_path = Path(args.test_root) / f"instance_data_{size}.pkl"
            raw = load_tsp_pickle(pkl_path)[: max(0, int(args.limit))]
            started = time.perf_counter()
            mean_dis, best_dis, worst_dis = _evaluate_code_eoh(code, raw, size, args.n_proc)
            wall_clock_sec = time.perf_counter() - started
            rows.append(
                {
                    "method": args.method_name,
                    "size": size,
                    "mean_distance": mean_dis,
                    "best_distance": best_dis,
                    "worst_distance": worst_dis,
                    "count": len(raw),
                    "wall_clock_sec": wall_clock_sec,
                }
            )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "best_population_file": str(best_path),
        "selection": args.selection,
        "algorithm": payload.get("algorithm"),
        "objective_train": payload.get("objective"),
        "dataset_source": dataset_source,
        "n_proc": max(1, int(args.n_proc)),
        "sizes": [row["size"] for row in rows],
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"size={row['size']}, mean={float(row['mean_distance']):.3f}, "
            f"best={float(row['best_distance']):.3f}, worst={float(row['worst_distance']):.3f}"
        )


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


def _evaluate_code_eoh(code: str, dataset, problem_size: int, n_proc: int) -> tuple[float, float, float]:
    if n_proc <= 1 or len(dataset) <= 1:
        _init_worker(code)
        results = [_evaluate_eoh_instance(item, problem_size) for item in dataset]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc, initializer=_init_worker, initargs=(code,)) as pool:
            results = list(pool.map(_evaluate_eoh_instance, dataset, [problem_size] * len(dataset), chunksize=1))
    arr = np.asarray(results, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def _evaluate_code_lehd(code: str, dataset, n_proc: int) -> tuple[float, float, float]:
    if n_proc <= 1 or len(dataset) <= 1:
        _init_worker(code)
        results = [_evaluate_lehd_item(item) for item in dataset]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc, initializer=_init_worker, initargs=(code,)) as pool:
            results = list(pool.map(_evaluate_lehd_item, dataset, chunksize=1))
    arr = np.asarray(results, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def _init_worker(code: str) -> None:
    global _HEURISTIC_MODULE
    module = types.ModuleType("heuristic_module")
    exec(code, module.__dict__)
    if not hasattr(module, "select_next_node") and hasattr(module, "select_next_node_v2"):
        module.select_next_node = module.select_next_node_v2
    _HEURISTIC_MODULE = module


def _evaluate_eoh_instance(item, problem_size: int) -> float:
    instance, distance_matrix = item
    return _evaluate_instance(np.asarray(instance), np.asarray(distance_matrix), problem_size)


def _evaluate_lehd_item(item) -> float:
    coords = np.asarray(item["coords"], dtype=np.float64)
    problem_size = int(len(coords))
    dmat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    return _evaluate_instance(coords, dmat, problem_size)


def _evaluate_instance(coords, distance_matrix, problem_size: int) -> float:
    module = _HEURISTIC_MODULE
    if module is None:
        raise RuntimeError("heuristic module was not initialized")
    route = np.zeros(problem_size, dtype=int)
    visited = np.zeros(problem_size, dtype=bool)
    visited[0] = True
    current_node = 0
    destination_node = 0
    neighbor_matrix = np.argsort(distance_matrix, axis=1)
    for i in range(1, problem_size - 1):
        near_nodes = neighbor_matrix[current_node]
        mask = ~visited[near_nodes]
        unvisited = near_nodes[mask]
        unvisited = unvisited[:problem_size]
        next_node = module.select_next_node(current_node, destination_node, unvisited.copy(), distance_matrix.copy())
        if isinstance(next_node, (bool, np.bool_)) or not isinstance(next_node, (int, np.integer)) or next_node not in unvisited:
            raise ValueError("selector must return an unvisited integer node")
        current_node = int(next_node)
        route[i] = current_node
        visited[current_node] = True
    last_node = np.flatnonzero(~visited)[0]
    route[problem_size - 1] = int(last_node)
    return _tour_cost(coords, route, problem_size)


def _generate_neighborhood_matrix(instance):
    instance = np.asarray(instance)
    n = len(instance)
    neighborhood_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        distances = np.linalg.norm(instance[i] - instance, axis=1)
        neighborhood_matrix[i] = np.argsort(distances)
    return neighborhood_matrix


def _tour_cost(instance, solution, problem_size):
    cost = 0.0
    for j in range(problem_size - 1):
        cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
    cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
    return float(cost)


if __name__ == "__main__":
    main()
