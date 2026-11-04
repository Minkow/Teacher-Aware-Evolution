from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import types
from pathlib import Path

import numpy as np

from tae.tasks.cvrp.data import load_cvrp_pickle, load_lehd_cvrp_txt

ROOT = Path(__file__).resolve().parent
_WORKER_SELECT_NEXT_NODE = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native EOH cvrp_construct output on a fixed CVRP test set.")
    parser.add_argument("--result-dir", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test-pkl")
    group.add_argument("--test-lehd-txt")
    parser.add_argument("--pop-id", type=int, default=10)
    parser.add_argument("--selection", choices=("objective", "align"), default="objective")
    parser.add_argument("--method-name", default="eoh_native_best")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    best_path, payload = _load_selected_payload(result_dir, args.pop_id, args.selection)
    code = str(payload["code"])

    if args.test_pkl:
        raw = load_cvrp_pickle(args.test_pkl)
        dataset_source = str(Path(args.test_pkl).resolve())
    else:
        raw = load_lehd_cvrp_txt(args.test_lehd_txt)
        dataset_source = str(Path(args.test_lehd_txt).resolve())
    if args.limit > 0:
        raw = raw[: args.limit]
    started = time.perf_counter()
    mean_dis, best_dis, worst_dis = _evaluate_code(code, raw, workers=max(1, args.workers))
    wall_clock_sec = time.perf_counter() - started
    ref_mean = None
    if raw and "reference_cost" in raw[0]:
        ref_mean = float(np.mean([float(item["reference_cost"]) for item in raw]))
    rows = [
        {
            "method": args.method_name,
            "dataset": Path(args.test_pkl or args.test_lehd_txt).stem,
            "size": int(len(raw[0]["coords"]) - 1) if raw else 0,
            "mean_distance": mean_dis,
            "best_distance": best_dis,
            "worst_distance": worst_dis,
            "reference_mean": ref_mean,
            "count": len(raw),
            "wall_clock_sec": wall_clock_sec,
        }
    ]

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
        "workers": max(1, args.workers),
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"method={args.method_name}, dataset={rows[0]['dataset']}, size={rows[0]['size']}, "
        f"mean={mean_dis:.3f}, best={best_dis:.3f}, worst={worst_dis:.3f}, "
        f"reference={ref_mean if ref_mean is not None else 'NA'}"
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


def _evaluate_code(code: str, dataset, *, workers: int = 1) -> tuple[float, float, float]:
    module = types.ModuleType("heuristic_module")
    exec(code, module.__dict__)
    if not hasattr(module, "select_next_node") and hasattr(module, "select_next_node_v2"):
        module.select_next_node = module.select_next_node_v2
    if workers <= 1 or len(dataset) <= 1:
        results = [_evaluate_one_item(item, module.select_next_node) for item in dataset]
    else:
        with mp.Pool(processes=min(workers, len(dataset)), initializer=_init_worker, initargs=(code,)) as pool:
            results = pool.map(_evaluate_one_item_worker, dataset)
    arr = np.asarray(results, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def _init_worker(code: str) -> None:
    global _WORKER_SELECT_NEXT_NODE
    module = types.ModuleType("heuristic_module_worker")
    exec(code, module.__dict__)
    if not hasattr(module, "select_next_node") and hasattr(module, "select_next_node_v2"):
        module.select_next_node = module.select_next_node_v2
    _WORKER_SELECT_NEXT_NODE = module.select_next_node


def _evaluate_one_item_worker(item) -> float:
    if _WORKER_SELECT_NEXT_NODE is None:
        raise RuntimeError("CVRP worker was not initialized")
    return _evaluate_one_item(item, _WORKER_SELECT_NEXT_NODE)


def _evaluate_one_item(item, select_next_node) -> float:
    coords = np.asarray(item["coords"], dtype=np.float64)
    demands = np.asarray(item["demands"], dtype=np.float64)
    capacity = float(item["capacity"])
    dmat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    visited = set()
    current = 0
    rest_capacity = capacity
    total = 0.0
    max_steps = max(20, (len(coords) - 1) * 4 + 8)
    steps = 0
    while len(visited) < len(coords) - 1:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("selector exceeded CVRP step limit during evaluation")
        unvisited = tuple(node for node in range(1, len(coords)) if node not in visited)
        next_node = select_next_node(current, 0, unvisited, rest_capacity, demands.copy(), dmat.copy())
        if isinstance(next_node, (bool, np.bool_)) or not isinstance(next_node, (int, np.integer)):
            raise ValueError("selector must return an integer node")
        next_node = int(next_node)
        if next_node == 0:
            if current == 0:
                raise RuntimeError("selector returned depot while already at depot")
            total += float(dmat[current][0])
            current = 0
            rest_capacity = capacity
            continue
        if next_node not in unvisited:
            raise RuntimeError(f"selector returned invalid customer {next_node}")
        demand = float(demands[next_node])
        if demand > rest_capacity + 1e-9:
            raise RuntimeError(f"selector returned infeasible customer {next_node}")
        total += float(dmat[current][next_node])
        current = next_node
        rest_capacity -= demand
        visited.add(next_node)
    total += float(dmat[current][0])
    return total


if __name__ == "__main__":
    main()
