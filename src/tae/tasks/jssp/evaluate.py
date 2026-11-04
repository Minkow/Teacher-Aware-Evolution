from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

from tae.tasks.jssp.data import load_instances
from tae.tasks.jssp.simulator import ScheduleState


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native EOH JSSP dispatching output.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--pop-id", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--machines", type=int, default=10)
    parser.add_argument("--test-npy")
    parser.add_argument("--test-jsonl")
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    best_path = result_dir / "results" / "pops_best" / f"population_generation_{args.pop_id}.json"
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    code = str(payload["code"])
    module = types.ModuleType("heuristic_module")
    exec(code, module.__dict__)
    if not hasattr(module, "score"):
        raise ValueError("best code does not define score")

    instances = load_instances(
        npy=args.test_npy,
        jsonl=args.test_jsonl,
        n_jobs=args.jobs,
        n_machines=args.machines,
        count=args.test_count,
        seed=args.seed,
    )
    makespans = [_rollout(instance, module) for instance in instances]
    row = {
        "method": "eoh_native_best",
        "jobs": args.jobs,
        "machines": args.machines,
        "count": len(makespans),
        "mean_makespan": sum(makespans) / len(makespans) if makespans else 0.0,
        "best_makespan": min(makespans) if makespans else 0,
        "worst_makespan": max(makespans) if makespans else 0,
    }
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    summary = {
        "best_population_file": str(best_path),
        "algorithm": payload.get("algorithm"),
        "objective_train": payload.get("objective"),
        "row": row,
        "makespans": makespans,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"jobs={args.jobs}, machines={args.machines}, mean={row['mean_makespan']:.3f}, "
        f"best={row['best_makespan']}, worst={row['worst_makespan']}"
    )


def _rollout(instance, module) -> int:
    state = ScheduleState(instance)
    while not state.done:
        features = state.candidate_features()
        action = int(max((float(module.score(feature, state)), -feature.op_id, feature.op_id) for feature in features)[2])
        feasible = {feature.op_id for feature in features}
        if action not in feasible:
            raise ValueError(f"invalid action {action}; feasible={sorted(feasible)}")
        state.dispatch(action)
    return int(state.makespan)


if __name__ == "__main__":
    main()
