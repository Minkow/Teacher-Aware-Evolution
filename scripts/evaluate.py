"""Evaluate a training-selected program on a bundled held-out dataset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/testing"
JSSP = {
    "jssp10": ("generatedData10_10_Seed300.npy", 10, 10, 100),
    "jssp15": ("generatedData15_15_Seed300.npy", 15, 15, 100),
    "jssp20": ("generatedData20_20_Seed300.npy", 20, 20, 100),
    "jssp30x20": ("generatedData30_20_Seed300.npy", 30, 20, 100),
    "jssp50x20": ("generatedData50_20_Seed300.npy", 50, 20, 100),
    "taillard20x20": ("tai20x20.npy", 20, 20, 10),
    "taillard30x20": ("tai30x20.npy", 30, 20, 10),
}
TSP = {"tsp20": 20, "tsp50": 50, "tsp100": 100, "tsp200": 200}
CVRP = {"cvrp50": ("test.pkl", 100), "cvrp200": ("vrp200_test_lkh.txt", 128),
        "vrplib192": ("vrplib_192instances.txt", 192)}
DATASETS = (*JSSP, *TSP, "tsplib70", *CVRP)


def make_plan(args):
    if args.workers < 1 or args.pop_id < 0:
        raise ValueError("Workers must be positive and population index nonnegative")
    dataset = args.dataset
    if dataset in JSSP:
        filename, jobs, machines, count = JSSP[dataset]
        family = "jssp"
        options = ["--test-npy", str(DATA / family / filename), "--jobs", str(jobs),
                   "--machines", str(machines), "--test-count", str(count)]
    elif dataset in TSP:
        family = "tsp"
        options = ["--test-root", str(DATA / family), "--sizes", str(TSP[dataset]),
                   "--limit", "100", "--n-proc", str(args.workers)]
    elif dataset == "tsplib70":
        family = "tsp"
        options = ["--test-lehd-txt", str(DATA / family / "TSPlib_70instances.txt"),
                   "--limit", "70", "--n-proc", str(args.workers)]
    else:
        family = "cvrp"
        filename, count = CVRP[dataset]
        flag = "--test-pkl" if dataset == "cvrp50" else "--test-lehd-txt"
        options = [flag, str(DATA / family / filename), "--limit", str(count),
                   "--workers", str(args.workers)]
    return [sys.executable, "-m", f"tae.tasks.{family}.evaluate",
            "--result-dir", str(args.result_dir.resolve()), "--pop-id", str(args.pop_id),
            *options, "--out-csv", str(args.output.resolve() / f"{dataset}.csv"),
            "--out-json", str(args.output.resolve() / f"{dataset}.json")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--result-dir", type=Path, required=True, help="Evolution output directory")
    parser.add_argument("--output", type=Path, default=Path("results/heldout"))
    parser.add_argument("--pop-id", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Routing evaluation workers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        command = make_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    if any((args.output / f"{args.dataset}.{suffix}").exists() for suffix in ("csv", "json")):
        parser.error("Evaluation output already exists; choose another output directory")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
