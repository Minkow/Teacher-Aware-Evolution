"""Portable entry point for the Inline Soft-Anchor implementation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("jssp10", "jssp15", "jssp20", "tsp50", "cvrp50", "maxcut", "jssp", "tsp", "cvrp")


def bundled_data(args, family):
    if family == "jssp":
        size = int(args.task.removeprefix("jssp") or "10")
        jobs, machines = args.jobs or size, args.machines or size
        if jobs == machines and jobs in {10, 15, 20}:
            return ROOT / "datasets" / f"jssp{jobs}" / "train.npy"
    elif family in {"tsp", "cvrp"} and (args.problem_size or 50) == 50:
        return ROOT / "datasets" / f"{family}50" / "train.pkl"
    elif family == "maxcut" and (args.nodes or 200) == 200:
        return ROOT / "datasets" / "maxcut"
    raise ValueError("No bundled dataset for these dimensions; supply --data explicitly")


def make_plan(args):
    family = "jssp" if args.task.startswith("jssp") else args.task.removesuffix("50")
    rollout_workers = args.rollout_workers if args.rollout_workers is not None else 12
    if rollout_workers < 1 or args.workers < 1:
        raise ValueError("Worker counts must be positive")
    applicable = {"jobs": {"jssp"}, "machines": {"jssp"}, "problem_size": {"tsp", "cvrp"},
                  "capacity": {"cvrp"}, "nodes": {"maxcut"}, "max_steps": {"maxcut"}}
    for name, families in applicable.items():
        value = getattr(args, name)
        if value is not None and (family not in families or not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and applicable to {family}")
    if args.train_count is not None and args.train_count <= 0:
        raise ValueError("--train-count must be positive")
    if args.data is None:
        args.data = bundled_data(args, family)
    env = {
        "EOH_MAX_FE": "100", "EOH_STRICT_FE": "0", "EOH_INITIAL_LLM_CHILDREN": "5",
        "EOH_INIT_SEED_FIRST": "1",
        "EOH_CHILDREN": "6", "EOH_S_CHILD_SCHEDULE": "8,8,5,5,4",
        "EOH_P_CHILD_SCHEDULE": "8,8,5,5,4", "EOH_M_CHILD_SCHEDULE": "8,8,5,5,4",
        "EOH_S_GEN1_PARENT_ALLOCATION": "3,3,2", "EOH_M_SUBJECT_MAX_PER_BATCH": "2",
        "EOH_P_ONE_SURVIVOR_PER_PARENT": "1", "EOH_M_ONE_SURVIVOR_PER_PAIR": "1",
        "EOH_TA_STAGE_STRATIFIED": "1", "EOH_TA_USE_LLM_ANALYZER": "1",
        "EOH_TA_STRICT_ANALYZER": "1", "EOH_TA_ANALYZER_JSON_RETRIES": "2",
        "EOH_TA_ANALYZER_FAILURE_POLICY": "reuse_last_or_empty",
        "EOH_TA_ANALYZER_TIMEOUT_S": "1800", "EOH_TA_ANALYZER_RETRIES": "5",
        "EOH_TA_ANALYZER_MODEL": args.model, "EOH_TA_ANALYZER_BASE_URL": args.base_url,
        "EOH_RESPONSE_PARSE_ATTEMPTS": "1", "EOH_DUPLICATE_GENERATION_RETRIES": "0",
        "EOH_OFFSPRING_ATTEMPTS": "1", "EOH_FE_TIMEOUT": "1800",
        "EOH_FE_ARCHIVE_PATH": str(args.output.resolve() / "evaluation_archive.json"),
        "EOH_FE_MANIFEST_PATH": str(args.output.resolve() / "fe_manifest.json"),
        "EOH_INSTANCE_ROLLOUT_WORKERS": str(rollout_workers),
        "EOH_MAXCUT_ROLLOUT_WORKERS": str(rollout_workers),
        "EOH_TA_RUNTIME_OBJECTIVE": "1", "EOH_PARETO_RUNTIME_WEIGHT": "0.10",
        "EOH_RUNTIME_FINAL_POP_MULTIPLIER": "5" if args.task in {"jssp20", "maxcut"} else "",
        "EOH_RUNTIME_MAX_MULTIPLIER": "", "EOH_GLOBAL_SEED": str(args.seed),
        "EOH_PROMPT_VERSION": "inline_soft_anchor_release_v1",
        "EOH_STRUCTURAL_MEMORY": "1" if family in {"cvrp", "maxcut"} else "0",
        "LLM_REASONING_EFFORT": "medium", "LLM_TEMPERATURE": "1.0", "LLM_TOP_P": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    }
    seed_bank = args.seed_programs.resolve() if args.seed_programs else "<initial-programs.json>"
    command = [sys.executable, "-m", f"tae.tasks.{family}.train",
               "--output-dir", str(args.output.resolve()), "--base-url", args.base_url,
               "--model", args.model, "--global-seed", str(args.seed),
               "--seed-programs", str(seed_bank), "--pop-size", "12", "--n-pop", "5",
               "--n-proc", str(args.workers), "--operators", "s,p,m",
               "--teacher-aware", "--multi-objective", "--teacher-device", args.device,
               "--teacher-max-states", "20", "--teacher-start-gen", "0",
               "--llm-timeout-s", "1800", "--llm-retries", "5", "--timeout", "1800"]
    if family == "jssp":
        size = int(args.task.removeprefix("jssp") or "10")
        command += ["--jobs", str(args.jobs or size), "--machines", str(args.machines or size),
                    "--train-count", str(args.train_count or 100), "--children", "6"]
        if args.data:
            command += ["--train-npy", str(args.data.resolve())]
        if args.teacher_root:
            command += ["--offline-ld-root", str(args.teacher_root.resolve())]
        if args.checkpoint:
            command += ["--offline-ld-checkpoint-dir", str(args.checkpoint.resolve())]
    elif family in {"tsp", "cvrp"}:
        command += ["--train-pkl", str(args.data.resolve()) if args.data else "<train.pkl>",
                    "--problem-size", str(args.problem_size or 50), "--train-count", str(args.train_count or 64)]
        if family == "cvrp":
            command += ["--teacher-stride", "3"]
            if args.capacity is not None:
                command += ["--capacity", str(args.capacity)]
        if args.teacher_root:
            env["TAE_LEHD_ROOT"] = str(args.teacher_root.resolve())
        if args.checkpoint:
            env[f"TAE_{family.upper()}_CHECKPOINT"] = str(args.checkpoint.resolve())
    else:
        command += ["--nodes", str(args.nodes or 200), "--instances", str(args.train_count or 30), "--max-steps", str(args.max_steps or 400),
                    "--interface", "action", "--attempts", "1"]
        if args.data:
            command += ["--bench-dir", str(args.data.resolve())]
        if args.teacher_root:
            command += ["--eco-root", str(args.teacher_root.resolve())]
        if args.checkpoint:
            command += ["--eco-ckpt", str(args.checkpoint.resolve())]
    return command, env


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-programs", type=Path, help="External JSON containing five initial programs")
    for name in ("jobs", "machines", "problem-size", "nodes", "max-steps", "train-count"):
        p.add_argument("--" + name, type=int, help="Override the task preset")
    p.add_argument("--capacity", type=float, help="CVRP vehicle capacity")
    p.add_argument("--output", type=Path, default=Path("runs/evolution"))
    p.add_argument("--data", type=Path, help="Fixed training data; defaults to the bundled task dataset")
    p.add_argument("--teacher-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rollout-workers", type=int, help="Instance workers: default 12")
    p.add_argument("--device", default="cuda")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-5.4"))
    p.add_argument("--dry-run", action="store_true", help="Print the plan without API calls or asset loading")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        command, settings = make_plan(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps({"command": command, "settings": settings}, indent=2))
        return
    for name in ("data", "teacher_root", "checkpoint", "seed_programs"):
        value = getattr(args, name)
        if value is None or not value.exists():
            p.error(f"--{name.replace('_', '-')} must point to an existing asset; see docs/assets.md")
    initial = json.loads(args.seed_programs.read_text())
    if not isinstance(initial, list) or len(initial) != 5:
        p.error("Initial program JSON must contain exactly five records")
    for item in initial:
        if not isinstance(item, dict) or not isinstance(item.get("code"), str):
            p.error("Each initial program requires a code string")
        compile(item["code"], "<initial-program>", "exec")
    if not os.environ.get("OPENAI_API_KEY"):
        p.error("Set OPENAI_API_KEY; credentials are not passed in the command line")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output directory is not empty; choose a distinct run directory")
    # Do not inherit hidden experiment switches from the development shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EOH_", "LLM_", "TAE_"))}
    env.update(settings)
    env["PYTHONPATH"] = str(ROOT / "src")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "launch.json").write_text(json.dumps({"command": command, "settings": settings}, indent=2) + "\n")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
