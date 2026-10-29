from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from tae.engine.eoh import EVOL  # type: ignore  # noqa: E402
from tae.engine.methods.management.pop_pareto import performance_slack
from tae.engine.utils.getParas import Paras  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native EOH cvrp_construct with fixed dataset and OpenAI-compatible endpoint.")
    parser.add_argument("--train-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--problem-size", type=int, default=50)
    parser.add_argument("--capacity", type=float, default=None,
                        help="Override vehicle capacity; default uses each dataset instance's capacity")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--n-pop", type=int, default=10)
    parser.add_argument("--n-proc", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--teacher-aware", action="store_true")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--teacher-stride", type=int, default=4)
    parser.add_argument("--teacher-max-states", type=int, default=18)
    parser.add_argument("--teacher-sample-instances", type=int, default=4)
    parser.add_argument("--teacher-start-gen", type=int, default=0)
    parser.add_argument("--multi-objective", action="store_true")
    parser.add_argument("--stagger-seconds", type=float, default=0.0)
    parser.add_argument("--llm-timeout-s", type=int, default=120)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff-cap-s", type=int, default=60)
    parser.add_argument("--pareto-align-weight", type=float, default=0.10)
    parser.add_argument("--pareto-fill-align-weight", type=float, default=0.35)
    parser.add_argument("--operators", default="", help="Optional comma-separated EC operators, e.g. s,p,m.")
    parser.add_argument("--seed-programs")
    parser.add_argument("--global-seed", type=int, default=2024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set OPENAI_API_KEY before starting evolution")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EOH_PROBLEM_FAMILY"] = "cvrp"
    os.environ["EOH_CVRP_TRAIN_PKL"] = str(Path(args.train_pkl).resolve())
    os.environ["EOH_CVRP_PROBLEM_SIZE"] = str(args.problem_size)
    os.environ["EOH_CVRP_N_INSTANCE"] = str(args.train_count)
    if args.capacity is None:
        os.environ.pop("EOH_CVRP_CAPACITY", None)
    else:
        os.environ["EOH_CVRP_CAPACITY"] = str(args.capacity)
    os.environ["EOH_TEACHER_AWARE"] = "1" if args.teacher_aware else "0"
    os.environ["EOH_TA_DEVICE"] = str(args.teacher_device)
    os.environ["EOH_TA_STRIDE"] = str(args.teacher_stride)
    os.environ["EOH_TA_MAX_STATES"] = str(args.teacher_max_states)
    os.environ["EOH_TA_SAMPLE_INSTANCES"] = str(args.teacher_sample_instances)
    os.environ["EOH_TA_START_GEN"] = str(args.teacher_start_gen)
    os.environ["EOH_TA_MULTI_OBJECTIVE"] = "1" if args.multi_objective else "0"
    os.environ["EOH_STAGGER_S"] = str(args.stagger_seconds)
    os.environ["LLM_TIMEOUT_S"] = str(args.llm_timeout_s)
    os.environ["LLM_RETRIES"] = str(args.llm_retries)
    os.environ["LLM_RETRY_BACKOFF_CAP_S"] = str(args.llm_retry_backoff_cap_s)
    os.environ["EOH_PARETO_ALIGN_WEIGHT"] = str(args.pareto_align_weight)
    os.environ["EOH_PARETO_FILL_ALIGN_WEIGHT"] = str(args.pareto_fill_align_weight)
    os.environ["EOH_GLOBAL_SEED"] = str(args.global_seed)

    ec_operators = (
        [op.strip() for op in args.operators.split(",") if op.strip()]
        if args.operators
        else (["e1", "e2", "m1", "m2", "m3"] if args.teacher_aware else ["e1", "e2", "m1", "m2"])
    )

    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem="cvrp_construct",
        llm_api_endpoint=args.base_url,
        llm_api_key=args.api_key,
        llm_model=args.model,
        ec_pop_size=args.pop_size,
        ec_n_pop=args.n_pop,
        ec_operators=ec_operators,
        ec_operator_weights=[1] * len(ec_operators),
        exp_n_proc=args.n_proc,
        eva_timeout=args.timeout,
        exp_use_seed=bool(args.seed_programs),
        exp_seed_path=str(Path(args.seed_programs).resolve()) if args.seed_programs else "./seeds/seeds.json",
        management="pop_pareto" if (args.teacher_aware and args.multi_objective) else "pop_greedy",
        selection="pareto_rank" if (args.teacher_aware and args.multi_objective) else "prob_rank",
        exp_debug_mode=args.debug,
        exp_output_path=str(out_dir) + "/",
    )

    meta = {
        "method": "eoh_native_cvrp_construct",
        "train_pkl": str(Path(args.train_pkl).resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "problem_size": args.problem_size,
        "capacity": args.capacity,
        "capacity_source": "dataset" if args.capacity is None else "override",
        "train_count": args.train_count,
        "ec_pop_size": args.pop_size,
        "ec_n_pop": args.n_pop,
        "exp_n_proc": args.n_proc,
        "eva_timeout": args.timeout,
        "teacher_aware": args.teacher_aware,
        "teacher_device": args.teacher_device,
        "teacher_stride": args.teacher_stride,
        "teacher_max_states": args.teacher_max_states,
        "teacher_sample_instances": args.teacher_sample_instances,
        "teacher_start_gen": args.teacher_start_gen,
        "alignment_metric": os.environ.get("EOH_ALIGN_METRIC", "top1"),
        "multi_objective": args.multi_objective,
        "pareto_performance_slack": performance_slack(),
        "stagger_seconds": args.stagger_seconds,
        "llm_timeout_s": args.llm_timeout_s,
        "llm_retries": args.llm_retries,
        "llm_retry_backoff_cap_s": args.llm_retry_backoff_cap_s,
        "pareto_align_weight": args.pareto_align_weight,
        "pareto_fill_align_weight": args.pareto_fill_align_weight,
        "ec_operators": ec_operators,
        "seed_programs": str(Path(args.seed_programs).resolve()) if args.seed_programs else "",
        "global_seed": args.global_seed,
        "max_fe": int(os.environ.get("EOH_MAX_FE", "0")),
        "strict_fe": os.environ.get("EOH_STRICT_FE", "0") == "1",
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    evolution = EVOL(paras)
    evolution.run()


if __name__ == "__main__":
    main()
