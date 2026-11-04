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
    parser = argparse.ArgumentParser(description="Run teacher-aware EOH/mEOH for Max-Cut flip local search.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--graph-set", default="", help="Optional tomdbar eco-dqn graph-set .pkl.")
    parser.add_argument("--bench-root", default="", help="Optional MaxCut-Bench repo root or data root.")
    parser.add_argument("--bench-dir", default="", help="Optional concrete MaxCut-Bench dataset directory with .npz files.")
    parser.add_argument("--bench-split", default="training", choices=["training", "validation", "testing"])
    parser.add_argument("--bench-distribution", default="", help="MaxCut-Bench distribution name, e.g. BA_20.")
    parser.add_argument(
        "--bench-distributions",
        default="",
        help="Optional comma-separated MaxCut-Bench distributions for mixed training, e.g. BA_20,Physics,BA_200vertices_weighted.",
    )
    parser.add_argument("--distribution", choices=["er", "ba"], default="er")
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--er-p", type=float, default=0.15)
    parser.add_argument("--ba-m", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--n-pop", type=int, default=10)
    parser.add_argument("--n-proc", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--teacher-aware", action="store_true")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--eco-root", default="")
    parser.add_argument("--eco-ckpt", default="")
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
    parser.add_argument("--operators", default="", help="Optional comma-separated EC operators, e.g. s or s,m1.")
    parser.add_argument("--interface", choices=["action", "score"], default="action")
    parser.add_argument("--seed-programs", default=str(ROOT / "seeds" / "maxcut_flip_seeds.json"))
    parser.add_argument("--global-seed", type=int, default=2024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set OPENAI_API_KEY before starting evolution")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EOH_PROBLEM_FAMILY"] = "maxcut"
    os.environ["EOH_MAXCUT_GRAPH_SET"] = str(Path(args.graph_set).resolve()) if args.graph_set else ""
    os.environ["EOH_MAXCUT_BENCH_ROOT"] = str(Path(args.bench_root).resolve()) if args.bench_root else ""
    os.environ["EOH_MAXCUT_BENCH_DIR"] = str(Path(args.bench_dir).resolve()) if args.bench_dir else ""
    os.environ["EOH_MAXCUT_BENCH_SPLIT"] = str(args.bench_split)
    os.environ["EOH_MAXCUT_BENCH_DISTRIBUTION"] = str(args.bench_distribution)
    os.environ["EOH_MAXCUT_BENCH_DISTRIBUTIONS"] = str(args.bench_distributions)
    os.environ["EOH_MAXCUT_DISTRIBUTION"] = args.distribution
    os.environ["EOH_MAXCUT_N_NODES"] = str(args.nodes)
    os.environ["EOH_MAXCUT_N_INSTANCE"] = str(args.instances)
    os.environ["EOH_MAXCUT_ER_P"] = str(args.er_p)
    os.environ["EOH_MAXCUT_BA_M"] = str(args.ba_m)
    os.environ["EOH_MAXCUT_MAX_STEPS"] = str(args.max_steps or (2 * args.nodes))
    os.environ["EOH_MAXCUT_ATTEMPTS"] = str(args.attempts)
    os.environ["EOH_MAXCUT_SEED"] = str(args.seed)
    os.environ["EOH_MAXCUT_INTERFACE"] = str(args.interface)
    os.environ["EOH_TEACHER_AWARE"] = "1" if args.teacher_aware else "0"
    os.environ["EOH_TA_DEVICE"] = str(args.teacher_device)
    os.environ["EOH_ECO_ROOT"] = str(Path(args.eco_root).resolve()) if args.eco_root else ""
    os.environ["EOH_ECO_CKPT"] = str(Path(args.eco_ckpt).resolve()) if args.eco_ckpt else ""
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

    seed_path = Path(args.seed_programs).resolve() if args.seed_programs else None
    ec_operators = (
        [op.strip() for op in args.operators.split(",") if op.strip()]
        if args.operators
        else (["e1", "e2", "m1", "m2", "m3"] if args.teacher_aware else ["e1", "e2", "m1", "m2"])
    )
    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem="maxcut_flip",
        llm_api_endpoint=args.base_url,
        llm_api_key=args.api_key,
        llm_model=args.model,
        ec_pop_size=args.pop_size,
        ec_n_pop=args.n_pop,
        ec_operators=ec_operators,
        ec_operator_weights=[1] * len(ec_operators),
        exp_n_proc=args.n_proc,
        eva_timeout=args.timeout,
        exp_use_seed=bool(seed_path),
        exp_seed_path=str(seed_path) if seed_path else "./seeds/seeds.json",
        management="pop_pareto" if (args.teacher_aware and args.multi_objective) else "pop_greedy",
        selection="pareto_rank" if (args.teacher_aware and args.multi_objective) else "prob_rank",
        exp_debug_mode=args.debug,
        exp_output_path=str(out_dir) + "/",
    )

    meta = {
        "method": "eoh_native_maxcut_flip",
        "base_url": args.base_url,
        "model": args.model,
        "graph_set": str(Path(args.graph_set).resolve()) if args.graph_set else "",
        "bench_root": str(Path(args.bench_root).resolve()) if args.bench_root else "",
        "bench_dir": str(Path(args.bench_dir).resolve()) if args.bench_dir else "",
        "bench_split": args.bench_split,
        "bench_distribution": args.bench_distribution,
        "bench_distributions": args.bench_distributions,
        "distribution": args.distribution,
        "nodes": args.nodes,
        "instances": args.instances,
        "er_p": args.er_p,
        "ba_m": args.ba_m,
        "max_steps": args.max_steps or (2 * args.nodes),
        "attempts": args.attempts,
        "seed": args.seed,
        "interface": args.interface,
        "ec_pop_size": args.pop_size,
        "ec_n_pop": args.n_pop,
        "exp_n_proc": args.n_proc,
        "eva_timeout": args.timeout,
        "teacher_aware": args.teacher_aware,
        "teacher_device": args.teacher_device,
        "eco_root": str(Path(args.eco_root).resolve()) if args.eco_root else "",
        "eco_ckpt": str(Path(args.eco_ckpt).resolve()) if args.eco_ckpt else "",
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
        "seed_programs": str(seed_path) if seed_path else "",
        "global_seed": args.global_seed,
        "max_fe": int(os.environ.get("EOH_MAX_FE", "0")),
        "strict_fe": os.environ.get("EOH_STRICT_FE", "0") == "1",
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    evolution = EVOL(paras)
    evolution.run()


if __name__ == "__main__":
    main()
