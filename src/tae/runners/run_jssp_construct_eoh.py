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

from tae.runners.jssp_construct import JSSPCONST  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native teacher-mEOH for JSSP dispatching heuristics.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--machines", type=int, default=10)
    parser.add_argument("--train-npy")
    parser.add_argument("--train-jsonl")
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-seed", type=int, default=2024)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--n-pop", type=int, default=10)
    parser.add_argument("--children", type=int)
    parser.add_argument("--n-proc", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--teacher-aware", action="store_true")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--offline-ld-root", default="third_party/Offline-LD")
    parser.add_argument("--offline-ld-checkpoint-root", default="results/offline_ld_full/checkpoints_100")
    parser.add_argument("--offline-ld-checkpoint-dir")
    parser.add_argument("--offline-ld-step", default="best")
    parser.add_argument("--teacher-stride", type=int, default=4)
    parser.add_argument("--teacher-max-states", type=int, default=18)
    parser.add_argument("--teacher-sample-instances", type=int, default=4)
    parser.add_argument("--teacher-start-gen", type=int, default=0)
    parser.add_argument("--multi-objective", action="store_true")
    parser.add_argument("--stagger-seconds", type=float, default=0.0)
    parser.add_argument("--llm-timeout-s", type=int, default=240)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff-cap-s", type=int, default=30)
    parser.add_argument("--pareto-align-weight", type=float, default=0.10)
    parser.add_argument("--pareto-fill-align-weight", type=float, default=0.35)
    parser.add_argument("--operators", default="", help="Optional comma-separated EC operators, e.g. s or s,m1.")
    parser.add_argument("--seed-programs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.api_key and not args.dry_run:
        parser.error("Set OPENAI_API_KEY before starting evolution")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EOH_PROBLEM_FAMILY"] = "jssp"
    os.environ["EOH_JSSP_N_JOBS"] = str(args.jobs)
    os.environ["EOH_JSSP_N_MACHINES"] = str(args.machines)
    os.environ["EOH_JSSP_N_INSTANCE"] = str(args.train_count)
    os.environ["EOH_JSSP_SEED"] = str(args.seed)
    os.environ["EOH_GLOBAL_SEED"] = str(args.global_seed)
    os.environ["EOH_JSSP_TRAIN_NPY"] = str(Path(args.train_npy).resolve()) if args.train_npy else ""
    os.environ["EOH_JSSP_TRAIN_JSONL"] = str(Path(args.train_jsonl).resolve()) if args.train_jsonl else ""
    os.environ["EOH_JSSP_OFFLINE_LD_ROOT"] = str(Path(args.offline_ld_root).resolve())
    os.environ["EOH_JSSP_TEACHER_CHECKPOINT_ROOT"] = str(Path(args.offline_ld_checkpoint_root).resolve())
    os.environ["EOH_JSSP_TEACHER_CHECKPOINT_DIR"] = (
        str(Path(args.offline_ld_checkpoint_dir).resolve()) if args.offline_ld_checkpoint_dir else ""
    )
    os.environ["EOH_JSSP_TEACHER_STEP"] = str(args.offline_ld_step)
    os.environ["EOH_CHILDREN"] = str(args.children if args.children is not None else args.pop_size)
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

    problem = JSSPCONST()
    ec_operators = (
        [op.strip() for op in args.operators.split(",") if op.strip()]
        if args.operators
        else (["e2", "m1", "m2", "m3"] if args.teacher_aware else ["e1", "e2", "m1", "m2"])
    )
    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem=problem,
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
        "method": "eoh_native_jssp_construct",
        "base_url": args.base_url,
        "model": args.model,
        "jobs": args.jobs,
        "machines": args.machines,
        "train_npy": os.environ["EOH_JSSP_TRAIN_NPY"],
        "train_jsonl": os.environ["EOH_JSSP_TRAIN_JSONL"],
        "train_count": args.train_count,
        "seed": args.seed,
        "global_seed": args.global_seed,
        "ec_pop_size": args.pop_size,
        "ec_n_pop": args.n_pop,
        "ec_children": int(os.environ["EOH_CHILDREN"]),
        "exp_n_proc": args.n_proc,
        "eva_timeout": args.timeout,
        "teacher_aware": args.teacher_aware,
        "teacher_device": args.teacher_device,
        "offline_ld_root": os.environ["EOH_JSSP_OFFLINE_LD_ROOT"],
        "offline_ld_checkpoint_root": os.environ["EOH_JSSP_TEACHER_CHECKPOINT_ROOT"],
        "offline_ld_checkpoint_dir": os.environ["EOH_JSSP_TEACHER_CHECKPOINT_DIR"],
        "offline_ld_step": args.offline_ld_step,
        "teacher_stride": args.teacher_stride,
        "teacher_max_states": args.teacher_max_states,
        "teacher_sample_instances": args.teacher_sample_instances,
        "teacher_stage_stratified": os.environ.get("EOH_TA_STAGE_STRATIFIED", "0") == "1",
        "teacher_start_gen": args.teacher_start_gen,
        "alignment_metric": os.environ.get("EOH_ALIGN_METRIC", "top1"),
        "multi_objective": args.multi_objective,
        "runtime_objective": os.environ.get("EOH_TA_RUNTIME_OBJECTIVE", "0") == "1",
        "runtime_metric": "heuristic thread CPU microseconds per dispatch decision",
        "stagger_seconds": args.stagger_seconds,
        "llm_timeout_s": args.llm_timeout_s,
        "llm_retries": args.llm_retries,
        "llm_retry_backoff_cap_s": args.llm_retry_backoff_cap_s,
        "llm_reasoning_effort": os.environ.get("LLM_REASONING_EFFORT", ""),
        "strict_fe": os.environ.get("EOH_STRICT_FE", "0") == "1",
        "strict_analyzer": os.environ.get("EOH_TA_STRICT_ANALYZER", "0") == "1",
        "prompt_version": os.environ.get("EOH_PROMPT_VERSION", ""),
        "structural_memory": os.environ.get("EOH_STRUCTURAL_MEMORY", "0") == "1",
        "structural_memory_limit": int(
            os.environ.get("EOH_STRUCTURAL_MEMORY_LIMIT", "100")
        ),
        "structural_memory_cases": int(
            os.environ.get("EOH_STRUCTURAL_MEMORY_CASES", "3")
        ),
        "structural_hypotheses": int(
            os.environ.get("EOH_STRUCTURAL_HYPOTHESES", "4")
        ),
        "structural_memory_operator": (
            "m" if os.environ.get("EOH_STRUCTURAL_MEMORY", "0") == "1" else ""
        ),
        "e2_realtime_analyzer_refresh": (
            os.environ.get("EOH_STRUCTURAL_MEMORY", "0") == "1"
        ),
        "analyzer_code_chars": int(
            os.environ.get("EOH_TA_ANALYZER_CODE_CHARS", "12000")
        ),
        "duplicate_generation_retries": int(
            os.environ.get("EOH_DUPLICATE_GENERATION_RETRIES", "2")
        ),
        "p_one_survivor_per_parent": (
            os.environ.get("EOH_P_ONE_SURVIVOR_PER_PARENT", "0") == "1"
        ),
        "teacher_trace_semantics": "paired_candidate_on_policy_states",
        "pareto_align_weight": args.pareto_align_weight,
        "pareto_fill_align_weight": args.pareto_fill_align_weight,
        "pareto_performance_slack": performance_slack(),
        "ec_operators": ec_operators,
        "seed_programs": str(Path(args.seed_programs).resolve()) if args.seed_programs else "",
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    evolution = EVOL(paras)
    if args.dry_run:
        print("dry_run_ok")
        return
    evolution.run()


if __name__ == "__main__":
    main()
