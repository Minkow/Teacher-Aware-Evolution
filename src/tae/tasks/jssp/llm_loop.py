from __future__ import annotations

from dataclasses import dataclass

from typing import Any


FEATURE_GUIDE = (
    "- feature.processing_time: duration of this candidate operation. Shorter operations can reduce local waiting, longer ones may clear critical work.\n"
    "- feature.remaining_work: total processing time still left in this job including this operation. High values indicate a long job tail.\n"
    "- feature.remaining_ops: number of operations still left in this job including this operation. Small values mean tail-completion stage.\n"
    "- feature.job_ready_time: earliest time when this job can start its next operation because of job precedence.\n"
    "- feature.machine_ready_time: latest end time among already scheduled operations on the candidate machine.\n"
    "- feature.earliest_start: earliest feasible insertion start time for this operation under current partial schedule.\n"
    "- feature.earliest_finish: earliest_start + processing_time. Lower values favor immediate completion.\n"
    "- feature.machine_queue_len: number of operations already inserted on this machine. Higher values indicate a busier machine history.\n"
    "- feature.machine_total_work: unscheduled work still requiring this machine across all jobs. High values indicate a bottleneck machine.\n"
    "- feature.job_progress: op_index / n_machines in [0, 1). Larger values mean the job is later in its route.\n"
    "- feature.lower_bound_after: simple lower-bound estimate after choosing this operation; lower values often mean less future tail risk.\n"
)

BASELINE_GUIDE = (
    "- fifo: first-in/operation-id order; a weak queue-order baseline.\n"
    "- spt: shortest processing time first; reduces local machine occupancy but can starve long jobs.\n"
    "- lpt: longest processing time first; clears long operations early but may delay quick completions.\n"
    "- mwkr: most work remaining; prioritizes jobs with large unfinished tails.\n"
    "- mor: most operations remaining; prioritizes jobs that still have many precedence steps.\n"
    "- fdd_mwkr: flow-due-date divided by remaining-work style rule; balances due-progress pressure and remaining job work.\n"
    "- eft: earliest finish time; greedily finishes the operation that can complete soonest.\n"
    "- critical_lb: lower-bound proxy after dispatch; estimates future tail risk after choosing a candidate.\n"
)

PROBLEM_CONTEXT = (
    "Problem context:\n"
    "Job Shop Scheduling (JSSP) has n jobs and m machines. Each job is an ordered chain of operations; "
    "operation k of a job must be processed on its required machine for its processing time, and a job "
    "cannot run two operations out of order. A machine can process only one operation at a time. During "
    "construction, the scheduler repeatedly sees one feasible next operation per unfinished job. Your "
    "score(feature, state) function assigns a priority to each feasible operation; the highest score is "
    "dispatched using earliest feasible insertion. The final objective is to minimize makespan, i.e. the "
    "maximum job completion time after all operations are scheduled."
)

TEACHER_CONTEXT = (
    "Teacher signal meaning:\n"
    "The teacher is an Offline-LD RL policy. It is not used to execute your rule. Instead, after a candidate "
    "rule creates partial schedules, we query the teacher on those same states and compare action rankings. "
    "rank_corr is Spearman correlation between your candidate scores and teacher Q-value ranking on feasible "
    "actions. top1 is exact agreement with the teacher's best action. teacher_percentile is the teacher rank "
    "percentile of the action chosen by your rule. These are training/evaluation diagnostics only; they are "
    "not fields available inside score(feature, state). A good rule should primarily reduce makespan; teacher "
    "agreement is a secondary guide when it does not hurt makespan."
)


@dataclass(frozen=True)
class RuleProgram:
    name: str
    source: str


def build_improvement_prompt(
    best_scores: list[Any],
    baseline_summary: str,
    teacher_summary: str = "",
) -> str:
    """Builds the prompt payload for an external LLM algorithm designer."""

    lines = [
        "You are designing an interpretable dispatching heuristic for JSSP.",
        "The algorithm must expose a function score(feature, state) -> float.",
        "At each step, the scheduler chooses the feasible operation with the highest score.",
        "",
        "Available candidate feature fields:",
        FEATURE_GUIDE,
        "",
        "Baseline summary:",
        baseline_summary,
    ]
    if teacher_summary:
        lines.extend(["", "L2D policy teacher summary:", teacher_summary])
    lines.extend(["", "Current best rules:"])
    for idx, score in enumerate(best_scores, start=1):
        alignment = _alignment_text(score)
        descriptor = (
            score.rule.describe()
            if hasattr(score.rule, "describe")
            else score.rule.source.strip().replace("\n", " ")[:240]
        )
        lines.append(
            f"{idx}. mean_makespan={score.evaluation.mean_makespan:.3f}, "
            f"alignment={alignment}, formula={descriptor}"
        )
    lines.extend(
        [
            "",
            "Return only a Python function body named score(feature, state).",
            "Keep it deterministic, side-effect free, and based only on the listed fields.",
        ]
    )
    return "\n".join(lines)


def build_program_prompt(
    best_scores: list[Any],
    parent_scores: tuple[Any, ...],
    baseline_summary: str,
    teacher_summary: str = "",
    proposer_guidance: str = "",
) -> str:
    lines = [
        PROBLEM_CONTEXT,
        "",
        "Task: generate a deterministic priority function for candidate selection.",
        "Evaluator: lower mean_makespan is better; higher rank_corr with teacher policy is better.",
        "Teacher signal may be OPD-style: the teacher ranks actions on states visited by the candidate rule.",
        "",
        "Return exactly one Python function: `def score(feature, state): ...`",
        "",
        "Constraints:",
        "- No imports.",
        "- No file/network/randomness/printing/global state.",
        "- Use only arithmetic, if/else, locals, safe builtins, and math.",
        "- Prefer feature-only rules; state-dependent rules receive weak alignment scores.",
        "- Do not invent fields. Metrics such as mean_makespan, rank_corr, top1, and teacher_percentile are evaluator outputs, not usable features.",
        "- Return one numeric score.",
        "",
        "Feature schema and semantics:",
        FEATURE_GUIDE,
        "",
        "Baseline heuristic meanings:",
        BASELINE_GUIDE,
        "",
        "Baseline metrics:",
        baseline_summary,
        "",
        TEACHER_CONTEXT,
        "",
        "Teacher/alignment summary:",
        teacher_summary,
    ]
    if proposer_guidance:
        lines.extend(["", "Proposer guidance from sampled teacher traces:", proposer_guidance])
    lines.extend(["", "Current Pareto candidates:"])
    for idx, score in enumerate(best_scores, start=1):
        lines.append(_score_summary(idx, score))
    lines.extend(["", "Parent programs to improve or recombine:"])
    for idx, score in enumerate(parent_scores, start=1):
        lines.append(_score_summary(idx, score))
        lines.append("```python")
        lines.append(score.rule.source.strip())
        lines.append("```")
    lines.extend(
        [
            "",
            "Generate one new rule that is materially different from the parents and targets the two metrics.",
            "Return only the Python code for `score`; do not include prose.",
        ]
    )
    return "\n".join(lines)


def build_proposer_prompt(
    *,
    baseline_summary: str,
    teacher_summary: str,
    trace_comparison_summary: str,
    current_scores: list[Any],
    generation: int | None = None,
) -> str:
    generation_text = "" if generation is None else f"Generation: {generation}\n"
    lines = [
        "You are a search proposer for LLM-based heuristic evolution on JSSP.",
        "Do not write code. Produce concise guidance for the next code-generating model.",
        "Your job is to compare traces from the best-performing rule, the most teacher-aligned rule, and the RL teacher policy.",
        "Explain which feature tradeoffs the next generation should try, and which misleading directions to avoid.",
        "",
        generation_text.rstrip(),
        PROBLEM_CONTEXT,
        "",
        "Available features:",
        FEATURE_GUIDE,
        "",
        "Baseline heuristic meanings:",
        BASELINE_GUIDE,
        "",
        TEACHER_CONTEXT,
        "",
        "Baseline metrics:",
        baseline_summary,
        "",
        "Teacher/alignment summary:",
        teacher_summary,
        "",
        "Trace comparison summary:",
        trace_comparison_summary,
        "",
        "Current candidates:",
    ]
    for idx, score in enumerate(current_scores, start=1):
        lines.append(_score_summary(idx, score))
    lines.extend(
        [
            "",
            "Return 5 to 8 short bullet points. Focus on feature tradeoffs likely to improve makespan.",
            "Use the trace comparison to distinguish makespan-improving behavior from teacher-imitation-only behavior.",
            "Do not invent feature names. Do not recommend copying the teacher blindly.",
        ]
    )
    return "\n".join(lines)


def _score_summary(idx: int, score: Any) -> str:
    return (
        f"{idx}. {score.rule.name}: mean_makespan={score.evaluation.mean_makespan:.3f}, "
        f"best={score.evaluation.best_makespan}, worst={score.evaluation.worst_makespan}, "
        f"{_alignment_text(score)}"
    )


def _alignment_text(score: Any) -> str:
    if hasattr(score, "alignment"):
        if score.alignment is None:
            return "rank_corr=n/a, top1=n/a"
        text = (
            f"rank_corr={score.alignment.mean_rank_correlation:.3f}, "
            f"top1={score.alignment.top1_agreement:.3f}"
        )
        if score.alignment.mean_teacher_action_percentile is not None:
            text += f", teacher_percentile={score.alignment.mean_teacher_action_percentile:.3f}"
        return text
    agreement = getattr(score, "teacher_agreement", None)
    if agreement is None:
        return "teacher_agreement=n/a"
    return f"teacher_agreement={agreement:.3f}"
