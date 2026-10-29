import random
import os


def parent_selection(pop, m):
    if not pop:
        return []

    ordered = sorted(
        pop,
        key=lambda item: (
            _pareto_rank_score(item),
            float(item.get("objective", float("inf"))),
            -float(item.get("alignment_score", 0.0)),
        ),
    )
    weights = [1.0 / (idx + 1) for idx in range(len(ordered))]
    if m >= len(ordered):
        return ordered[:]
    return random.choices(ordered, weights=weights, k=m)


def _pareto_rank_score(item):
    objective = float(item.get("objective", float("inf")))
    alignment = float(item.get("alignment_score", 0.0))
    align_weight = float(os.environ.get("EOH_PARETO_ALIGN_WEIGHT", "0.10"))
    score = objective - align_weight * alignment
    if os.environ.get("EOH_TA_RUNTIME_OBJECTIVE", "0") == "1":
        runtime = float(item.get("runtime_per_decision_us", float("inf")))
        runtime_weight = float(os.environ.get("EOH_PARETO_RUNTIME_WEIGHT", "0.10"))
        score += runtime_weight * runtime
    return score
