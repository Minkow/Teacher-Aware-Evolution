from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

from tae.tasks.jssp.core import CandidateFeature
from tae.tasks.jssp.traces import TeacherTrace


@dataclass(frozen=True)
class AlignmentResult:
    mean_rank_correlation: float
    top1_agreement: float
    states: int
    mean_teacher_action_percentile: float | None = None
    diagnostic: str = ""


def policy_ranking_alignment(
    rule,
    traces: list[TeacherTrace],
    *,
    mode: str = "direct",
    mc_samples: int = 32,
    seed: int = 0,
) -> AlignmentResult:
    """Compare rule scores with L2D candidate preferences.

    `direct` ranks candidates by policy probability.
    `mc` samples actions from the policy and ranks candidates by empirical frequency.
    """

    if mode not in {"direct", "mc"}:
        raise ValueError("alignment mode must be `direct` or `mc`")
    rng = random.Random(seed)
    correlations = []
    agreements = []
    for trace in traces:
        for step in trace.steps:
            if len(step.candidate_features) < 2:
                continue
            try:
                rule_scores = [rule.score(feature, None) for feature in step.candidate_features]
            except Exception:
                correlations.append(-1.0)
                agreements.append(0)
                continue
            if mode == "direct":
                policy_scores = list(step.policy_probs)
            else:
                policy_scores = _mc_policy_frequencies(step.policy_probs, mc_samples, rng)
            correlations.append(spearman_correlation(rule_scores, policy_scores))
            agreements.append(
                int(
                    _argmax(rule_scores, step.candidate_features)
                    == _argmax(policy_scores, step.candidate_features)
                )
            )
    return AlignmentResult(
        mean_rank_correlation=mean(correlations) if correlations else 0.0,
        top1_agreement=mean(agreements) if agreements else 0.0,
        states=len(correlations),
    )


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("rank correlation inputs must have equal length")
    if len(xs) < 2:
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mean_x = mean(rx)
    mean_y = mean(ry)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    den_x = sum((x - mean_x) ** 2 for x in rx)
    den_y = sum((y - mean_y) ** 2 for y in ry)
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / ((den_x * den_y) ** 0.5)


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and ordered[end][1] == ordered[pos][1]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0
        for idx in range(pos, end):
            ranks[ordered[idx][0]] = avg_rank
        pos = end
    return ranks


def _argmax(scores: list[float], features: tuple[CandidateFeature, ...]) -> int:
    return max((score, -feature.op_id, feature.op_id) for score, feature in zip(scores, features))[2]


def _mc_policy_frequencies(
    probs: tuple[float, ...],
    samples: int,
    rng: random.Random,
) -> list[float]:
    total = sum(max(0.0, p) for p in probs)
    if total <= 0:
        return [1.0 / len(probs) for _ in probs]
    normalized = [max(0.0, p) / total for p in probs]
    counts = [0 for _ in probs]
    for _ in range(samples):
        draw = rng.random()
        acc = 0.0
        for idx, prob in enumerate(normalized):
            acc += prob
            if draw <= acc:
                counts[idx] += 1
                break
    return [count / samples for count in counts]
