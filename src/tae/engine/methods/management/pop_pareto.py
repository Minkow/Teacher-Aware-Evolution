import os


PERFORMANCE_SLACK = {"jssp": 0.02, "tsp": 0.05, "cvrp": 0.05, "maxcut": 0.02}


def performance_slack():
    return PERFORMANCE_SLACK.get(os.environ.get("EOH_PROBLEM_FAMILY", ""))


def population_management(pop, size):
    pop = [individual for individual in pop if individual.get("objective") is not None]
    if not pop:
        return []
    if size > len(pop):
        size = len(pop)
    pop = _performance_filtered(pop, size)

    unique = []
    seen = set()
    for individual in pop:
        code = individual.get("code") or ""
        objective = float(individual["objective"])
        alignment = float(individual.get("alignment_score", 0.0))
        key = (round(objective, 8), round(alignment, 8), code)
        if key in seen:
            continue
        seen.add(key)
        unique.append(individual)

    if os.environ.get("EOH_TA_RUNTIME_OBJECTIVE", "0") == "1":
        return _three_objective_management(unique, size)

    front = _pareto_front(unique)
    ordered_front = sorted(front, key=lambda item: (float(item["objective"]), -float(item.get("alignment_score", 0.0))))
    if len(ordered_front) >= size:
        return ordered_front[:size]

    remaining = [item for item in unique if item not in front]
    enriched = []
    if remaining:
        fill_align_weight = float(os.environ.get("EOH_PARETO_FILL_ALIGN_WEIGHT", "0.35"))
        obj_sorted = sorted(remaining, key=lambda item: float(item["objective"]))
        align_sorted = sorted(remaining, key=lambda item: float(item.get("alignment_score", 0.0)), reverse=True)
        obj_rank = {id(item): idx for idx, item in enumerate(obj_sorted)}
        align_rank = {id(item): idx for idx, item in enumerate(align_sorted)}
        for item in remaining:
            score = obj_rank[id(item)] + fill_align_weight * align_rank[id(item)]
            enriched.append((score, float(item["objective"]), -float(item.get("alignment_score", 0.0)), item))
        enriched.sort(key=lambda row: (row[0], row[1], row[2]))

    result = list(ordered_front)
    for _score, _obj, _align, item in enriched:
        if len(result) >= size:
            break
        result.append(item)
    return result[:size]


def _performance_filtered(pop, size):
    slack = performance_slack()
    if slack is None:
        return pop
    best = min(float(item["objective"]) for item in pop)
    cutoff = best + abs(best) * slack
    eligible = [item for item in pop if float(item["objective"]) <= cutoff]
    if len(eligible) >= size:
        return eligible
    ranked = sorted(pop, key=lambda item: float(item["objective"]))
    for item in ranked:
        if item not in eligible:
            eligible.append(item)
        if len(eligible) >= size:
            break
    return eligible


def _pareto_front(pop):
    front = []
    for candidate in pop:
        dominated = False
        for other in pop:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _dominates(lhs, rhs):
    lhs_obj = float(lhs["objective"])
    rhs_obj = float(rhs["objective"])
    lhs_align = float(lhs.get("alignment_score", 0.0))
    rhs_align = float(rhs.get("alignment_score", 0.0))
    better_or_equal = lhs_obj <= rhs_obj and lhs_align >= rhs_align
    strictly_better = lhs_obj < rhs_obj or lhs_align > rhs_align
    return better_or_equal and strictly_better


def _three_objective_management(pop, size):
    remaining = list(pop)
    result = []
    while remaining and len(result) < size:
        front = _pareto_front_runtime(remaining)
        slots = size - len(result)
        if len(front) <= slots:
            result.extend(
                sorted(
                    front,
                    key=lambda item: (
                        float(item["objective"]),
                        -float(item.get("alignment_score", 0.0)),
                        _runtime(item),
                    ),
                )
            )
        else:
            crowding = _crowding_distance(front)
            front = sorted(
                front,
                key=lambda item: (
                    -crowding[id(item)],
                    float(item["objective"]),
                    -float(item.get("alignment_score", 0.0)),
                    _runtime(item),
                ),
            )
            result.extend(front[:slots])
        remaining = [item for item in remaining if item not in front]
    return result[:size]


def _pareto_front_runtime(pop):
    return [
        candidate
        for candidate in pop
        if not any(
            other is not candidate and _dominates_runtime(other, candidate)
            for other in pop
        )
    ]


def _dominates_runtime(lhs, rhs):
    lhs_values = (
        float(lhs["objective"]),
        -float(lhs.get("alignment_score", 0.0)),
        _runtime(lhs),
    )
    rhs_values = (
        float(rhs["objective"]),
        -float(rhs.get("alignment_score", 0.0)),
        _runtime(rhs),
    )
    return all(left <= right for left, right in zip(lhs_values, rhs_values)) and any(
        left < right for left, right in zip(lhs_values, rhs_values)
    )


def _runtime(item):
    value = item.get("runtime_per_decision_us")
    return float("inf") if value is None else float(value)


def _crowding_distance(front):
    distance = {id(item): 0.0 for item in front}
    if len(front) <= 2:
        return {id(item): float("inf") for item in front}
    objectives = (
        lambda item: float(item["objective"]),
        lambda item: -float(item.get("alignment_score", 0.0)),
        _runtime,
    )
    for objective in objectives:
        ordered = sorted(front, key=objective)
        distance[id(ordered[0])] = float("inf")
        distance[id(ordered[-1])] = float("inf")
        low = objective(ordered[0])
        high = objective(ordered[-1])
        if not (high > low):
            continue
        for idx in range(1, len(ordered) - 1):
            if distance[id(ordered[idx])] == float("inf"):
                continue
            distance[id(ordered[idx])] += (
                objective(ordered[idx + 1]) - objective(ordered[idx - 1])
            ) / (high - low)
    return distance
