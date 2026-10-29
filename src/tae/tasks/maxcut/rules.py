from __future__ import annotations

from tae.tasks.maxcut.program import ProgramActionScorer


GREEDY_GAIN = ProgramActionScorer(
    "greedy_gain",
    """
def select_action(state):
    best_action = state.feasible_actions[0]
    best_score = -10**18
    for feature in state.candidate_features:
        score = feature.flip_gain
        if score > best_score:
            best_score = score
            best_action = feature.action
    return best_action
""",
)


NORMALIZED_GAIN = ProgramActionScorer(
    "normalized_gain",
    """
def select_action(state):
    best_action = state.feasible_actions[0]
    best_score = -10**18
    for feature in state.candidate_features:
        score = feature.normalized_gain + 0.03 * feature.flip_gain
        if score > best_score:
            best_score = score
            best_action = feature.action
    return best_action
""",
)


TABU_GAIN = ProgramActionScorer(
    "tabu_gain",
    """
def select_action(state):
    best_action = state.feasible_actions[0]
    best_score = -10**18
    for feature in state.candidate_features:
        tabu_penalty = 0.0
        if feature.time_since_flip <= 2:
            tabu_penalty = 2.0
        score = feature.flip_gain + 0.05 * feature.weighted_degree - tabu_penalty
        if score > best_score:
            best_score = score
            best_action = feature.action
    return best_action
""",
)


BALANCE_AWARE = ProgramActionScorer(
    "balance_aware",
    """
def select_action(state):
    best_action = state.feasible_actions[0]
    best_score = -10**18
    for feature in state.candidate_features:
        if feature.spin == 0:
            balance_after = abs((feature.side1_fraction + 1.0 / state.n_nodes) - 0.5)
        else:
            balance_after = abs((feature.side1_fraction - 1.0 / state.n_nodes) - 0.5)
        score = feature.flip_gain + 0.02 * feature.weighted_degree - 0.5 * balance_after
        if score > best_score:
            best_score = score
            best_action = feature.action
    return best_action
""",
)


DEFAULT_RULES = (GREEDY_GAIN, NORMALIZED_GAIN, TABU_GAIN, BALANCE_AWARE)
