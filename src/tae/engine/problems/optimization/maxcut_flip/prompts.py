import os


class GetPrompts():
    def __init__(self):
        interface = os.environ.get("EOH_MAXCUT_INTERFACE", "action").strip().lower()
        if interface == "score":
            self.prompt_task = (
                "Given one candidate node from a Max-Cut local-search state, assign a priority score to flipping "
                "that node. The evaluator computes this score for every node and flips the node with the highest "
                "score. The objective is to maximize the best cut value reached during the rollout."
            )
            self.prompt_func_name = "score"
            self.prompt_func_inputs = ["feature", "state"]
            self.prompt_func_outputs = ["priority_score"]
            self.prompt_inout_inf = (
                "'feature' describes one candidate flip and has attributes: node_id, spin, flip_gain, "
                "normalized_gain, same_weight, opposite_weight, weighted_degree, degree, time_since_flip, "
                "recent_flip, side0_fraction, side1_fraction, step_fraction, aspiration_gain, balance_after, "
                "same_weight_share, opposite_weight_share, boundary_mix, neighbor_gain_mean, neighbor_gain_max, "
                "positive_neighbor_gain_count, positive_neighbor_gain_sum, neighbor_weighted_degree_mean, "
                "neighbor_weighted_degree_max, two_hop_degree_proxy, after_positive_gain_count, "
                "after_positive_gain_count_delta, after_max_gain, after_top5_gain_mean, after_positive_gain_sum, "
                "current_cut, and best_cut. 'flip_gain' is the immediate cut-value change from flipping this node. "
                "'aspiration_gain' is max(0, current_cut + flip_gain - best_cut). "
                "'state' has attributes n_nodes, step, max_steps, current_cut, best_cut, spins, adjacency, "
                "flip_history, candidate_features, and features_by_action."
            )
            self.prompt_other_inf = (
                "Use standard Python or NumPy operations. Return only one numeric priority score for this feature. "
                "Do not choose or return a node id. Do not use randomness, file I/O, global state, or external solvers."
            )
        else:
            self.prompt_task = (
                "Given an undirected weighted graph and a current 0/1 partition for Max-Cut, "
                "design a deterministic and interpretable decision function that selects the next node to flip. "
                "This function is called at each local-search step and should use the current state to choose the node_id "
                "that is most promising for improving the best cut value reached during the rollout."
            )
            self.prompt_func_name = "select_flip_node"
            self.prompt_func_inputs = [
                "spins",
                "adjacency",
                "current_cut",
                "best_cut",
                "step",
                "max_steps",
                "flip_history",
            ]
            self.prompt_func_outputs = ["node_id"]
            self.prompt_inout_inf = (
                "'spins' is a list/array of 0/1 node assignments. "
                "'adjacency[i][j]' is the symmetric edge weight between nodes i and j. "
                "'current_cut' is the current cut value and 'best_cut' is the best cut observed in this rollout. "
                "'step' and 'max_steps' describe local-search progress. "
                "'flip_history' is the sequence of previously flipped node ids. "
                "Flipping node i changes the cut by sum of weights from i to same-side nodes minus sum of weights from i to opposite-side nodes. "
                "This sign is important: immediate_gain = same_side_weight - opposite_side_weight, not the reverse."
            )
            self.prompt_other_inf = (
                "Use standard Python or NumPy operations. "
                "Return an integer node_id in [0, len(spins)-1]. "
                "Do not use randomness, file I/O, global state, or external solvers."
            )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
