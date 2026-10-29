class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "Given a depot, customer coordinates, customer demands, and a vehicle capacity, "
            "you need to build short CVRP routes that serve every customer while never exceeding capacity. "
            "The task is solved step-by-step by choosing either a feasible next customer or the depot to restart a route. "
            "Help me design a novel route-construction algorithm that is different from standard nearest-neighbor or savings rules."
        )
        self.prompt_func_name = "select_next_node"
        self.prompt_func_inputs = ["current_node", "depot", "unvisited_nodes", "rest_capacity", "demands", "distance_matrix"]
        self.prompt_func_outputs = ["next_node"]
        self.prompt_inout_inf = (
            "'current_node', 'depot', and values inside 'unvisited_nodes' are node IDs. "
            "'rest_capacity' is the remaining capacity on the current route. "
            "'demands[node]' is the demand of customer node. "
            "'distance_matrix[i][j]' is the Euclidean distance between nodes i and j."
        )
        self.prompt_other_inf = (
            "Use standard Python or NumPy operations. "
            "Return either a feasible customer ID from unvisited_nodes or the depot ID 0 when the route should restart. "
            "Do not return a customer whose demand exceeds rest_capacity."
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

