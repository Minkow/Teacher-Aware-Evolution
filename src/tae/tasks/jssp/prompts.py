class GetPrompts:
    def __init__(self) -> None:
        self.prompt_task = (
            "Given one feasible operation from a partial Job Shop Scheduling Problem (JSSP) schedule, assign "
            "a priority score to that operation. Each job is an ordered chain of operations, each operation has "
            "a processing time and a required machine, and the scheduler dispatches the feasible operation with "
            "the highest score at its earliest feasible start. The objective is to minimize final makespan. "
            "Help me design a novel dispatching priority heuristic."
        )
        self.prompt_func_name = "score"
        self.prompt_func_inputs = ["feature", "state"]
        self.prompt_func_outputs = ["priority_score"]
        self.prompt_inout_inf = (
            "'feature' is a CandidateFeature object for one feasible operation. It has attributes: "
            "op_id, job_id, op_index, machine_id, processing_time, "
            "remaining_work, remaining_ops, job_ready_time, machine_ready_time, earliest_start, "
            "earliest_finish, machine_queue_len, machine_total_work, job_progress, lower_bound_after. "
            "'state' has attributes actions, job_ready, next_op, makespan, and instance. "
            "'state.instance' has durations, machines, n_jobs, n_machines, and n_operations. These declared "
            "state and instance fields may be indexed or aggregated to derive scalar priority statistics."
        )
        self.prompt_other_inf = (
            "Use standard Python built-ins or math operations; do not import modules. Return a numeric score. The scheduler will evaluate "
            "this function for every feasible operation and choose the op_id with the highest score, breaking "
            "ties by smaller op_id. Return only a scalar score; do not return an op_id."
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

