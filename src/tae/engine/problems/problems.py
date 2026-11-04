# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from tae.tasks.tsp import problem as run
            self.prob = run.TSPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "cvrp_construct":
            from tae.tasks.cvrp import problem as run
            self.prob = run.CVRPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "maxcut_flip":
            from tae.tasks.maxcut import problem as run
            self.prob = run.MAXCUTFLIP()
            print("- Prob "+paras.problem+" loaded ")
        else:
            raise ValueError("Unknown problem: " + paras.problem)


    def get_problem(self):

        return self.prob
