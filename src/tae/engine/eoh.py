
import os
import random
import numpy as np

from .utils import createFolders
from .methods import methods
from .problems import problems

# main class for AEL
class EVOL:

    # initilization
    def __init__(self, paras, prob=None, **kwargs):

        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # Create folder #
        createFolders.create_folders(paras.exp_output_path)
        print("- output folder created -")

        self.paras = paras

        print("-  parameters loaded -")

        self.prob = prob

        # Set a reproducible global seed for all Python / NumPy sampling used by EOH.
        global_seed = int(os.environ.get("EOH_GLOBAL_SEED", "2024"))
        random.seed(global_seed)
        np.random.seed(global_seed)

        
    # run methods
    def run(self):

        problemGenerator = problems.Probs(self.paras)

        problem = problemGenerator.get_problem()

        try:
            methodGenerator = methods.Methods(self.paras,problem)
            method = methodGenerator.get_method()
            method.run()
        finally:
            pool = getattr(problem, "rollout_pool", None)
            if pool is not None:
                pool.shutdown()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")
