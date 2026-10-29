
from .selection import prob_rank,equal,roulette_wheel,tournament,pareto_rank
from .management import pop_greedy, pop_pareto

class Methods():
    def __init__(self,paras,problem) -> None:
        self.paras = paras      
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        elif paras.selection == 'pareto_rank':
            self.select = pareto_rank
        else:
            print("selection method "+paras.selection+" has not been implemented !")
            exit()

        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == "pop_pareto":
            self.manage = pop_pareto
        else:
            print("management method "+paras.management+" has not been implemented !")
            exit()

        
    def get_method(self):

        if self.paras.method == "eoh":
            from .eoh.eoh import EOH
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            raise ValueError("This release supports the EoH-based TAE engine only")
