import numpy as np
import json
import random
import time
import os

from .eoh_interface_EC import InterfaceEC
from tae.analysis.population import operator_guidance
from tae.analysis.memory import build_transition_records
from ..selection import prob_rank, pareto_rank
from ..management import pop_greedy, pop_pareto

# main class for eoh
class EOH:

    @staticmethod
    def _fmt_metric(ind):
        obj = ind.get('objective')
        align = float(ind.get('alignment_score', 0.0) or 0.0)
        runtime = ind.get('runtime_per_decision_us')
        obj_text = "None" if obj is None else f"{float(obj):.5f}"
        runtime_text = "" if runtime is None else f", runtime_us={float(runtime):.3f}"
        return f"obj={obj_text}, align={align:.3f}{runtime_text}"

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator
        self.teacher_aware = os.environ.get("EOH_TEACHER_AWARE", "0") == "1"
        self.teacher_start_gen = max(0, int(os.environ.get("EOH_TA_START_GEN", "0")))
        self.teacher_multi_objective = os.environ.get("EOH_TA_MULTI_OBJECTIVE", "1") == "1"
        self.teacher_disable_guidance = os.environ.get("EOH_TA_DISABLE_GUIDANCE", "0") == "1"
        self.base_select = prob_rank
        self.base_manage = pop_greedy
        self.ta_select = pareto_rank
        self.ta_manage = pop_pareto
        self.structural_memory_enabled = (
            os.environ.get("EOH_STRUCTURAL_MEMORY", "0") == "1"
        )
        self.structural_memory_limit = max(
            1,
            int(os.environ.get("EOH_STRUCTURAL_MEMORY_LIMIT", "100")),
        )
        self.structural_transitions = []
        self.structural_archive_path = os.environ.get(
            "EOH_STRUCTURAL_ARCHIVE_PATH",
            os.path.join(self.output_path, "structural_transition_archive.json"),
        ).strip()
        if self.structural_memory_enabled and os.path.exists(self.structural_archive_path):
            try:
                with open(self.structural_archive_path, encoding="utf-8") as file:
                    restored = json.load(file)
                if isinstance(restored, list):
                    self.structural_transitions = restored[-self.structural_memory_limit :]
            except (OSError, json.JSONDecodeError):
                self.structural_transitions = []

        print("- EoH parameters loaded -")
        # Set a reproducible global seed for all Python / NumPy sampling used by EOH.
        self.global_seed = int(os.environ.get("EOH_GLOBAL_SEED", "2024"))
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)

    # add new individual to population
    def add2pop(self, population, offspring):
        existing_codes = {
            str(ind.get('code') or '')
            for ind in population
            if isinstance(ind, dict) and ind.get('code')
        }
        for off in offspring:
            if not isinstance(off, dict) or off.get('objective') is None:
                continue
            code = str(off.get('code') or '')
            if code and code in existing_codes:
                if self.debug_mode:
                    print("duplicated code, skipping population insertion ... ")
                continue
            population.append(off)
            if code:
                existing_codes.add(code)
    

    def _maintain_generation_runtime(self, population, candidates):
        multiplier_text = os.environ.get(
            "EOH_RUNTIME_FINAL_POP_MULTIPLIER", ""
        ).strip()
        if not multiplier_text or not population:
            return population
        try:
            multiplier = float(multiplier_text)
        except ValueError:
            return population
        if multiplier <= 0:
            return population

        measured = [
            float(item["runtime_per_decision_us"])
            for item in population
            if item.get("runtime_per_decision_us") is not None
            and float(item["runtime_per_decision_us"]) > 0
        ]
        if not measured:
            return population

        cutoff = min(measured) * multiplier
        retained = [
            item
            for item in population
            if item.get("runtime_per_decision_us") is not None
            and 0 < float(item["runtime_per_decision_us"]) <= cutoff
        ]
        removed = [item for item in population if item not in retained]
        if not removed:
            print(
                "runtime_generation_maintenance "
                f"multiplier={multiplier:g} cutoff_us={cutoff:.3f} "
                f"removed=0 filled=0 final={len(population)}"
            )
            return population

        retained_codes = {
            str(item.get("code") or "") for item in retained if item.get("code")
        }
        replacements = []
        seen_codes = set(retained_codes)
        for item in candidates:
            if not isinstance(item, dict) or item.get("objective") is None:
                continue
            runtime = item.get("runtime_per_decision_us")
            if runtime is None or not (0 < float(runtime) <= cutoff):
                continue
            code = str(item.get("code") or "")
            if code and code in seen_codes:
                continue
            if code:
                seen_codes.add(code)
            replacements.append(item)
        replacements.sort(
            key=lambda item: (
                float(item["runtime_per_decision_us"]),
                float(item["objective"]),
                -float(item.get("alignment_score", 0.0)),
            )
        )

        target_size = len(population)
        result = list(retained)
        result.extend(replacements[: max(0, target_size - len(result))])
        print(
            "runtime_generation_maintenance "
            f"multiplier={multiplier:g} cutoff_us={cutoff:.3f} "
            f"removed={len(removed)} filled={len(result) - len(retained)} "
            f"final={len(result)}"
        )
        return result

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        # initialization
        population = []
        if self.load_pop:  # load population from files
            print("load initial population from " + self.load_pop_path)
            with open(self.load_pop_path) as file:
                data = json.load(file)
            for individual in data:
                population.append(individual)
            print("initial population has been loaded!")
            n_start = self.load_pop_id
        else:  # create new population, then optionally merge seed programs
            print("creating initial population:")
            seed_first = self.use_seed and os.environ.get("EOH_INIT_SEED_FIRST", "0") == "1"
            skip_i1 = os.environ.get("EOH_SKIP_INITIAL_I1", "0") == "1"
            seed_population = []
            if seed_first:
                with open(self.seed_path) as file:
                    data = json.load(file)
                print(f"loading seed programs before i1: {self.seed_path}")
                seed_population = interface_ec.population_generation_seed(data, self.exp_n_proc)
                population = list(seed_population)
                print(f"seed-first initial pool seeds={len(seed_population)}")
            else:
                population = []
            if not skip_i1:
                generated_population = interface_ec.population_generation()
                self.add2pop(population, generated_population)
            else:
                print("skip i1 initial generation; using seed programs only")
            if self.use_seed:
                if not seed_first:
                    with open(self.seed_path) as file:
                        data = json.load(file)
                    print(f"loading seed programs into initial pool: {self.seed_path}")
                    seed_population = interface_ec.population_generation_seed(data, self.exp_n_proc)
                    self.add2pop(population, seed_population)
                print(f"initial pool total={len(population)} seeds={len(seed_population)}")
            if len(population) == 0:
                raise RuntimeError("initial population generation failed: no valid offspring or seed programs")
            population = self.manage.population_management(population, self.pop_size)

            # print(len(population))
            # if len(population)<self.pop_size:
            #     for op in [self.operators[0],self.operators[2]]:
            #         _,new_ind = interface_ec.get_algorithm(population, op)
            #         self.add2pop(population, new_ind)
            #         population = self.manage.population_management(population, self.pop_size)
            #         if len(population) >= self.pop_size:
            #             break
            #         print(len(population))
 
            
            print(f"Pop initial: ")
            for off in population:
                print(f" {self._fmt_metric(off)}", end=" |")
            print()
            print("initial population has been created!")
            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):  
            current_gen = pop + 1
            interface_ec.set_generation_context(current_gen)
            teacher_active = (not self.teacher_aware) or current_gen >= self.teacher_start_gen
            use_ta_objective = self.teacher_aware and teacher_active and self.teacher_multi_objective
            active_select = self.ta_select if use_ta_objective else self.base_select
            active_manage = self.ta_manage if use_ta_objective else self.base_manage
            active_operators = self.operators if teacher_active else [op for op in self.operators if op not in {"m", "e3", "m3", "s", "p"}]
            if "e2" in active_operators and "e3" in active_operators:
                active_operators = [op for op in active_operators if op != "e3"]
            interface_ec.select = active_select
            generation_candidates = list(population)
            if teacher_active and not self.teacher_disable_guidance:
                interface_ec.evol.set_teacher_context(
                    operator_guidance(
                        population,
                        structural_transitions=self.structural_transitions,
                    )
                )
            else:
                interface_ec.evol.set_teacher_context({})
            mode = (
                "teacher-aware-multiobj"
                if use_ta_objective
                else "teacher-aware-singleobj"
                if (self.teacher_aware and teacher_active)
                else "plain-eoh"
            )
            print(f" generation_mode={mode} start_gen={self.teacher_start_gen} current_gen={current_gen}")
            if self.teacher_disable_guidance:
                print(" teacher_guidance=disabled")
            #print(f" [{na + 1} / {self.pop_size}] ", end="|")         
            active_weights = []
            for op in active_operators:
                idx = self.operators.index(op)
                active_weights.append(self.operator_weights[idx])
            n_active_op = len(active_operators)
            for i in range(n_active_op):
                op = active_operators[i]
                print(f" OP: {op} [{i + 1}/{n_active_op}] ", end="|")
                op_w = active_weights[i]
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                else:
                    parents, offsprings = [], []
                self._record_structural_transitions(
                    generation=current_gen,
                    operator=op,
                    parent_batches=parents,
                    offspring=offsprings,
                )
                self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                self.add2pop(generation_candidates, offsprings)
                for off in offsprings:
                    print(f" {self._fmt_metric(off)}", end=" |")
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                # populatin management
                size_act = min(len(population), self.pop_size)
                if size_act == 0:
                    print("population became empty after operator step, skip management")
                    continue
                population = active_manage.population_management(population, size_act)
                print()


            population = self._maintain_generation_runtime(
                population,
                generation_candidates,
            )

            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            if len(population) == 0:
                raise RuntimeError(f"population empty after generation {pop + 1}")
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)


            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Population: ", end="")
            for i in range(len(population)):
                print(f"[{self._fmt_metric(population[i])}] ", end="")
            print()

        interface_ec.flush_fe_artifacts()
        strict_fe = os.environ.get("EOH_STRICT_FE", "0") == "1"
        max_fe = max(0, int(os.environ.get("EOH_MAX_FE", "0")))
        if strict_fe and interface_ec.fe_count != max_fe:
            raise RuntimeError(
                f"Strict FE audit failed: fe_count={interface_ec.fe_count}, max_fe={max_fe}"
            )
        final_population_path = os.path.join(
            self.output_path,
            "final_retained_children.json",
        )
        with open(final_population_path, "w", encoding="utf-8") as file:
            json.dump(population, file, indent=2)
        self._flush_structural_transitions()

    def _record_structural_transitions(
        self,
        *,
        generation,
        operator,
        parent_batches,
        offspring,
    ):
        if not self.structural_memory_enabled:
            return
        records = build_transition_records(
            generation=generation,
            operator=operator,
            parent_batches=parent_batches,
            offspring=offspring,
        )
        if not records:
            return
        self.structural_transitions.extend(records)
        self.structural_transitions = self.structural_transitions[
            -self.structural_memory_limit :
        ]
        self._flush_structural_transitions()

    def _flush_structural_transitions(self):
        if not self.structural_memory_enabled or not self.structural_archive_path:
            return
        directory = os.path.dirname(self.structural_archive_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.structural_archive_path, "w", encoding="utf-8") as file:
            json.dump(self.structural_transitions, file, indent=2, allow_nan=True)
