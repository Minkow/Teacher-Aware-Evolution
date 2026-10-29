import numpy as np
import time
import os
import json
import threading
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import hashlib
from tae.engine.evaluation_pool import EvaluationBudget


def mark_p_sibling_population_eligibility(results):
    """Keep one valid P child per immediate parent without changing FE accounting."""
    groups = {}
    for result_index, result in enumerate(results):
        if not result:
            continue
        _, offspring = result
        if not isinstance(offspring, dict) or offspring.get("objective") is None:
            continue
        parent_refs = offspring.get("parent_refs") or []
        parent_hash = None
        if parent_refs and isinstance(parent_refs[0], dict):
            parent_hash = parent_refs[0].get("code_sha256")
        if not parent_hash:
            parent_hash = f"missing-parent:{result_index}:{offspring.get('fe_index')}"
        groups.setdefault(str(parent_hash), []).append(offspring)

    for parent_hash, siblings in groups.items():
        winner = min(
            siblings,
            key=lambda item: (
                float(item["objective"]),
                -float(item.get("alignment_score") or 0.0),
                float(item.get("runtime_per_decision_us") or float("inf")),
                int(item.get("fe_index") or 0),
            ),
        )
        winner_fe = winner.get("fe_index")
        for sibling in siblings:
            sibling["p_population_eligible"] = sibling is winner
            sibling["p_sibling_parent_hash"] = parent_hash
            sibling["p_sibling_group_size"] = len(siblings)
            sibling["p_sibling_kept_fe_index"] = winner_fe
    return results


def mark_m_pair_population_eligibility(results):
    """Keep one valid fusion child per exact, order-independent parent pair."""
    groups = {}
    for result_index, result in enumerate(results):
        if not result:
            continue
        _, offspring = result
        if not isinstance(offspring, dict) or offspring.get("objective") is None:
            continue
        parent_refs = offspring.get("parent_refs") or []
        parent_hashes = sorted(
            str(ref.get("code_sha256"))
            for ref in parent_refs
            if isinstance(ref, dict) and ref.get("code_sha256")
        )
        pair_key = "|".join(parent_hashes) if len(parent_hashes) == 2 else None
        if not pair_key:
            pair_key = f"missing-pair:{result_index}:{offspring.get('fe_index')}"
        groups.setdefault(pair_key, []).append(offspring)

    for pair_key, siblings in groups.items():
        winner = min(
            siblings,
            key=lambda item: (
                float(item["objective"]),
                -float(item.get("alignment_score") or 0.0),
                float(item.get("runtime_per_decision_us") or float("inf")),
                int(item.get("fe_index") or 0),
            ),
        )
        winner_fe = winner.get("fe_index")
        for sibling in siblings:
            sibling["m_pair_population_eligible"] = sibling is winner
            sibling["m_sibling_pair_key"] = pair_key
            sibling["m_sibling_pair_group_size"] = len(siblings)
            sibling["m_sibling_pair_kept_fe_index"] = winner_fe
    return results

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.child_count = max(1, int(os.environ.get("EOH_CHILDREN", str(pop_size))))
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.fe_timeout = max(
            1,
            int(os.environ.get("EOH_FE_TIMEOUT", str(self.timeout))),
        )
        self.operator_batch_timeout = max(
            self.timeout,
            int(os.environ.get("EOH_OPERATOR_BATCH_TIMEOUT", str(self.timeout))),
        )
        self.use_numba = use_numba
        backend_override = os.environ.get("EOH_PARALLEL_BACKEND", "").strip().lower()
        if backend_override not in {"", "threading"}:
            raise ValueError("Offspring scheduling requires threading for shared FE accounting; "
                             "use rollout workers for process-based instance evaluation")
        self.parallel_backend = "threading"
        self.stagger_s = max(0.0, float(os.environ.get("EOH_STAGGER_S", "0")))
        self.strict_fe = os.environ.get("EOH_STRICT_FE", "0") == "1"
        self.max_fe = max(0, int(os.environ.get("EOH_MAX_FE", "0")))
        self.fe_count = 0
        self.fe_lock = threading.Lock()
        self.fe_archive = []
        self.fe_archive_lock = threading.Lock()
        self.fe_archive_path = os.environ.get("EOH_FE_ARCHIVE_PATH", "").strip()
        self.fe_manifest_path = os.environ.get("EOH_FE_MANIFEST_PATH", "").strip()
        self.generated_code_hashes = set()
        self.generated_code_lock = threading.Lock()
        self.duplicate_generation_retries = max(
            0,
            int(os.environ.get("EOH_DUPLICATE_GENERATION_RETRIES", "2")),
        )
        self.current_generation = 0
        self.operator_child_schedules = {
            "s": self._parse_child_schedule(
                "EOH_S_CHILD_SCHEDULE", "8,8,5,5,4"
            ),
            "p": self._parse_child_schedule(
                "EOH_P_CHILD_SCHEDULE", "8,8,5,5,4"
            ),
            "m": self._parse_child_schedule(
                "EOH_M_CHILD_SCHEDULE", "8,8,5,5,4"
            ),
        }

    @staticmethod
    def _parse_child_schedule(env_name, default):
        values = tuple(
            int(value.strip())
            for value in os.environ.get(env_name, default).split(",")
            if value.strip()
        )
        if not values or any(value < 0 for value in values):
            raise ValueError(f"{env_name} must contain non-negative integers")
        return values

    def set_generation_context(self, generation):
        self.current_generation = max(1, int(generation))

    def _set_operator_child_count(self, operator):
        schedule = self.operator_child_schedules.get(operator)
        if schedule is None or self.current_generation <= 0:
            return
        schedule_idx = min(self.current_generation - 1, len(schedule) - 1)
        self.child_count = schedule[schedule_idx]
        print(
            f"operator_budget generation={self.current_generation} "
            f"operator={operator} child_count={self.child_count} schedule={schedule}"
        )

    def _reserve_fe(self, phase):
        if self.max_fe <= 0:
            return -1
        with self.fe_lock:
            if self.fe_count >= self.max_fe:
                return None
            self.fe_count += 1
            return self.fe_count

    def _archive_attempt(self, individual):
        if not isinstance(individual, dict):
            return
        with self.fe_archive_lock:
            self.fe_archive.append(dict(individual))

    def flush_fe_artifacts(self):
        if self.fe_archive_path:
            os.makedirs(os.path.dirname(self.fe_archive_path), exist_ok=True)
            with self.fe_archive_lock:
                archive = sorted(
                    self.fe_archive,
                    key=lambda item: int(item.get("fe_index", 0)),
                )
            with open(self.fe_archive_path, "w", encoding="utf-8") as file:
                json.dump(
                    archive,
                    file,
                    indent=2,
                    allow_nan=True,
                    default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
                )
        if self.fe_manifest_path:
            os.makedirs(os.path.dirname(self.fe_manifest_path), exist_ok=True)
            phase_counts = {}
            with self.fe_archive_lock:
                for individual in self.fe_archive:
                    phase = str(individual.get("fe_phase", "unknown"))
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
            with open(self.fe_manifest_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "max_fe": self.max_fe,
                        "fe_count": self.fe_count,
                        "complete": self.max_fe > 0 and self.fe_count == self.max_fe,
                        "strict_fe": self.strict_fe,
                        "phase_counts": phase_counts,
                    },
                    file,
                    indent=2,
                )
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def _claim_generated_code(self, code):
        """Atomically reserve generated code before objective evaluation."""
        code_text = str(code or "").strip()
        if not code_text:
            return False
        digest = hashlib.sha256(code_text.encode("utf-8")).hexdigest()
        with self.generated_code_lock:
            if digest in self.generated_code_hashes:
                return False
            self.generated_code_hashes.add(digest)
        return True

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        init_children = os.environ.get("EOH_INITIAL_LLM_CHILDREN", "").strip()
        if init_children:
            target = max(0, int(init_children))
            if target == 0:
                return []
            previous_child_count = self.child_count
            try:
                self.child_count = target
                _, population = self.get_algorithm([], "i1")
            finally:
                self.child_count = previous_child_count
            return [
                individual
                for individual in population
                if individual.get("objective") is not None
            ]

        n_create = 2
        population = []
        for _ in range(n_create):
            _, pop = self.get_algorithm([], "i1")
            for individual in pop:
                if individual.get("objective") is not None:
                    population.append(individual)
        return population
    
    def _seed_to_individual(self, seed, fitness):
        seed_alg = {
            'algorithm': seed['algorithm'],
            'code': seed['code'],
            'objective': None,
            'other_inf': None,
            'alignment_score': None,
            'teacher_value_score': None,
            'top1_agreement': None,
            'teacher_percentile': None,
            'case_summary': None,
            'stage_summary': None,
            'structure_cases': None,
            'parameter_cases': None,
            'structure_guidance': None,
            'parameter_guidance': None,
            'feature_preference_summary': None,
            'trace_summary': None,
            'teacher_trace_summary': None,
            'trace_delta_summary': None,
            'trace_stage_offsets': None,
            'teacher_trace_stage_offsets': None,
            'teacher_trace_semantics': None,
            'runtime_per_decision_us': None,
            'runtime_per_score_call_us': None,
        }
        if isinstance(fitness, dict):
            objective = fitness.get("objective")
            if objective is None or not np.isfinite(float(objective)):
                raise ValueError("seed objective must be finite")
            seed_alg['objective'] = np.round(objective, 5)
            seed_alg['alignment_score'] = float(fitness.get("alignment_score", 0.0))
            seed_alg['teacher_value_score'] = float(fitness.get("teacher_value_score", seed_alg['alignment_score']))
            seed_alg['top1_agreement'] = float(fitness.get("top1_agreement", 0.0))
            seed_alg['teacher_percentile'] = float(fitness.get("teacher_percentile", 0.0))
            seed_alg['case_summary'] = fitness.get("case_summary")
            seed_alg['stage_summary'] = fitness.get("stage_summary")
            seed_alg['structure_cases'] = fitness.get("structure_cases")
            seed_alg['parameter_cases'] = fitness.get("parameter_cases")
            seed_alg['structure_guidance'] = fitness.get("structure_guidance")
            seed_alg['parameter_guidance'] = fitness.get("parameter_guidance")
            seed_alg['feature_preference_summary'] = fitness.get("feature_preference_summary")
            seed_alg['trace_summary'] = fitness.get("trace_summary")
            seed_alg['teacher_trace_summary'] = fitness.get("teacher_trace_summary")
            seed_alg['trace_delta_summary'] = fitness.get("trace_delta_summary")
            seed_alg['trace_stage_offsets'] = fitness.get("trace_stage_offsets")
            seed_alg['teacher_trace_stage_offsets'] = fitness.get("teacher_trace_stage_offsets")
            seed_alg['runtime_per_decision_us'] = float(fitness.get("runtime_per_decision_us", 0.0))
            seed_alg['runtime_per_score_call_us'] = float(fitness.get("runtime_per_score_call_us", 0.0))
            seed_alg['runtime_decisions'] = int(fitness.get("runtime_decisions", 0))
            seed_alg['other_inf'] = fitness.get("diagnostic")
        else:
            if not np.isfinite(float(fitness)):
                raise ValueError("seed objective must be finite")
            seed_alg['objective'] = np.round(float(fitness), 5)
        return seed_alg

    def _evaluate_seed(self, code):
        budget = EvaluationBudget(time.monotonic() + min(self.timeout, self.fe_timeout))
        function = (self.interface_eval.evaluate_metadata
                    if hasattr(self.interface_eval, "evaluate_metadata") else self.interface_eval.evaluate)
        try:
            return budget.run(function, code)
        except Exception as exc:
            return {"objective": None, "evaluation_error": f"{type(exc).__name__}: {exc}"}

    def population_generation_seed(self,seeds,n_p):

        population = []
        seed_entries = []
        for seed in seeds:
            fe_index = self._reserve_fe("seed")
            if fe_index is None:
                break
            seed_entries.append((seed, fe_index))
        seeds = [entry[0] for entry in seed_entries]

        fitness = Parallel(n_jobs=n_p, backend=self.parallel_backend)(
            delayed(self._evaluate_seed)(seed['code']) for seed in seeds
        )

        for i in range(len(seeds)):
            fe_index = seed_entries[i][1]
            try:
                if fitness[i] is None:
                    raise ValueError("seed evaluation returned None")
                if isinstance(fitness[i], dict) and fitness[i].get("evaluation_error"):
                    raise ValueError(fitness[i]["evaluation_error"])
                seed_alg = self._seed_to_individual(seeds[i], fitness[i])
                if fe_index >= 0:
                    seed_alg["fe_index"] = fe_index
                    seed_alg["fe_phase"] = "seed"
                self._archive_attempt(seed_alg)
                population.append(seed_alg)

            except Exception as e:
                print(f"Error in seed algorithm {i}: {type(e).__name__}: {e}")
                failed = {
                    "algorithm": seeds[i].get("algorithm", ""),
                    "code": seeds[i].get("code"),
                    "objective": None,
                    "fe_index": fe_index,
                    "fe_phase": "seed",
                    "evaluation_error": f"{type(e).__name__}: {e}",
                }
                self._archive_attempt(failed)

        self.flush_fe_artifacts()
        print("Initiliazation finished! Get "+str(len(population))+"/"+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator,m_task=None,s_parent=None):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None,
            'alignment_score': None,
            'teacher_value_score': None,
            'top1_agreement': None,
            'teacher_percentile': None,
            'case_summary': None,
            'stage_summary': None,
            'structure_cases': None,
            'parameter_cases': None,
            'structure_guidance': None,
            'parameter_guidance': None,
            'feature_preference_summary': None,
            'trace_summary': None,
            'teacher_trace_summary': None,
            'trace_delta_summary': None,
            'trace_stage_offsets': None,
            'teacher_trace_stage_offsets': None,
            'runtime_per_decision_us': None,
            'runtime_per_score_call_us': None,
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()            
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            parents = self._select_fusion_parents(pop)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents)
        elif operator == "m":
            if not isinstance(m_task, dict):
                raise ValueError("m requires a prepared parent-pair task")
            parents = m_task["parents"]
            hint = m_task["hint"]
            [offspring['code'],offspring['algorithm']] = self.evol.m(parents, hint=hint)
            offspring["m_hint"] = hint
            offspring["m_pair_id"] = int(m_task["pair_id"])
            offspring["m_analyzer_calls"] = int(m_task["analyzer_calls"])
            offspring["m_analyzer_workers"] = int(m_task["analyzer_workers"])
            offspring["m_analysis_mode"] = str(
                m_task.get("analysis_mode", "inline_generator")
            )
            offspring["m_better_parent_index"] = int(
                m_task["better_parent_index"]
            )
            for metadata_key in (
                "m_subject_key",
                "m_subject_batch_count",
                "m_subject_batch_cap",
            ):
                if metadata_key in m_task:
                    offspring[metadata_key] = m_task[metadata_key]
        elif operator == "e3":
            parents = self._select_fusion_parents(pop)
            [offspring['code'],offspring['algorithm']] = self.evol.e3(parents)
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])   
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0]) 
        elif operator == "m3":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0]) 
        elif operator == "s":
            parents = [s_parent] if s_parent is not None else self._select_proposer_parent(pop)
            reference = self._best_performance_reference(pop, parents[0])
            [offspring['code'],offspring['algorithm']] = self.evol.s(parents[0], reference)
        elif operator == "p":
            parents = self._select_proposer_parent(pop)
            reference = self._best_performance_reference(pop, parents[0])
            [offspring['code'],offspring['algorithm']] = self.evol.p(parents[0], reference)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        structural_hypothesis = self.evol.consume_last_structural_hypothesis()
        if structural_hypothesis:
            offspring["structural_hypothesis"] = structural_hypothesis
        offspring["parent_refs"] = [self._individual_ref(parent) for parent in (parents or [])]
        if operator in {"s", "p"}:
            offspring["performance_reference"] = self._individual_ref(reference) if reference else None
        return parents, offspring

    def _select_proposer_parent(self, pop):
        if not pop:
            return []
        ranked = sorted(
            pop,
            key=lambda item: (float(item["objective"]), -float(item.get("alignment_score", 0.0))),
        )
        pool_size = max(1, int(os.environ.get("EOH_S_OBJ_POOL", "3")))
        return [self._sample_ranked_parent(ranked[:pool_size])]

    def _prepare_s_parent_batch(self, pop, child_count):
        allocation_text = os.environ.get("EOH_S_GEN1_PARENT_ALLOCATION", "").strip()
        if self.current_generation != 1 or not allocation_text:
            return None
        allocation = tuple(
            int(value.strip())
            for value in allocation_text.split(",")
            if value.strip()
        )
        if not allocation or any(value <= 0 for value in allocation):
            raise ValueError(
                "EOH_S_GEN1_PARENT_ALLOCATION must contain positive integers"
            )
        if sum(allocation) != child_count:
            raise ValueError(
                "generation-1 S allocation must match the S child count: "
                f"allocation={allocation}, child_count={child_count}"
            )

        ranked = sorted(
            pop,
            key=lambda item: (
                float(item["objective"]),
                -float(item.get("alignment_score", 0.0)),
            ),
        )
        distinct = []
        seen_codes = set()
        for candidate in ranked:
            code_digest = hashlib.sha256(
                str(candidate.get("code") or "").encode("utf-8")
            ).hexdigest()
            if code_digest in seen_codes:
                continue
            seen_codes.add(code_digest)
            distinct.append(candidate)
            if len(distinct) == len(allocation):
                break
        if len(distinct) != len(allocation):
            raise ValueError(
                "generation-1 S allocation requires enough code-distinct parents: "
                f"required={len(allocation)}, available={len(distinct)}"
            )

        remaining = list(allocation)
        parents = []
        while any(count > 0 for count in remaining):
            for rank, parent in enumerate(distinct):
                if remaining[rank] <= 0:
                    continue
                parents.append(parent)
                remaining[rank] -= 1
        print(
            "s_gen1_parent_allocation "
            f"counts={allocation} "
            f"objectives={tuple(float(parent['objective']) for parent in distinct)} "
            f"code_hashes={tuple(hashlib.sha256(str(parent.get('code') or '').encode('utf-8')).hexdigest()[:12] for parent in distinct)}"
        )
        return parents

    def _best_performance_reference(self, pop, parent):
        parent_code = str(parent.get("code") or "")
        for candidate in sorted(
            pop,
            key=lambda item: (float(item["objective"]), -float(item.get("alignment_score", 0.0))),
        ):
            if candidate is parent:
                continue
            if str(candidate.get("code") or "") == parent_code:
                continue
            if float(candidate["objective"]) < float(parent["objective"]):
                return candidate
        return None

    @staticmethod
    def _individual_ref(individual):
        if not individual:
            return None
        code = str(individual.get("code") or "")
        return {
            "objective": individual.get("objective"),
            "alignment_score": individual.get("alignment_score"),
            "top1_agreement": individual.get("top1_agreement"),
            "fe_index": individual.get("fe_index"),
            "fe_phase": individual.get("fe_phase"),
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None,
        }

    def _select_fusion_parents(self, pop):
        teacher_guided = bool(
            (self.evol.teacher_context or {}).get("e2")
            or (self.evol.teacher_context or {}).get("m")
            or (self.evol.teacher_context or {}).get("e3")
        )
        family = os.environ.get("EOH_PROBLEM_FAMILY", "").strip().lower()
        if family not in {"maxcut", "jssp", "tsp", "cvrp"} or not teacher_guided:
            return self.select.parent_selection(pop, self.m)
        if not pop:
            return []
        obj_pool_size = max(1, int(os.environ.get("EOH_E3_OBJ_POOL", "3")))
        align_pool_size = max(1, int(os.environ.get("EOH_E3_ALIGN_POOL", "3")))

        obj_ranked = sorted(pop, key=lambda item: (float(item["objective"]), -float(item.get("alignment_score", 0.0))))
        with_align = [item for item in pop if item.get("alignment_score") is not None]
        align_ranked = sorted(
            with_align if with_align else pop,
            key=lambda item: (-float(item.get("alignment_score", 0.0)), float(item["objective"])),
        )

        parents = [self._sample_ranked_parent(obj_ranked[:obj_pool_size])]
        align_pool = [item for item in align_ranked[:align_pool_size] if item is not parents[0]]
        if align_pool:
            parents.append(self._sample_ranked_parent(align_pool))
        elif len(obj_ranked) > 1:
            parents.append(obj_ranked[1])
        if len(parents) < self.m:
            for candidate in self.select.parent_selection(pop, self.m):
                if any(candidate is existing for existing in parents):
                    continue
                parents.append(candidate)
                if len(parents) >= self.m:
                    break
        return parents[: self.m]


    def _prepare_m_batch(self, pop, child_count):
        if child_count <= 0:
            return []
        if len(pop) < 2:
            raise ValueError("m requires at least two distinct parents")

        obj_ranked = sorted(
            pop,
            key=lambda item: (
                float(item["objective"]),
                -float(item.get("alignment_score", 0.0)),
            ),
        )
        with_align = [item for item in pop if item.get("alignment_score") is not None]
        align_ranked = sorted(
            with_align if with_align else pop,
            key=lambda item: (
                -float(item.get("alignment_score", 0.0)),
                float(item["objective"]),
            ),
        )
        obj_limit = min(
            len(obj_ranked),
            max(1, int(os.environ.get("EOH_E3_OBJ_POOL", "3"))),
        )
        align_limit = min(
            len(align_ranked),
            max(1, int(os.environ.get("EOH_E3_ALIGN_POOL", "3"))),
        )
        subject_cap = max(1, int(os.environ.get("EOH_M_SUBJECT_MAX_PER_BATCH", "2")))

        selected_rows = None
        while selected_rows is None:
            candidates = {}
            for obj_rank, obj_parent in enumerate(obj_ranked[:obj_limit], start=1):
                for align_rank, align_parent in enumerate(align_ranked[:align_limit], start=1):
                    if obj_parent is align_parent:
                        continue
                    parents = [obj_parent, align_parent]
                    pair_key = self.evol._m_pair_key(parents)
                    better_index = min(
                        range(len(parents)),
                        key=lambda idx: (
                            float(parents[idx]["objective"]),
                            -float(parents[idx].get("alignment_score", 0.0)),
                            hashlib.sha256(str(parents[idx].get("code") or "").encode("utf-8")).hexdigest(),
                        ),
                    )
                    subject = parents[better_index]
                    subject_key = hashlib.sha256(
                        str(subject.get("code") or "").encode("utf-8")
                    ).hexdigest()
                    weight = 1.0 / float(obj_rank * align_rank)
                    entry = candidates.get(pair_key)
                    if entry is None:
                        candidates[pair_key] = {
                            "parents": parents,
                            "weight": weight,
                            "subject_key": subject_key,
                            "better_index": better_index,
                        }
                    else:
                        entry["weight"] += weight

            rows = list(candidates.values())
            for _attempt in range(256):
                remaining = list(rows)
                chosen = []
                subject_counts = {}
                while remaining and len(chosen) < child_count:
                    eligible = [
                        row
                        for row in remaining
                        if subject_counts.get(row["subject_key"], 0) < subject_cap
                    ]
                    if not eligible:
                        break
                    weights = np.asarray([row["weight"] for row in eligible], dtype=np.float64)
                    weights /= weights.sum()
                    row = eligible[int(np.random.choice(len(eligible), p=weights))]
                    chosen.append(row)
                    subject_counts[row["subject_key"]] = subject_counts.get(row["subject_key"], 0) + 1
                    remaining.remove(row)
                if len(chosen) == child_count:
                    selected_rows = chosen
                    break

            if selected_rows is not None:
                break
            next_obj_limit = min(len(obj_ranked), obj_limit + 1)
            next_align_limit = min(len(align_ranked), align_limit + 1)
            if next_obj_limit == obj_limit and next_align_limit == align_limit:
                raise ValueError(
                    f"m cannot allocate {child_count} distinct pairs with "
                    f"subject cap {subject_cap} from {len(candidates)} candidates"
                )
            obj_limit = next_obj_limit
            align_limit = next_align_limit

        parent_batches = [row["parents"] for row in selected_rows]
        hints, pair_ids = self.evol.build_m_batch_hints(parent_batches)
        if len(hints) != child_count or len(pair_ids) != child_count:
            raise ValueError("m batch analyzer returned an inconsistent task count")
        stats = getattr(self.evol, "_last_m_analyzer_stats", {})
        analyzer_calls = int(stats.get("analyzer_calls", len(set(pair_ids))))
        analyzer_workers = int(stats.get("analyzer_workers", 1))
        subject_counts = {}
        for row in selected_rows:
            subject_counts[row["subject_key"]] = subject_counts.get(row["subject_key"], 0) + 1
        tasks = [
            {
                "parents": row["parents"],
                "hint": hint,
                "pair_id": pair_id,
                "analyzer_calls": analyzer_calls,
                "analyzer_workers": analyzer_workers,
                "analysis_mode": str(stats.get("analysis_mode", "inline_generator")),
                "better_parent_index": row["better_index"],
                "m_subject_key": row["subject_key"],
                "m_subject_batch_count": subject_counts[row["subject_key"]],
                "m_subject_batch_cap": subject_cap,
            }
            for row, hint, pair_id in zip(selected_rows, hints, pair_ids)
        ]
        print(
            f"m_pair_subject_allocation children={child_count} "
            f"unique_pairs={len(set(pair_ids))} unique_subjects={len(subject_counts)} "
            f"max_subject_count={max(subject_counts.values())} subject_cap={subject_cap} "
            f"analyzer_calls={analyzer_calls} analyzer_workers={analyzer_workers}"
        )
        return tasks

    def _sample_ranked_parent(self, ranked):
        if not ranked:
            return None
        if len(ranked) == 1:
            return ranked[0]
        weights = np.array([1.0 / (idx + 1) for idx in range(len(ranked))], dtype=np.float64)
        weights /= weights.sum()
        choice = int(np.random.choice(len(ranked), p=weights))
        return ranked[choice]

    def get_offspring(
        self,
        pop,
        operator,
        m_task=None,
        s_parent=None,
        deadline=None,
    ):
        max_retry = 1 if self.strict_fe else max(1, int(os.environ.get("EOH_OFFSPRING_ATTEMPTS", "1")))
        last_exception = None
        p = None
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }

        for _ in range(max_retry):
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("per-FE deadline exhausted before generation")
                p, offspring = self._get_alg(
                    pop,
                    operator,
                    m_task=m_task,
                    s_parent=s_parent,
                )

                duplicate_retry = 0
                while True:
                    if offspring['code'] is None:
                        raise ValueError("empty offspring code")
                    is_duplicate = self.check_duplicate(pop, offspring['code'])
                    if not is_duplicate:
                        is_duplicate = not self._claim_generated_code(offspring['code'])
                    if not is_duplicate:
                        break
                    if duplicate_retry >= self.duplicate_generation_retries:
                        raise ValueError(
                            "duplicate offspring code after "
                            f"{duplicate_retry} regeneration attempt(s)"
                        )
                    duplicate_retry += 1
                    if self.debug:
                        print("duplicated code, regenerating before evaluation ... ")
                    p, offspring = self._get_alg(
                        pop,
                        operator,
                        m_task=m_task,
                        s_parent=s_parent,
                    )

                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    if match is None:
                        raise ValueError("function definition not found")
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                remaining = float(self.timeout)
                if deadline is not None:
                    remaining = min(remaining, deadline - time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("per-FE deadline exhausted before evaluation")
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                evaluation_budget = EvaluationBudget(time.monotonic() + remaining)
                future = None
                try:
                    if hasattr(self.interface_eval, "evaluate_metadata"):
                        future = executor.submit(evaluation_budget.run, self.interface_eval.evaluate_metadata, code)
                    else:
                        future = executor.submit(evaluation_budget.run, self.interface_eval.evaluate, code)
                    fitness = future.result(timeout=remaining)
                except concurrent.futures.TimeoutError:
                    evaluation_budget.cancelled.set()
                    # Give rollout watchdogs time to kill/reap before returning.
                    concurrent.futures.wait([future], timeout=5)
                    raise
                finally:
                    if future is not None:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)

                if fitness is None:
                    raise ValueError("evaluation returned None")

                if isinstance(fitness, dict):
                    objective = fitness.get("objective")
                    if objective is None or not np.isfinite(float(objective)):
                        raise ValueError("evaluation objective must be finite")
                    offspring['objective'] = np.round(objective, 5)
                    offspring['alignment_score'] = float(fitness.get("alignment_score", 0.0))
                    offspring['teacher_value_score'] = float(fitness.get("teacher_value_score", offspring['alignment_score']))
                    offspring['top1_agreement'] = float(fitness.get("top1_agreement", 0.0))
                    offspring['teacher_percentile'] = float(fitness.get("teacher_percentile", 0.0))
                    offspring['case_summary'] = fitness.get("case_summary")
                    offspring['stage_summary'] = fitness.get("stage_summary")
                    offspring['structure_cases'] = fitness.get("structure_cases")
                    offspring['parameter_cases'] = fitness.get("parameter_cases")
                    offspring['structure_guidance'] = fitness.get("structure_guidance")
                    offspring['parameter_guidance'] = fitness.get("parameter_guidance")
                    offspring['feature_preference_summary'] = fitness.get("feature_preference_summary")
                    offspring['trace_summary'] = fitness.get("trace_summary")
                    offspring['teacher_trace_summary'] = fitness.get("teacher_trace_summary")
                    offspring['trace_delta_summary'] = fitness.get("trace_delta_summary")
                    offspring['trace_stage_offsets'] = fitness.get("trace_stage_offsets")
                    offspring['teacher_trace_stage_offsets'] = fitness.get("teacher_trace_stage_offsets")
                    offspring['teacher_trace_semantics'] = fitness.get("teacher_trace_semantics")
                    offspring['runtime_per_decision_us'] = float(fitness.get("runtime_per_decision_us", 0.0))
                    offspring['runtime_per_score_call_us'] = float(fitness.get("runtime_per_score_call_us", 0.0))
                    offspring['runtime_decisions'] = int(fitness.get("runtime_decisions", 0))
                    offspring['other_inf'] = fitness.get("diagnostic")
                else:
                    if not np.isfinite(float(fitness)):
                        raise ValueError("evaluation objective must be finite")
                    offspring['objective'] = np.round(fitness, 5)
                return p, offspring
            except Exception as e:
                last_exception = e
                offspring['objective'] = None
                if self.debug:
                    print(f"offspring retry failed: {e}")
                if isinstance(e, (TimeoutError, concurrent.futures.TimeoutError)):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break

        if last_exception is not None:
            offspring['evaluation_error'] = (
                f"{type(last_exception).__name__}: {last_exception}"
            )
            print(
                f"offspring failed after retries for operator={operator}: "
                f"{type(last_exception).__name__}: {last_exception}"
            )

        # Round the objective values
        return p, offspring
    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()
                
    #     return result

    def _get_offspring_with_stagger(
        self,
        pop,
        operator,
        task_idx,
        m_task=None,
        s_parent=None,
    ):
        fe_index = self._reserve_fe(operator)
        if fe_index is None:
            return None, None
        if self.stagger_s > 0:
            slot = task_idx % max(1, self.n_p)
            if slot > 0:
                time.sleep(slot * self.stagger_s)
        deadline = time.monotonic() + float(self.fe_timeout)
        self.evol.interface_llm.set_deadline(deadline)
        try:
            parents, offspring = self.get_offspring(
                pop,
                operator,
                m_task=m_task,
                s_parent=s_parent,
                deadline=deadline,
            )
        except Exception as exc:
            parents = None
            offspring = {
                "algorithm": None,
                "code": None,
                "objective": None,
                "other_inf": None,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self.evol.interface_llm.clear_deadline()
        if offspring is None:
            offspring = {
                "algorithm": None,
                "code": None,
                "objective": None,
                "other_inf": None,
                "evaluation_error": "offspring task returned no result",
            }
        if fe_index >= 0:
            offspring["fe_index"] = fe_index
            offspring["fe_phase"] = operator
        return parents, offspring

    def _get_algorithm_threaded(
        self,
        pop,
        operator,
        m_tasks=None,
        s_parents=None,
    ):
        results = []
        start_ts = time.time()
        batch_deadline = self.operator_batch_timeout + 15
        print(
            f"operator_batch_start operator={operator} "
            f"workers={self.n_p} pop_size={self.pop_size} child_count={self.child_count} "
            f"evaluation_timeout_s={self.timeout} batch_timeout_s={batch_deadline} "
            f"start_ts={start_ts:.3f}"
        )
        remaining = self.child_count
        if self.max_fe > 0:
            remaining = min(self.child_count, max(0, self.max_fe - self.fe_count))
        if operator == "m" and len(m_tasks or []) != remaining:
            raise ValueError("prepared m task count does not match remaining FE")
        if operator == "s" and s_parents is not None and len(s_parents) != remaining:
            raise ValueError("prepared S parent count does not match remaining FE")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_p)
        futures = {}
        for task_idx in range(remaining):
            m_task = m_tasks[task_idx] if operator == "m" else None
            future = executor.submit(
                self._get_offspring_with_stagger,
                pop,
                operator,
                task_idx,
                m_task,
                s_parents[task_idx] if s_parents is not None else None,
            )
            futures[future] = task_idx
        completed = 0
        try:
            for future in concurrent.futures.as_completed(futures, timeout=batch_deadline):
                task_idx = futures[future]
                try:
                    results.append(future.result())
                    completed += 1
                except Exception as e:
                    print(
                        f"offspring task failed for operator={operator} task_idx={task_idx}: "
                        f"{type(e).__name__}: {e}"
                    )
        except concurrent.futures.TimeoutError:
            pending = [task_idx for future, task_idx in futures.items() if not future.done()]
            elapsed = time.time() - start_ts
            print(
                f"Parallel batch timed out for operator={operator}: "
                f"completed={completed}/{self.child_count}, pending={pending}, "
                f"elapsed_s={elapsed:.3f}, configured_batch_timeout_s={batch_deadline}"
            )
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        elapsed = time.time() - start_ts
        print(
            f"operator_batch_end operator={operator} "
            f"completed={completed}/{self.child_count} results={len(results)} "
            f"elapsed_s={elapsed:.3f}"
        )
        return results

    
    def get_algorithm(self, pop, operator):
        self._set_operator_child_count(operator)
        results = []
        remaining = self.child_count
        if self.max_fe > 0:
            remaining = min(self.child_count, max(0, self.max_fe - self.fe_count))
        m_tasks = None
        s_parents = None
        if operator == "m":
            m_tasks = self._prepare_m_batch(pop, remaining)
        elif operator == "s":
            s_parents = self._prepare_s_parent_batch(pop, remaining)
        if self.parallel_backend == "threading":
            results = self._get_algorithm_threaded(
                pop,
                operator,
                m_tasks=m_tasks,
                s_parents=s_parents,
            )
        else:
            try:
                results = Parallel(
                    n_jobs=self.n_p,
                    timeout=self.timeout + 15,
                    backend=self.parallel_backend,
                )(
                    delayed(self._get_offspring_with_stagger)(
                        pop,
                        operator,
                        task_idx,
                        m_tasks[task_idx] if operator == "m" else None,
                        s_parents[task_idx] if s_parents is not None else None,
                    )
                    for task_idx in range(remaining)
                )
            except Exception as e:
                print(
                    f"Parallel batch failed for operator={operator}: "
                    f"{type(e).__name__}: {e}"
                )
             
        time.sleep(2)

        if (
            operator == "p"
            and os.environ.get("EOH_P_ONE_SURVIVOR_PER_PARENT", "0") == "1"
        ):
            mark_p_sibling_population_eligibility(results)
        if (
            operator == "m"
            and os.environ.get("EOH_M_ONE_SURVIVOR_PER_PAIR", "0") == "1"
        ):
            mark_m_pair_population_eligibility(results)

        out_p = []
        out_off = []

        for result in results:
            if not result:
                continue
            p, off = result
            if off is None:
                continue
            self._archive_attempt(off)
            if off.get('objective') is None:
                continue
            if operator == "p" and off.get("p_population_eligible") is False:
                print(
                    "p_sibling_prune "
                    f"parent={str(off.get('p_sibling_parent_hash', ''))[:12]} "
                    f"group={off.get('p_sibling_group_size')} "
                    f"pruned_fe={off.get('fe_index')} "
                    f"kept_fe={off.get('p_sibling_kept_fe_index')}"
                )
                continue
            if operator == "m" and off.get("m_pair_population_eligible") is False:
                print(
                    "m_pair_prune "
                    f"pair={str(off.get('m_sibling_pair_key', ''))[:25]} "
                    f"group={off.get('m_sibling_pair_group_size')} "
                    f"pruned_fe={off.get('fe_index')} "
                    f"kept_fe={off.get('m_sibling_pair_kept_fe_index')}"
                )
                continue
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        self.flush_fe_artifacts()
        return out_p, out_off
    # def get_algorithm(self,pop,operator, pop_size, n_p):
        
    #     # perform it pop_size times with n_p processes in parallel
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring
