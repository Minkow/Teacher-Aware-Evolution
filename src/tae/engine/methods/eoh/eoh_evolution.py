import json
import concurrent.futures
import re
import time
import os
import random
import threading
from ...llm.interface_LLM import InterfaceLLM

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)
        self.teacher_context = {"e1": "", "e2": "", "m": "", "e3": "", "m1": "", "m2": "", "m3": "", "s": "", "p": ""}
        self._hypothesis_lock = threading.Lock()
        self._hypothesis_cursor = {}
        self._hypothesis_local = threading.local()
        self._hypothesis_rng = random.Random(
            int(os.environ.get("EOH_GLOBAL_SEED", "2024")) + 1701
        )

    def set_teacher_context(self, context):
        self.teacher_context = dict(context or {})
        with self._hypothesis_lock:
            self._hypothesis_cursor = {}
            for key in ("s_hypotheses", "m_hypotheses"):
                value = self.teacher_context.get(key)
                if not isinstance(value, list):
                    self.teacher_context[key] = []
                    continue
                shuffled = list(value)
                self._hypothesis_rng.shuffle(shuffled)
                self.teacher_context[key] = shuffled

    def consume_last_structural_hypothesis(self):
        hypothesis = getattr(
            self._hypothesis_local,
            "structural_hypothesis",
            None,
        )
        self._hypothesis_local.structural_hypothesis = None
        return hypothesis

    def _extract_algorithm(self, response):
        if not response:
            return None
        matches = re.findall(r"\{(.*?)\}", response, re.DOTALL)
        if matches:
            return matches[0].strip()
        fence_split = re.split(r"```(?:python)?", response, maxsplit=1, flags=re.IGNORECASE)
        if fence_split:
            head = fence_split[0].strip()
            if head:
                return head.splitlines()[0].strip()
        head = response.split("def ", 1)[0].strip()
        if head:
            return head.splitlines()[0].strip()
        return None

    def _extract_code(self, response):
        if not response:
            return None
        fence = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        candidate = fence.group(1) if fence else response
        lines = candidate.splitlines()
        start = None
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                start = i
                break
            if stripped.startswith(f"def {self.prompt_func_name}"):
                start = i
                break
        if start is None:
            return None
        code = "\n".join(lines[start:]).strip()
        if f"def {self.prompt_func_name}" not in code:
            return None
        if "return" not in code:
            return None
        return code

    def get_prompt_i1(self):
        
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    @staticmethod
    def _compact_text(value, limit):
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 28)].rstrip() + "\n[truncated for prompt budget]"

    @staticmethod
    def _parent_metrics(parent):
        return (
            f"objective={float(parent.get('objective', 0.0)):.6f}, "
            f"alignment={float(parent.get('alignment_score', 0.0)):.3f}, "
            f"top1={float(parent.get('top1_agreement', 0.0)):.3f}, "
            f"origin={parent.get('fe_phase', 'unknown')}#{parent.get('fe_index', 'n/a')}"
        )

    def _append_guidance(self, prompt_content, operator, parents=None):
        self._hypothesis_local.structural_hypothesis = None
        parts = [prompt_content]
        global_guidance = self._compact_text(
            (self.teacher_context or {}).get(operator, ""),
            4200,
        )
        if global_guidance:
            parts.append(
                "Population-level analyzer hypothesis (derived from representative programs, "
                "not necessarily from the selected parent below). Treat it as a hypothesis and "
                "override it when the selected-parent evidence conflicts:\n" + global_guidance
            )

        structural_hypothesis = self._sample_structural_hypothesis(operator)
        if structural_hypothesis:
            self._hypothesis_local.structural_hypothesis = structural_hypothesis
            parts.append(
                "Sampled structural-transition hypothesis for this child:\n"
                + self._format_structural_hypothesis(structural_hypothesis)
                + "\nTreat it as one experiment to test against the actual selected parent. "
                "Do not force it when the parent code or interface makes it incoherent."
            )

        if operator in {"s", "p"} and parents:
            parent = parents[0]
            evidence = [f"Selected parent metrics: {self._parent_metrics(parent)}"]
            trace_delta = self._compact_text(parent.get("trace_delta_summary"), 1500)
            pref_summary = self._compact_text(parent.get("feature_preference_summary"), 1000)
            cases = self._compact_text(
                parent.get("structure_cases") or parent.get("case_summary"),
                1600,
            )
            if trace_delta:
                evidence.append("Paired candidate-teacher trace delta:\n" + trace_delta)
            if pref_summary:
                evidence.append("Teacher-feature mismatch:\n" + pref_summary)
            if cases:
                evidence.append("Representative disagreements:\n" + cases)
            parts.append(
                "Evidence tied to the actual selected parent; this evidence has priority:\n"
                + "\n".join(evidence)
            )

        if operator in {"e2", "m", "e3"} and parents:
            parent_blocks = []
            roles = ["objective-oriented Pareto sibling", "alignment-oriented Pareto sibling"]
            for idx, parent in enumerate(parents, start=1):
                role = roles[idx - 1] if idx <= len(roles) else "additional selected parent"
                block = [f"Parent {idx} role: {role}", self._parent_metrics(parent)]
                trace_delta = self._compact_text(parent.get("trace_delta_summary"), 1100)
                pref_summary = self._compact_text(parent.get("feature_preference_summary"), 750)
                cases = self._compact_text(
                    parent.get("structure_cases") or parent.get("case_summary"),
                    900,
                )
                if trace_delta:
                    block.append("Paired candidate-teacher trace delta:\n" + trace_delta)
                if pref_summary:
                    block.append("Teacher-feature mismatch:\n" + pref_summary)
                if cases:
                    block.append("Representative disagreement:\n" + cases)
                parent_blocks.append("\n".join(block))
            parts.append(
                "Evidence tied to the actual selected parents; this evidence has priority:\n"
                + "\n\n".join(parent_blocks)
            )

        return "\n\n".join(part for part in parts if part)

    def _sample_structural_hypothesis(self, operator):
        key = {
            "s": "s_hypotheses",
            "m": "m_hypotheses",
        }.get(operator)
        if key is None:
            return None
        hypotheses = (self.teacher_context or {}).get(key)
        if not isinstance(hypotheses, list) or not hypotheses:
            return None
        with self._hypothesis_lock:
            cursor = int(self._hypothesis_cursor.get(key, 0))
            hypothesis = hypotheses[cursor % len(hypotheses)]
            self._hypothesis_cursor[key] = cursor + 1
        return hypothesis

    @staticmethod
    def _format_structural_hypothesis(hypothesis):
        if isinstance(hypothesis, str):
            return hypothesis.strip()
        if not isinstance(hypothesis, dict):
            return str(hypothesis)
        lines = []
        for key, label in (
            ("hypothesis", "Transformation"),
            ("evidence", "Evidence"),
            ("avoid", "Avoid"),
        ):
            value = str(hypothesis.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def _output_contract(self, description_request):
        return (
            description_request
            + " The description must be one sentence inside braces. Then implement it in Python "
            + "as a function named " + self.prompt_func_name + " accepting "
            + str(len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs
            + " and returning " + str(len(self.prompt_func_outputs)) + " output(s): "
            + self.joined_outputs + ". " + self.prompt_inout_inf + " "
            + self.prompt_other_inf + "\nDo not give additional explanations."
        )

    def _performance_reference_block(self, reference):
        if not reference:
            return ""
        return (
            "A better-performing population member is provided only as a performance contrast. "
            "Infer which concrete behavior may explain its advantage; do not copy it wholesale.\n"
            f"Reference metrics: {self._parent_metrics(reference)}\n"
            f"Reference description: {reference.get('algorithm', '')}\n"
            f"Reference code:\n{reference.get('code', '')}"
        )

        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return self._append_guidance(prompt_content, "e1", indivs)
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm by combining useful ideas from the given parents into a structurally new decision algorithm. "\
"The result should be a real fusion or reorganization of decision logic, not a tiny extension or superficial rewrite of one parent. "\
"If extra guidance is provided, use it only as auxiliary evidence when deciding how to combine or reorganize the parent ideas. \n"\
"First, describe the fused structure and why it improves on the parents in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return self._append_guidance(prompt_content, "e2", indivs)

    def get_prompt_e3(self,indivs):
        return self.get_prompt_e2(indivs)

    def _m_parent_evidence(self, parent, index):
        parts = [
            f"Parent {index} metrics: {self._parent_metrics(parent)}",
            f"Parent {index} description: {parent.get('algorithm', '')}",
            f"Parent {index} code:\n{self._compact_text(parent.get('code'), 7000)}",
        ]
        trace_delta = self._compact_text(parent.get("trace_delta_summary"), 2200)
        preference = self._compact_text(parent.get("feature_preference_summary"), 1600)
        cases = self._compact_text(
            parent.get("structure_cases") or parent.get("case_summary"),
            2200,
        )
        if trace_delta:
            parts.append(f"Parent {index} on-policy candidate-teacher trace delta:\n{trace_delta}")
        if preference:
            parts.append(f"Parent {index} on-policy teacher-feature mismatch:\n{preference}")
        if cases:
            parts.append(f"Parent {index} on-policy disagreement cases:\n{cases}")
        return "\n".join(parts)

    @staticmethod
    def _m_pair_key(parents):
        return tuple(
            (
                parent.get("fe_phase"),
                parent.get("fe_index"),
                str(parent.get("code") or ""),
            )
            for parent in parents
        )

    def build_m_batch_hints(self, parent_batches):
        """Assign pair ids; pair analysis is folded into child generation."""
        unique_pairs = []
        pair_key_to_id = {}
        child_pair_ids = []
        for parents in parent_batches:
            key = self._m_pair_key(parents)
            pair_id = pair_key_to_id.get(key)
            if pair_id is None:
                pair_id = len(unique_pairs)
                pair_key_to_id[key] = pair_id
                unique_pairs.append(parents)
            child_pair_ids.append(pair_id)

        self._last_m_analyzer_stats = {
            "children": len(parent_batches),
            "unique_pairs": len(unique_pairs),
            "analyzer_calls": 0,
            "analyzer_workers": 0,
            "analysis_mode": "inline_generator",
        }
        return ["" for _ in parent_batches], child_pair_ids


    def _m_local_parent_evidence(self, parent, label):
        block = [
            f"Parent {label} metrics: {self._parent_metrics(parent)}",
            f"Parent {label} description: {parent.get('algorithm', '')}",
            f"Parent {label} code:\n{self._compact_text(parent.get('code'), 6000)}",
        ]
        self_trace = self._compact_text(parent.get("trace_summary"), 1100)
        teacher_trace = self._compact_text(parent.get("teacher_trace_summary"), 1100)
        trace_delta = self._compact_text(parent.get("trace_delta_summary"), 1100)
        preference = self._compact_text(parent.get("feature_preference_summary"), 900)
        cases = self._compact_text(
            parent.get("structure_cases") or parent.get("case_summary"),
            1200,
        )
        if self_trace:
            block.append(f"Parent {label} self trace:\n{self_trace}")
        if teacher_trace:
            block.append(
                f"Teacher trace on Parent {label}'s on-policy states:\n{teacher_trace}"
            )
        if trace_delta:
            block.append(f"Parent {label} on-policy candidate-teacher trace delta:\n{trace_delta}")
        if preference:
            block.append(f"Parent {label} on-policy teacher-feature mismatch:\n{preference}")
        if cases:
            block.append(f"Parent {label} on-policy disagreement cases:\n{cases}")
        return "\n".join(block)



    def get_prompt_m(self, indivs, hint=""):
        if len(indivs) != 2:
            raise ValueError("m requires exactly two selected parents")
        ordered = sorted(
            indivs,
            key=lambda parent: (
                float(parent["objective"]),
                -float(parent.get("alignment_score", 0.0)),
            ),
        )
        parent_blocks = [
            self._m_local_parent_evidence(parent, label)
            for parent, label in zip(ordered, ("Better", "Other"))
        ]
        prompt_content = (
            self.prompt_task + "\n"
            + "Create one child from the two selected parent algorithms below. Aim to improve the task objective "
            + "over Better, which has the lower task objective. Better is the performance reference, not a required "
            + "code backbone. Compare the parents' "
            + "executable mechanisms together with their self traces and on-policy teacher traces. Let that evidence "
            + "determine whether the child should preserve, replace, transform, or combine mechanisms from either parent. "
            + "Other may also serve as negative evidence when one of its mechanisms explains worse objective "
            + "behavior. Teacher agreement is behavioral evidence, not the target. Produce one coherent, structurally distinct decision rule "
            + "for the task objective. Do not force a fusion or average formulas, "
            + "and do not stop at a coefficient-only, threshold-only, or gate-only edit. Do not output a separate "
            + "analysis; express the decision through the final algorithm description and executable code.\n\n"
            + "\n\n".join(parent_blocks)
        )
        return prompt_content + "\n\n" + self._output_contract(
            "Describe the resulting decision rule."
        )
    
    def get_prompt_m1(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return self._append_guidance(prompt_content, "m1", [indiv1])
    
    def get_prompt_m2(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please identify the main decision parameters of this algorithm and create a new version with a different parameterization of the same underlying score or priority logic. "\
"If extra guidance is provided, treat it only as auxiliary evidence for parameter-level calibration. "\
"Do not change the core decision structure. \n"\
"First, describe the retained backbone and the parameter-level changes in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return self._append_guidance(prompt_content, "m2", [indiv1])
    
    def get_prompt_m3(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please preserve the strongest useful strengths of this algorithm, but create a new version by changing one core decision module while keeping the overall problem-solving intent recognizable. "\
"If extra guidance is provided, use it only as auxiliary evidence for what kind of structural mutation may be helpful. "\
"Do not merely retune constants or rename the existing score. \n"\
"First, describe the retained strengths and the structural mutation in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return self._append_guidance(prompt_content, "m3", [indiv1])

    def get_prompt_s(self,indiv1,reference=None):
        prompt_content = (
            self.prompt_task + "\n"
            + "Selected parent description: " + indiv1["algorithm"] + "\n"
            + "Selected parent code:\n" + indiv1["code"] + "\n"
            + self._performance_reference_block(reference) + "\n"
            + "Treat the selected parent as behavioral evidence and a starting point, not as a mandatory code "
            + "template. Use objective results and teacher traces to choose one coherent structural alternative, "
            + "including what should be retained, removed, reorganized, or replaced. Choose the change yourself; "
            + "teacher evidence diagnoses behavioral consequences and is not an imitation target. Keep the child "
            + "no more asymptotically expensive per decision than the parent. In a per-candidate score function, do "
            + "not scan all feasible actions, the full instance, or large state arrays inside each score call; use "
            + "supplied candidate fields and only O(1) scalar state summaries. The result must stay inside the "
            + "declared function interface and remain deterministic. Do not spend the generation on coefficient-only "
            + "tuning, renaming, cosmetic edits, or a tie-break-only patch."
        )
        prompt_content = self._append_guidance(prompt_content, "s", [indiv1])
        return prompt_content + "\n\n" + self._output_contract(
            "Describe the teacher-conditioned diagnosis, backbone decision, and structural experiment."
        )

    def get_prompt_p(self,indiv1,reference=None):
        prompt_content = (
            self.prompt_task + "\n"
            + "Selected parent description: " + indiv1["algorithm"] + "\n"
            + "Selected parent code:\n" + indiv1["code"] + "\n"
            + self._performance_reference_block(reference) + "\n"
            + "Calibrate the selected parent's existing strategy using the evidence. Preserve its broad "
            + "decision structure, but adjust a small coherent set of numeric tradeoffs, transforms, gates, "
            + "or phase boundaries. You may remove redundant terms when that makes the tradeoff clearer. "
            + "Do not introduce an unrelated decision backbone, and do not change many independent choices "
            + "without a single evidence-backed rationale."
        )
        prompt_content = self._append_guidance(prompt_content, "p", [indiv1])
        return prompt_content + "\n\n" + self._output_contract(
            "Describe the retained strategy, diagnosed calibration issue, and parameter-level correction."
        )


    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)
        algorithm = self._extract_algorithm(response)
        code = self._extract_code(response)

        n_retry = 1
        while (algorithm is None or code is None):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)
            algorithm = self._extract_algorithm(response)
            code = self._extract_code(response)
                
            if n_retry > 3:
                break
            n_retry +=1

        if algorithm is None or code is None:
            raise ValueError("algorithm or code not identified from LLM response")

        return [code, algorithm]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e3(self,parents):

        prompt_content = self.get_prompt_e3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m(self,parents, hint=""):

        prompt_content = self.get_prompt_m(parents, hint=hint)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m3(self,parents):
      
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def s(self,parent,reference=None):

        prompt_content = self.get_prompt_s(parent, reference)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ s ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def p(self,parent,reference=None):

        prompt_content = self.get_prompt_p(parent, reference)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ p ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
