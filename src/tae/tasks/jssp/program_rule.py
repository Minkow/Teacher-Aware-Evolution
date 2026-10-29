from __future__ import annotations

import ast
import math
import textwrap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tae.tasks.jssp.core import CandidateFeature
from tae.tasks.jssp.simulator import ScheduleState


class RuleProgramError(ValueError):
    pass


_SAFE_BUILTINS = MappingProxyType(
    {
        "abs": abs,
        "bool": bool,
        "float": float,
        "int": int,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
        "sum": sum,
    }
)

_SAFE_GLOBALS = MappingProxyType(
    {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
    }
)

_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)

_FORBIDDEN_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "hasattr",
    "input",
    "locals",
    "open",
    "setattr",
}

_ALLOWED_FEATURE_ATTRS = set(CandidateFeature.__dataclass_fields__)
_ALLOWED_STATE_ATTRS = {
    "actions",
    "candidate_ops",
    "done",
    "instance",
    "job_ready",
    "machine_slots",
    "makespan",
    "next_op",
}
_ALLOWED_INSTANCE_ATTRS = {
    "durations",
    "machines",
    "n_jobs",
    "n_machines",
    "n_operations",
}


@dataclass(frozen=True)
class ProgramRule:
    name: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_fn", compile_rule_source(self.source))

    def score(self, feature: CandidateFeature, state: ScheduleState) -> float:
        try:
            value = self._fn(feature, state)
        except Exception as exc:
            raise RuleProgramError(f"{self.name} failed during score(): {exc}") from exc
        try:
            return float(value)
        except Exception as exc:
            raise RuleProgramError(f"{self.name} returned a non-numeric score: {value!r}") from exc


def compile_rule_source(source: str):
    normalized = _normalize_source(source)
    tree = ast.parse(normalized, mode="exec")
    _validate_tree(tree)
    namespace: dict[str, Any] = {}
    exec(compile(tree, filename="<llm-rule>", mode="exec"), dict(_SAFE_GLOBALS), namespace)
    fn = namespace.get("score")
    if not callable(fn):
        raise RuleProgramError("rule source must define `def score(feature, state): ...`")
    return fn


def extract_rule_source(text: str) -> str:
    """Extract a Python rule from an LLM response."""

    if "```" not in text:
        return _normalize_source(text)
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        if block.lstrip().startswith("python"):
            block = block.lstrip()[len("python") :]
        if "def score" in block:
            return _normalize_source(block)
    return _normalize_source(text)


def _normalize_source(source: str) -> str:
    source = textwrap.dedent(source).strip()
    if "def score" not in source:
        source = "def score(feature, state):\n" + textwrap.indent(source, "    ")
    return source + "\n"


def _validate_tree(tree: ast.AST) -> None:
    score_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score"]
    if len(score_defs) != 1:
        raise RuleProgramError("source must contain exactly one `score` function")
    if len(tree.body) != 1:
        raise RuleProgramError("source may only define `score`; module-level setup is not allowed")

    fn = score_defs[0]
    arg_names = [arg.arg for arg in fn.args.args]
    if arg_names != ["feature", "state"]:
        raise RuleProgramError("score signature must be `score(feature, state)`")

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise RuleProgramError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            _validate_call(node)
        if isinstance(node, ast.Attribute):
            _validate_attribute(node)
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RuleProgramError("dunder names are not allowed")


def _validate_call(node: ast.Call) -> None:
    if isinstance(node.func, ast.Name):
        if node.func.id in _FORBIDDEN_CALL_NAMES:
            raise RuleProgramError(f"forbidden call: {node.func.id}")
        if node.func.id not in _SAFE_BUILTINS:
            raise RuleProgramError(f"call to unknown function `{node.func.id}` is not allowed")
    elif isinstance(node.func, ast.Attribute):
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "math":
            raise RuleProgramError("only `math.<fn>()` attribute calls are allowed")
        if node.func.attr.startswith("_") or not hasattr(math, node.func.attr):
            raise RuleProgramError(f"unknown math function: {node.func.attr}")
    else:
        raise RuleProgramError("dynamic calls are not allowed")


def _validate_attribute(node: ast.Attribute) -> None:
    if node.attr.startswith("_"):
        raise RuleProgramError("private attributes are not allowed")
    if isinstance(node.value, ast.Name) and node.value.id == "feature":
        if node.attr not in _ALLOWED_FEATURE_ATTRS:
            raise RuleProgramError(f"unknown feature attribute: {node.attr}")
    elif isinstance(node.value, ast.Name) and node.value.id == "state":
        if node.attr not in _ALLOWED_STATE_ATTRS:
            raise RuleProgramError(f"unknown state attribute: {node.attr}")
    elif isinstance(node.value, ast.Attribute):
        if (
            isinstance(node.value.value, ast.Name)
            and node.value.value.id == "state"
            and node.value.attr == "instance"
        ):
            if node.attr not in _ALLOWED_INSTANCE_ATTRS:
                raise RuleProgramError(f"unknown instance attribute: {node.attr}")
        elif isinstance(node.value.value, ast.Name) and node.value.value.id == "math":
            return
        else:
            raise RuleProgramError("nested attributes are limited to `state.instance.<field>`")
    elif isinstance(node.value, ast.Name) and node.value.id == "math":
        return
    else:
        raise RuleProgramError("attribute access is limited to feature/state/math")
