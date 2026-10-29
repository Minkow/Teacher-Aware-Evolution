from __future__ import annotations

import ast
import math
import textwrap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class ActionProgramError(ValueError):
    pass


_SAFE_BUILTINS = MappingProxyType(
    {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "sum": sum,
        "tuple": tuple,
    }
)
_SAFE_GLOBALS = MappingProxyType({"__builtins__": _SAFE_BUILTINS, "math": math})
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
    "print",
    "setattr",
}


@dataclass(frozen=True)
class ProgramActionScorer:
    name: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_fn", compile_action_source(self.source))

    def select_action(self, state) -> int:
        try:
            value = self._fn(state)
        except Exception as exc:
            raise ActionProgramError(f"{self.name} failed during select_action(): {exc}") from exc
        action = int(value)
        if action not in state.feasible_actions:
            raise ActionProgramError(f"{self.name} selected invalid action {action}")
        return action


def compile_action_source(source: str):
    normalized = _normalize_source(source)
    tree = ast.parse(normalized, mode="exec")
    _validate_tree(tree)
    namespace: dict[str, Any] = {}
    exec(compile(tree, filename="<maxcut-action-scorer>", mode="exec"), dict(_SAFE_GLOBALS), namespace)
    fn = namespace.get("select_action")
    if not callable(fn):
        raise ActionProgramError("source must define `def select_action(state): ...`")
    return fn


def extract_action_source(text: str) -> str:
    if "```" not in text:
        return _normalize_source(text)
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        if block.lstrip().startswith("python"):
            block = block.lstrip()[len("python") :]
        if "def select_action" in block:
            return _normalize_source(block)
    return _normalize_source(text)


def _normalize_source(source: str) -> str:
    source = textwrap.dedent(source).strip()
    if "def select_action" not in source:
        source = "def select_action(state):\n" + textwrap.indent(source, "    ")
    return source + "\n"


def _validate_tree(tree: ast.AST) -> None:
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_action"]
    if len(defs) != 1:
        raise ActionProgramError("source must contain exactly one `select_action` function")
    if len(tree.body) != 1:
        raise ActionProgramError("source may only define `select_action`; module-level setup is not allowed")
    fn = defs[0]
    if [arg.arg for arg in fn.args.args] != ["state"]:
        raise ActionProgramError("select_action signature must be `select_action(state)`")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ActionProgramError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            _validate_call(node)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ActionProgramError("private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ActionProgramError("dunder names are not allowed")


def _validate_call(node: ast.Call) -> None:
    if isinstance(node.func, ast.Name):
        if node.func.id in _FORBIDDEN_CALL_NAMES:
            raise ActionProgramError(f"forbidden call: {node.func.id}")
        if node.func.id not in _SAFE_BUILTINS:
            raise ActionProgramError(f"call to unknown function `{node.func.id}` is not allowed")
    elif isinstance(node.func, ast.Attribute):
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "math":
            raise ActionProgramError("only `math.<fn>()` calls are allowed")
        if node.func.attr.startswith("_") or not hasattr(math, node.func.attr):
            raise ActionProgramError(f"unknown math function: {node.func.attr}")
    else:
        raise ActionProgramError("dynamic calls are not allowed")
