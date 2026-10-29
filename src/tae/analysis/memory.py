from __future__ import annotations

from collections.abc import Iterable
from typing import Any


STRUCTURAL_OPERATORS = frozenset({"s", "e1", "e2", "m", "e3", "m3"})


def build_transition_records(
    *,
    generation: int,
    operator: str,
    parent_batches: Iterable[object],
    offspring: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Build serializable parent-to-child records for one operator batch."""
    records: list[dict[str, object]] = []
    for parents_value, child in zip(parent_batches, offspring):
        if not isinstance(child, dict) or child.get("objective") is None:
            continue
        parents = [
            _program_snapshot(parent)
            for parent in (parents_value or [])
            if isinstance(parent, dict) and parent.get("objective") is not None
        ]
        if not parents:
            continue

        child_snapshot = _program_snapshot(child)
        best_parent = min(parents, key=lambda item: float(item["objective"]))
        child_objective = float(child_snapshot["objective"])
        parent_objective = float(best_parent["objective"])
        child_alignment = _optional_float(child_snapshot.get("alignment_score"))
        parent_alignment = _optional_float(best_parent.get("alignment_score"))
        child_runtime = _optional_float(child_snapshot.get("runtime_per_decision_us"))
        parent_runtime = _optional_float(best_parent.get("runtime_per_decision_us"))

        records.append(
            {
                "generation": int(generation),
                "operator": str(operator),
                "parents": parents,
                "child": child_snapshot,
                "best_parent_objective": parent_objective,
                "objective_improvement": parent_objective - child_objective,
                "alignment_change": _difference(child_alignment, parent_alignment),
                "runtime_change_us": _difference(child_runtime, parent_runtime),
                "structural_hypothesis": child.get("structural_hypothesis"),
            }
        )
    return records


def select_transition_examples(
    records: Iterable[dict[str, object]],
    *,
    success_limit: int = 3,
    failure_limit: int = 3,
) -> dict[str, list[dict[str, object]]]:
    """Select objective-improving and objective-worsening structural transitions."""
    structural = [
        record
        for record in records
        if str(record.get("operator")) in STRUCTURAL_OPERATORS
        and record.get("objective_improvement") is not None
    ]
    improving = sorted(
        (record for record in structural if float(record["objective_improvement"]) > 0.0),
        key=lambda record: float(record["objective_improvement"]),
        reverse=True,
    )
    worsening = sorted(
        (record for record in structural if float(record["objective_improvement"]) <= 0.0),
        key=lambda record: float(record["objective_improvement"]),
    )
    return {
        "successful": improving[: max(0, int(success_limit))],
        "failed": worsening[: max(0, int(failure_limit))],
    }


def transition_evidence_text(
    records: Iterable[dict[str, object]],
    *,
    success_limit: int = 3,
    failure_limit: int = 3,
    code_limit: int = 1400,
) -> str:
    examples = select_transition_examples(
        records,
        success_limit=success_limit,
        failure_limit=failure_limit,
    )
    if not examples["successful"] and not examples["failed"]:
        return ""

    sections = [
        "Structural transition evidence from earlier evaluated children.",
        "Positive objective improvement means that the child outperformed its best selected parent.",
        "Infer reusable program transformations rather than copying an entire program.",
    ]
    for label, rows in (
        ("Successful transitions", examples["successful"]),
        ("Failed transitions", examples["failed"]),
    ):
        if not rows:
            continue
        sections.append(label + ":")
        for index, record in enumerate(rows, start=1):
            parent = min(
                record["parents"],
                key=lambda item: float(item["objective"]),
            )
            child = record["child"]
            hypothesis = _hypothesis_text(record.get("structural_hypothesis"))
            sections.append(
                "\n".join(
                    [
                        (
                            f"{index}. operator={record['operator']}, "
                            f"generation={record['generation']}, "
                            f"objective_improvement={float(record['objective_improvement']):+.6f}, "
                            f"alignment_change={_format_optional(record.get('alignment_change'))}, "
                            f"runtime_change_us={_format_optional(record.get('runtime_change_us'))}"
                        ),
                        f"Applied hypothesis: {hypothesis or 'none (unconditioned structural proposal)'}",
                        (
                            f"Parent objective={float(parent['objective']):.6f}; "
                            f"description={parent.get('algorithm') or 'n/a'}"
                        ),
                        "Parent code:\n" + _compact(parent.get("code"), code_limit),
                        (
                            f"Child objective={float(child['objective']):.6f}; "
                            f"description={child.get('algorithm') or 'n/a'}"
                        ),
                        "Child code:\n" + _compact(child.get("code"), code_limit),
                    ]
                )
            )
    return "\n\n".join(sections)


def _program_snapshot(program: dict[str, object]) -> dict[str, object]:
    return {
        "objective": float(program["objective"]),
        "alignment_score": _optional_float(program.get("alignment_score")),
        "runtime_per_decision_us": _optional_float(program.get("runtime_per_decision_us")),
        "algorithm": str(program.get("algorithm") or "").strip(),
        "code": str(program.get("code") or "").strip(),
        "fe_index": program.get("fe_index"),
        "fe_phase": program.get("fe_phase"),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional(value: object) -> str:
    parsed = _optional_float(value)
    return "n/a" if parsed is None else f"{parsed:+.6f}"


def _compact(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + "\n[truncated]"


def _hypothesis_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in ("hypothesis", "evidence", "avoid"):
            item = str(value.get(key) or "").strip()
            if item:
                parts.append(f"{key}={item}")
        return "; ".join(parts)
    return ""
