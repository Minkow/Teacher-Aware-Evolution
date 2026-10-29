from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tae.tasks.jssp.core import CandidateFeature


@dataclass(frozen=True)
class TeacherStep:
    instance_name: str
    step: int
    candidate_ops: tuple[int, ...]
    policy_probs: tuple[float, ...]
    teacher_action: int
    greedy_rank: int
    candidate_features: tuple[CandidateFeature, ...]


@dataclass(frozen=True)
class TeacherTrace:
    instance_name: str
    makespan: int
    actions: tuple[int, ...]
    steps: tuple[TeacherStep, ...]


def dump_traces_jsonl(traces: Iterable[TeacherTrace], path: str | Path) -> None:
    with Path(path).open("w") as fh:
        for trace in traces:
            fh.write(json.dumps(asdict(trace)) + "\n")


def load_traces_jsonl(path: str | Path) -> list[TeacherTrace]:
    traces = []
    with Path(path).open() as fh:
        for line in fh:
            row = json.loads(line)
            steps = []
            for step in row["steps"]:
                features = tuple(CandidateFeature(**feature) for feature in step["candidate_features"])
                steps.append(
                    TeacherStep(
                        instance_name=step["instance_name"],
                        step=step["step"],
                        candidate_ops=tuple(step["candidate_ops"]),
                        policy_probs=tuple(step["policy_probs"]),
                        teacher_action=step["teacher_action"],
                        greedy_rank=step["greedy_rank"],
                        candidate_features=features,
                    )
                )
            traces.append(
                TeacherTrace(
                    instance_name=row["instance_name"],
                    makespan=row["makespan"],
                    actions=tuple(row["actions"]),
                    steps=tuple(steps),
                )
            )
    return traces

