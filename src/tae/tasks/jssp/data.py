from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from tae.tasks.jssp.core import JSSPInstance


def generate_uniform_instances(
    n_jobs: int,
    n_machines: int,
    count: int,
    low: int = 1,
    high: int = 99,
    seed: int = 0,
) -> list[JSSPInstance]:
    rng = random.Random(seed)
    instances = []
    for idx in range(count):
        durations = [
            [rng.randint(low, high) for _ in range(n_machines)]
            for _ in range(n_jobs)
        ]
        machines = []
        for _ in range(n_jobs):
            row = list(range(n_machines))
            rng.shuffle(row)
            machines.append(row)
        instances.append(JSSPInstance.from_lists(durations, machines, name=f"u{idx:04d}"))
    return instances


def load_l2d_npy(path: str | Path, limit: int | None = None) -> list[JSSPInstance]:
    import numpy as np

    arr = np.load(Path(path), allow_pickle=True)
    instances = []
    for idx, item in enumerate(arr[:limit]):
        durations = item[0].astype(int).tolist()
        machines = item[1].astype(int).tolist()
        instances.append(JSSPInstance.from_lists(durations, machines, name=f"{Path(path).stem}_{idx}"))
    return instances


def load_jsonl(path: str | Path, limit: int | None = None) -> list[JSSPInstance]:
    instances = []
    with Path(path).open() as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            row = json.loads(line)
            instances.append(
                JSSPInstance.from_lists(
                    row["durations"],
                    row["machines"],
                    name=row.get("name", f"{Path(path).stem}_{idx}"),
                )
            )
    return instances


def dump_jsonl(instances: Iterable[JSSPInstance], path: str | Path) -> None:
    with Path(path).open("w") as fh:
        for instance in instances:
            fh.write(
                json.dumps(
                    {
                        "name": instance.name,
                        "durations": instance.durations,
                        "machines": instance.machines,
                    }
                )
                + "\n"
            )


def load_instances(
    *,
    npy: str | Path | None = None,
    jsonl: str | Path | None = None,
    n_jobs: int = 6,
    n_machines: int = 6,
    count: int = 20,
    seed: int = 0,
    low: int = 1,
    high: int = 99,
) -> list[JSSPInstance]:
    if npy:
        return load_l2d_npy(npy, limit=count)
    if jsonl:
        return load_jsonl(jsonl, limit=count)
    return generate_uniform_instances(n_jobs, n_machines, count, low=low, high=high, seed=seed)

