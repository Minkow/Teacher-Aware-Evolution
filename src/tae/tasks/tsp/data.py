from __future__ import annotations

import ast
import pickle
import sys
from pathlib import Path

import numpy as np


def load_tsp_pickle(path: str | Path):
    _install_numpy_pickle_aliases()
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_lehd_tsp_txt(path: str | Path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(_parse_lehd_tsp_row(line))
    if not rows:
        raise ValueError(f"no TSP rows loaded from {path}")
    return rows


def _parse_lehd_tsp_row(line: str):
    parts = line.strip().split()
    if "output" not in parts:
        return _parse_tsplib_row(line)
    output_idx = parts.index("output")
    coord_vals = [float(x) for x in parts[:output_idx]]
    if len(coord_vals) % 2 != 0:
        raise ValueError("LEHD TSP coordinate section must have even length")
    coords = np.asarray(coord_vals, dtype=np.float64).reshape(-1, 2).astype(np.float32)

    route = [int(x) - 1 for x in parts[output_idx + 1 :]]
    if len(route) == len(coords) + 1 and route[0] == route[-1]:
        route = route[:-1]
    if len(route) != len(coords):
        raise ValueError(f"route length {len(route)} does not match node count {len(coords)}")
    reference_tour = np.asarray(route, dtype=np.int64)
    reference_cost = _tour_cost(coords, reference_tour)
    return {
        "coords": coords,
        "reference_tour": reference_tour,
        "reference_cost": float(reference_cost),
    }


def _parse_tsplib_row(line: str):
    row = ast.literal_eval(line)
    if len(row) < 4:
        raise ValueError("TSPlib row must include name, reference cost, and coordinates")
    name = str(row[0])
    reference_cost = float(row[1])
    coord_vals = [float(x) for x in row[2:]]
    if len(coord_vals) % 2 != 0:
        raise ValueError(f"TSPlib coordinate section for {name} must have even length")
    coords = np.asarray(coord_vals, dtype=np.float64).reshape(-1, 2).astype(np.float32)
    return {
        "name": name,
        "coords": coords,
        "reference_tour": np.arange(len(coords), dtype=np.int64),
        "reference_cost": reference_cost,
    }


def _tour_cost(coords: np.ndarray, tour: np.ndarray) -> float:
    ordered = coords[tour]
    rolled = np.roll(ordered, shift=-1, axis=0)
    return float(np.linalg.norm(ordered - rolled, axis=1).sum())


def _install_numpy_pickle_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
