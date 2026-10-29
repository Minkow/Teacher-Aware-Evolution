from __future__ import annotations

import ast
import pickle
import sys
from pathlib import Path

import numpy as np


def load_cvrp_pickle(path: str | Path):
    _install_numpy_pickle_aliases()
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def load_lehd_cvrp_txt(path: str | Path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(_parse_lehd_row(line))
    if not rows:
        raise ValueError(f"no CVRP rows loaded from {path}")
    return rows


def _parse_lehd_row(line: str):
    if line.startswith("["):
        parts = ast.literal_eval(line)
    else:
        parts = [token.strip() for token in line.split(",")]

    def marker(name: str) -> int:
        try:
            return parts.index(name)
        except ValueError as exc:
            raise ValueError(f"missing marker {name!r} in LEHD CVRP row") from exc

    depot_idx = marker("depot")
    customer_idx = marker("customer")
    capacity_idx = marker("capacity")
    demand_idx = marker("demand")
    cost_idx = marker("cost")
    node_flag_idx = parts.index("node_flag") if "node_flag" in parts else None
    name_idx = parts.index("name") if "name" in parts else None

    depot_vals = [float(x) for x in parts[depot_idx + 1 : customer_idx]]
    marker_indices = [idx for idx in (capacity_idx, demand_idx, cost_idx, node_flag_idx, name_idx) if idx is not None]
    customer_end = min(idx for idx in marker_indices if idx > customer_idx)
    demand_end = min(
        [idx for idx in marker_indices + [len(parts)] if idx > demand_idx],
        default=len(parts),
    )
    customer_vals = [float(x) for x in parts[customer_idx + 1 : customer_end]]
    demand_vals = [float(x) for x in parts[demand_idx + 1 : demand_end]]
    capacity = float(parts[capacity_idx + 1])
    cost = float(parts[cost_idx + 1])
    name = parts[name_idx + 1] if name_idx is not None else ""

    if len(depot_vals) != 2:
        raise ValueError("LEHD CVRP row must contain exactly one depot coordinate pair")
    if len(customer_vals) % 2 != 0:
        raise ValueError("LEHD CVRP customer coordinate section must have even length")

    customers = np.asarray(customer_vals, dtype=np.float64).reshape(-1, 2)
    n_customers = customers.shape[0]
    if len(demand_vals) == n_customers + 1:
        demand_vals = demand_vals[1:]
    if len(demand_vals) != n_customers:
        raise ValueError(
            f"LEHD CVRP row demand length {len(demand_vals)} does not match customer count {n_customers}"
        )

    coords = np.vstack([np.asarray(depot_vals, dtype=np.float64)[None, :], customers]).astype(np.float32)
    demands = np.zeros(n_customers + 1, dtype=np.float32)
    demands[1:] = np.asarray(demand_vals, dtype=np.float32)

    row = {
        "name": name,
        "coords": coords,
        "demands": demands,
        "capacity": float(capacity),
        "reference_cost": float(cost),
    }
    if node_flag_idx is not None:
        row["node_flag"] = [int(float(x)) for x in parts[node_flag_idx + 1 :]]
    return row


def _install_numpy_pickle_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
