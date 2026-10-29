from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import networkx as nx
import numpy as np

from tae.tasks.maxcut.core import MaxCutInstance


def random_er_instances(count: int, n_nodes: int, p: float, *, seed: int = 0) -> list[MaxCutInstance]:
    rng = np.random.default_rng(seed)
    instances = []
    for idx in range(count):
        graph_seed = int(rng.integers(0, 2**31 - 1))
        graph = nx.erdos_renyi_graph(n_nodes, p, seed=graph_seed)
        adj = nx.to_numpy_array(graph, dtype=float)
        instances.append(MaxCutInstance.from_array(adj, name=f"ER_{n_nodes}_{idx:04d}"))
    return instances


def random_ba_instances(count: int, n_nodes: int, m: int, *, seed: int = 0) -> list[MaxCutInstance]:
    rng = np.random.default_rng(seed)
    instances = []
    for idx in range(count):
        graph_seed = int(rng.integers(0, 2**31 - 1))
        graph = nx.barabasi_albert_graph(n_nodes, m, seed=graph_seed)
        adj = nx.to_numpy_array(graph, dtype=float)
        instances.append(MaxCutInstance.from_array(adj, name=f"BA_{n_nodes}_{idx:04d}"))
    return instances


def load_eco_graph_set(path: str | Path) -> list[MaxCutInstance]:
    with Path(path).open("rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = pickle.load(f)
    instances = []
    for idx, graph in enumerate(payload):
        if hasattr(graph, "toarray"):
            adj = graph.toarray()
        elif hasattr(graph, "nodes") and hasattr(graph, "edges"):
            adj = nx.to_numpy_array(graph, dtype=float)
        else:
            adj = np.asarray(graph, dtype=float)
        instances.append(MaxCutInstance.from_array(adj, name=f"{Path(path).stem}_{idx:04d}"))
    return instances


def load_maxcut_bench_directory(path: str | Path, *, limit: int | None = None) -> list[MaxCutInstance]:
    root = Path(path)
    files = sorted(root.glob("*.npz"))
    if limit is not None:
        files = files[: max(0, int(limit))]
    instances: list[MaxCutInstance] = []
    for file_path in files:
        adj = _load_sparse_npz_as_dense(file_path)
        instances.append(MaxCutInstance.from_array(adj, name=file_path.stem))
    return instances


def load_maxcut_bench_split(
    root: str | Path,
    *,
    split: str,
    distribution: str,
    limit: int | None = None,
) -> list[MaxCutInstance]:
    bench_root = Path(root)
    dataset_dir = bench_root / "data" / split / distribution
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"MaxCut-Bench dataset directory not found: {dataset_dir}")
    return load_maxcut_bench_directory(dataset_dir, limit=limit)


def load_maxcut_bench_mixed_split(
    root: str | Path,
    *,
    split: str,
    distributions: list[str],
    limit: int | None = None,
) -> list[MaxCutInstance]:
    bench_root = Path(root)
    dist_names = [item.strip() for item in distributions if item and item.strip()]
    if not dist_names:
        return []

    if limit is None:
        quotas = [None] * len(dist_names)
    else:
        total = max(0, int(limit))
        base = total // len(dist_names)
        rem = total % len(dist_names)
        quotas = [base + (1 if idx < rem else 0) for idx in range(len(dist_names))]

    grouped: list[list[MaxCutInstance]] = []
    for idx, dist_name in enumerate(dist_names):
        dataset_dir = bench_root / "data" / split / dist_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"MaxCut-Bench dataset directory not found: {dataset_dir}")
        quota = quotas[idx]
        bucket = []
        for item in load_maxcut_bench_directory(dataset_dir, limit=quota):
            bucket.append(MaxCutInstance.from_array(item.as_array(), name=f"{dist_name}::{item.name}"))
        grouped.append(bucket)

    mixed: list[MaxCutInstance] = []
    cursor = 0
    while True:
        advanced = False
        for bucket in grouped:
            if cursor < len(bucket):
                mixed.append(bucket[cursor])
                advanced = True
        if not advanced:
            break
        cursor += 1
    return mixed


def load_maxcut_bench_optimal(path: str | Path) -> dict[str, float]:
    import pandas as pd

    target = Path(path)
    if target.is_dir():
        target = target / "optimal"
    if not target.is_file():
        return {}
    payload = pd.read_pickle(target)
    records: dict[str, float] = {}
    if hasattr(payload, "iterrows"):
        for _, row in payload.iterrows():
            name = row.get("graph") or row.get("Instances") or row.get("instance")
            if name is None:
                continue
            records[Path(str(name)).stem] = float(row["OPT"])
    return records


def _load_sparse_npz_as_dense(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "data" not in payload or "indices" not in payload or "indptr" not in payload or "shape" not in payload:
            raise ValueError(f"Unsupported sparse npz format: {path}")
        data = payload["data"]
        indices = payload["indices"]
        indptr = payload["indptr"]
        shape = tuple(int(x) for x in payload["shape"])
        if "format" in payload:
            raw_fmt = payload["format"]
            if hasattr(raw_fmt, "item"):
                raw_fmt = raw_fmt.item()
            if isinstance(raw_fmt, bytes):
                fmt = raw_fmt.decode("utf-8", errors="ignore")
            else:
                fmt = str(raw_fmt)
        else:
            fmt = "csr"
    if len(shape) != 2:
        raise ValueError(f"Expected 2D sparse matrix in {path}, got shape={shape}")
    if fmt not in {"csr", "csc"}:
        raise ValueError(f"Unsupported sparse format {fmt!r} in {path}")
    dense = np.zeros(shape, dtype=data.dtype)
    if fmt == "csr":
        for row in range(shape[0]):
            start = int(indptr[row])
            end = int(indptr[row + 1])
            dense[row, indices[start:end]] = data[start:end]
        return dense
    for col in range(shape[1]):
        start = int(indptr[col])
        end = int(indptr[col + 1])
        dense[indices[start:end], col] = data[start:end]
    return dense
