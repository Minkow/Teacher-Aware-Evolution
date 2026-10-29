from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reusable weighted Max-Cut datasets in MaxCut-Bench npz format.")
    parser.add_argument("--graph-type", choices=["ba", "er"], required=True)
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--er-p", type=float, default=0.15)
    parser.add_argument("--ba-m", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    files: list[str] = []
    for idx in range(max(0, int(args.count))):
        if args.graph_type == "ba":
            matrix = generate_weighted_ba(args.nodes, args.ba_m, rng)
            stem = f"BA_{args.nodes}vertices_graph_{idx:04d}"
        else:
            matrix = generate_weighted_er(args.nodes, args.er_p, rng)
            stem = f"ER_{args.nodes}vertices_graph_{idx:04d}"
        path = out_dir / f"{stem}.npz"
        sparse.save_npz(path, sparse.csr_matrix(matrix))
        files.append(path.name)

    meta = {
        "graph_type": args.graph_type,
        "nodes": int(args.nodes),
        "count": int(args.count),
        "seed": int(args.seed),
        "er_p": float(args.er_p),
        "ba_m": int(args.ba_m),
        "files": files,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "count": len(files), "sample": files[:3]}, indent=2))


def generate_weighted_er(n_nodes: int, p: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random((n_nodes, n_nodes)) < float(p)
    upper = np.triu(mask, k=1)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_nodes, n_nodes))
    weighted = upper.astype(np.float32) * np.triu(signs, k=1)
    matrix = weighted + weighted.T
    np.fill_diagonal(matrix, 0.0)
    return matrix


def generate_weighted_ba(n_nodes: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if m < 1 or m >= n_nodes:
        raise ValueError(f"invalid ba-m={m} for n_nodes={n_nodes}")

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    # Match the edge count pattern used by networkx.barabasi_albert_graph:
    # start from a star graph on m+1 nodes, then attach each new node with m edges.
    repeated_nodes: list[int] = []
    for target in range(1, m + 1):
        weight = float(rng.choice((-1.0, 1.0)))
        adjacency[0, target] = weight
        adjacency[target, 0] = weight
        repeated_nodes.extend([0, target])

    source = m + 1
    while source < n_nodes:
        targets = random_subset(repeated_nodes, m, rng)
        for target in targets:
            weight = float(rng.choice((-1.0, 1.0)))
            adjacency[source, target] = weight
            adjacency[target, source] = weight
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        source += 1

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def random_subset(repeated_nodes: list[int], m: int, rng: np.random.Generator) -> list[int]:
    targets: set[int] = set()
    while len(targets) < m:
        idx = int(rng.integers(0, len(repeated_nodes)))
        targets.add(int(repeated_nodes[idx]))
    return list(targets)


if __name__ == "__main__":
    main()
