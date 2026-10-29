from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from tae.tasks.maxcut.alignment import TeacherPreference


class EcoDQNTeacher:
    """Adapter for ECO-DQN checkpoints from either the original repo or MaxCut-Bench."""

    def __init__(
        self,
        *,
        eco_root: str | Path,
        checkpoint: str | Path,
        device: str | None = None,
        step_factor: int = 2,
    ):
        self.eco_root = _resolve_eco_root(eco_root)
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.step_factor = int(step_factor)
        self.backend = _detect_backend(self.eco_root)
        self.network = self._build_network().to(self.device)
        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.network.load_state_dict(state_dict)
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

    def _build_network(self):
        with _prepend_sys_path(self.eco_root):
            if self.backend == "bench":
                from src.networks.models import MPNN  # type: ignore

                return MPNN(dim_in=7, dim_embedding=64, num_layers=3)
            from src.networks.mpnn import MPNN  # type: ignore

            return MPNN(
                n_obs_in=7,
                n_layers=3,
                n_features=64,
                n_hid_readout=[],
                tied_weights=False,
            )

    @torch.no_grad()
    def preference(self, state) -> TeacherPreference:
        if self.backend == "bench":
            batch = _bench_graph_data_from_state(state).to(self.device)
            qs = self.network(batch).detach().cpu().numpy().reshape(-1)
        else:
            obs = _eco_observation_from_state(state)
            qs = self.network(torch.FloatTensor(obs).to(self.device)).detach().cpu().numpy().reshape(-1)
        actions = tuple(range(state.n_nodes))
        scores = tuple(float(qs[action]) for action in actions)
        best_idx = int(np.argmax(qs[: state.n_nodes]))
        return TeacherPreference(candidate_actions=actions, teacher_scores=scores, teacher_action=best_idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["network"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.network is None:
            self.network = self._build_network().to(self.device)
            state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.network.load_state_dict(state_dict)
            self.network.eval()
            for param in self.network.parameters():
                param.requires_grad = False


@contextmanager
def _prepend_sys_path(path: Path):
    text = str(path)
    old = list(sys.path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        sys.path[:] = old


def _resolve_eco_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if (root / "src").is_dir():
        return root
    candidate = root / "solvers" / "ECO-DQN"
    if (candidate / "src").is_dir():
        return candidate
    raise FileNotFoundError(f"Could not resolve ECO-DQN root from {path}")


def _detect_backend(root: Path) -> str:
    if (root / "src" / "networks" / "models.py").is_file():
        return "bench"
    if (root / "src" / "networks" / "mpnn.py").is_file():
        return "eco"
    raise FileNotFoundError(f"Unsupported ECO-DQN layout under {root}")


def _eco_observation_from_state(state) -> np.ndarray:
    adj = np.asarray(state.adjacency, dtype=np.float32)
    spins = np.asarray(state.spins, dtype=np.float32)
    gains = np.asarray([feature.flip_gain for feature in state.candidate_features()], dtype=np.float32)
    max_local = float(max(np.abs(adj).sum(axis=1).max(), 1e-6))
    greedy_fraction = float(1.0 - np.mean(gains <= 0.0))
    time_since = _time_since_flip(state)
    best_spins = np.asarray(state.best_spins, dtype=np.float32)
    hamming_to_best = (best_spins != spins).astype(np.float32)
    distance_from_best_score = np.full(
        state.n_nodes,
        abs(float(state.current_cut) - float(state.best_cut)) / max_local,
        dtype=np.float32,
    )
    features = np.stack(
        [
            spins,
            gains / max_local,
            time_since,
            distance_from_best_score,
            hamming_to_best,
            np.full(state.n_nodes, greedy_fraction, dtype=np.float32),
            np.full(state.n_nodes, state.step / max(1, state.max_steps), dtype=np.float32),
        ],
        axis=0,
    )
    return np.vstack([features, adj]).astype(np.float32)


def _bench_graph_data_from_state(state):
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import dense_to_sparse

    adj = np.asarray(state.adjacency, dtype=np.float32)
    gains = np.asarray([feature.flip_gain for feature in state.candidate_features()], dtype=np.float32)
    max_local = float(max(np.abs(adj).sum(axis=1).max(), 1e-6))
    spins = np.asarray(state.spins, dtype=np.float32)
    best_spins = np.asarray(state.best_spins, dtype=np.float32)
    node_features = np.stack(
        [
            spins,
            gains / max_local,
            _time_since_flip(state),
            np.full(
                state.n_nodes,
                abs(float(state.current_cut) - float(state.best_cut)) / max_local,
                dtype=np.float32,
            ),
            (best_spins != spins).astype(np.float32),
            np.full(state.n_nodes, 1.0 - float(np.mean(gains <= 0.0)), dtype=np.float32),
            np.full(state.n_nodes, 1.0 - state.step / max(1, state.max_steps), dtype=np.float32),
        ],
        axis=1,
    )
    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(adj))
    data = Data(
        x=torch.from_numpy(node_features),
        edge_index=edge_index,
        edge_attr=edge_attr.unsqueeze(-1).to(torch.float32),
    )
    return Batch.from_data_list([data], follow_batch=None, exclude_keys=None)


def _time_since_flip(state) -> np.ndarray:
    actions = list(state.actions)
    last_flip = {node: idx for idx, node in enumerate(actions)}
    values = np.asarray(
        [state.step - last_flip.get(node, -1) for node in range(state.n_nodes)],
        dtype=np.float32,
    )
    return np.clip(values / max(1, state.max_steps), 0.0, 1.0)
