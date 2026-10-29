from __future__ import annotations

import contextlib
import csv
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tae.tasks.jssp.core import JSSPInstance


@dataclass(frozen=True)
class TeacherPreference:
    candidate_ops: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    teacher_action: int


@contextlib.contextmanager
def _prepend_sys_path(path: str | Path):
    path_str = str(Path(path).resolve())
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path_str)


def _install_gym_shims() -> None:
    """Install the tiny gym/gymnasium surface needed by Offline-LD's SJSSP env."""
    if "gym" not in sys.modules and importlib.util.find_spec("gym") is None:
        gym_mod = types.ModuleType("gym")
        gym_utils = types.ModuleType("gym.utils")

        class Env:
            def reset(self, **kwargs):
                return None

        class EzPickle:
            def __init__(self, *args, **kwargs):
                pass

        gym_utils.EzPickle = EzPickle
        gym_mod.Env = Env
        gym_mod.utils = gym_utils
        sys.modules["gym"] = gym_mod
        sys.modules["gym.utils"] = gym_utils

    if "gymnasium" not in sys.modules and importlib.util.find_spec("gymnasium") is None:
        gymnasium_mod = types.ModuleType("gymnasium")
        spaces_mod = types.ModuleType("gymnasium.spaces")

        class Env:
            def reset(self, **kwargs):
                return None

        class Space:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Box(Space):
            pass

        class Discrete(Space):
            pass

        class Dict(Space):
            pass

        class MultiBinary(Space):
            pass

        class MultiDiscrete(Space):
            pass

        spaces_mod.Box = Box
        spaces_mod.Discrete = Discrete
        spaces_mod.Dict = Dict
        spaces_mod.MultiBinary = MultiBinary
        spaces_mod.MultiDiscrete = MultiDiscrete
        gymnasium_mod.Env = Env
        gymnasium_mod.spaces = spaces_mod
        sys.modules["gymnasium"] = gymnasium_mod
        sys.modules["gymnasium.spaces"] = spaces_mod


class OfflineLDTeacher:
    """Offline-LD QRDQN-CQL policy adapter for OPD-style program scoring."""

    def __init__(
        self,
        offline_root: str | Path,
        checkpoint_dir: str | Path,
        *,
        checkpoint_step: int | str = "best",
        device: str = "cpu",
        eval_n_jobs: int | None = None,
        eval_n_machines: int | None = None,
    ):
        self.offline_root = Path(offline_root).resolve()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_step = checkpoint_step
        self.device = device
        self.eval_n_jobs = eval_n_jobs
        self.eval_n_machines = eval_n_machines
        self._loaded = False

    @classmethod
    def from_checkpoint_root(
        cls,
        offline_root: str | Path,
        checkpoint_root: str | Path,
        n_jobs: int,
        n_machines: int,
        *,
        checkpoint_step: int | str = "best",
        device: str = "cpu",
        checkpoint_n_jobs: int | None = None,
        checkpoint_n_machines: int | None = None,
    ) -> "OfflineLDTeacher":
        root = Path(checkpoint_root).resolve()
        ckpt_jobs = checkpoint_n_jobs or n_jobs
        ckpt_machines = checkpoint_n_machines or n_machines
        pattern = f"QRDQN-CQL-full-{ckpt_jobs}x{ckpt_machines}*-{ckpt_jobs}x{ckpt_machines}-*"
        candidates = sorted(path for path in root.glob(pattern) if path.is_dir())
        if not candidates:
            raise FileNotFoundError(f"no Offline-LD checkpoint dir matching {pattern} under {root}")
        candidates = sorted(candidates, key=lambda path: _checkpoint_dir_score(path, checkpoint_step))
        return cls(
            offline_root,
            candidates[-1],
            checkpoint_step=checkpoint_step,
            device=device,
            eval_n_jobs=n_jobs,
            eval_n_machines=n_machines,
        )

    @property
    def n_jobs(self) -> int:
        self._load()
        return int(self.eval_n_jobs or self.config["n_jobs"])

    @property
    def n_machines(self) -> int:
        self._load()
        return int(self.eval_n_machines or self.config["n_machines"])

    def _load(self) -> None:
        if self._loaded:
            return

        import torch
        import yaml
        import torch.nn as nn

        config_path = self.checkpoint_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        self.config: dict[str, Any] = yaml.safe_load(config_path.read_text())
        checkpoint_path = self._resolve_checkpoint_path()

        _install_gym_shims()
        with _prepend_sys_path(self.offline_root):
            from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.L2D.JSSP_Env import SJSSP  # type: ignore
            from offline_jssp_rl.networks.gin import GraphCNN, aggr_obs, g_pool_cal  # type: ignore

            class Scalar(nn.Module):
                def __init__(self, init_value: float):
                    super().__init__()
                    self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

                def forward(self) -> nn.Parameter:
                    return self.constant

            class FeatureExtractor(nn.Module):
                def __init__(
                    self,
                    n_j: int,
                    n_m: int,
                    num_layers: int,
                    hidden_dim: int,
                    learn_eps: bool,
                    input_dim: int,
                    num_mlp_layers_feature_extract: int,
                    neighbor_pooling_type: str,
                    device: str,
                    dropout: float = 0.0,
                    orthogonal_init: bool = False,
                ):
                    super().__init__()
                    self.n_j = n_j
                    self.n_m = n_m
                    self.n_ops_perjob = n_m
                    self.device = device
                    self.feature_extract = GraphCNN(
                        num_layers=num_layers,
                        num_mlp_layers=num_mlp_layers_feature_extract,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        learn_eps=learn_eps,
                        neighbor_pooling_type=neighbor_pooling_type,
                        device=device,
                        dropout=dropout,
                    ).to(device)

                def forward(self, adj, features, candidate, graph_pool):
                    h_pooled, h_nodes = self.feature_extract(
                        x=features,
                        graph_pool=graph_pool,
                        padded_nei=None,
                        adj=adj,
                    )
                    dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
                    candidate_feature = torch.gather(
                        h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)),
                        1,
                        dummy,
                    )
                    h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
                    return torch.cat((candidate_feature, h_pooled_repeated), dim=-1), h_pooled

            class QRDQN_net(nn.Module):
                def __init__(
                    self,
                    n_j: int,
                    n_m: int,
                    num_layers: int,
                    hidden_dim: int,
                    learn_eps: bool,
                    input_dim: int,
                    action_dim: int,
                    num_mlp_layers_feature_extract: int,
                    neighbor_pooling_type: str,
                    device: str,
                    N: int = 32,
                    q_dropout: float = 0.0,
                    gnn_dropout: float = 0.0,
                    orthogonal_init: bool = False,
                    hidden_dim_q: int = 256,
                    n_hidden_q: int = 3,
                    activation_fn: type[nn.Module] = nn.ReLU,
                    dueling: bool = True,
                ):
                    super().__init__()
                    self.N = N
                    self.feature_extractor = FeatureExtractor(
                        n_j=n_j,
                        n_m=n_m,
                        num_layers=num_layers,
                        hidden_dim=hidden_dim,
                        learn_eps=learn_eps,
                        input_dim=input_dim,
                        num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                        neighbor_pooling_type=neighbor_pooling_type,
                        device=device,
                        dropout=gnn_dropout,
                        orthogonal_init=orthogonal_init,
                    )
                    self.dueling = dueling
                    layers: list[nn.Module] = []
                    if q_dropout > 0:
                        layers.append(nn.Dropout(p=q_dropout))
                    layers.append(nn.Linear(hidden_dim * 2, hidden_dim_q))
                    layers.append(activation_fn())
                    for _ in range(n_hidden_q - 1):
                        if q_dropout > 0:
                            layers.append(nn.Dropout(p=q_dropout))
                        layers.append(nn.Linear(hidden_dim_q, hidden_dim_q))
                        layers.append(activation_fn())
                    if q_dropout > 0:
                        layers.append(nn.Dropout(p=q_dropout))
                    layers.append(nn.Linear(hidden_dim_q, self.N))
                    self.network = nn.Sequential(*layers)
                    self.action_dim = action_dim

                def forward(self, adj, features, candidate, graph_pool, action_mask):
                    out_net, _ = self.feature_extractor(adj, features, candidate, graph_pool)
                    return self.network(out_net)

                def get_q_values(self, adj, features, candidate, graph_pool, action_mask):
                    q_values = self.forward(adj, features, candidate, graph_pool, action_mask)
                    q_values = q_values.mean(dim=2)
                    return q_values.masked_fill(~action_mask, -np.inf)

            activation = nn.ReLU if self.config.get("activation_fn", "relu") == "relu" else nn.Tanh
            eval_n_jobs = int(self.eval_n_jobs or self.config["n_jobs"])
            eval_n_machines = int(self.eval_n_machines or self.config["n_machines"])
            self.actor = QRDQN_net(
                n_j=eval_n_jobs,
                n_m=eval_n_machines,
                input_dim=2,
                neighbor_pooling_type=self.config.get("neighbor_pooling_type", "sum"),
                hidden_dim=int(self.config.get("hidden_dim_fe", 64)),
                num_layers=int(self.config.get("n_dim_fe", 2)),
                num_mlp_layers_feature_extract=int(self.config.get("n_dim_fe_mlp", 2)),
                learn_eps=False,
                device=self.device,
                gnn_dropout=float(self.config.get("gnn_dropout", 0)),
                q_dropout=float(self.config.get("q_dropout", 0.4)),
                orthogonal_init=bool(self.config.get("orthogonal_init", False)),
                action_dim=eval_n_jobs,
                hidden_dim_q=int(self.config.get("q_hidden_size", 32)),
                n_hidden_q=int(self.config.get("q_n_hidden_layers", 1)),
                activation_fn=activation,
                dueling=bool(self.config.get("dqn_dueling", False)),
            ).to(self.device)
            # Official checkpoints were created by executing train_cql_qrdqn.py
            # as __main__, so pickle looks for __main__.Scalar when loading the
            # full trainer state. We only use dqn_net, but the container has to
            # unpickle first.
            setattr(sys.modules["__main__"], "Scalar", Scalar)
            state = torch.load(
                checkpoint_path,
                map_location=torch.device(self.device),
                weights_only=False,
            )
            self.actor.load_state_dict(state["dqn_net"])
            self.actor.eval()
            self.env = SJSSP(eval_n_jobs, eval_n_machines)
            self.graph_pool = g_pool_cal(
                self.config.get("graph_pool_type", "average"),
                torch.Size([1, eval_n_jobs * eval_n_machines, eval_n_jobs * eval_n_machines]),
                eval_n_jobs * eval_n_machines,
                self.device,
            )
            self._aggr_obs = aggr_obs
            self._g_pool_cal = g_pool_cal

        self.state_mean, self.state_std = self._compute_feature_stats()
        self.torch = torch
        self._loaded = True

    def _resolve_checkpoint_path(self) -> Path:
        if isinstance(self.checkpoint_step, int):
            path = self.checkpoint_dir / f"checkpoint_{self.checkpoint_step}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        if self.checkpoint_step == "final":
            steps = self._result_steps()
            return self.checkpoint_dir / f"checkpoint_{max(steps)}.pt"
        if self.checkpoint_step == "best":
            best_step = self._best_result_step()
            return self.checkpoint_dir / f"checkpoint_{best_step}.pt"
        path = Path(str(self.checkpoint_step)).expanduser()
        if not path.is_absolute():
            path = self.checkpoint_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _result_steps(self) -> list[int]:
        results_path = self.checkpoint_dir / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        with results_path.open() as fh:
            return [int(float(row["step"])) for row in csv.DictReader(fh)]

    def _best_result_step(self) -> int:
        results_path = self.checkpoint_dir / "results.csv"
        with results_path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"empty results file: {results_path}")
        best = min(rows, key=lambda row: float(row["make_span"]))
        return int(float(best["step"]))

    def _compute_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        cached = self.checkpoint_dir / "feature_stats.npz"
        if cached.exists():
            data = np.load(cached)
            return data["mean"].astype(np.float32), data["std"].astype(np.float32)

        import minari

        dataset = minari.load_dataset(str(self.config["dataset"]))
        limit = self.config.get("num_instance")
        total = 0
        feature_sum: np.ndarray | None = None
        feature_sq_sum: np.ndarray | None = None
        for idx, episode in enumerate(dataset):
            if limit is not None and idx >= int(limit):
                break
            features = np.asarray(episode.observations["fea"][:-1], dtype=np.float64)
            flat = features.reshape(-1, features.shape[-1])
            if feature_sum is None:
                feature_sum = flat.sum(axis=0)
                feature_sq_sum = np.square(flat).sum(axis=0)
            else:
                feature_sum += flat.sum(axis=0)
                feature_sq_sum += np.square(flat).sum(axis=0)
            total += flat.shape[0]
        if total <= 0 or feature_sum is None or feature_sq_sum is None:
            raise ValueError(f"empty Offline-LD dataset: {self.config['dataset']}")
        mean = feature_sum / total
        var = np.maximum(feature_sq_sum / total - np.square(mean), 0.0)
        mean = mean.astype(np.float32)
        std = (np.sqrt(var) + 1e-3).astype(np.float32)
        np.savez(cached, mean=mean, std=std)
        return mean, std

    def reset_instance(self, instance: JSSPInstance):
        self._load()
        instance = instance.normalized()
        if instance.n_jobs != self.n_jobs or instance.n_machines != self.n_machines:
            raise ValueError("instance shape must match Offline-LD teacher shape")
        data = (
            np.asarray(instance.durations, dtype=np.single),
            np.asarray(instance.machines, dtype=np.int64),
        )
        state, _ = self.env.reset(options={"JSM_env": data})
        return state

    def step(self, action: int):
        self._load()
        return self.env.step(int(action))

    def preference(self, state) -> TeacherPreference:
        return self.preference_batch([state])[0]

    def preference_batch(self, states) -> list[TeacherPreference]:
        """Evaluate fixed-shape JSSP states in one GNN forward pass."""
        self._load()
        states = list(states)
        if not states:
            return []
        torch = self.torch
        feasible_masks = [~np.asarray(state["mask"], dtype=bool) for state in states]
        candidate_rows = [
            tuple(int(op) for op, feasible in zip(state["omega"], mask) if feasible)
            for state, mask in zip(states, feasible_masks)
        ]
        if any(not candidates for candidates in candidate_rows):
            raise ValueError("cannot score terminal state")

        features = np.stack(
            [
                (np.asarray(state["fea"], dtype=np.float32) - self.state_mean)
                / self.state_std
                for state in states
            ]
        )
        n_nodes = features.shape[1]
        fea_tensor = torch.as_tensor(
            features.reshape(-1, features.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        adj_tensor = torch.as_tensor(
            np.stack([np.asarray(state["adj"], dtype=np.float32) for state in states]),
            dtype=torch.float32,
            device=self.device,
        ).to_sparse()
        adj_tensor = self._aggr_obs(adj_tensor, n_nodes)
        omega_tensor = torch.as_tensor(
            np.stack([np.asarray(state["omega"], dtype=np.int64) for state in states]),
            dtype=torch.long,
            device=self.device,
        )
        mask_tensor = torch.as_tensor(
            np.stack(feasible_masks),
            dtype=torch.bool,
            device=self.device,
        )
        graph_pool = self._g_pool_cal(
            self.config.get("graph_pool_type", "average"),
            torch.Size([len(states), n_nodes, n_nodes]),
            n_nodes,
            self.device,
        )
        with torch.no_grad():
            q_values = self.actor.get_q_values(
                adj_tensor,
                fea_tensor,
                omega_tensor,
                graph_pool,
                mask_tensor,
            )
        raw_rows = q_values.detach().cpu().numpy().astype(float)
        outputs = []
        for raw_scores, feasible_mask, candidate_ops in zip(
            raw_rows,
            feasible_masks,
            candidate_rows,
        ):
            scores = tuple(
                float(score)
                for score, feasible in zip(raw_scores.tolist(), feasible_mask)
                if feasible
            )
            best_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -candidate_ops[idx]))
            outputs.append(
                TeacherPreference(
                    candidate_ops=candidate_ops,
                    teacher_scores=scores,
                    teacher_action=candidate_ops[best_idx],
                )
            )
        return outputs


def _checkpoint_dir_score(path: Path, checkpoint_step: int | str) -> tuple[int, int, float]:
    """Prefer usable completed runs over stale failed retry directories."""

    checkpoints = list(path.glob("checkpoint_*.pt"))
    if isinstance(checkpoint_step, int):
        usable = int((path / f"checkpoint_{checkpoint_step}.pt").exists())
    elif checkpoint_step == "best":
        usable = int((path / "results.csv").exists() and bool(checkpoints))
    else:
        usable = int(bool(checkpoints))
    latest = max(
        [path.stat().st_mtime]
        + [checkpoint.stat().st_mtime for checkpoint in checkpoints]
        + ([(path / "results.csv").stat().st_mtime] if (path / "results.csv").exists() else [])
    )
    return usable, len(checkpoints), latest
