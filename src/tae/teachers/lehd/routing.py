from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LEHD_ROOT = Path(os.environ.get("TAE_LEHD_ROOT", str(ROOT / "NCO_code" / "single_objective")))
TSP_CKPT = Path(os.environ.get("TAE_TSP_CHECKPOINT", str(LEHD_ROOT / "LEHD" / "TSP" / "result" / "20230509_153705_train" / "checkpoint-150.pt")))
CVRP_CKPT = Path(os.environ.get("TAE_CVRP_CHECKPOINT", str(LEHD_ROOT / "LEHD" / "CVRP" / "result" / "20230817_235537_train" / "checkpoint-40.pt")))


@dataclass(frozen=True)
class TSPInstance:
    coords: np.ndarray
    name: str = ""


@dataclass(frozen=True)
class CVRPInstance:
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    name: str = ""


@dataclass(frozen=True)
class RoutingFeature:
    node: int
    via_depot: int
    distance_from_current: float
    distance_to_depot: float
    distance_to_start: float
    nearest_unvisited_distance: float
    demand_ratio: float
    remaining_capacity_ratio: float
    capacity_after_ratio: float
    is_capacity_tight: float
    unvisited_fraction: float
    polar_angle: float
    radial_distance: float
    saving_via_depot: float


@dataclass(frozen=True)
class TeacherStep:
    features: tuple[RoutingFeature, ...]
    teacher_scores: tuple[float, ...]
    teacher_action_index: int


@dataclass(frozen=True)
class Rollout:
    objective: float
    actions: tuple[tuple[int, int], ...]
    steps: tuple[TeacherStep, ...]


@contextlib.contextmanager
def _lehd_import_path():
    path = str(LEHD_ROOT)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path)


def random_tsp_instances(count: int, n_nodes: int, seed: int = 0) -> list[TSPInstance]:
    rng = np.random.default_rng(seed)
    return [TSPInstance(rng.random((n_nodes, 2), dtype=np.float32), f"tsp_{seed}_{i}") for i in range(count)]


def random_cvrp_instances(
    count: int,
    n_customers: int,
    seed: int = 0,
    capacity: float = 50.0,
) -> list[CVRPInstance]:
    rng = np.random.default_rng(seed)
    instances = []
    for idx in range(count):
        coords = rng.random((n_customers + 1, 2), dtype=np.float32)
        coords[0] = np.array([0.5, 0.5], dtype=np.float32)
        demands = np.zeros(n_customers + 1, dtype=np.float32)
        demands[1:] = rng.integers(1, 10, size=n_customers).astype(np.float32)
        instances.append(CVRPInstance(coords, demands, capacity, f"cvrp_{seed}_{idx}"))
    return instances


class LEHDTSPTeacher:
    def __init__(self, checkpoint: str | Path = TSP_CKPT, device: str = "cpu"):
        self.device = torch.device(device)
        with _lehd_import_path():
            from LEHD.TSP.TSPModel import TSPModel

        params = _model_params("test")
        self.model = TSPModel(**params).to(self.device)
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def rollout(self, instance: TSPInstance, *, collect_steps: bool = True) -> Rollout:
        coords = _as_float_tensor(instance.coords, self.device)
        n_nodes = coords.shape[0]
        selected = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        visited = {0}
        actions: list[tuple[int, int]] = [(0, 0)]
        steps: list[TeacherStep] = []
        encoded = self.model.encoder(coords.unsqueeze(0))

        while len(visited) < n_nodes:
            with torch.no_grad():
                probs = self.model.decoder(encoded, selected).detach().cpu().numpy()[0]
            candidates = [node for node in range(n_nodes) if node not in visited]
            scores = np.array([float(probs[node]) for node in candidates], dtype=np.float64)
            choice_pos = int(np.argmax(scores))
            choice = candidates[choice_pos]
            if collect_steps:
                feats = tuple(_tsp_feature(instance.coords, selected[0, -1].item(), node, visited) for node in candidates)
                steps.append(TeacherStep(feats, tuple(scores.tolist()), choice_pos))
            selected = torch.cat([selected, torch.tensor([[choice]], dtype=torch.long, device=self.device)], dim=1)
            visited.add(choice)
            actions.append((choice, 0))

        tour = tuple(node for node, _ in actions)
        return Rollout(_tour_length(instance.coords, tour), tuple(actions), tuple(steps))


class LEHDCVRPTeacher:
    def __init__(self, checkpoint: str | Path = CVRP_CKPT, device: str = "cpu"):
        self.device = torch.device(device)
        with _lehd_import_path():
            from LEHD.CVRP.VRPModel import VRPModel

        params = _model_params("test")
        self.model = VRPModel(**params).to(self.device)
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def rollout(self, instance: CVRPInstance, *, collect_steps: bool = True) -> Rollout:
        coords = np.asarray(instance.coords, dtype=np.float32)
        demands = np.asarray(instance.demands, dtype=np.float32)
        capacity = float(instance.capacity)
        selected = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        visited: set[int] = set()
        current = 0
        remaining = capacity
        actions: list[tuple[int, int]] = []
        steps: list[TeacherStep] = []

        while len(visited) < len(coords) - 1:
            problem = _cvrp_problem_tensor(coords, demands, capacity, remaining, self.device)
            with torch.no_grad():
                encoded = self.model.encoder(problem, capacity)
                probs = self.model.decoder(
                    encoded,
                    selected,
                    capacity,
                    torch.tensor([remaining], dtype=torch.float32, device=self.device),
                ).detach().cpu().numpy()[0]
            candidates: list[tuple[int, int]] = []
            scores: list[float] = []
            features: list[RoutingFeature] = []
            n_customers = len(coords) - 1
            for customer in range(1, n_customers + 1):
                if customer in visited:
                    continue
                demand = float(demands[customer])
                if demand <= remaining + 1e-9:
                    idx = customer - 1
                    candidates.append((customer, 0))
                    scores.append(float(probs[idx]))
                    features.append(_cvrp_feature(coords, demands, capacity, current, remaining, customer, 0, visited))
                idx = n_customers + customer - 1
                candidates.append((customer, 1))
                scores.append(float(probs[idx]))
                features.append(_cvrp_feature(coords, demands, capacity, current, remaining, customer, 1, visited))
            choice_pos = int(np.argmax(np.asarray(scores, dtype=np.float64)))
            customer, via_depot = candidates[choice_pos]
            if collect_steps:
                steps.append(TeacherStep(tuple(features), tuple(scores), choice_pos))
            if via_depot:
                remaining = capacity
            remaining -= float(demands[customer])
            visited.add(customer)
            current = customer
            selected = torch.cat([selected, torch.tensor([[customer]], dtype=torch.long, device=self.device)], dim=1)
            actions.append((customer, via_depot))

        return Rollout(_cvrp_length(coords, actions), tuple(actions), tuple(steps))


def feature_matrix(features: Iterable[RoutingFeature]) -> np.ndarray:
    return np.asarray([_feature_vector(feature) for feature in features], dtype=np.float64)


def feature_names() -> tuple[str, ...]:
    return (
        "bias",
        "via_depot",
        "distance_from_current",
        "distance_to_depot",
        "distance_to_start",
        "nearest_unvisited_distance",
        "demand_ratio",
        "remaining_capacity_ratio",
        "capacity_after_ratio",
        "is_capacity_tight",
        "unvisited_fraction",
        "cos_angle",
        "sin_angle",
        "radial_distance",
        "saving_via_depot",
    )


def _model_params(mode: str) -> dict[str, float | int | str]:
    return {
        "mode": mode,
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "decoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
    }


def _as_float_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32, device=device)


def _cvrp_problem_tensor(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    remaining_capacity: float,
    device: torch.device,
) -> torch.Tensor:
    cap = np.full((len(coords), 1), remaining_capacity, dtype=np.float32)
    data = np.concatenate([coords.astype(np.float32), demands[:, None].astype(np.float32), cap], axis=1)
    data[:, 3] = remaining_capacity
    data[0, 3] = capacity
    return torch.as_tensor(data[None, :, :], dtype=torch.float32, device=device)


def _tsp_feature(coords: np.ndarray, current: int, node: int, visited: set[int]) -> RoutingFeature:
    unvisited = [idx for idx in range(len(coords)) if idx not in visited and idx != node]
    return _base_feature(coords, np.zeros(len(coords), dtype=np.float32), 1.0, current, 1.0, node, 0, visited, unvisited)


def _cvrp_feature(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    current: int,
    remaining: float,
    node: int,
    via_depot: int,
    visited: set[int],
) -> RoutingFeature:
    unvisited = [idx for idx in range(1, len(coords)) if idx not in visited and idx != node]
    return _base_feature(coords, demands, capacity, current, remaining, node, via_depot, visited, unvisited)


def _base_feature(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    current: int,
    remaining: float,
    node: int,
    via_depot: int,
    visited: set[int],
    unvisited: list[int],
) -> RoutingFeature:
    depot = 0
    start = 0
    origin = depot if via_depot else current
    d_current = _dist(coords[origin], coords[node])
    d_depot = _dist(coords[node], coords[depot])
    nearest = min((_dist(coords[node], coords[idx]) for idx in unvisited), default=0.0)
    demand = float(demands[node]) if node < len(demands) else 0.0
    capacity_after = (capacity - demand) if via_depot else (remaining - demand)
    center = coords[node] - coords[depot]
    angle = math.atan2(float(center[1]), float(center[0]))
    saving = _dist(coords[current], coords[depot]) + _dist(coords[depot], coords[node]) - _dist(coords[current], coords[node])
    return RoutingFeature(
        node=int(node),
        via_depot=int(via_depot),
        distance_from_current=float(d_current),
        distance_to_depot=float(d_depot),
        distance_to_start=float(_dist(coords[node], coords[start])),
        nearest_unvisited_distance=float(nearest),
        demand_ratio=float(demand / capacity) if capacity else 0.0,
        remaining_capacity_ratio=float(remaining / capacity) if capacity else 0.0,
        capacity_after_ratio=float(capacity_after / capacity) if capacity else 0.0,
        is_capacity_tight=float(capacity_after < 0.15 * capacity),
        unvisited_fraction=float(len(unvisited) / max(1, len(coords) - 1)),
        polar_angle=float(angle),
        radial_distance=float(np.linalg.norm(center)),
        saving_via_depot=float(saving),
    )


def _feature_vector(feature: RoutingFeature) -> list[float]:
    return [
        1.0,
        float(feature.via_depot),
        feature.distance_from_current,
        feature.distance_to_depot,
        feature.distance_to_start,
        feature.nearest_unvisited_distance,
        feature.demand_ratio,
        feature.remaining_capacity_ratio,
        feature.capacity_after_ratio,
        feature.is_capacity_tight,
        feature.unvisited_fraction,
        math.cos(feature.polar_angle),
        math.sin(feature.polar_angle),
        feature.radial_distance,
        feature.saving_via_depot,
    ]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _tour_length(coords: np.ndarray, tour: tuple[int, ...]) -> float:
    total = 0.0
    for a, b in zip(tour, tour[1:]):
        total += _dist(coords[a], coords[b])
    total += _dist(coords[tour[-1]], coords[tour[0]])
    return total


def _cvrp_length(coords: np.ndarray, actions: Iterable[tuple[int, int]]) -> float:
    current = 0
    total = 0.0
    for customer, via_depot in actions:
        if via_depot:
            total += _dist(coords[current], coords[0])
            current = 0
        total += _dist(coords[current], coords[customer])
        current = customer
    total += _dist(coords[current], coords[0])
    return total
