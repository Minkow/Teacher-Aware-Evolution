from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class MaxCutInstance:
    adjacency: tuple[tuple[float, ...], ...]
    name: str = ""

    @classmethod
    def from_array(cls, adjacency, name: str = "") -> "MaxCutInstance":
        arr = np.asarray(adjacency, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not np.allclose(arr, arr.T):
            raise ValueError("Max-Cut adjacency must be symmetric")
        arr = arr.copy()
        np.fill_diagonal(arr, 0.0)
        return cls(tuple(tuple(float(x) for x in row) for row in arr), name=name)

    @property
    def n_nodes(self) -> int:
        return len(self.adjacency)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.adjacency, dtype=float)


@dataclass(frozen=True)
class CandidateFeature:
    action: int
    spin: int
    flip_gain: float
    normalized_gain: float
    same_weight: float
    opposite_weight: float
    weighted_degree: float
    degree: int
    side0_fraction: float
    side1_fraction: float
    time_since_flip: int
    step_fraction: float

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MaxCutDecisionState:
    n_nodes: int
    step: int
    max_steps: int
    step_fraction: float
    current_cut: float
    best_cut: float
    spins: tuple[int, ...]
    actions: tuple[int, ...]
    feasible_actions: tuple[int, ...]
    adjacency: tuple[tuple[float, ...], ...]
    candidate_features: tuple[CandidateFeature, ...]
    features_by_action: MappingProxyType


class MaxCutState:
    def __init__(self, instance: MaxCutInstance, *, max_steps: int | None = None, init_spins=None):
        self.instance = instance
        self.adjacency = instance.as_array()
        self.n_nodes = instance.n_nodes
        self.max_steps = int(max_steps or 2 * self.n_nodes)
        if init_spins is None:
            spins = np.zeros(self.n_nodes, dtype=np.int8)
        else:
            spins = np.asarray(init_spins, dtype=np.int8)
            if spins.shape != (self.n_nodes,):
                raise ValueError("init_spins length must match graph size")
            spins = (spins != 0).astype(np.int8)
        self.spins = spins
        self.step = 0
        self.actions: list[int] = []
        self.current_cut = cut_value(self.adjacency, self.spins)
        self.best_cut = self.current_cut
        self.best_spins = self.spins.copy()

    @property
    def done(self) -> bool:
        return self.step >= self.max_steps

    @property
    def feasible_actions(self) -> tuple[int, ...]:
        return tuple(range(self.n_nodes))

    def candidate_features(self) -> list[CandidateFeature]:
        total_weight = np.maximum(np.abs(self.adjacency).sum(axis=1), 1e-9)
        side1 = float(np.mean(self.spins))
        side0 = 1.0 - side1
        last_flip = {node: idx for idx, node in enumerate(self.actions)}
        features: list[CandidateFeature] = []
        for node in range(self.n_nodes):
            row = self.adjacency[node]
            same = self.spins == self.spins[node]
            same[node] = False
            opposite = self.spins != self.spins[node]
            same_weight = float(row[same].sum())
            opposite_weight = float(row[opposite].sum())
            gain = same_weight - opposite_weight
            features.append(
                CandidateFeature(
                    action=node,
                    spin=int(self.spins[node]),
                    flip_gain=gain,
                    normalized_gain=float(gain / total_weight[node]),
                    same_weight=same_weight,
                    opposite_weight=opposite_weight,
                    weighted_degree=float(total_weight[node]),
                    degree=int(np.count_nonzero(row)),
                    side0_fraction=side0,
                    side1_fraction=side1,
                    time_since_flip=self.step - last_flip.get(node, -1),
                    step_fraction=self.step / max(1, self.max_steps),
                )
            )
        return features

    def decision_state(self) -> MaxCutDecisionState:
        features = tuple(self.candidate_features())
        return MaxCutDecisionState(
            n_nodes=self.n_nodes,
            step=self.step,
            max_steps=self.max_steps,
            step_fraction=self.step / max(1, self.max_steps),
            current_cut=float(self.current_cut),
            best_cut=float(self.best_cut),
            spins=tuple(int(x) for x in self.spins),
            actions=tuple(self.actions),
            feasible_actions=tuple(range(self.n_nodes)),
            adjacency=self.instance.adjacency,
            candidate_features=features,
            features_by_action=MappingProxyType({feature.action: feature for feature in features}),
        )

    def dispatch(self, action: int) -> None:
        if self.done:
            raise ValueError("cannot dispatch terminal state")
        action = int(action)
        if action < 0 or action >= self.n_nodes:
            raise ValueError(f"invalid Max-Cut flip action: {action}")
        feature = self.candidate_features()[action]
        self.spins[action] = 1 - self.spins[action]
        self.current_cut += feature.flip_gain
        self.step += 1
        self.actions.append(action)
        if self.current_cut > self.best_cut:
            self.best_cut = float(self.current_cut)
            self.best_spins = self.spins.copy()


def cut_value(adjacency: np.ndarray, spins: np.ndarray) -> float:
    diff = spins[:, None] != spins[None, :]
    return float(np.triu(adjacency * diff, k=1).sum())
