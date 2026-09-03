from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

OBS_DIM = 6


@dataclass
class Episode:
    observations: np.ndarray
    target: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.observations, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != OBS_DIM:
            raise ValueError(f"observations must have shape [T,{OBS_DIM}], got {array.shape}")
        self.observations = array
        self.target = float(self.target)


class TaskFamily(Protocol):
    name: str

    def generate(self, seed: int, split: str) -> Episode: ...
    def oracle(self, episode: Episode) -> float: ...
    def success(self, prediction: float, target: float) -> bool: ...


class StatefulTaskAdapter:
    """Stateful reset/observe/step/target/done interface required by the research spec."""

    def __init__(self, task: TaskFamily):
        self.task = task
        self.episode: Episode | None = None
        self.index = 0

    def reset(self, seed: int, split: str = "development") -> np.ndarray:
        self.episode = self.task.generate(seed, split)
        self.index = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        if self.episode is None:
            raise RuntimeError("reset must be called first")
        return self.episode.observations[self.index].copy()

    def step(self, action_or_input=None) -> tuple[np.ndarray | None, bool]:
        if self.episode is None:
            raise RuntimeError("reset must be called first")
        self.index += 1
        if self.index >= len(self.episode.observations):
            return None, True
        return self.observe(), False

    def target(self) -> float:
        if self.episode is None:
            raise RuntimeError("reset must be called first")
        return self.episode.target

    def done(self) -> bool:
        return self.episode is not None and self.index >= len(self.episode.observations) - 1


def common_prelude(rng: np.random.Generator) -> np.ndarray:
    """Identically distributed first observation across every task family."""
    return rng.normal(0.0, 1.0, size=(OBS_DIM,)).astype(np.float32)


def nuisance_vector(rng: np.random.Generator, scale: float = 0.25) -> np.ndarray:
    return rng.normal(0.0, scale, size=(OBS_DIM,)).astype(np.float32)
