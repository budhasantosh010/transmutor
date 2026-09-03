from __future__ import annotations

import numpy as np

from ..common.task_interface import Episode, common_prelude, nuisance_vector


class VariableCompositionTask:
    name = "variable_composition"

    def generate(self, seed: int, split: str) -> Episode:
        rng = np.random.default_rng(seed)
        depth = int(rng.integers(4, 7)) if split == "fresh_audit" else int(rng.integers(1, 4))
        value = float(rng.uniform(-0.8, 0.8))
        initial = value
        rows = [common_prelude(rng)]
        for index in range(depth):
            gain = float(rng.uniform(0.45, 0.95))
            drive = float(rng.uniform(-0.35, 0.35))
            value = float(np.tanh(gain * value + drive))
            row = nuisance_vector(rng, 0.18)
            row[0] = initial if index == 0 else float(rng.normal(0.0, 0.18))
            row[1] = gain
            row[2] = drive
            rows.append(row)
        return Episode(np.stack(rows), value, {"family": self.name, "depth": depth})

    def oracle(self, episode: Episode) -> float:
        return float(episode.target)

    def success(self, prediction: float, target: float) -> bool:
        return abs(prediction - target) <= 0.16
