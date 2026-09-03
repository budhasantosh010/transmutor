from __future__ import annotations

import numpy as np

from ..common.task_interface import Episode, common_prelude, nuisance_vector


class IterativeStateTask:
    name = "iterative_state"

    def generate(self, seed: int, split: str) -> Episode:
        rng = np.random.default_rng(seed)
        length = int(rng.integers(11, 31)) if split == "fresh_audit" else int(rng.integers(3, 11))
        values = rng.choice([-1.0, 1.0], size=length).astype(np.float32)
        state = 0.0
        rows = [common_prelude(rng)]
        for value in values:
            state = 0.65 * state + 0.35 * float(value)
            row = nuisance_vector(rng, 0.22)
            row[0] = float(value)
            rows.append(row)
        return Episode(np.stack(rows), state, {"family": self.name, "length": length})

    def oracle(self, episode: Episode) -> float:
        return float(episode.target)

    def success(self, prediction: float, target: float) -> bool:
        return abs(prediction - target) <= 0.16
