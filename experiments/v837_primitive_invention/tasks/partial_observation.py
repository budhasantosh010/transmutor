from __future__ import annotations

import numpy as np

from ..common.task_interface import Episode, common_prelude, nuisance_vector


class PartialObservationTask:
    name = "partial_observation"

    def generate(self, seed: int, split: str) -> Episode:
        rng = np.random.default_rng(seed)
        length = int(rng.integers(10, 19)) if split == "fresh_audit" else int(rng.integers(5, 11))
        rho = float(rng.uniform(0.72, 0.94))
        noise = 0.28 if split == "fresh_audit" else 0.20
        z = float(rng.normal(0.0, 0.5))
        rows = [common_prelude(rng)]
        for _ in range(length):
            z = rho * z + float(rng.normal(0.0, 0.18))
            observed = z + float(rng.normal(0.0, noise))
            row = nuisance_vector(rng, 0.20)
            row[0] = float(np.tanh(observed))
            row[1] = float(rng.normal(0.0, 0.30))
            rows.append(row)
        next_z = rho * z
        target = float(np.tanh(next_z))
        return Episode(np.stack(rows), target, {"family": self.name, "length": length, "rho": rho, "hidden_target": target})

    def oracle(self, episode: Episode) -> float:
        return float(episode.metadata["hidden_target"])

    def success(self, prediction: float, target: float) -> bool:
        return abs(prediction - target) <= 0.20
