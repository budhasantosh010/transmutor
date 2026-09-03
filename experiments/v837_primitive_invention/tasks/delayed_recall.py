from __future__ import annotations

import numpy as np

from ..common.task_interface import Episode, OBS_DIM, common_prelude, nuisance_vector


class DelayedRecallTask:
    name = "delayed_recall"

    def generate(self, seed: int, split: str) -> Episode:
        rng = np.random.default_rng(seed)
        delay = int(rng.integers(13, 25)) if split == "fresh_audit" else int(rng.integers(4, 13))
        value = float(rng.choice([-1.0, 1.0]))
        rows = [common_prelude(rng)]
        present = nuisance_vector(rng, 0.30)
        present[0] = value
        rows.append(present)
        for _ in range(delay):
            distractor = nuisance_vector(rng, 0.45)
            distractor[0] = float(rng.choice([-1.0, 1.0])) * 0.35
            rows.append(distractor)
        query = nuisance_vector(rng, 0.30)
        query[5] = float(rng.normal(0.0, 0.30))
        rows.append(query)
        return Episode(np.stack(rows), value, {"family": self.name, "delay": delay})

    def oracle(self, episode: Episode) -> float:
        return float(episode.target)

    def success(self, prediction: float, target: float) -> bool:
        return abs(prediction - target) <= 0.35
