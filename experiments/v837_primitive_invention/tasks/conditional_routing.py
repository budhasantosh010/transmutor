from __future__ import annotations

import numpy as np

from ..common.task_interface import Episode, common_prelude, nuisance_vector


class ConditionalRoutingTask:
    name = "conditional_routing"

    def generate(self, seed: int, split: str) -> Episode:
        rng = np.random.default_rng(seed)
        control = float(rng.choice([-1.0, 1.0]))
        scale = 0.95 if split == "fresh_audit" else 0.85
        payload_a = float(rng.uniform(-scale, scale))
        payload_b = float(rng.uniform(-scale, scale))
        rows = [common_prelude(rng)]
        r1 = nuisance_vector(rng, 0.25); r1[0] = control; rows.append(r1)
        r2 = nuisance_vector(rng, 0.25); r2[1] = payload_a; rows.append(r2)
        r3 = nuisance_vector(rng, 0.25); r3[2] = payload_b; rows.append(r3)
        extra = 2 if split == "fresh_audit" else int(rng.integers(0, 2))
        for _ in range(extra):
            rows.append(nuisance_vector(rng, 0.35))
        target = payload_a if control > 0 else payload_b
        return Episode(np.stack(rows), target, {"family": self.name, "control": control})

    def oracle(self, episode: Episode) -> float:
        return float(episode.target)

    def success(self, prediction: float, target: float) -> bool:
        return abs(prediction - target) <= 0.25
