from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass
class ResourceAccounting:
    candidate_evaluations: int = 0
    optimizer_steps: int = 0
    environment_steps: int = 0
    examples_processed: int = 0
    mutation_count: int = 0
    model_fits: int = 0
    search_expansions: int = 0
    archive_lookups: int = 0
    primitive_calls: int = 0
    wall_seconds: float = 0.0
    peak_cells: int = 0
    peak_edges: int = 0
    final_cells: int = 0
    final_edges: int = 0
    parameter_count: int = 0
    max_candidate_budget: int = 0
    max_optimizer_step_budget: int = 0

    def merge(self, other: "ResourceAccounting") -> "ResourceAccounting":
        for field_name in asdict(self):
            if field_name in {"peak_cells", "peak_edges"}:
                setattr(self, field_name, max(getattr(self, field_name), getattr(other, field_name)))
            elif field_name in {"final_cells", "final_edges", "parameter_count", "max_candidate_budget", "max_optimizer_step_budget"}:
                setattr(self, field_name, getattr(other, field_name) or getattr(self, field_name))
            else:
                setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class WallTimer:
    def __enter__(self) -> "WallTimer":
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.seconds = time.perf_counter() - self.start
