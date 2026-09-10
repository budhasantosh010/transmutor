from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalPrimitiveIR:
    family: str
    semantic_variable: str
    semantic_dimension: int
    phase_machine: tuple[str, ...]
    canonical_inputs: tuple[str, ...]
    transition_law: str
    output_law: str
    valid_semantic_range: Any

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phase_machine"] = list(self.phase_machine)
        d["canonical_inputs"] = list(self.canonical_inputs)
        return d


IR = {
    "conditional_routing": CanonicalPrimitiveIR(
        "conditional_routing", "control", 1,
        ("CONTROL_WRITE", "PAYLOAD_A", "PAYLOAD_B", "DECIDE"),
        ("control", "payload_a", "payload_b"),
        "CONTROL_WRITE: z=observed_control; PAYLOAD_A/PAYLOAD_B: z_next=z",
        "SELECT(control,A,B): A if z>0 else B", [-1.0, 1.0],
    ),
    "delayed_recall": CanonicalPrimitiveIR(
        "delayed_recall", "remembered_value", 1,
        ("WRITE", "DELAY", "READ"), ("value",),
        "WRITE: z=presented_value; DELAY: z_next=z", "READ: y=z", [-1.0, 1.0],
    ),
    "iterative_state": CanonicalPrimitiveIR(
        "iterative_state", "iterative_running_state", 1,
        ("EARLY", "MIDDLE", "LATE"), ("x",),
        "z_next=0.65*z+0.35*x", "benchmark final output is the final running state", "DISCOVERY_P05_P95",
    ),
    "variable_composition": CanonicalPrimitiveIR(
        "variable_composition", "running_composition_state", 1,
        ("EARLY", "MIDDLE", "LATE"), ("gain", "drive"),
        "z_next=tanh(g*z+d)", "benchmark final output is the final composition state", [-1.0, 1.0],
    ),
}


def canonical_ir(family: str) -> CanonicalPrimitiveIR:
    return IR[family]


def all_ir_dicts() -> dict[str, dict]:
    return {k: v.to_dict() for k, v in IR.items()}
