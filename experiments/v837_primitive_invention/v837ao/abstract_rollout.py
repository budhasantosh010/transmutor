from __future__ import annotations

import numpy as np


def next_canonical_state(family: str, z: float, inputs: dict) -> float:
    if family in {"conditional_routing", "delayed_recall"}:
        return float(z)
    if family == "iterative_state":
        return 0.65 * float(z) + 0.35 * float(inputs["x"])
    if family == "variable_composition":
        return float(np.tanh(float(inputs["gain"]) * float(z) + float(inputs["drive"])))
    raise KeyError(family)


def rollout_from_setpoint(family: str, episode, timestep: int, z_star: float) -> dict:
    z = float(z_star)
    traj = [z]
    if family == "conditional_routing":
        a = float(episode.causal_inputs["payload_a"])
        b = float(episode.causal_inputs["payload_b"])
        target = a if z > 0 else b
        remaining = max(0, len(episode.observations) - int(timestep) - 1)
        traj.extend([z] * remaining)
        return {"trajectory": traj, "final_target": target, "decision": "A" if z > 0 else "B"}
    if family == "delayed_recall":
        remaining = max(0, len(episode.observations) - int(timestep) - 1)
        traj.extend([z] * remaining)
        return {"trajectory": traj, "final_target": z, "decision": "READ"}
    if family == "iterative_state":
        xs = np.asarray(episode.causal_inputs["x_t"], dtype=np.float64)
        for x in xs[int(timestep):]:
            z = next_canonical_state(family, z, {"x": float(x)})
            traj.append(z)
        return {"trajectory": traj, "final_target": z}
    if family == "variable_composition":
        gains = np.asarray(episode.causal_inputs["gain"], dtype=np.float64)
        drives = np.asarray(episode.causal_inputs["drive"], dtype=np.float64)
        for g, d in zip(gains[int(timestep):], drives[int(timestep):]):
            z = next_canonical_state(family, z, {"gain": float(g), "drive": float(d)})
            traj.append(z)
        return {"trajectory": traj, "final_target": z}
    raise KeyError(family)
