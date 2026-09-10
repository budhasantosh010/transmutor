from __future__ import annotations

BACKEND_ORDER = ("B0_GLOBAL_K1", "B1_PHASE_GAUGE_K1", "B2_PHASE_K1")
PHASES = {
    "conditional_routing": ("POST_CONTROL", "POST_PAYLOAD_A", "POST_PAYLOAD_B"),
    "delayed_recall": ("POST_WRITE", "MID_DELAY", "PRE_READ"),
    "iterative_state": ("EARLY", "MIDDLE", "LATE"),
    "variable_composition": ("EARLY", "MIDDLE", "LATE"),
}


def phase_timesteps(episode) -> dict[str, list[int]]:
    family = episode.family
    if family == "conditional_routing":
        return {"POST_CONTROL": [1], "POST_PAYLOAD_A": [2], "POST_PAYLOAD_B": [3]}
    if family == "delayed_recall":
        delay = int(episode.latent_variables["delay"])
        return {"POST_WRITE": [1], "MID_DELAY": [2 + (delay - 1) // 2], "PRE_READ": [1 + delay]}
    if family in {"iterative_state", "variable_composition"}:
        count = int(episode.latent_variables["length"] if family == "iterative_state" else episode.latent_variables["depth"])
        out = {"EARLY": [], "MIDDLE": [], "LATE": []}
        for j in range(count):
            frac = (j + 0.5) / max(count, 1)
            phase = "EARLY" if frac < 1 / 3 else ("MIDDLE" if frac < 2 / 3 else "LATE")
            out[phase].append(1 + j)
        return out
    raise KeyError(family)


def representative_phase_timesteps(episode) -> dict[str, int]:
    out = {}
    for phase, times in phase_timesteps(episode).items():
        if times:
            out[phase] = times[len(times) // 2]
    return out


def minimum_complexity_winner(pass_by_variant: dict[str, bool]):
    for variant in BACKEND_ORDER:
        if pass_by_variant.get(variant):
            return variant
    return None
