from __future__ import annotations

from .oracle_macrostate import InstrumentedEpisode


def primary_timestep(ep: InstrumentedEpisode) -> int:
    if ep.family == "conditional_routing": return 1
    if ep.family == "delayed_recall":
        delay=int(ep.latent_variables["delay"]); return 2 + (delay - 1) // 2
    if ep.family == "iterative_state":
        length=int(ep.latent_variables["length"]); j=(length - 1)//2; return 1+j
    if ep.family == "partial_observation":
        length=int(ep.latent_variables["length"]); return 1 + (length - 1)//2
    if ep.family == "variable_composition":
        depth=int(ep.latent_variables["depth"]); return 1 + (depth - 1)//2
    raise KeyError(ep.family)


def routing_window(ep: InstrumentedEpisode) -> list[int]:
    if ep.family == "conditional_routing": return list(range(1, min(len(ep.observations), 4)))
    if ep.family == "delayed_recall": return [primary_timestep(ep)]
    t=primary_timestep(ep); return [t] + ([t+1] if t+1 < len(ep.observations) else [])


def delayed_recall_phase_timesteps(ep: InstrumentedEpisode) -> dict[str,int]:
    if ep.family != "delayed_recall": raise ValueError("delayed recall only")
    delay=int(ep.latent_variables["delay"])
    return {"WRITE":1,"MID_DELAY":2+(delay-1)//2,"PRE_QUERY":1+delay}
