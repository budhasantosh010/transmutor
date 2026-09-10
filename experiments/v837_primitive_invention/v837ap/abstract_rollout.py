from __future__ import annotations
import numpy as np

def next_state(family,z,inputs):
    if family in {"conditional_routing","delayed_recall"}:return float(z)
    if family=="iterative_state":return .65*float(z)+.35*float(inputs["x"])
    if family=="variable_composition":return float(np.tanh(float(inputs["gain"])*float(z)+float(inputs["drive"])))
    raise KeyError(family)
def rollout_from_setpoint(family,episode,timestep,z_star):
    z=float(z_star);traj=[z]
    if family=="conditional_routing":
        a=float(episode.causal_inputs["payload_a"]);b=float(episode.causal_inputs["payload_b"]);return {"trajectory":traj+[z]*max(0,len(episode.observations)-timestep-1),"final_target":a if z>0 else b,"decision":"A" if z>0 else "B"}
    if family=="delayed_recall":return {"trajectory":traj+[z]*max(0,len(episode.observations)-timestep-1),"final_target":z,"decision":"READ"}
    if family=="iterative_state":
        xs=np.asarray(episode.causal_inputs["x_t"],dtype=np.float64)
        for x in xs[int(timestep):]:z=next_state(family,z,{"x":x});traj.append(z)
        return {"trajectory":traj,"final_target":z}
    if family=="variable_composition":
        gs=np.asarray(episode.causal_inputs["gain"],dtype=np.float64);ds=np.asarray(episode.causal_inputs["drive"],dtype=np.float64)
        for g,d in zip(gs[int(timestep):],ds[int(timestep):]):z=next_state(family,z,{"gain":g,"drive":d});traj.append(z)
        return {"trajectory":traj,"final_target":z}
    raise KeyError(family)
