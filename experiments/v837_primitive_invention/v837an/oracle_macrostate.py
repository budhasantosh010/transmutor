from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.v837_primitive_invention.common.task_interface import Episode, common_prelude, nuisance_vector
from experiments.v837_primitive_invention.tasks import task_by_name

from .authorization import PRIMARY_PROBES
from .utils import HERE, write_json


@dataclass
class InstrumentedEpisode:
    family: str
    seed: int
    split: str
    observations: np.ndarray
    target: float
    phase_labels: list[str]
    macrostate_traces: dict[str, np.ndarray]
    causal_inputs: dict[str, Any]
    latent_variables: dict[str, Any]
    metadata: dict[str, Any]

    def as_episode(self) -> Episode:
        return Episode(self.observations.copy(), float(self.target), dict(self.metadata))


def _routing(seed: int, split: str) -> InstrumentedEpisode:
    rng = np.random.default_rng(seed)
    control = float(rng.choice([-1.0, 1.0]))
    scale = 0.95 if split == "fresh_audit" else 0.85
    a = float(rng.uniform(-scale, scale)); b = float(rng.uniform(-scale, scale))
    rows = [common_prelude(rng)]
    r1 = nuisance_vector(rng, .25); r1[0] = control; rows.append(r1)
    r2 = nuisance_vector(rng, .25); r2[1] = a; rows.append(r2)
    r3 = nuisance_vector(rng, .25); r3[2] = b; rows.append(r3)
    extra = 2 if split == "fresh_audit" else int(rng.integers(0, 2))
    for _ in range(extra): rows.append(nuisance_vector(rng, .35))
    target = a if control > 0 else b
    T = len(rows)
    control_trace = np.full(T, np.nan, dtype=np.float64); control_trace[1:] = control
    selected = np.full(T, np.nan, dtype=np.float64); selected[3:] = target
    phases = ["PRELUDE", "CONTROL", "PAYLOAD_A", "PAYLOAD_B"] + ["POST_PAYLOAD"] * extra
    return InstrumentedEpisode("conditional_routing", seed, split, np.stack(rows), target, phases,
        {"ROUTING_CONTROL_STATE": control_trace, "ROUTING_SELECTED_VALUE": selected},
        {"control": control, "payload_a": a, "payload_b": b, "selected_payload": target}, {},
        {"family": "conditional_routing", "control": control})


def _recall(seed: int, split: str) -> InstrumentedEpisode:
    rng = np.random.default_rng(seed)
    delay = int(rng.integers(13, 25)) if split == "fresh_audit" else int(rng.integers(4, 13))
    value = float(rng.choice([-1.0, 1.0]))
    rows = [common_prelude(rng)]
    present = nuisance_vector(rng, .30); present[0] = value; rows.append(present)
    for _ in range(delay):
        d = nuisance_vector(rng, .45); d[0] = float(rng.choice([-1.0, 1.0])) * .35; rows.append(d)
    query = nuisance_vector(rng, .30); query[5] = float(rng.normal(0.0, .30)); rows.append(query)
    T = len(rows); memory = np.full(T, np.nan, dtype=np.float64); memory[1:] = value
    phases = ["PRELUDE", "WRITE"] + ["DELAY"] * delay + ["READ"]
    return InstrumentedEpisode("delayed_recall", seed, split, np.stack(rows), value, phases,
        {"RECALL_MAINTAIN": memory, "RECALL_WRITE": memory.copy(), "RECALL_READ": memory.copy()},
        {"value": value}, {"delay": delay}, {"family": "delayed_recall", "delay": delay})


def _iterative(seed: int, split: str) -> InstrumentedEpisode:
    rng = np.random.default_rng(seed)
    length = int(rng.integers(11, 31)) if split == "fresh_audit" else int(rng.integers(3, 11))
    values = rng.choice([-1.0, 1.0], size=length).astype(np.float32)
    state = 0.0; states = [0.0]; before = []; after = []; rows = [common_prelude(rng)]
    for value in values:
        before.append(state); state = .65 * state + .35 * float(value); after.append(state); states.append(state)
        row = nuisance_vector(rng, .22); row[0] = float(value); rows.append(row)
    return InstrumentedEpisode("iterative_state", seed, split, np.stack(rows), state, ["PRELUDE"] + ["ITERATE"] * length,
        {"ITERATIVE_RUNNING_STATE": np.asarray(states, dtype=np.float64)},
        {"x_t": values.astype(np.float64), "s_before": np.asarray(before), "s_after": np.asarray(after)},
        {"length": length}, {"family": "iterative_state", "length": length})


def _partial(seed: int, split: str) -> InstrumentedEpisode:
    rng = np.random.default_rng(seed)
    length = int(rng.integers(10, 19)) if split == "fresh_audit" else int(rng.integers(5, 11))
    rho = float(rng.uniform(.72, .94)); noise_scale = .28 if split == "fresh_audit" else .20
    z0 = float(rng.normal(0.0, .5)); z = z0; rows = [common_prelude(rng)]
    innovations=[]; obs_noise=[]; z_before=[]; z_after=[]; observed_raw=[]
    for _ in range(length):
        z_before.append(z); eps=float(rng.normal(0.0,.18)); innovations.append(eps); z=rho*z+eps; z_after.append(z)
        eta=float(rng.normal(0.0,noise_scale)); obs_noise.append(eta); observed=z+eta; observed_raw.append(observed)
        row=nuisance_vector(rng,.20); row[0]=float(np.tanh(observed)); row[1]=float(rng.normal(0.0,.30)); rows.append(row)
    target=float(np.tanh(rho*z))
    z_trace=np.asarray([z0]+z_after,dtype=np.float64)
    pred_trace=np.asarray([np.tanh(rho*z0)]+[np.tanh(rho*v) for v in z_after],dtype=np.float64)
    return InstrumentedEpisode("partial_observation",seed,split,np.stack(rows),target,["PRELUDE"]+["OBSERVE"]*length,
        {"PARTIAL_LATENT_Z":z_trace,"PARTIAL_PREDICTIVE_MEAN":pred_trace},
        {"rho":rho,"innovation":np.asarray(innovations),"observation_noise":np.asarray(obs_noise)},
        {"initial_z0":z0,"z_before":np.asarray(z_before),"z_after":np.asarray(z_after),"observed_raw":np.asarray(observed_raw),"noise_scale":noise_scale,"length":length},
        {"family":"partial_observation","length":length,"rho":rho,"hidden_target":target})


def _composition(seed: int, split: str) -> InstrumentedEpisode:
    rng=np.random.default_rng(seed)
    depth=int(rng.integers(4,7)) if split=="fresh_audit" else int(rng.integers(1,4))
    initial=float(rng.uniform(-.8,.8)); value=initial; rows=[common_prelude(rng)]
    gains=[];drives=[];before=[];after=[];running=[initial]
    for index in range(depth):
        gain=float(rng.uniform(.45,.95));drive=float(rng.uniform(-.35,.35));gains.append(gain);drives.append(drive);before.append(value)
        value=float(np.tanh(gain*value+drive));after.append(value);running.append(value)
        row=nuisance_vector(rng,.18);row[0]=initial if index==0 else float(rng.normal(0.0,.18));row[1]=gain;row[2]=drive;rows.append(row)
    return InstrumentedEpisode("variable_composition",seed,split,np.stack(rows),value,["PRELUDE"]+["COMPOSE"]*depth,
        {"COMPOSITION_RUNNING_STATE":np.asarray(running,dtype=np.float64)},
        {"initial_value":initial,"gain":np.asarray(gains),"drive":np.asarray(drives),"v_before":np.asarray(before),"v_after":np.asarray(after)},
        {"depth":depth},{"family":"variable_composition","depth":depth})


def instrument_episode(family: str, seed: int, split: str = "development") -> InstrumentedEpisode:
    fn={"conditional_routing":_routing,"delayed_recall":_recall,"iterative_state":_iterative,"partial_observation":_partial,"variable_composition":_composition}.get(family)
    if fn is None: raise KeyError(family)
    return fn(int(seed),split)


def primary_value(episode: InstrumentedEpisode, timestep: int) -> float:
    return float(episode.macrostate_traces[PRIMARY_PROBES[episode.family]][int(timestep)])


def verify_oracle_instrumentation() -> dict:
    families=sorted(PRIMARY_PROBES); rows=[]; max_obs=0.0; max_target=0.0; exact_arrays=0
    for family in families:
        task=task_by_name(family)
        for seed in range(10000,10512):
            got=instrument_episode(family,seed,"development"); expected=task.generate(seed,"development")
            if got.observations.shape!=expected.observations.shape or len(got.phase_labels)!=len(expected.observations): raise RuntimeError("ORACLE_INSTRUMENTATION_MISMATCH")
            delta=float(np.max(np.abs(got.observations-expected.observations))) if got.observations.size else 0.0
            tdelta=abs(float(got.target)-float(expected.target)); max_obs=max(max_obs,delta);max_target=max(max_target,tdelta)
            if np.array_equal(got.observations,expected.observations): exact_arrays+=1
            if delta>1e-7 or tdelta>1e-12: raise RuntimeError(f"ORACLE_INSTRUMENTATION_MISMATCH:{family}:{seed}:{delta}:{tdelta}")
        rows.append({"family":family,"seeds":512,"primary_probe":PRIMARY_PROBES[family]})
    payload={"version":"V837an","pass":True,"families":rows,"episodes_verified":len(families)*512,"exact_observation_arrays":exact_arrays,"max_observation_abs_error":max_obs,"max_target_abs_error":max_target,"fresh_audit_consumed":False}
    write_json(HERE/"raw/oracle_instrumentation_verification.json",payload);write_json(HERE/"diagnostics/oracle_equivalence.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(verify_oracle_instrumentation(),indent=2))
