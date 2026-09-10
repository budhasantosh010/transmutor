from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from experiments.v837_primitive_invention.common.task_interface import Episode
from experiments.v837_primitive_invention.v837an.oracle_macrostate import InstrumentedEpisode, instrument_episode


@dataclass
class OperatorCase:
    family: str
    seed: int
    case_id: str
    intervention: str
    phase: str
    horizon: int
    order: int
    base_episode: Episode
    intervened_episode: Episode
    oracle_response: float
    requested_delta: float | None
    actual_delta: float | None
    time_index: int | None
    zero_effect_expected: bool
    interaction_group: str | None = None
    component_labels: tuple[str, str] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family, "seed": self.seed, "case_id": self.case_id,
            "intervention": self.intervention, "phase": self.phase, "horizon": self.horizon,
            "order": self.order, "oracle_response": self.oracle_response,
            "requested_delta": self.requested_delta, "actual_delta": self.actual_delta,
            "time_index": self.time_index, "zero_effect_expected": self.zero_effect_expected,
            "interaction_group": self.interaction_group, "component_labels": list(self.component_labels) if self.component_labels else None,
        }


def _episode(obs: np.ndarray, target: float, family: str, **meta) -> Episode:
    return Episode(np.asarray(obs, dtype=np.float32), float(target), {"family": family, **meta})


def _clip_delta(value: float, requested: float, lo: float, hi: float) -> tuple[float, float]:
    new = float(np.clip(value + requested, lo, hi))
    return new, new - float(value)


def _timing_indices(n: int) -> dict[str, int]:
    if n <= 1:
        return {"EARLY": 0, "MIDDLE": 0, "LATE": 0}
    return {"EARLY": 0, "MIDDLE": (n - 1) // 2, "LATE": n - 1}


def _iterative_target(values: np.ndarray) -> tuple[float, np.ndarray]:
    z = 0.0; states = [0.0]
    for x in values:
        z = 0.65 * z + 0.35 * float(x); states.append(z)
    return float(z), np.asarray(states, dtype=np.float64)


def _composition_target(initial: float, gains: np.ndarray, drives: np.ndarray) -> tuple[float, np.ndarray]:
    z = float(initial); states = [z]
    for g, d in zip(gains, drives):
        z = float(np.tanh(float(g) * z + float(d))); states.append(z)
    return z, np.asarray(states, dtype=np.float64)


def _routing_cases(seed: int, split: str) -> list[OperatorCase]:
    base = instrument_episode("conditional_routing", seed, split)
    c = float(base.causal_inputs["control"]); a = float(base.causal_inputs["payload_a"]); b = float(base.causal_inputs["payload_b"])
    out: list[OperatorCase] = []
    def add(name: str, obs: np.ndarray, control: float, aa: float, bb: float, req: float | None, act: float | None, phase: str, zero: bool = False):
        target = aa if control > 0 else bb
        out.append(OperatorCase("conditional_routing", seed, f"{seed}:{name}:{len(out)}", name, phase, 1, 1,
                                base.as_episode(), _episode(obs, target, "conditional_routing", control=control),
                                float(target - base.target), req, act, None, zero))
    obs = base.observations.copy(); obs[1, 0] = -c
    add("CONTROL_FLIP", obs, -c, a, b, -2*c, -2*c, "CONTROL")
    for req in (-0.40, 0.40):
        aa, act = _clip_delta(a, req, -0.85, 0.85); obs = base.observations.copy(); obs[2, 1] = aa
        add("PAYLOAD_A_PERTURB", obs, c, aa, b, req, act, "PAYLOAD_A", zero=(c < 0))
        bb, actb = _clip_delta(b, req, -0.85, 0.85); obs = base.observations.copy(); obs[3, 2] = bb
        add("PAYLOAD_B_PERTURB", obs, c, a, bb, req, actb, "PAYLOAD_B", zero=(c > 0))
    # Wrong-phase/nuisance control through a legitimate non-semantic observation channel.
    obs = base.observations.copy(); obs[0, 5] = float(obs[0, 5] + 0.50)
    add("WRONG_PHASE_NUISANCE", obs, c, a, b, 0.50, 0.50, "PRELUDE", zero=True)
    return out


def _recall_cases(seed: int, split: str) -> list[OperatorCase]:
    base = instrument_episode("delayed_recall", seed, split); value = float(base.causal_inputs["value"]); delay = int(base.latent_variables["delay"])
    out: list[OperatorCase] = []
    def add(name: str, obs: np.ndarray, target: float, req: float | None, act: float | None, phase: str, zero: bool):
        out.append(OperatorCase("delayed_recall", seed, f"{seed}:{name}:{len(out)}", name, phase, 1, 1,
                                base.as_episode(), _episode(obs, target, "delayed_recall", delay=len(obs)-3),
                                float(target - base.target), req, act, None, zero))
    obs = base.observations.copy(); obs[1, 0] = -value
    add("WRITE_FLIP", obs, -value, -2*value, -2*value, "WRITE", False)
    delay_indices = _timing_indices(delay)
    for phase, j in delay_indices.items():
        t = 2 + j; old = float(base.observations[t, 0]); obs = base.observations.copy(); obs[t, 0] = -old
        add("DISTRACTOR_FLIP", obs, value, -2*old, -2*old, f"{phase}_DELAY", True)
    for req in (-0.30, 0.30):
        obs = base.observations.copy(); old = float(obs[-1, 5]); obs[-1, 5] = old + req
        add("QUERY_NUISANCE", obs, value, req, req, "READ", True)
    if delay > 4:
        obs = np.concatenate([base.observations[:-2], base.observations[-1:]], axis=0)
        add("DELAY_SHORTEN", obs, value, -1.0, -1.0, "HOLD", True)
    # Extend by duplicating an actually observed natural distractor row.
    obs = np.concatenate([base.observations[:-1], base.observations[2:3], base.observations[-1:]], axis=0)
    add("DELAY_EXTEND", obs, value, 1.0, 1.0, "HOLD", True)
    return out


def _iterative_cases(seed: int, split: str, *, reality: bool = False) -> list[OperatorCase]:
    base = instrument_episode("iterative_state", seed, split); x = np.asarray(base.causal_inputs["x_t"], dtype=np.float64)
    out: list[OperatorCase] = []; timings = _timing_indices(len(x)); magnitudes = (-1.0, -0.5, 0.5, 1.0)
    for phase, j in timings.items():
        for req in magnitudes:
            new, act = _clip_delta(float(x[j]), req, -1.0, 1.0)
            if abs(act) < 1e-12:
                continue
            xp = x.copy(); xp[j] = new; _, bp_states = _iterative_target(x); _, cp_states = _iterative_target(xp)
            for h in (1, 2, 4, 8):
                end_input = j + h
                if end_input > len(x):
                    continue
                base_obs = base.observations[:1 + end_input].copy(); cf_obs = base_obs.copy(); cf_obs[1+j, 0] = new
                bt = float(bp_states[end_input]); ct = float(cp_states[end_input])
                oracle = 0.35 * (0.65 ** (h - 1)) * act
                if abs((ct - bt) - oracle) > 1e-10:
                    raise RuntimeError("V837AQ_ITERATIVE_ORACLE_CONSTRUCTION_DRIFT")
                out.append(OperatorCase("iterative_state", seed, f"{seed}:INPUT_PERTURB:{phase}:{req}:{h}", "INPUT_PERTURB", phase, h, 1,
                                        _episode(base_obs, bt, "iterative_state", length=end_input), _episode(cf_obs, ct, "iterative_state", length=end_input),
                                        float(oracle), req, act, j, False))
    # A wrong-phase prelude nuisance change should not alter the abstract state.
    obs = base.observations.copy(); obs[0, 5] = float(obs[0, 5] + 0.50)
    out.append(OperatorCase("iterative_state", seed, f"{seed}:WRONG_PHASE_NUISANCE", "WRONG_PHASE_NUISANCE", "PRELUDE", len(x), 1,
                            base.as_episode(), _episode(obs, base.target, "iterative_state", length=len(x)), 0.0, 0.50, 0.50, 0, True))
    return out


def _composition_cases(seed: int, split: str) -> list[OperatorCase]:
    base = instrument_episode("variable_composition", seed, split)
    initial = float(base.causal_inputs["initial_value"]); gains = np.asarray(base.causal_inputs["gain"], dtype=np.float64); drives = np.asarray(base.causal_inputs["drive"], dtype=np.float64)
    out: list[OperatorCase] = []
    base_final, base_states = _composition_target(initial, gains, drives)
    def add(name: str, obs: np.ndarray, target: float, req: float | None, act: float | None, phase: str, t: int | None, horizon: int = 1, base_ep: Episode | None = None, zero: bool = False):
        out.append(OperatorCase("variable_composition", seed, f"{seed}:{name}:{phase}:{req}:{horizon}:{len(out)}", name, phase, horizon, 1,
                                base_ep or base.as_episode(), _episode(obs, target, "variable_composition", depth=len(obs)-1),
                                float(target - (base_ep.target if base_ep else base_final)), req, act, t, zero))
    for req in (-0.25, 0.25):
        new, act = _clip_delta(initial, req, -0.8, 0.8); tp, _ = _composition_target(new, gains, drives); obs = base.observations.copy(); obs[1, 0] = new
        add("INITIAL_VALUE_PERTURB", obs, tp, req, act, "INITIAL", 0)
    for phase, j in _timing_indices(len(gains)).items():
        for kind, arr, col, lim, mags in (("GAIN_PERTURB", gains, 1, (0.45, 0.95), (-0.15, 0.15)), ("DRIVE_PERTURB", drives, 2, (-0.35, 0.35), (-0.15, 0.15))):
            for req in mags:
                new, act = _clip_delta(float(arr[j]), req, lim[0], lim[1])
                if abs(act) < 1e-12: continue
                gp = gains.copy(); dp = drives.copy()
                if kind == "GAIN_PERTURB": gp[j] = new
                else: dp[j] = new
                for h in (1, 2):
                    end = j + h
                    if end > len(gains): continue
                    btarget = float(base_states[end]); target, cstates = _composition_target(initial, gp[:end], dp[:end]); target = float(target)
                    bobs = base.observations[:1+end].copy(); obs = bobs.copy(); obs[1+j, col] = new
                    bep = _episode(bobs, btarget, "variable_composition", depth=end)
                    add(kind, obs, target, req, act, phase, j, h, bep)
    obs = base.observations.copy(); obs[0, 5] = float(obs[0, 5] + 0.50)
    add("WRONG_PHASE_NUISANCE", obs, base_final, 0.50, 0.50, "PRELUDE", None, 1, zero=True)
    return out


def build_first_order_cases(family: str, seed_values: list[int], split: str = "development", *, reality: bool = False) -> list[OperatorCase]:
    fn = {"conditional_routing": _routing_cases, "delayed_recall": _recall_cases, "iterative_state": _iterative_cases, "variable_composition": _composition_cases}[family]
    rows: list[OperatorCase] = []
    for s in seed_values:
        if family == "iterative_state": rows.extend(fn(int(s), split, reality=reality))
        else: rows.extend(fn(int(s), split))
    return rows


def build_compound_cases(family: str, seed_values: list[int], split: str = "development") -> list[OperatorCase]:
    out: list[OperatorCase] = []
    for seed in seed_values:
        base = instrument_episode(family, int(seed), split)
        if family == "conditional_routing":
            c=float(base.causal_inputs["control"]); a=float(base.causal_inputs["payload_a"]); b=float(base.causal_inputs["payload_b"])
            aa, act=_clip_delta(a, 0.40, -0.85, 0.85); obs=base.observations.copy();obs[1,0]=-c;obs[2,1]=aa;target=aa if -c>0 else b
            out.append(OperatorCase(family,seed,f"{seed}:CONTROLxA","CONTROL_X_PAYLOAD_A","CONTROL+PAYLOAD_A",1,2,base.as_episode(),_episode(obs,target,family,control=-c),float(target-base.target),0.40,act,None,False,f"{seed}:CONTROLxA",("CONTROL_FLIP","PAYLOAD_A_PERTURB")))
        elif family == "delayed_recall":
            value=float(base.causal_inputs["value"]); delay=int(base.latent_variables["delay"]); j=(delay-1)//2;t=2+j;obs=base.observations.copy();obs[1,0]=-value;old=float(obs[t,0]);obs[t,0]=-old
            out.append(OperatorCase(family,seed,f"{seed}:WRITExDIST","WRITE_X_DISTRACTOR","WRITE+MID_DELAY",1,2,base.as_episode(),_episode(obs,-value,family,delay=delay),float(-value-base.target),None,None,t,False,f"{seed}:WRITExDIST",("WRITE_FLIP","DISTRACTOR_FLIP")))
        elif family == "iterative_state":
            x=np.asarray(base.causal_inputs["x_t"],dtype=np.float64); js=sorted(set([0,max(0,len(x)//2)])); xp=x.copy(); deltas=[];obs=base.observations.copy()
            for j in js:
                new=-float(xp[j]);d=new-float(xp[j]);xp[j]=new;obs[1+j,0]=new;deltas.append((j,d))
            target,_=_iterative_target(xp)
            out.append(OperatorCase(family,seed,f"{seed}:TWO_IMPULSES","TWO_INPUT_IMPULSES","EARLY+MIDDLE",len(x),2,base.as_episode(),_episode(obs,target,family,length=len(x)),float(target-base.target),None,None,None,False,f"{seed}:TWO_IMPULSES",("INPUT_PERTURB_EARLY","INPUT_PERTURB_MIDDLE")))
        elif family == "variable_composition":
            initial=float(base.causal_inputs["initial_value"]);g=np.asarray(base.causal_inputs["gain"],dtype=np.float64);d=np.asarray(base.causal_inputs["drive"],dtype=np.float64);j=0
            gn,ga=_clip_delta(float(g[j]),0.15,0.45,0.95);dn,da=_clip_delta(float(d[j]),0.15,-0.35,0.35);gp=g.copy();dp=d.copy();gp[j]=gn;dp[j]=dn;target,_=_composition_target(initial,gp,dp);obs=base.observations.copy();obs[1+j,1]=gn;obs[1+j,2]=dn
            out.append(OperatorCase(family,seed,f"{seed}:GAINxDRIVE","GAIN_X_DRIVE","EARLY",len(g),2,base.as_episode(),_episode(obs,target,family,depth=len(g)),float(target-base.target),None,None,j,False,f"{seed}:GAINxDRIVE",("GAIN_PERTURB","DRIVE_PERTURB")))
        else: raise KeyError(family)
    return out


def intervention_library_manifest() -> dict:
    """Materialize the already-frozen coordinate-free semantic intervention inventory."""
    return {
        "version": "V837aq",
        "coordinate_free": True,
        "hidden_state_set_used": False,
        "families": {
            "conditional_routing": {
                "first_order": ["CONTROL_FLIP", "PAYLOAD_A_PERTURB", "PAYLOAD_B_PERTURB", "WRONG_PHASE_NUISANCE"],
                "second_order": ["CONTROL_X_PAYLOAD_A"],
                "program_word": "SELECT o LOAD o CONTROL",
            },
            "delayed_recall": {
                "first_order": ["WRITE_FLIP", "DISTRACTOR_FLIP", "QUERY_NUISANCE", "DELAY_SHORTEN", "DELAY_EXTEND"],
                "second_order": ["WRITE_X_DISTRACTOR"],
                "program_word": "READ o HOLD^n o WRITE",
            },
            "iterative_state": {
                "first_order": ["INPUT_PERTURB", "WRONG_PHASE_NUISANCE"],
                "second_order": ["TWO_INPUT_IMPULSES"],
                "program_word": "U(x2) o U(x1)",
                "oracle_kernel": "0.35*(0.65**(h-1))*delta",
            },
            "variable_composition": {
                "first_order": ["INITIAL_VALUE_PERTURB", "GAIN_PERTURB", "DRIVE_PERTURB", "WRONG_PHASE_NUISANCE"],
                "second_order": ["GAIN_X_DRIVE"],
                "program_word": "A(g2,d2) o A(g1,d1)",
            },
        },
        "third_order_forbidden": True,
        "fresh_audit_consumed": False,
    }
