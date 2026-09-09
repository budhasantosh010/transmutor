from __future__ import annotations

import json

from .utils import HERE, read_json, write_json


class ContextAlignedPrimitiveTransplant:
    """Gate-owned V837am transplant wrapper.

    The cross-organism context-conditioned runtime is only scientifically
    reachable when an AM-A hypothesis survives selection, meta-confirmation,
    final development confirmation, and final validation.  The frozen V837am
    execution must not instantiate it earlier.  Identity self-transplant is
    deliberately just the immutable recipient model.
    """

    def __init__(self, recipient_model, selected_config: dict | None = None):
        self.recipient_model = recipient_model
        self.selected_config = selected_config or {}

    def identity_self(self, observations, lengths, return_trace=False):
        return self.recipient_model(observations, lengths, return_trace=return_trace)

    def __call__(self, observations, lengths, *, return_trace=False):
        if not self.selected_config:
            return self.identity_self(observations, lengths, return_trace=return_trace)
        raise RuntimeError("V837AM_CONTEXT_CLOSED_LOOP_CROSS_DONOR_REQUIRES_AUTHORIZED_AM_A_FINAL_PASS")


def run_closed_loop():
    final_path = HERE / "raw/final_validation.json"
    frozen_path = HERE / "raw/selected_final_hypothesis.json"
    final = read_json(final_path) if final_path.is_file() else {}
    frozen = read_json(frozen_path) if frozen_path.is_file() else {}
    selected = frozen.get("selected")
    if not final.get("run") or not final.get("pass"):
        payload = {
            "version": "V837am",
            "stage": "CLOSED_LOOP",
            "run": False,
            "pass": False,
            "reason": "FINAL_VALIDATION_PASS_REQUIRED",
            "adapter_cheating_detected": False,
        }
    elif selected is None or selected.get("branch") != "AM-A":
        payload = {
            "version": "V837am",
            "stage": "CLOSED_LOOP",
            "run": False,
            "pass": False,
            "reason": "ONLY_AM_A_AUTHORIZES_IMMEDIATE_CLOSED_LOOP",
            "selected_branch": None if selected is None else selected.get("branch"),
            "adapter_cheating_detected": False,
        }
    else:
        # This branch is intentionally fail-closed until the frozen AM-A gates
        # authorize it.  With the immutable V837am source evidence AM-A has no
        # SELECT_PASS, so reaching here would indicate source/result drift.
        raise RuntimeError("V837AM_UNEXPECTED_AM_A_FINAL_PASS_RUNTIME_DRIFT")
    write_json(HERE / "raw/closed_loop_results.json", payload)
    write_json(HERE / "diagnostics/adapter_cheating_controls.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run_closed_loop(), indent=2))
