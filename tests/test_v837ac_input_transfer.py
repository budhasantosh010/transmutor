from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837ac.shared_input_transfer import CONDITIONS, ControllerInputTransferY3, exact_t2_projection_from_seed

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/"experiments/v837_primitive_invention/v837ac/config.json").read_text(encoding="utf-8"))
DECISION=json.loads((ROOT/"experiments/v837_primitive_invention/v837ab/diagnostics/decision_state.json").read_text(encoding="utf-8"))


def model(condition:str,family:str="conditional_routing",replicate:int=0):
    coupling=deterministic_int(CONFIG["training"]["coupling_seed_namespace"],CONFIG["training"]["coupling_seed_condition"],replicate)
    projection=deterministic_int(CONFIG["training"]["projection_initialization_namespace"],family,replicate)
    return ControllerInputTransferY3(high_capacity_generic_graph(replicate),condition=condition,coupling_initialization_seed=coupling,projection_seed=projection)


class V837acInputTransferTests(unittest.TestCase):
    def test_v837ac_requires_v837ab_decision_state(self):
        self.assertTrue(DECISION["v837ab_complete"]); self.assertTrue(DECISION["neutral_transfer_allowed"]); self.assertEqual(DECISION["authorized_v837ac_mode"],"TRAINABLE_CONTROLLER_INPUT_FACTORIZATION")
    def test_only_authorized_transfer_mode_executes(self): self.assertEqual(CONFIG["authorized_mode"],"TRAINABLE_CONTROLLER_INPUT_FACTORIZATION")
    def test_no_unauthorized_fallback(self): self.assertEqual(set(CONFIG["conditions"]),CONDITIONS)
    def test_y3_parent_hashes_frozen(self):
        self.assertEqual(CONFIG["v837y_results_sha256"],"e4cb9004c7c68a25d5feb84cb6c25263459a0610a74350f08b3f1d3b999fa7e6")
    def test_exact_t2_projection_reproduces_reference_initialization(self):
        seed=deterministic_int("v837j-primary-init","conditional_routing",0)
        A,a=exact_t2_projection_from_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); layer=torch.nn.Linear(6,6)
        self.assertTrue(torch.equal(A,layer.weight)); self.assertTrue(torch.equal(a,layer.bias))
    def test_one_shared_projection_computed_once_per_timestep(self):
        torch.manual_seed(11); m=model("AC1_controller_input_factorization"); calls=[]; hook=m.shared_input_projection.register_forward_hook(lambda *args: calls.append(1))
        m(torch.randn(2,7,6),torch.tensor([7,6])); hook.remove(); self.assertEqual(len(calls),7)
    def test_global_controller_receives_projected_input(self):
        torch.manual_seed(12); m=model("AC1_controller_input_factorization"); self.assertTrue(m.projection_active); w,b=m.effective_controller_input(); self.assertEqual(tuple(w.shape),(6,)); self.assertEqual(tuple(b.shape),(1,))
    def test_candidate_cells_use_raw_input_when_controller_only(self):
        torch.manual_seed(13); a=model("AC1_controller_input_factorization"); torch.manual_seed(13); f=model("AC1F_folded_control");
        for i in range(10): self.assertTrue(torch.equal(a.base.cell_wx[i],f.base.cell_wx[i]))
    def _equivalence(self):
        torch.manual_seed(21); a=model("AC1_controller_input_factorization"); torch.manual_seed(21); f=model("AC1F_folded_control")
        x=torch.randn(4,8,6); lengths=torch.tensor([8,7,6,5])
        with torch.no_grad(): pa,ta=a(x,lengths,return_trace=True); pf,tf=f(x,lengths,return_trace=True)
        return {"prediction":float((pa-pf).abs().max()),"candidate_input":float((ta.input_terms-tf.input_terms).abs().max()),"gate":float((ta.global_gates-tf.global_gates).abs().max()),"candidate":float((ta.candidate_states-tf.candidate_states).abs().max()),"state":float((ta.states-tf.states).abs().max())}
    def test_ac1_ac1f_candidate_input_terms_equal(self): self.assertLessEqual(self._equivalence()["candidate_input"],1e-6)
    def test_ac1_ac1f_global_gate_equal(self): self.assertLessEqual(self._equivalence()["gate"],1e-6)
    def test_ac1_ac1f_candidate_states_equal(self): self.assertLessEqual(self._equivalence()["candidate"],1e-6)
    def test_ac1_ac1f_next_states_equal(self): self.assertLessEqual(self._equivalence()["state"],1e-6)
    def test_ac1_ac1f_prediction_equal(self): self.assertLessEqual(self._equivalence()["prediction"],1e-6)
    def test_controller_only_has_no_deshared_control(self): self.assertFalse(CONFIG["deshared_control_applicable"]); self.assertNotIn("AC1D",CONFIG["conditions"])
    def test_projection_added_only_to_ac1(self):
        self.assertFalse(model("AC0_y3_parent").projection_active); self.assertTrue(model("AC1_controller_input_factorization").projection_active); self.assertFalse(model("AC1F_folded_control").projection_active)
    def test_projection_adds_42_parameters_and_36_macs(self):
        torch.manual_seed(30); a=model("AC1_controller_input_factorization"); torch.manual_seed(30); f=model("AC1F_folded_control"); self.assertEqual(a.parameter_count()-f.parameter_count(),42); self.assertEqual(a.projection_specific_macs,36)
    def test_science_locks(self):
        self.assertFalse(CONFIG["fresh_audit_consumed"]); self.assertFalse(CONFIG["structural_search_allowed"]); self.assertFalse(CONFIG["primitive_mining_allowed"]); self.assertFalse(CONFIG["v838_started"]); self.assertEqual(CONFIG["primitives_promoted"],0)

if __name__=="__main__": unittest.main()
