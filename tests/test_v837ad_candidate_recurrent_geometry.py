from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.v837ad.candidate_recurrent_geometry import (
    CONDITION_SPECS,
    H40,
    SPARSE_TOPOLOGIES,
    CandidateGeometrySpec,
    CandidateRecurrentGeometryGRU,
    block_diagonal_mask,
    candidate_mask,
    dense_mask,
    degree4_global_sparse_mask,
    mask_integrity,
    raw_initialization_signature,
)
from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import DynamicGranularityGRU

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837ad"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


class V837adCandidateRecurrentGeometryTests(unittest.TestCase):
    def test_dense_mask_all_ones(self):
        self.assertTrue(torch.equal(dense_mask(40), torch.ones(40, 40)))

    def test_2x20_mask_exact(self):
        m = block_diagonal_mask(40, 20)
        self.assertEqual(int(m.sum()), 800)
        self.assertTrue(torch.all(m[:20, :20] == 1) and torch.all(m[20:, 20:] == 1))
        self.assertTrue(torch.all(m[:20, 20:] == 0) and torch.all(m[20:, :20] == 0))

    def test_5x8_mask_exact(self):
        m = block_diagonal_mask(40, 8)
        self.assertEqual(int(m.sum()), 320)
        for i in range(5): self.assertTrue(torch.all(m[i*8:(i+1)*8, i*8:(i+1)*8] == 1))

    def test_10x4_mask_exact(self):
        m = block_diagonal_mask(40, 4)
        self.assertEqual(int(m.sum()), 160)
        for i in range(10): self.assertTrue(torch.all(m[i*4:(i+1)*4, i*4:(i+1)*4] == 1))

    def test_ad4_has_160_active_weights(self):
        self.assertEqual(int(candidate_mask(CONDITION_SPECS["AD4_H40_10x4"]).sum()), 160)

    def test_ad4s_has_160_active_weights(self):
        for sid in SPARSE_TOPOLOGIES:
            self.assertEqual(int(degree4_global_sparse_mask(sid).sum()), 160)

    def test_ad4s_four_inputs_per_output(self):
        for sid in SPARSE_TOPOLOGIES:
            self.assertEqual(set(int(v) for v in degree4_global_sparse_mask(sid).sum(1).tolist()), {4})

    def test_ad4s_four_outputs_per_input(self):
        for sid in SPARSE_TOPOLOGIES:
            self.assertEqual(set(int(v) for v in degree4_global_sparse_mask(sid).sum(0).tolist()), {4})

    def test_ad4s_no_same_block_edges(self):
        for sid in SPARSE_TOPOLOGIES:
            self.assertEqual(mask_integrity(CONDITION_SPECS[f"AD4S_{sid}"])["same_logical_4d_block_edges"], 0)

    def test_ad4s_strongly_connected(self):
        for sid in SPARSE_TOPOLOGIES:
            self.assertTrue(mask_integrity(CONDITION_SPECS[f"AD4S_{sid}"])["strongly_connected"])

    def test_all_sparse_topologies_unique(self):
        masks = [degree4_global_sparse_mask(sid) for sid in SPARSE_TOPOLOGIES]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                self.assertFalse(torch.equal(masks[i], masks[j]))

    def test_masks_task_independent(self):
        self.assertEqual(set(SPARSE_TOPOLOGIES), {"S0", "S1", "S2", "S3", "S4"})
        self.assertNotIn("family", json.dumps(SPARSE_TOPOLOGIES).lower())
        self.assertNotIn("seed", json.dumps(SPARSE_TOPOLOGIES).lower())

    def test_masks_nontrainable(self):
        torch.manual_seed(1); model = CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD4S_S0"])
        self.assertNotIn("candidate_mask", dict(model.named_parameters()))
        self.assertIn("candidate_mask", dict(model.named_buffers()))
        self.assertFalse(model.candidate_mask.requires_grad)

    def test_reject_arbitrary_mask_mode(self):
        with self.assertRaises(ValueError):
            CandidateGeometrySpec("bad", 40, "arbitrary").validate()

    def test_ad0_matches_t2_equation(self):
        torch.manual_seed(1234); ref = DynamicGranularityGRU(13, 6, condition="T2_scalarized_update_no_reset")
        torch.manual_seed(1234); model = CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD0_H13_dense"])
        torch.manual_seed(12); x = torch.randn(9, 8, 6); lengths = torch.tensor([8,8,8,7,7,6,5,4,3])
        p0, t0 = ref(x, lengths, return_trace=True); p1, t1 = model(x, lengths, return_trace=True)
        self.assertLessEqual(float(torch.max(torch.abs(p0-p1))), 1e-6)
        self.assertLessEqual(float(torch.max(torch.abs(t0.states-t1.states))), 1e-6)
        self.assertLessEqual(float(torch.max(torch.abs(t0.candidates-t1.candidates))), 1e-6)
        self.assertLessEqual(float(torch.max(torch.abs(t0.updates-t1.updates))), 1e-6)

    def test_candidate_mask_affects_candidate_hidden_slice_only(self):
        seed=456
        torch.manual_seed(seed); dense = CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD1_H40_dense"])
        torch.manual_seed(seed); block = CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD4_H40_10x4"])
        a=raw_initialization_signature(dense); b=raw_initialization_signature(block)
        for key in a: self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertFalse(torch.equal(dense.candidate_mask, block.candidate_mask))
        self.assertTrue(torch.equal(dense.update_weight(), block.update_weight()))

    def test_update_hidden_slice_unchanged(self):
        seed=457
        torch.manual_seed(seed); a=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD1_H40_dense"])
        torch.manual_seed(seed); b=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD4S_S0"])
        self.assertTrue(torch.equal(a.update_weight(), b.update_weight()))
        self.assertTrue(torch.all(a.candidate_mask == 1)); self.assertEqual(int(b.candidate_mask.sum()), 160)

    def test_input_projection_unchanged(self):
        seed=458
        models=[]
        for name in ["AD1_H40_dense","AD2_H40_2x20","AD3_H40_5x8","AD4_H40_10x4","AD4S_S0"]:
            torch.manual_seed(seed); models.append(CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS[name]))
        for m in models[1:]:
            self.assertTrue(torch.equal(models[0].input_projection_weight,m.input_projection_weight))
            self.assertTrue(torch.equal(models[0].input_projection_bias,m.input_projection_bias))

    def test_reset_off_all_conditions(self):
        x=torch.randn(3,4,6)
        for name in ["AD0_H13_dense","AD1_H40_dense","AD2_H40_2x20","AD3_H40_5x8","AD4_H40_10x4","AD4S_S0"]:
            torch.manual_seed(2); m=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS[name]); _,t=m(x,return_trace=True)
            self.assertTrue(torch.all(t.resets == 1), name)

    def test_post_sigmoid_scalarization_unchanged(self):
        torch.manual_seed(3); m=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD1_H40_dense"]); _,t=m(torch.randn(3,5,6),return_trace=True)
        self.assertLessEqual(float(torch.max(torch.abs(t.updates - t.updates[..., :1]))), 1e-7)

    def test_carry_equation_unchanged(self):
        torch.manual_seed(4); m=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD1_H40_dense"]); _,t=m(torch.randn(2,1,6),return_trace=True)
        expected=(1.0-t.updates[:,0])*t.candidates[:,0]
        self.assertLessEqual(float(torch.max(torch.abs(t.states[:,0]-expected))),1e-7)

    def test_h40_raw_initialization_paired(self):
        seed=9876; names=["AD1_H40_dense","AD2_H40_2x20","AD3_H40_5x8","AD4_H40_10x4","AD4S_S0","AD4S_S1","AD4S_S2","AD4S_S3","AD4S_S4"]
        sigs=[]
        for name in names:
            torch.manual_seed(seed); sigs.append(raw_initialization_signature(CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS[name])))
        for sig in sigs[1:]:
            for key in sigs[0]: self.assertTrue(torch.equal(sigs[0][key],sig[key]),key)

    def test_programmatic_parameter_counts(self):
        expected={"AD0_H13_dense":(875,602,530),"AD1_H40_dense":(5843,3923,3716),"AD2_H40_2x20":(5843,3123,2916),"AD3_H40_5x8":(5843,2643,2436),"AD4_H40_10x4":(5843,2483,2276),"AD4S_S0":(5843,2483,2276)}
        for name, values in expected.items():
            torch.manual_seed(8); m=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS[name]); self.assertEqual((m.nominal_parameter_count(),m.active_parameter_count(),m.total_active_macs_per_timestep),values)

    def test_ad4_ad4s_equal_candidate_compute(self):
        torch.manual_seed(9); a=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD4_H40_10x4"])
        torch.manual_seed(9); b=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD4S_S0"])
        self.assertEqual(a.active_candidate_recurrent_weights,b.active_candidate_recurrent_weights)
        self.assertEqual(a.candidate_recurrent_macs,b.candidate_recurrent_macs)
        self.assertEqual(a.total_active_macs_per_timestep,b.total_active_macs_per_timestep)

    def test_masked_effective_weights_zero(self):
        torch.manual_seed(10); m=CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS["AD4_H40_10x4"])
        w=m.candidate_weight_effective().detach(); self.assertTrue(torch.all(w[m.candidate_mask==0] == 0))

    def test_4x_data_only(self):
        self.assertEqual(CONFIG["data_regime"],"4x_unique"); self.assertEqual(CONFIG["unique_seed_defined_episodes"],3200)
        self.assertEqual(CONFIG["training"]["train_episodes"],512); self.assertEqual(CONFIG["training"]["validation_episodes"],128)

    def test_training_budget_locked(self):
        self.assertEqual(CONFIG["training"]["steps"],192); self.assertEqual(CONFIG["training"]["replicates"],5)
        self.assertEqual(CONFIG["training"]["development_seed_range"],[10000,10511]); self.assertEqual(CONFIG["training"]["validation_seed_range"],[20000,20127])

    def test_science_locks(self):
        self.assertFalse(CONFIG["fresh_audit_consumed"]); self.assertFalse(CONFIG["structural_search_allowed"]); self.assertFalse(CONFIG["primitive_mining_allowed"]); self.assertFalse(CONFIG["v838_started"])
        audit=json.loads((ROOT/"experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8")); self.assertEqual(audit["episodes_consumed"],0)
        self.assertFalse((ROOT/"experiments/v837_primitive_invention/v838").exists())

    def test_no_neutral_y3_modification_declared(self):
        self.assertEqual(CONFIG["parent"],"V837ac")
        self.assertIn("candidate recurrence", CONFIG["single_change"])


if __name__ == "__main__": unittest.main()
