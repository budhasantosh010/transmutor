from __future__ import annotations

import unittest
from experiments.v837_primitive_invention.v837al.adapter_fit import fit_all_adapters
from experiments.v837_primitive_invention.v837al.scope_localization import run_scope_localization
from experiments.v837_primitive_invention.v837al.select_interface import freeze_selected

class TestV837alPipelineRunner(unittest.TestCase):
    def test_10_fit_adapters(self):
        p=fit_all_adapters();self.assertEqual(p['gradient_steps_used_for_adapters'],0);self.assertGreater(len(p['rows']),0)
    def test_20_scope_localization(self):
        p=run_scope_localization();self.assertEqual(p['configs_evaluated'],253)
    def test_30_freeze_selected(self):
        p=freeze_selected();self.assertTrue(p['frozen_before_align_test']);self.assertIn('selected_interface_sha256',p)
    def test_40_pairwise_gate(self):
        from experiments.v837_primitive_invention.v837al.heldout_pairwise import run_pairwise_test
        from experiments.v837_primitive_invention.v837al.port_necessity import run_port_necessity
        p=run_pairwise_test();n=run_port_necessity();self.assertFalse(p['global_track']['run']);self.assertFalse(p['causal_track']['run']);self.assertFalse(n['global_track']['run']);self.assertFalse(n['causal_track']['run'])
    def test_50_canonical_gate(self):
        from experiments.v837_primitive_invention.v837al.canonical_interface import run_canonical_test
        p=run_canonical_test();self.assertFalse(p['global_track']['run']);self.assertFalse(p['causal_track']['run'])
    def test_60_transplant_and_frontier_gate(self):
        from experiments.v837_primitive_invention.v837al.closed_loop import run_closed_loop
        from experiments.v837_primitive_invention.v837al.alignment_data_frontier import run_alignment_data_frontier
        p=run_closed_loop();f=run_alignment_data_frontier();self.assertFalse(p['run']);self.assertFalse(f['run'])
    def test_70_analyze(self):
        from experiments.v837_primitive_invention.v837al.analyze_results import analyze
        p=analyze();self.assertEqual(p['diagnosis'],'LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT');self.assertTrue(p['decision_state']['source_integrity']);self.assertTrue(p['decision_state']['legacy_baseline_reproduced']);self.assertEqual(p['decision_state']['configs_evaluated'],253);self.assertEqual(p['decision_state']['new_model_fits'],0);self.assertEqual(p['decision_state']['optimizer_steps'],0)

if __name__=='__main__':unittest.main()
