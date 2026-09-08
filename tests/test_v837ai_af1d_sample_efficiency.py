from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CandidateInputFactorizationY3
from experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency import (
    CONFIG, CONDITION, FAMILIES, ai4_reuse_diagnostic, architecture_lock, build_af1d,
    coupling_seed, data_nesting_diagnostic, development_seeds, historical_v837l_comparison,
    initialization_pairing_diagnostic, load_ai4_rows, projection_seed, trainable_tensor_fingerprint,
    validation_seeds,
)

ROOT = Path(__file__).resolve().parents[1]

class TestV837aiArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = build_af1d(FAMILIES[0], 0)

    def test_v837ai_uses_exact_af1d_class(self): self.assertIs(type(self.model), CandidateInputFactorizationY3)
    def test_v837ai_condition_is_af1d(self): self.assertEqual(CONDITION, "AF1D_deshared_candidate_input_factorization"); self.assertEqual(self.model.transfer_condition, CONDITION)
    def test_ten_candidate_projections_present(self): self.assertEqual(len(self.model.cell_candidate_projections), 10)
    def test_candidate_projections_untied(self):
        layers=self.model.cell_candidate_projections; self.assertEqual(len({id(x.weight) for x in layers}),10); self.assertEqual(len({id(x.bias) for x in layers}),10)
    def test_projection_initialization_identical_within_fit(self):
        layers=self.model.cell_candidate_projections; self.assertTrue(all(torch.equal(layers[0].weight,l.weight) and torch.equal(layers[0].bias,l.bias) for l in layers[1:]))
    def test_rank4_coupling_unchanged(self): self.assertEqual(self.model.interaction_spec.candidate_coupling_mode,"rank4_cross_block"); self.assertEqual(self.model.interaction_spec.coupling_rank,4)
    def test_global_controller_unchanged(self): self.assertTrue(self.model.global_scalar_control); self.assertEqual(self.model.controller_param_count,47)
    def test_message_schedule_unchanged(self): self.assertEqual(len(self.model.graph.edges),55); self.assertEqual([(e.src,e.dst,e.recurrent) for e in self.model.graph.edges],[(e.src,e.dst,e.recurrent) for e in high_capacity_generic_graph(0).edges])
    def test_readout_unchanged(self): self.assertEqual(tuple(self.model.base.readout.weight.shape),(1,40))
    def test_parameter_count_constant_across_multipliers(self): self.assertEqual(self.model.parameter_count(),1643); self.assertEqual(CONFIG['expected_active_parameters'],1643)
    def test_macs_constant_across_multipliers(self): self.assertEqual(self.model.total_recurrent_controller_projection_macs,1206); self.assertEqual(CONFIG['expected_active_macs_per_timestep'],1206)

class TestV837aiData(unittest.TestCase):
    def test_1x_has_128_development_episodes(self): self.assertEqual(len(development_seeds(1)),128)
    def test_2x_has_256_development_episodes(self): self.assertEqual(len(development_seeds(2)),256)
    def test_4x_has_512_development_episodes(self): self.assertEqual(len(development_seeds(4)),512)
    def test_validation_fixed_128(self): self.assertEqual(len(validation_seeds()),128)
    def test_1x_subset_of_2x(self): self.assertLess(set(development_seeds(1)),set(development_seeds(2)))
    def test_2x_subset_of_4x(self): self.assertLess(set(development_seeds(2)),set(development_seeds(4)))
    def test_no_development_validation_overlap(self): self.assertFalse(set(development_seeds(4)) & set(validation_seeds()))
    def test_no_duplicate_seeds(self):
        for m in (1,2,4): self.assertEqual(len(development_seeds(m)),len(set(development_seeds(m))))
    def test_same_task_generators(self): self.assertEqual(len(FAMILIES),5)
    def test_same_validation_rows(self): self.assertEqual(validation_seeds(),list(range(20000,20128)))
    def test_exact_data_nesting_diagnostic(self):
        d=data_nesting_diagnostic(); self.assertTrue(d['exact']); self.assertTrue(d['strict_nesting']); self.assertEqual(d['total_unique_family_seed_episodes']['union'],3200)

class TestV837aiInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ai4=load_ai4_rows(); cls.pair=initialization_pairing_diagnostic(cls.ai4)
    def test_ai1_ai2_initialization_identical(self): self.assertTrue(self.pair['pairing_exact'])
    def test_ai1_reconstructs_v837af_initialization(self): self.assertTrue(all(r['historical_seed_fields_match'] for r in self.pair['rows']))
    def test_projection_seed_law_unchanged(self): self.assertEqual(projection_seed(FAMILIES[0],0),int(next(r for r in self.ai4 if r['family']==FAMILIES[0] and r['replicate_id']==0)['projection_initialization_seed']))
    def test_coupling_seed_law_unchanged(self): self.assertEqual(coupling_seed(0),int(next(r for r in self.ai4 if r['family']==FAMILIES[0] and r['replicate_id']==0)['coupling_initialization_seed']))
    def test_graph_parameter_seeds_unchanged(self): self.assertTrue(self.pair['pairing_exact'])
    def test_fingerprint_deterministic(self): self.assertEqual(trainable_tensor_fingerprint(build_af1d(FAMILIES[0],0)),trainable_tensor_fingerprint(build_af1d(FAMILIES[0],0)))

class TestV837aiAI4Reuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.guard, cls.rows=ai4_reuse_diagnostic()
    def test_ai4_source_is_v837af_af1d(self): self.assertEqual(self.guard['source_version'],'V837af'); self.assertEqual(self.guard['source_condition'],CONDITION)
    def test_ai4_source_sha_exact(self): self.assertEqual(self.guard['source_sha'],'1e2be191b39f2c624c043d47ce11fc602b95be1e')
    def test_ai4_config_compatible(self): self.assertTrue(self.guard['checks']['training_protocol']); self.assertTrue(self.guard['checks']['architecture'])
    def test_ai4_has_25_rows(self): self.assertEqual(len(self.rows),25)
    def test_ai4_has_five_families(self): self.assertEqual(set(r['family'] for r in self.rows),set(FAMILIES))
    def test_ai4_has_five_replicates_each(self): self.assertTrue(all(sum(r['family']==f for r in self.rows)==5 for f in FAMILIES))
    def test_ai4_reanalysis_matches_v837af(self): self.assertTrue(self.guard['checks']['exact_reanalysis']); self.assertEqual(self.guard['reanalyzed']['families_passing'],4)

class TestV837aiScienceLocks(unittest.TestCase):
    def test_optimizer_is_adamw(self): self.assertEqual(CONFIG['training']['optimizer'],'AdamW')
    def test_steps_are_192(self): self.assertEqual(CONFIG['training']['steps'],192)
    def test_lr_is_005(self): self.assertEqual(CONFIG['training']['learning_rate'],0.005)
    def test_weight_decay_is_00001(self): self.assertEqual(CONFIG['training']['weight_decay'],0.0001)
    def test_gradient_clip_is_5(self): self.assertEqual(CONFIG['training']['gradient_clip'],5.0)
    def test_no_architecture_tuning(self): self.assertTrue(json.loads((ROOT/'experiments/v837_primitive_invention/v837ai/frozen_sample_efficiency_gate.json').read_text())['architecture_frozen'])
    def test_no_fresh_audit(self): self.assertFalse(CONFIG['fresh_audit_allowed']); self.assertEqual(json.loads((ROOT/'experiments/v837_primitive_invention/audit/audit_results.json').read_text())['episodes_consumed'],0)
    def test_no_structural_search(self): self.assertFalse(CONFIG['structural_search_execution_allowed'])
    def test_no_primitive_mining(self): self.assertFalse(CONFIG['primitive_mining_allowed'])
    def test_v837ae_absent(self): self.assertFalse((ROOT/'experiments/v837_primitive_invention/v837ae').exists())
    def test_v837ag_absent(self): self.assertFalse((ROOT/'experiments/v837_primitive_invention/v837ag').exists())
    def test_v837ah_absent(self): self.assertFalse((ROOT/'experiments/v837_primitive_invention/v837ah').exists())
    def test_v838_absent(self): self.assertFalse((ROOT/'experiments/v837_primitive_invention/v838').exists())
    def test_architecture_hash_lock(self): self.assertTrue(architecture_lock()['compatible'])
    def test_historical_v837l_comparator(self): self.assertTrue(historical_v837l_comparison()['compatible'])

if __name__ == '__main__': unittest.main()
