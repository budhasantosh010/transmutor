from __future__ import annotations

import itertools, json, unittest
from pathlib import Path
import numpy as np
import torch

from experiments.v837_primitive_invention.v837an.authorization import (
    CARRIERS, LATENT_DIMS, MIN_ELIGIBLE, PARTITIONS, PRIMARY_PROBES, ROUTING_PREFIXES,
)
from experiments.v837_primitive_invention.v837an.causal_metrics import recovery, direction_agreement
from experiments.v837_primitive_invention.v837an.coalition_scan import all_cell_subsets
from experiments.v837_primitive_invention.v837an.counterfactual_subspace import fit_difference_subspace, projected_delta
from experiments.v837_primitive_invention.v837an.counterfactual_tasks import make_counterfactual
from experiments.v837_primitive_invention.v837an.failure_ledger import load as load_ledger
from experiments.v837_primitive_invention.v837an.oracle_macrostate import instrument_episode
from experiments.v837_primitive_invention.v837an.random_controls import random_subspaces
from experiments.v837_primitive_invention.v837an.semantic_compiler import fit_semantic_compiler, compile_delta
from experiments.v837_primitive_invention.v837an.source_integrity import verify_source_integrity
from experiments.v837_primitive_invention.v837an.utils import HERE, ROOT, START_SHA, read_json


class TestV837anSourceIntegrity(unittest.TestCase):
    def test_start_sha_exact(self): self.assertEqual(START_SHA,"f556c92a895b141f73f214e5eda3ac29b1927ea9")
    def test_v837am_diagnosis_exact(self): self.assertEqual(read_json(ROOT/"experiments/v837_primitive_invention/v837am/diagnostics/decision_state.json")["diagnosis"],"DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE")
    def test_v837am_next_program_exact(self): self.assertEqual(read_json(ROOT/"experiments/v837_primitive_invention/v837am/diagnostics/decision_state.json")["next_program"],"V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED")
    def test_v837ak_population_50(self): self.assertEqual(len(read_json(ROOT/"experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json")["rows"]),50)
    def test_competent_population_40(self): self.assertEqual(sum(bool(r["competent"]) for r in read_json(ROOT/"experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json")["rows"]),40)
    def test_v837an_never_requires_cross_organism_state_invertibility(self): self.assertFalse(read_json(HERE/"frozen_causal_primitive_gate.json")["cross_organism_state_invertibility_required"])
    def test_source_checkpoint_integrity_artifact(self): self.assertTrue(read_json(HERE/"diagnostics/source_integrity.json")["valid"])


class TestV837anHistoricalRefinement(unittest.TestCase):
    def test_v837am_105_invalid_am_a_configs_recorded(self): self.assertEqual(read_json(HERE/"frozen_causal_primitive_gate.json")["historical_refinements"]["am_a_invalid_state_configs"],105)
    def test_v837am_am_c_zero_rows_recorded(self): self.assertEqual(read_json(HERE/"frozen_causal_primitive_gate.json")["historical_refinements"]["am_c_zero_valid_rows"],7)
    def test_historical_diagnosis_not_rewritten(self): self.assertTrue(read_json(HERE/"frozen_causal_primitive_gate.json")["historical_refinements"]["diagnosis_preserved"])
    def test_refinement_is_append_only(self):
        ids={e["failure_id"] for e in load_ledger()["entries"]};self.assertTrue({"REF-AN-001","REF-AN-002","REF-AN-003"}.issubset(ids))


class TestV837anDataLocks(unittest.TestCase):
    def test_partitions_exact(self):
        exp={"AN_FIT":[10000,10127],"AN_SELECT":[10128,10255],"AN_ROUTING_FIT":[10256,10319],"AN_ROUTING_SELECT":[10320,10383],"AN_META_CONFIRM":[10384,10447],"AN_FINAL_DEV":[10448,10511],"AN_FINAL_VALIDATION":[20000,20127],"FRESH_AUDIT":[90000,90499]};self.assertEqual(PARTITIONS,exp)
    def test_development_partitions_disjoint(self):
        keys=[k for k in PARTITIONS if k not in {"AN_FINAL_VALIDATION","FRESH_AUDIT"}];sets={k:set(range(PARTITIONS[k][0],PARTITIONS[k][1]+1)) for k in keys}
        for i,a in enumerate(keys):
            for b in keys[i+1:]:self.assertTrue(sets[a].isdisjoint(sets[b]))
    def test_no_fresh_audit(self): self.assertFalse(read_json(HERE/"frozen_causal_primitive_gate.json")["fresh_audit_consumed"])


class TestV837anOracleAndCounterfactuals(unittest.TestCase):
    def test_all_512_dev_seeds_match(self): self.assertEqual(read_json(HERE/"diagnostics/oracle_equivalence.json")["episodes_verified"],2560)
    def test_oracle_exact(self):
        d=read_json(HERE/"diagnostics/oracle_equivalence.json");self.assertEqual(d["max_observation_abs_error"],0.0);self.assertEqual(d["max_target_abs_error"],0.0)
    def test_routing_control_flip(self):
        p=make_counterfactual("conditional_routing",10000);self.assertAlmostEqual(p.counterfactual_episode.causal_inputs["control"],-p.base_episode.causal_inputs["control"])
    def test_recall_value_flip(self):
        p=make_counterfactual("delayed_recall",10000);self.assertAlmostEqual(p.counterfactual_episode.target,-p.base_episode.target)
    def test_iterative_mid_input_flip(self):
        p=make_counterfactual("iterative_state",10000);a=np.asarray(p.base_episode.causal_inputs["x_t"]);b=np.asarray(p.counterfactual_episode.causal_inputs["x_t"]);self.assertEqual(np.sum(a!=b),1)
    def test_partial_initial_latent_sign_flip(self):
        p=make_counterfactual("partial_observation",10000);self.assertAlmostEqual(p.counterfactual_episode.latent_variables["initial_z0"],-p.base_episode.latent_variables["initial_z0"])
    def test_composition_initial_sign_flip(self):
        p=make_counterfactual("variable_composition",10000);self.assertAlmostEqual(p.counterfactual_episode.causal_inputs["initial_value"],-p.base_episode.causal_inputs["initial_value"])
    def test_counterfactual_length_preserved(self):
        for f in PRIMARY_PROBES:
            p=make_counterfactual(f,10017);self.assertEqual(len(p.base_episode.observations),len(p.counterfactual_episode.observations))
    def test_no_new_random_seed(self): self.assertEqual(read_json(HERE/"diagnostics/counterfactual_validity.json")["new_random_task_seeds"],0)


class TestV837anRuntime(unittest.TestCase):
    def test_runtime_equivalence(self):
        d=read_json(HERE/"diagnostics/instrumented_runtime_equivalence.json");self.assertTrue(d["pass"]);self.assertTrue(all(v<=1e-6 for v in d["max_abs"].values()))
    def test_intervention_hooks(self): self.assertEqual(set(read_json(HERE/"diagnostics/instrumented_runtime_equivalence.json")["intervention_hooks"]),{"patch_state","patch_output","patch_message","patch_global_term","patch_gate"})


class TestV837anCausalSubspace(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(837013);self.x=self.rng.normal(size=(128,40));v=self.rng.normal(size=(40,2));self.y=self.x+self.rng.normal(size=(128,2))@v.T
    def test_delta_svd_deterministic(self):
        a=fit_difference_subspace(self.x,self.y,2);b=fit_difference_subspace(self.x,self.y,2);np.testing.assert_array_equal(a["q"],b["q"])
    def test_basis_orthonormal(self):
        q=fit_difference_subspace(self.x,self.y,4)["q"];np.testing.assert_allclose(q.T@q,np.eye(4),atol=1e-10)
    def test_k_1_2_4_8_exact(self): self.assertEqual(LATENT_DIMS["STATE40"],[1,2,4,8])
    def test_source_swap_projection_formula(self):
        q=fit_difference_subspace(self.x,self.y,2)["q"];d=self.y-self.x;np.testing.assert_allclose(projected_delta(d,q),d@q@q.T)
    def test_semantic_compiler_zero_intercept(self):
        q=fit_difference_subspace(self.x,self.y,2)["q"];z=np.linspace(-1,1,len(self.x));m=fit_semantic_compiler(z,self.y-self.x,q);zero=compile_delta(np.zeros(5),m);self.assertLess(np.max(np.abs(zero)),1e-12)
    def test_32_random_subspaces(self): self.assertEqual(len(random_subspaces(40,4,32,123)),32)


class TestV837anMetrics(unittest.TestCase):
    def test_recovery_formula(self):
        b=np.array([0.]);c=np.array([1.]);p=np.array([1.]);self.assertAlmostEqual(float(recovery(b,c,p)[0]),1.0)
    def test_direction_agreement(self):
        b=np.array([0.,0.]);c=np.array([1.,-1.]);p=np.array([.2,-.1]);np.testing.assert_array_equal(direction_agreement(b,c,p),np.array([1.,1.]))


class TestV837anRoutingAndCoalitions(unittest.TestCase):
    def test_prefix_sizes_exact(self): self.assertEqual(ROUTING_PREFIXES,{"message":[1,2,4,8,16,32],"global":[1,2,4,8,10],"combined":[1,2,4,8,16,32]})
    def test_exact_1023_nonempty_cell_subsets(self): self.assertEqual(len(all_cell_subsets()),1023)
    def test_singletons_pairs_triples_full_present(self):
        s=set(all_cell_subsets());self.assertIn((0,),s);self.assertIn((0,1),s);self.assertIn((0,1,2),s);self.assertIn(tuple(range(10)),s)


class TestV837anScienceLocks(unittest.TestCase):
    def test_new_model_fits_zero(self): self.assertEqual(read_json(HERE/"frozen_causal_primitive_gate.json")["new_model_fits"],0)
    def test_optimizer_steps_zero(self): self.assertEqual(read_json(HERE/"frozen_causal_primitive_gate.json")["optimizer_steps"],0)
    def test_adapter_gradient_steps_zero(self): self.assertEqual(read_json(HERE/"frozen_causal_primitive_gate.json")["adapter_gradient_steps"],0)
    def test_no_primitive_archive_population(self): self.assertFalse(read_json(HERE/"frozen_causal_primitive_gate.json")["primitive_archive_allowed"])
    def test_primitives_promoted_zero(self): self.assertEqual(read_json(HERE/"frozen_causal_primitive_gate.json")["primitives_promoted"],0)
    def test_v838_false(self): self.assertFalse(read_json(HERE/"frozen_causal_primitive_gate.json")["v838_started"])


if __name__=="__main__": unittest.main()
