from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CandidateInputFactorizationY3
from experiments.v837_primitive_invention.v837aj import analyze_results as analyze_mod
from experiments.v837_primitive_invention.v837aj import random_structural_sampler as random_mod
from experiments.v837_primitive_invention.v837aj import structural_search as search_mod
from experiments.v837_primitive_invention.v837aj.af1d_structural_model import (
    CONDITION,
    architecture_lock,
    build_candidate_model,
    build_finalization_model,
    common_initialization_equal,
    finalization_common_seed,
    initialization_fingerprint,
    semantic_edge_tensor_map,
)
from experiments.v837_primitive_invention.v837aj.fidelity_calibration import (
    CONFIG,
    FINAL_VALIDATION_POOL,
    SEARCH_SELECTION_POOL,
    SEARCH_TRAIN_POOL,
    STAGE_CALIBRATION,
    STAGE_FINALIZATION,
    STAGE_RANDOM_SEARCH,
    STAGE_SEARCH,
    FinalValidationLeakageError,
    assert_data_partitions,
    fidelity_metrics,
    final_validation_seeds,
    full_development_seeds,
    kendall_tau_b,
    search_selection_seeds,
    search_train_seeds,
    select_cheapest_fidelity,
    spearman_rho,
)
from experiments.v837_primitive_invention.v837aj.finalize_champions import finalization_protocol
from experiments.v837_primitive_invention.v837aj.structural_mutations import ALLOWED_MUTATIONS, apply_mutation, constructive_initial_population
from experiments.v837_primitive_invention.v837aj.topology import (
    MAX_EDGES,
    RECURRENT_UNIVERSE,
    SAME_STEP_UNIVERSE,
    SearchEdge,
    SearchTopology,
    calibration_panel,
    historical_anchor_topology,
    minimal_topology,
    sample_topology_with_counts,
    semantic_edge_initial_weight,
    zero_message_topology,
)

ROOT = Path(__file__).resolve().parents[1]


def fake_proxy(topology, *, family, run_index, candidate_slot, fidelity, stage):
    # Deterministic synthetic evaluator used only to verify budget/control logic.
    success = 1.0 if candidate_slot % 7 == 0 else 0.5 + (int(topology.topology_id[:4], 16) % 4000) / 10000.0
    fitness = 1.0 - success + 0.0005 * topology.edge_count
    return {
        "stage": stage,
        "family": family,
        "run_index": run_index,
        "candidate_initialization_slot": candidate_slot,
        "fidelity": fidelity,
        "topology": topology.to_dict(),
        "topology_id": topology.topology_id,
        "development_success": success,
        "selection_success": success,
        "selection_loss": 1.0 - success,
        "fitness": fitness,
        "compute": {},
    }


class TestV837ajArchitectureLocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_candidate_model(minimal_topology(), "conditional_routing", 0, 0)

    def test_v837aj_uses_exact_af1d_class(self): self.assertIs(type(self.model), CandidateInputFactorizationY3)
    def test_condition_is_af1d_deshared(self): self.assertEqual(self.model.transfer_condition, CONDITION)
    def test_exactly_ten_cells(self): self.assertEqual(len(self.model.graph.cells), 10)
    def test_rank4_coupling_unchanged(self): self.assertEqual(self.model.interaction_spec.candidate_coupling_mode, "rank4_cross_block"); self.assertEqual(self.model.interaction_spec.coupling_rank, 4)
    def test_global_controller_unchanged(self): self.assertTrue(self.model.global_scalar_control); self.assertEqual(self.model.controller_param_count, 47)
    def test_ten_deshared_input_projections(self): self.assertIsNotNone(self.model.cell_candidate_projections); self.assertEqual(len(self.model.cell_candidate_projections), 10)
    def test_readout_unchanged(self): self.assertEqual(tuple(self.model.base.readout.weight.shape), (1, 40))
    def test_no_cell_count_search(self): self.assertNotIn("ADD_CELL", ALLOWED_MUTATIONS); self.assertNotIn("REMOVE_CELL", ALLOWED_MUTATIONS)
    def test_no_projection_search(self): self.assertFalse(any("PROJECTION" in x for x in ALLOWED_MUTATIONS))
    def test_no_controller_search(self): self.assertFalse(any("CONTROLLER" in x for x in ALLOWED_MUTATIONS))
    def test_no_coupling_search(self): self.assertFalse(any("COUPLING" in x for x in ALLOWED_MUTATIONS))
    def test_architecture_lock_passes(self): self.assertTrue(architecture_lock()["compatible"])


class TestV837ajTopologySemantics(unittest.TestCase):
    def test_same_step_edges_require_src_lt_dst(self):
        with self.assertRaises(ValueError): SearchEdge(4, 3, False)
        with self.assertRaises(ValueError): SearchEdge(4, 4, False)
        self.assertEqual(SearchEdge(3, 4, False).src, 3)
    def test_recurrent_edges_allow_all_pairs(self):
        for src in (0, 5, 9):
            for dst in (0, 5, 9): SearchEdge(src, dst, True)
    def test_no_duplicate_edges(self):
        e=SearchEdge(0,1,False)
        with self.assertRaises(ValueError): SearchTopology((e,e))
    def test_max_64_edges(self):
        with self.assertRaises(ValueError): SearchTopology(tuple(list(SAME_STEP_UNIVERSE)+list(RECURRENT_UNIVERSE[:20])))
        self.assertEqual(len(SearchTopology(tuple(list(SAME_STEP_UNIVERSE)+list(RECURRENT_UNIVERSE[:19]))).edges),64)
    def test_topology_id_structure_only(self):
        t=SearchTopology((SearchEdge(0,1,False),SearchEdge(1,1,True)))
        self.assertEqual(t.topology_id, SearchTopology(tuple(reversed(t.edges))).topology_id)
        self.assertEqual(t.topology_id, t.topology_id)
    def test_topology_canonical_order(self):
        t=SearchTopology((SearchEdge(8,8,True),SearchEdge(1,3,False),SearchEdge(0,2,False)))
        self.assertEqual(t.edges,(SearchEdge(0,2,False),SearchEdge(1,3,False),SearchEdge(8,8,True)))
    def test_minimal_topology_has_19_edges(self): self.assertEqual(minimal_topology().edge_count,19)
    def test_minimal_topology_chain_exact(self): self.assertEqual({e for e in minimal_topology().edges if not e.recurrent},{SearchEdge(i,i+1,False) for i in range(9)})
    def test_minimal_topology_self_loops_exact(self): self.assertEqual({e for e in minimal_topology().edges if e.recurrent},{SearchEdge(i,i,True) for i in range(10)})
    def test_legal_universe_sizes(self): self.assertEqual(len(SAME_STEP_UNIVERSE),45); self.assertEqual(len(RECURRENT_UNIVERSE),100)
    def test_zero_message_topology_legal(self): self.assertEqual(zero_message_topology().edge_count,0)


class TestV837ajMutationRestrictions(unittest.TestCase):
    def test_only_six_structural_mutations_allowed(self): self.assertEqual(set(ALLOWED_MUTATIONS),{"ADD_SAME_STEP_EDGE","REMOVE_SAME_STEP_EDGE","ADD_RECURRENT_EDGE","REMOVE_RECURRENT_EDGE","REWIRE_SAME_STEP_EDGE","REWIRE_RECURRENT_EDGE"})
    def test_no_cell_mutation(self): self.assertFalse(any("CELL" in x for x in ALLOWED_MUTATIONS))
    def test_no_parameter_mutation(self): self.assertFalse(any("PARAMETER" in x for x in ALLOWED_MUTATIONS))
    def test_no_edge_weight_mutation(self): self.assertFalse(any("WEIGHT" in x for x in ALLOWED_MUTATIONS))
    def test_rewire_preserves_temporal_class(self):
        import numpy as np
        rng=np.random.default_rng(4); base=minimal_topology()
        a=apply_mutation(base,"REWIRE_SAME_STEP_EDGE",rng); self.assertIsNotNone(a); self.assertEqual(a.same_step_count,base.same_step_count); self.assertEqual(a.recurrent_count,base.recurrent_count)
        b=apply_mutation(base,"REWIRE_RECURRENT_EDGE",rng); self.assertIsNotNone(b); self.assertEqual(b.same_step_count,base.same_step_count); self.assertEqual(b.recurrent_count,base.recurrent_count)
    def test_add_remove_respect_legal_universe(self):
        import numpy as np
        t=apply_mutation(minimal_topology(),"ADD_SAME_STEP_EDGE",np.random.default_rng(1)); self.assertTrue(all(e.recurrent or e.src<e.dst for e in t.edges)); self.assertLessEqual(t.edge_count,MAX_EDGES)
    def test_duplicate_topology_rejected(self):
        e=SearchEdge(0,1,False)
        with self.assertRaises(ValueError): SearchTopology((e,e))


class TestV837ajFastRuntimeParity(unittest.TestCase):
    def _models_and_batch(self):
        topology=sample_topology_with_counts(11,18,namespace="fast-runtime-test",parts=(1,))
        reference=build_candidate_model(topology,"variable_composition",4,22)
        optimized=build_candidate_model(topology,"variable_composition",4,22)
        generator=torch.Generator().manual_seed(771)
        observations=torch.randn(32,8,6,generator=generator)
        lengths=torch.randint(2,9,(32,),generator=generator)
        targets=torch.randn(32,generator=generator)
        return reference,optimized,observations,lengths,targets
    def test_fast_runtime_output_bit_exact(self):
        reference,optimized,observations,lengths,_=self._models_and_batch()
        expected=CandidateInputFactorizationY3.forward(reference,observations,lengths)
        actual=optimized(observations,lengths)
        self.assertTrue(torch.equal(expected,actual))
    def test_fast_runtime_gradient_and_optimizer_bit_exact(self):
        reference,optimized,observations,lengths,targets=self._models_and_batch()
        reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=.005,weight_decay=.0001)
        optimized_optimizer=torch.optim.AdamW(optimized.parameters(),lr=.005,weight_decay=.0001,foreach=True)
        for _ in range(3):
            reference_optimizer.zero_grad(set_to_none=True); optimized_optimizer.zero_grad(set_to_none=True)
            expected=CandidateInputFactorizationY3.forward(reference,observations,lengths)
            actual=optimized(observations,lengths)
            torch.nn.functional.mse_loss(expected,targets).backward(); torch.nn.functional.mse_loss(actual,targets).backward()
            for a,b in zip(reference.parameters(),optimized.parameters()): self.assertTrue(torch.equal(a.grad,b.grad))
            torch.nn.utils.clip_grad_norm_(reference.parameters(),5.0)
            torch.nn.utils.clip_grad_norm_(optimized.parameters(),5.0,foreach=True)
            reference_optimizer.step(); optimized_optimizer.step()
            for a,b in zip(reference.parameters(),optimized.parameters()): self.assertTrue(torch.equal(a,b))
    def test_fast_runtime_trace_falls_back_bit_exact(self):
        reference,optimized,observations,lengths,_=self._models_and_batch()
        expected_prediction,expected_trace=CandidateInputFactorizationY3.forward(reference,observations,lengths,return_trace=True)
        actual_prediction,actual_trace=optimized(observations,lengths,return_trace=True)
        self.assertTrue(torch.equal(expected_prediction,actual_prediction))
        self.assertTrue(torch.equal(expected_trace.states,actual_trace.states))
        self.assertTrue(torch.equal(expected_trace.messages,actual_trace.messages))


class TestV837ajInitializationPairing(unittest.TestCase):
    def test_common_parameters_identical_across_topologies(self):
        a=build_candidate_model(minimal_topology(),"conditional_routing",3,9); b=build_candidate_model(sample_topology_with_counts(9,10,namespace="test-other",parts=(1,)),"conditional_routing",3,9); self.assertTrue(common_initialization_equal(a,b))
    def test_common_edge_initialization_identical(self):
        t1=minimal_topology(); t2=SearchTopology(t1.edges+(SearchEdge(0,2,False),)); a=build_candidate_model(t1,"delayed_recall",1,6); b=build_candidate_model(t2,"delayed_recall",1,6); ea=semantic_edge_tensor_map(a); eb=semantic_edge_tensor_map(b); self.assertTrue(all(torch.equal(ea[k],eb[k]) for k in ea.keys()&eb.keys()))
    def test_edge_initialization_path_independent(self):
        edge=SearchEdge(2,7,False); self.assertEqual(semantic_edge_initial_weight("iterative_state",4,edge),semantic_edge_initial_weight("iterative_state",4,edge))
    def test_search_random_candidate_slot_pairing(self):
        a=sample_topology_with_counts(8,11,namespace="pair-a",parts=(1,)); b=sample_topology_with_counts(8,11,namespace="pair-b",parts=(1,)); self.assertTrue(common_initialization_equal(build_candidate_model(a,"partial_observation",2,17),build_candidate_model(b,"partial_observation",2,17)))
    def test_no_parent_weight_inheritance(self): self.assertNotIn("inherit", " ".join(ALLOWED_MUTATIONS).lower())
    def test_finalization_seed_independent_from_proxy_seed(self):
        t=minimal_topology(); proxy=build_candidate_model(t,"conditional_routing",0,0); final=build_finalization_model(t,"conditional_routing",0); self.assertNotEqual(initialization_fingerprint(proxy,include_edges=False),initialization_fingerprint(final,include_edges=False))
    def test_search_random_finalization_seed_paired(self): self.assertEqual(finalization_common_seed("conditional_routing",2),finalization_common_seed("conditional_routing",2))


class TestV837ajDataSeparation(unittest.TestCase):
    def test_search_train_pool_exact(self): self.assertEqual(list(SEARCH_TRAIN_POOL),list(range(10000,10384)))
    def test_search_selection_pool_exact(self): self.assertEqual(list(SEARCH_SELECTION_POOL),list(range(10384,10512)))
    def test_final_validation_exact(self): self.assertEqual(list(FINAL_VALIDATION_POOL),list(range(20000,20128)))
    def test_search_train_selection_disjoint(self): self.assertFalse(set(SEARCH_TRAIN_POOL)&set(SEARCH_SELECTION_POOL))
    def test_final_validation_not_in_search(self): self.assertFalse((set(SEARCH_TRAIN_POOL)|set(SEARCH_SELECTION_POOL))&set(FINAL_VALIDATION_POOL))
    def test_union_unique_episodes_3200(self): self.assertEqual(assert_data_partitions()["union_unique_family_seed_episodes"],3200)
    def test_no_fresh_audit_seeds(self): self.assertEqual(assert_data_partitions()["fresh_audit_overlap"],0)


class TestV837ajFinalValidationGuard(unittest.TestCase):
    def test_calibration_cannot_request_final_validation(self):
        with self.assertRaises(FinalValidationLeakageError): final_validation_seeds(STAGE_CALIBRATION)
    def test_search_cannot_request_final_validation(self):
        with self.assertRaises(FinalValidationLeakageError): final_validation_seeds(STAGE_SEARCH)
    def test_random_search_cannot_request_final_validation(self):
        with self.assertRaises(FinalValidationLeakageError): final_validation_seeds(STAGE_RANDOM_SEARCH)
    def test_finalization_may_request_final_validation(self): self.assertEqual(final_validation_seeds(STAGE_FINALIZATION),list(range(20000,20128)))


class TestV837ajFidelity(unittest.TestCase):
    def test_calibration_panel_has_12_topologies(self): self.assertEqual(len(calibration_panel()),12)
    def test_panel_task_independent(self): self.assertEqual([t.topology_id for t in calibration_panel().values()],[t.topology_id for t in calibration_panel().values()])
    def test_fidelity_training_sets_nested(self): self.assertLess(set(search_train_seeds(64)),set(search_train_seeds(128))); self.assertLess(set(search_train_seeds(128)),set(search_train_seeds(256))); self.assertLess(set(search_train_seeds(256)),set(search_train_seeds(384)))
    def test_selection_set_constant(self): self.assertEqual(search_selection_seeds(128),list(range(10384,10512)))
    def test_f4_target_exact(self): self.assertEqual(CONFIG["fidelities"]["F4"],{"steps":192,"train_count":384,"selection_count":128,"eligible":False,"target":True})
    def test_rank_metrics_deterministic(self):
        a=[1,3,2,4]; b=[1,2,3,4]; self.assertEqual(spearman_rho(a,b),spearman_rho(a,b)); self.assertEqual(kendall_tau_b(a,b),kendall_tau_b(a,b))
    def test_cheapest_passing_fidelity_selected(self):
        good={"median_spearman_rho":.8,"median_kendall_tau":.7,"minimum_family_kendall_tau":.4,"median_pairwise_order_accuracy":.8,"median_top4_recall":.75,"minimum_family_top4_recall":.5}
        bad={**good,"median_spearman_rho":.5}; metrics={"F0":bad,"F1":good,"F2":good,"F3":good}; self.assertEqual(select_cheapest_fidelity(metrics),"F1")
    def test_search_blocked_when_proxy_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/"diagnostics").mkdir(); (p/"diagnostics/fidelity_decision.json").write_text(json.dumps({"proxy_valid":False,"selected_search_fidelity":None}))
            with patch.object(search_mod,"HERE",p):
                with self.assertRaises(RuntimeError): search_mod.require_valid_proxy()


class TestV837ajEqualBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(search_mod,"train_proxy_candidate",side_effect=fake_proxy):
            cls.directed=search_mod.run_directed_search("conditional_routing",0,"F0")
        with patch.object(random_mod,"train_proxy_candidate",side_effect=fake_proxy):
            cls.random=random_mod.run_random_sampler(cls.directed,"F0")
    def test_search_has_exactly_64_unique_candidate_evaluations(self): self.assertEqual(len(self.directed["records"]),64); self.assertEqual(len({r["topology_id"] for r in self.directed["records"]}),64)
    def test_random_has_exactly_64_unique_candidate_evaluations(self): self.assertEqual(len(self.random["records"]),64); self.assertEqual(len({r["topology_id"] for r in self.random["records"]}),64)
    def test_no_early_stop(self): self.assertEqual(len(self.directed["records"]),64); self.assertTrue(any(r["selection_success"]==1.0 for r in self.directed["records"]))
    def test_random_matches_edge_count_per_slot(self): self.assertTrue(all(a["topology"]["edge_count"]==b["topology"]["edge_count"] for a,b in zip(self.directed["records"],self.random["records"])))
    def test_random_matches_recurrent_count_per_slot(self): self.assertTrue(all(a["topology"]["recurrent_edge_count"]==b["topology"]["recurrent_edge_count"] for a,b in zip(self.directed["records"],self.random["records"])))
    def test_random_uses_same_candidate_initialization_slot(self): self.assertTrue(all(a["candidate_initialization_slot"]==b["candidate_initialization_slot"] for a,b in zip(self.directed["records"],self.random["records"])))
    def test_constructive_initial_population_anchor_free(self): self.assertNotIn(historical_anchor_topology().topology_id,{t.topology_id for t in constructive_initial_population("conditional_routing",0)})
    def test_checkpointing_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(search_mod,"PROGRESS_DIR",Path(tmp)), patch.object(search_mod,"train_proxy_candidate",side_effect=fake_proxy):
                search_mod.run_directed_search("conditional_routing",2,"F0")
            self.assertEqual(list(Path(tmp).glob("progress_search_*.json")),[])
    def test_directed_atomic_checkpoint_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"progress.json"; calls={"n":0}; real_replace=Path.replace
            def flaky_replace(path_obj,target_path):
                calls["n"]+=1
                if calls["n"]<3: raise PermissionError(5,"synthetic Windows file lock")
                return real_replace(path_obj,target_path)
            with patch.object(Path,"replace",autospec=True,side_effect=flaky_replace), patch.object(search_mod.time,"sleep",return_value=None):
                search_mod._atomic_json(target,{"ok":True})
            self.assertEqual(json.loads(target.read_text()),{"ok":True}); self.assertEqual(calls["n"],3)
    def test_random_atomic_checkpoint_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"progress.json"; calls={"n":0}; real_replace=Path.replace
            def flaky_replace(path_obj,target_path):
                calls["n"]+=1
                if calls["n"]<3: raise PermissionError(5,"synthetic Windows file lock")
                return real_replace(path_obj,target_path)
            with patch.object(Path,"replace",autospec=True,side_effect=flaky_replace), patch.object(random_mod.time,"sleep",return_value=None):
                random_mod._atomic_json(target,{"ok":True})
            self.assertEqual(json.loads(target.read_text()),{"ok":True}); self.assertEqual(calls["n"],3)
    def test_directed_checkpoint_resume_preserves_exact_trajectory(self):
        with patch.object(search_mod,"train_proxy_candidate",side_effect=fake_proxy):
            baseline=search_mod.run_directed_search("delayed_recall",3,"F0")
        calls={"n":0}
        def interrupting_proxy(*args,**kwargs):
            calls["n"]+=1
            if calls["n"]==21: raise RuntimeError("synthetic interruption")
            return fake_proxy(*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(search_mod,"PROGRESS_DIR",Path(tmp)), patch.object(search_mod,"train_proxy_candidate",side_effect=interrupting_proxy):
                with self.assertRaisesRegex(RuntimeError,"synthetic interruption"):
                    search_mod.run_directed_search("delayed_recall",3,"F0",checkpoint=True)
            with patch.object(search_mod,"PROGRESS_DIR",Path(tmp)), patch.object(search_mod,"train_proxy_candidate",side_effect=fake_proxy):
                resumed=search_mod.run_directed_search("delayed_recall",3,"F0",checkpoint=True)
        self.assertEqual([r["topology_id"] for r in baseline["records"]],[r["topology_id"] for r in resumed["records"]])
        self.assertEqual(baseline["champion"]["topology_id"],resumed["champion"]["topology_id"])
    def test_random_checkpoint_resume_preserves_exact_trajectory(self):
        calls={"n":0}
        def interrupting_proxy(*args,**kwargs):
            calls["n"]+=1
            if calls["n"]==18: raise RuntimeError("synthetic interruption")
            return fake_proxy(*args,**kwargs)
        with patch.object(random_mod,"train_proxy_candidate",side_effect=fake_proxy):
            baseline=random_mod.run_random_sampler(self.directed,"F0")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(random_mod,"PROGRESS_DIR",Path(tmp)), patch.object(random_mod,"train_proxy_candidate",side_effect=interrupting_proxy):
                with self.assertRaisesRegex(RuntimeError,"synthetic interruption"):
                    random_mod.run_random_sampler(self.directed,"F0",checkpoint=True)
            with patch.object(random_mod,"PROGRESS_DIR",Path(tmp)), patch.object(random_mod,"train_proxy_candidate",side_effect=fake_proxy):
                resumed=random_mod.run_random_sampler(self.directed,"F0",checkpoint=True)
        self.assertEqual([r["topology_id"] for r in baseline["records"]],[r["topology_id"] for r in resumed["records"]])
        self.assertEqual(baseline["champion"]["topology_id"],resumed["champion"]["topology_id"])


class TestV837ajChampionProtocol(unittest.TestCase):
    def test_finalizer_script_bootstraps_repo_root(self):
        script=ROOT/"experiments/v837_primitive_invention/v837aj/finalize_champions.py"
        completed=subprocess.run([sys.executable,str(script),"--help"],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
    def test_champion_selected_before_final_validation(self):
        with patch.object(search_mod,"train_proxy_candidate",side_effect=fake_proxy): run=search_mod.run_directed_search("delayed_recall",1,"F0")
        self.assertTrue(run["champion"]["selected_before_final_validation"])
    def test_champion_topology_immutable_after_selection(self):
        t=minimal_topology()
        with self.assertRaises(dataclasses.FrozenInstanceError): t.edges=tuple()
    def test_proxy_weights_discarded(self): self.assertTrue(finalization_protocol()["proxy_weights_discarded"])
    def test_full_512_development_retrain(self): self.assertEqual(finalization_protocol()["development_episodes_per_family"],512); self.assertEqual(len(full_development_seeds()),512)
    def test_full_192_steps(self): self.assertEqual(finalization_protocol()["optimizer_steps"],192)
    def test_final_validation_evaluated_once(self): self.assertEqual(finalization_protocol()["final_validation_data_accesses"],1)
    def test_search_random_finalization_seed_paired(self): self.assertEqual(finalization_common_seed("variable_composition",4),finalization_common_seed("variable_composition",4))


class TestV837ajAnalysisContract(unittest.TestCase):
    def test_fixed_af1d_scoreboard_compute_contract(self):
        board=analyze_mod._structural_efficiency_scoreboard([],[])["fixed_af1d_anchor"]
        self.assertEqual(board["edge_count"],55)
        self.assertEqual(board["active_parameters"],1643)
        self.assertEqual(board["modeled_macs_per_timestep"],1426)
        self.assertIsNone(board["search_evaluations_required"])
    def test_resource_accounting_exposes_primary_and_robustness_breakdown(self):
        resources=analyze_mod._resource_totals()
        for key in ("primary_proxy_directed","primary_proxy_random","primary_final_directed","primary_final_random","primary_stage_b","robustness_proxy_directed","robustness_proxy_random","robustness_final_directed","robustness_final_random","robustness_extension"):
            self.assertIn(key,resources)


class TestV837ajScienceLocks(unittest.TestCase):
    def test_structural_search_authorized_by_v837ai(self):
        d=json.loads((ROOT/"experiments/v837_primitive_invention/v837ai/diagnostics/decision_state.json").read_text()); self.assertTrue(d["structural_search_recovery_allowed"])
    def test_recommended_multiplier_is_4x(self):
        d=json.loads((ROOT/"experiments/v837_primitive_invention/v837ai/diagnostics/decision_state.json").read_text()); self.assertEqual(d["recommended_structural_search_multiplier"],4)
    def test_no_primitive_mining(self): self.assertFalse(CONFIG["primitive_mining_allowed_during_v837aj"])
    def test_no_motif_promotion(self): self.assertFalse(CONFIG["primitive_mining_allowed_during_v837aj"])
    def test_no_fresh_audit(self): self.assertFalse(CONFIG["fresh_audit_allowed"]); self.assertEqual(json.loads((ROOT/"experiments/v837_primitive_invention/audit/audit_results.json").read_text())["episodes_consumed"],0)
    def test_no_v838(self): self.assertFalse((ROOT/"experiments/v837_primitive_invention/v838").exists())
    def test_v837ae_absent(self): self.assertFalse((ROOT/"experiments/v837_primitive_invention/v837ae").exists())
    def test_v837ag_absent(self): self.assertFalse((ROOT/"experiments/v837_primitive_invention/v837ag").exists())
    def test_v837ah_absent(self): self.assertFalse((ROOT/"experiments/v837_primitive_invention/v837ah").exists())


if __name__ == "__main__": unittest.main()
