from __future__ import annotations

import json, unittest
from pathlib import Path
import numpy as np
import torch

from experiments.v837_primitive_invention.v837am.authorization import PARTITIONS, CONTEXT_SETS, PORT_BUNDLES, AM_B_BOUNDARIES, AM_C, GATE
from experiments.v837_primitive_invention.v837am.branch_a_context import configs as a_configs
from experiments.v837_primitive_invention.v837am.branch_b_boundary import configs as b_configs
from experiments.v837_primitive_invention.v837am.branch_c_temporal_interaction import configs as c_configs
from experiments.v837_primitive_invention.v837am.conditional_adapter import fit_map,apply_np,inverse_np,conditioning
from experiments.v837_primitive_invention.v837am.context_features import context_tensor,gate_context_tensor
from experiments.v837_primitive_invention.v837am.failure_ledger import RAW as LEDGER, CENTRAL
from experiments.v837_primitive_invention.v837am.interface_data import collect_trace
from experiments.v837_primitive_invention.v837am.temporal_context import HISTORIES
from experiments.v837_primitive_invention.v837am.utils import HERE, ROOT, START_SHA, read_json


class TestV837amSource(unittest.TestCase):
    def test_start_sha(self): self.assertEqual(START_SHA,"7d3b70ee6907b5502e0f229116d74eff4745c71e")
    def test_v837al_diagnosis_and_253_failures(self):
        d=read_json(ROOT/"experiments/v837_primitive_invention/v837al/diagnostics/decision_state.json")
        self.assertEqual(d["diagnosis"],"LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT");self.assertEqual(d["configs_evaluated"],253);self.assertIsNone(d["global_selected_config"]);self.assertIsNone(d["causal_selected_config"])
    def test_primary_causal_class_powered(self):
        g=read_json(GATE);self.assertEqual(g["causal_recipient_count"],38);self.assertLess(g["minimum_possible_p"],.01)
    def test_source_locks(self):
        g=read_json(GATE);self.assertFalse(g["fresh_audit_consumed"]);self.assertFalse(g["v838_started"]);self.assertEqual(g["primitives_promoted"],0);self.assertEqual(g["new_model_fits"],0);self.assertEqual(g["organism_optimizer_steps"],0);self.assertEqual(g["adapter_gradient_steps"],0)


class TestV837amFailureMemory(unittest.TestCase):
    def test_failure_analysis_exists_before_science(self): self.assertTrue((HERE/"FAILURE_ANALYSIS.md").is_file())
    def test_machine_failure_ledger_exists(self): self.assertTrue(LEDGER.is_file());self.assertTrue((HERE/"diagnostics/failure_ledger.json").is_file())
    def test_historical_failures_backfilled(self):
        ids={e["failure_id"] for e in read_json(LEDGER)["entries"]};self.assertIn("V837ak-BOUNDARY-INTERCHANGEABILITY-NOT-ESTABLISHED",ids);self.assertIn("V837al-STATIC-LINEAR-INTERFACE-EXHAUSTED",ids)
    def test_engineering_scientific_failure_distinction(self):
        types={e["failure_type"] for e in read_json(LEDGER)["entries"]};self.assertIn("ENGINEERING_FAILURE",types);self.assertIn("SCIENTIFIC_FAILURE",types)
    def test_schema1_invalidation_is_append_only_correction(self):
        p=read_json(LEDGER);c=next(e for e in p["entries"] if e["failure_id"]=="V837am-CORRECTION-AM-A-SCHEMA1-INVALIDATED");self.assertEqual(len(c["exact_implementation"]["invalidated_failure_ids"]),126);self.assertTrue(c["do_not_repeat_unchanged"])
    def test_central_failure_ledger_append_only(self):
        txt=CENTRAL.read_text(encoding="utf-8");self.assertIn("Append-only research memory",txt);self.assertIn("V837al-STATIC-LINEAR-INTERFACE-EXHAUSTED",txt)
    def test_failure_entries_have_required_evidence(self):
        for e in read_json(LEDGER)["entries"]:
            self.assertIn("metrics",e);self.assertIn("failed_conditions",e);self.assertIn("reproduction_command",e);self.assertIn("failure_type",e);self.assertIn("source_artifact_hashes",e)


class TestV837amData(unittest.TestCase):
    def test_all_partitions_exact(self):
        exp={"AM_A_FIT":[10000,10063],"AM_A_SELECT":[10064,10127],"AM_B_FIT":[10128,10191],"AM_B_SELECT":[10192,10255],"AM_C_FIT":[10256,10319],"AM_C_SELECT":[10320,10383],"META_CONFIRM":[10384,10447],"FINAL_DEV_CONFIRM":[10448,10511],"FINAL_VALIDATION":[20000,20127],"FRESH_AUDIT":[90000,90499]};self.assertEqual(PARTITIONS,exp)
    def test_development_partitions_disjoint(self):
        names=[k for k in PARTITIONS if k not in {"FINAL_VALIDATION","FRESH_AUDIT"}];sets={k:set(range(PARTITIONS[k][0],PARTITIONS[k][1]+1)) for k in names}
        for i,a in enumerate(names):
            for b in names[i+1:]:self.assertTrue(sets[a].isdisjoint(sets[b]))
    def test_final_validation_not_used_before_freeze(self): self.assertTrue(read_json(HERE/"diagnostics/data_partition_lock.json")["final_validation_not_used_before_freeze"])
    def test_no_fresh_audit(self): self.assertFalse(read_json(HERE/"diagnostics/data_partition_lock.json")["fresh_audit_consumed"])


class TestV837amGrid(unittest.TestCase):
    def test_exact_126_am_a_configs(self): self.assertEqual(len(a_configs()),126);self.assertEqual(len({x["config_id"] for x in a_configs()}),126)
    def test_exact_20_am_b_configs(self): self.assertEqual(len(b_configs()),20);self.assertEqual(len(AM_B_BOUNDARIES),10)
    def test_exact_7_am_c_configs(self): self.assertEqual(len(c_configs()),7);self.assertEqual(len(AM_C),7)
    def test_context_sets_and_port_bundles_exact(self): self.assertEqual(len(CONTEXT_SETS),5);self.assertEqual(len(PORT_BUNDLES),6)
    def test_whole_system_diagnostic_not_promotable(self): self.assertTrue(next(x for x in c_configs() if x["boundary"]=="WHOLE_SYSTEM")["diagnostic_only"])


class TestV837amConditionalAdapter(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(837_012);self.z=self.rng.normal(size=(512,4));self.c=self.rng.normal(size=(512,3))
    def test_context_additive_closed_form(self):
        A=np.eye(4)+.05;D=self.rng.normal(size=(3,4))*.1;b=np.arange(4.)*.1;y=self.z@A+b+self.c@D;m=fit_map(self.z,y,self.c,"CONTEXT_ADDITIVE");self.assertLess(np.mean((apply_np(self.z,self.c,m)-y)**2),1e-10)
    def test_bilinear_rank1_factorized(self):
        cf=self.rng.normal(size=(3,1));dm=self.rng.normal(size=(1,4,4))*.05;alpha=self.c@cf;y=self.z+np.einsum("nr,rij,ni->nj",alpha,dm,self.z);m=fit_map(self.z,y,self.c,"CONTEXT_MOD_R1");self.assertEqual(m["deployment_representation"],"FACTORED_LOW_RANK");self.assertEqual(np.asarray(m["context_factors"]).shape[1],1);self.assertLess(np.mean((apply_np(self.z,self.c,m)-y)**2),1e-8)
    def test_rank2_rank4_compression(self):
        for fam,r in (("CONTEXT_MOD_R2",2),("CONTEXT_MOD_R4",4)):
            y=self.z+self.c[:,0,None]*self.z*.02;m=fit_map(self.z,y,self.c,fam);self.assertLessEqual(m["compression"]["effective_rank"],r);self.assertEqual(np.asarray(m["context_factors"]).shape[1],m["compression"]["effective_rank"])
    def test_adapter_deterministic(self):
        y=self.z+.1*self.c[:,0,None];a=fit_map(self.z,y,self.c,"CONTEXT_MOD_R2");b=fit_map(self.z,y,self.c,"CONTEXT_MOD_R2");self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True))
    def test_dynamic_state_inverse(self):
        y=self.z@np.diag([1.2,.9,1.1,1.3])+.05*self.c[:,0,None];m=fit_map(self.z,y,self.c,"CONTEXT_ADDITIVE",state=True);rt=inverse_np(apply_np(self.z,self.c,m),self.c,m);self.assertLess(np.max(np.abs(rt-self.z)),1e-8)
    def test_state_map_condition_guard_metric(self):
        y=self.z@np.diag([1.1,1.2,.9,1.05]);m=fit_map(self.z,y,self.c,"CONTEXT_ADDITIVE",state=True);d=conditioning(self.c,m);self.assertTrue(d["valid"]);self.assertGreaterEqual(d["minimum_singular_value"],.05);self.assertLessEqual(d["maximum_condition_number"],1000)
    def test_optimizer_steps_zero(self): self.assertEqual(read_json(GATE)["adapter_gradient_steps"],0)


class TestV837amContextIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=read_json(HERE/"raw/frozen_pairs.json")["pairs"][0];cls.occ=p["recipient"];cls.trace=collect_trace(cls.occ["organism_id"],cls.occ["nodes"],list(range(10000,10064)))
    def test_self_port_exclusion(self):
        base=context_tensor(self.trace,self.occ["nodes"],0,"C5_FULL_LOCAL_CONTEXT","output").shape[-1];self.assertEqual(base,20);self.assertEqual(context_tensor(self.trace,self.occ["nodes"],0,"C5_FULL_LOCAL_CONTEXT","external_messages").shape[-1],16);self.assertEqual(context_tensor(self.trace,self.occ["nodes"],0,"C5_FULL_LOCAL_CONTEXT","projected_input").shape[-1],14)
    def test_gate_context_position_invariant(self):
        g=gate_context_tensor(self.trace,self.occ["nodes"],"C5_FULL_LOCAL_CONTEXT");self.assertEqual(g.shape[-1],19)
    def test_context_uses_recipient_observable_signals_only(self):
        d=read_json(HERE/"diagnostics/context_feature_integrity.json");self.assertTrue(d["recipient_observable_only"]);self.assertTrue(d["no_future_state"]);self.assertTrue(d["no_target"]);self.assertTrue(d["no_task_label"])
    def test_history_lengths_2_4_8(self): self.assertEqual(HISTORIES,(2,4,8))


class TestV837amConditionalArtifacts(unittest.TestCase):
    def test_every_corrected_failed_config_has_ledger_entry_when_selection_exists(self):
        path=HERE/"raw/am_a_selection.json"
        if not path.is_file():self.skipTest("corrected AM-A selection not complete yet")
        ids={e["failure_id"] for e in read_json(LEDGER)["entries"]};sel=read_json(path)
        for r in sel["results"]:
            if not r["pass"]:self.assertIn(f"V837am-SCHEMA2-{r['config_id']}",ids)
    def test_every_am_b_c_failure_has_ledger_entry(self):
        ids={e["failure_id"] for e in read_json(LEDGER)["entries"]}
        for rel in ("raw/am_b_selection.json","raw/am_c_selection.json"):
            p=read_json(HERE/rel)
            for r in p["results"]:
                if not r["pass"]:self.assertIn(f"V837am-{r['config_id']}",ids)
    def test_no_hidden_post_test_configs(self):
        self.assertEqual(len(a_configs()),126);self.assertEqual(len(b_configs()),20);self.assertEqual(len(c_configs()),7)
    def test_no_branch_winner_and_no_validation_use(self):
        for rel in ("raw/am_a_selection.json","raw/am_b_selection.json","raw/am_c_selection.json"):
            self.assertIsNone(read_json(HERE/rel)["winner"])
        self.assertIsNone(read_json(HERE/"raw/selected_final_hypothesis.json")["selected"])
        self.assertFalse(read_json(HERE/"raw/final_validation.json")["run"])
        self.assertFalse(read_json(HERE/"raw/closed_loop_results.json")["run"])
    def test_final_diagnosis_and_failure_accounting(self):
        d=read_json(HERE/"diagnostics/decision_state.json")
        self.assertEqual(d["diagnosis"],"DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE")
        self.assertEqual((d["am_a_pass_count"],d["am_b_pass_count"],d["am_c_pass_count"]),(0,0,0))
        self.assertEqual(d["failure_entries"],286);self.assertEqual(d["scientific_failures"],155);self.assertEqual(d["engineering_failures"],5);self.assertEqual(d["invalidated_measurement_records"],126)
    def test_invalidated_measurements_preserved_but_separate(self):
        inv=read_json(HERE/"raw/invalidated_am_a_selection_schema1.json");self.assertEqual(len(inv["results"]),126)
        text=(HERE/"FAILURE_ANALYSIS.md").read_text(encoding="utf-8");self.assertIn("Preserved invalidated schema-1",text)
    def test_science_locks_final(self):
        d=read_json(HERE/"diagnostics/decision_state.json");self.assertEqual(d["new_model_fits"],0);self.assertEqual(d["optimizer_steps"],0);self.assertEqual(d["primitives_promoted"],0);self.assertFalse(d["fresh_audit_consumed"]);self.assertFalse(d["v838_started"]);self.assertFalse(d["primitive_archive_allowed_next"])

if __name__=="__main__":unittest.main()
