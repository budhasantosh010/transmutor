from __future__ import annotations

import inspect,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/"experiments/v837_primitive_invention/v837ar";START="287d37743751554c68742bc275b558d7f9bb1f95"
from experiments.v837_primitive_invention.v837ar.program_ir import ProgramIR,canonical_bytes,ir_sha256,validate_ir
from experiments.v837_primitive_invention.v837ar.program_ir_interpreter import run
from experiments.v837_primitive_invention.v837ar.shared_fit import equal_organism_weights
from experiments.v837_primitive_invention.v837ar.word_partitions import PARTITIONS,assert_disjoint
from experiments.v837_primitive_invention.v837ar.operator_words import ROLE_SEEDS
from experiments.v837_primitive_invention.v837ar.memory_ir import GRID
from experiments.v837_primitive_invention.v837ar.composition import COMPOSITION_SEEDS

def j(rel):return json.loads((HERE/rel).read_text())
def _need(rel):
    if not (HERE/rel).is_file():pytest.skip(f"artifact pending: {rel}")
    return j(rel)
def test_start_sha_exact():assert START=="287d37743751554c68742bc275b558d7f9bb1f95"
def test_v837aq_contracts():
    r=json.loads((ROOT/"experiments/v837_primitive_invention/v837aq/results.json").read_text());c=json.loads((ROOT/"experiments/v837_primitive_invention/v837aq/raw/frozen_program_operator_contracts.json").read_text());assert r["diagnosis"]=="CAUSAL_OPERATOR_FOUND_NOT_COMPOSITIONALLY_CLOSED" and r["heldout_confirmed_operator_family_count"]==3 and r["second_order_required_families"]==["conditional_routing"] and r["predictive_response_rank_99"]["delayed_recall"]==1 and c["family_contracts"]["iterative_state"]["compositionally_closed_in_discovery"] is True and c["family_contracts"]["variable_composition"] is None
def test_science_locks_predecessor():
    r=json.loads((ROOT/"experiments/v837_primitive_invention/v837aq/results.json").read_text());assert r["fresh_audit_consumed"] is False and not (ROOT/"experiments/v837_primitive_invention/v838").exists()
def test_basis_contract():
    x=_need("raw/canonical_response_basis.json");assert x["response_basis_sha256"]==_need("diagnostics/response_basis_integrity.json")["basis_hash"] and x["organism_id_in_basis"] is False and x["semantic_units_only"] is True and "interaction_order" in x["ordering"]
def test_partitions():
    assert assert_disjoint(_need("raw/frozen_operator_word_partitions.json"));assert PARTITIONS["iterative_state"]["FIT_WORDS"]==["single_early","single_middle","single_late","two_adjacent"];assert PARTITIONS["delayed_recall"]["FINAL_UNSEEN_WORDS"]==["delay_10","delay_11","delay_12"];assert "control_A_B_triple" in PARTITIONS["conditional_routing"]["FINAL_UNSEEN_WORDS"];assert list(COMPOSITION_SEEDS)==list(range(11320,11384));assert not set(COMPOSITION_SEEDS).intersection(*(set(v) for v in ROLE_SEEDS.values()))
def test_iterative_grammar_and_no_oracle_fit():
    f=_need("raw/iterative_ir_fits.json")["fits"];assert set(("I0_CONSTANT_RESPONSE","I1_IDENTITY_STATE","I2_AFFINE_UPDATE","I3_QUADRATIC_UPDATE")).issubset(f)
    from experiments.v837_primitive_invention.v837ar import iterative_ir
    s=inspect.getsource(iterative_ir.fit_candidates);assert "0.65" not in s and "0.35" not in s
def test_interpreter_affine_and_route():
    ir=ProgramIR(family="iterative_state",operator_name="U",operator_class="LINEAR_STATE_UPDATE_1D",granularity="STATE_UPDATE",parameters={"a":.5,"b":.5,"c":0.0},predictive_state={"dimension":1}).to_dict();assert abs(run(ir,{"ops":[{"phase":"UPDATE","x":1.0},{"phase":"UPDATE","x":-1.0}]})+.25)<1e-12
    rr=ProgramIR(family="conditional_routing",operator_name="ROUTE",operator_class="BILINEAR_STATIC_MAP",granularity="ATOMIC",parameters={"b0":0,"bc":0,"bA":0,"bB":0,"bcA":1,"bcB":0}).to_dict();assert run(rr,{"control":2,"A":3,"B":4})==6
def test_routing_ladder():
    f=_need("raw/routing_ir_fits.json");assert set(("R0_FINE","R1_TWO_STAGE","R2_FIRST_ORDER","R2_BILINEAR","R2_FULL_SECOND_ORDER")).issubset(f["fits"]) and f["third_order_used"] is False
def test_memory_ladder_and_lambda():
    f=_need("raw/memory_ir_fits.json");assert set(("M0_CONSTANT","M1_DELAY_TABLE","M2A_RANK1_SIMPLE","M2_RANK1_MEMORY")).issubset(f["fits"]) and f["read_gain_fixed"]==1.0 and len(GRID)==2049 and GRID[0]==-1.25 and GRID[-1]==1.25
def test_ir_rejects_neural_payloads():
    for key in ("q_vector","state40","checkpoint_parameters"):
        with pytest.raises(ValueError):validate_ir({"operator_class":"LINEAR_STATE_UPDATE_1D",key:[1]})
def test_interpreter_is_neural_free():
    src=(HERE/"program_ir_interpreter.py").read_text().lower();assert all(x not in src for x in ("torch","checkpoint","state40","q_vector","semantic_execution"))
def test_canonical_hash_and_equal_weighting():
    ir=ProgramIR(family="x",operator_name="x",operator_class="LINEAR_STATE_UPDATE_1D",granularity="x",parameters={"a":1,"b":1,"c":0}).to_dict();assert canonical_bytes(ir) and ir_sha256(ir)==ir_sha256(ir);w=equal_organism_weights(["a","a","b"]);assert abs(w[:2].sum()-.5)<1e-12 and abs(w[2]-.5)<1e-12
def test_hankel_and_composition_contracts():
    assert all(all(k in d for k in ("rank90","rank95","rank99","rank999")) for d in _need("raw/predictive_ranks.json")["families"].values());c=_need("raw/composition_results.json");assert c["composition_seed_range"]==[11320,11383] and c["fit_or_selection_rows_used"] is False;d=next(iter(c["families"].values()));assert all(k in d for k in ("C1_repeated_operator","C2_family_program_words","C3_suboperator_factorization","minimum_closed_granularity"))
def test_meta_freeze_and_final_isolation():
    m=_need("raw/meta_confirmation.json");f=_need("raw/frozen_canonical_program_irs.json");u=_need("raw/final_unseen_word_results.json");h=_need("raw/reused_aq_heldout_results.json");assert m["no_refit"] and m["no_fallback"] and f["final_unseen_words_opened"] is False and f["freeze_sha256"]==_need("diagnostics/program_ir_freeze.json")["freeze_sha256"] and u["no_refit"] and h["label"]=="REUSED_AQ_HELDOUT_ORGANISM_CONFIRMATION" and h["no_refit"] and h["no_gain_calibration"] and h["no_bias_calibration"]
def test_shared_fit_and_compression():
    a=_need("raw/cross_organism_agreement.json");assert a["diagnostic_only"] and a["shared_ir_is_deployed"]
    vals=_need("raw/compression_results.json")["families"].values();assert all("ir_parameters" in d and "ir_bytes" in d and "execution_cost" in d and ((not d["archiveable"]) or d["ir_parameters"]<=256) for d in vals)
def test_failure_memory():
    d=_need("raw/failure_ledger.json");assert d["append_only"] and all(e["failure_type"] in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE","INVALID_EVIDENCE"} and e["do_not_retry_unchanged"] and e["reproduction_command"] for e in d["entries"])
def test_final_locks_and_negative_control():
    d=_need("diagnostics/decision_state.json");assert d["new_source_model_fits"]==0 and d["source_optimizer_steps"]==0 and d["primitive_archive_population"] is False and d["primitives_promoted"]==0 and d["fresh_audit_consumed"] is False and d["v838_started"] is False and _need("raw/variable_composition_negative_control.json")["new_variable_composition_primitive_fit"] is False
    s=json.loads((ROOT/"experiments/v837_primitive_invention/causal_operator_canonicalization_program_status.json").read_text());assert s["version"]=="V837ar" and s["status"]=="COMPLETE" and s["fresh_audit_consumed"] is False and s["v838_started"] is False
