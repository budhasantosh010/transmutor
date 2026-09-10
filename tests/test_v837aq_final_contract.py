from __future__ import annotations
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/"experiments/v837_primitive_invention/v837aq"
def j(rel):return json.loads((HERE/rel).read_text(encoding="utf-8"))

def test_contract_hash_fixed():assert j("raw/frozen_program_operator_contracts.json")["contract_sha256"]=="461faadf9cb9cc1262c9a288236c08875ae5212685bb8dd06e6a4ed92768d9af"
def test_gate_hash_fixed():assert j("frozen_program_level_causal_operator_gate.json")["gate_sha256"]=="a65124362fe2991d822d326b7bce0640492de0643cd56ae09e16a70292b51154"
def test_three_frozen_operators():assert j("raw/frozen_program_operator_contracts.json")["confirmed_families"]==["conditional_routing","delayed_recall","iterative_state"]
@pytest.mark.parametrize("family",["conditional_routing","delayed_recall","iterative_state"])
def test_frozen_coordinate_free(family):
    f=j("raw/frozen_program_operator_contracts.json")["family_contracts"][family];assert f["coordinate_free"] and f["oracle_relative"] and not f["hidden_state_alignment"] and not f["q_alignment"]
def test_routing_order2_frozen():assert j("raw/frozen_program_operator_contracts.json")["family_contracts"]["conditional_routing"]["operator_order"]==2
def test_iterative_program_closed_frozen():assert j("raw/frozen_program_operator_contracts.json")["family_contracts"]["iterative_state"]["program_closed_in_meta"] is True
def test_routing_not_rescued():assert j("raw/frozen_program_operator_contracts.json")["family_contracts"]["conditional_routing"]["compositionally_closed_in_discovery"] is False
def test_recall_not_rescued():assert j("raw/frozen_program_operator_contracts.json")["family_contracts"]["delayed_recall"]["compositionally_closed_in_discovery"] is False
def test_heldout_contract_hash_matches():assert j("raw/heldout_confirmation.json")["contract_sha256"]==j("raw/frozen_program_operator_contracts.json")["contract_sha256"]
def test_resource_zero_training():
    r=j("v837aq_resource_accounting.json");assert r["new_source_model_fits"]==r["source_optimizer_steps"]==r["source_training_examples"]==r["source_architecture_changes"]==0
def test_resource_no_fresh_audit():assert j("v837aq_resource_accounting.json")["fresh_audit_episodes"]==0
def test_primary_artifact_pass():assert (HERE/"PASS.md").is_file()
def test_status_matches_result():
    s=json.loads((ROOT/"experiments/v837_primitive_invention/program_level_causal_operator_localization_program_status.json").read_text());r=j("results.json");assert s["diagnosis"]==r["diagnosis"] and s["next_program"]==r["next_program"]
