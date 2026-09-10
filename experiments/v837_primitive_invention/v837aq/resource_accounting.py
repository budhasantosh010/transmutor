from __future__ import annotations

from .utils import HERE, read_json, write_json


def _load(name,default=None):
    p=HERE/name;return read_json(p) if p.is_file() else default


def build_resource_accounting()->dict:
    reality=_load("raw/operator_reality_gate.json",{});disc=_load("raw/operator_discovery.json",{});loc=_load("raw/localization_results.json",{});comp=_load("raw/composition_results.json",{});meta=_load("raw/meta_confirmation.json",{});held=_load("raw/heldout_confirmation.json",{});tensors=_load("raw/response_tensor_manifest.json",{})
    reality_rows=sum(len(r.get("rows",[])) for r in reality.get("organisms",[])) if isinstance(reality.get("organisms"),list) else 0
    tensor_rows=sum(v.get("rows",0) for v in tensors.get("families",{}).values())
    localization_singletons=sum(int(r.get("candidate_units",0)) for r in loc.get("rows",[]))
    localization_prefixes=sum(len(r.get("prefixes",[])) for r in loc.get("rows",[]))
    localization_random_masks=sum(16*len(r.get("prefixes",[])) for r in loc.get("rows",[]))
    payload={
      "version":"V837aq","all_accounted":True,
      "new_source_model_fits":0,"source_optimizer_steps":0,"source_training_examples":0,"source_architecture_changes":0,"gpu_seconds":0.0,
      "reality_response_rows":reality_rows,"coordinate_free_response_tensor_rows":tensor_rows,
      "operator_discovery_organism_rows":len(disc.get("rows",[])),"incompetent_control_rows":len(disc.get("incompetent_controls",[])),
      "localization_singleton_masks":localization_singletons,"localization_prefix_masks":localization_prefixes,"localization_random_control_masks":localization_random_masks,
      "composition_organism_rows":len(comp.get("rows",[])),"meta_organism_rows":len(meta.get("rows",[])),"heldout_organism_rows":len(held.get("rows",[])),
      "hidden_state_set_primary_evidence":False,"cross_organism_state_alignment":False,"cross_organism_q_alignment":False,
      "fresh_audit_episodes":0,"primitives_promoted":0,"primitive_archive_population":False,"v838_started":False,
      "note":"Counts are exact artifact/evaluation-row counts; low-level framework forward-call batching is intentionally not reconstructed from wall-clock logs."
    }
    write_json(HERE/"v837aq_resource_accounting.json",payload);write_json(HERE/"diagnostics/resource_accounting.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(build_resource_accounting(),indent=2))
