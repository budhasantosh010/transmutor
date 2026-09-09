from __future__ import annotations

from .utils import HERE,read_json,sha256_json,write_json

FAMILY_ORDER={"SIGNED_PERMUTATION":0,"DIAGONAL_AFFINE":1,"RIGID_AFFINE":2,"FULL_AFFINE":3,"IDENTITY":-1}


def _winner(results,track):
    passing=[r for r in results if r[track]["pass"]]
    if not passing:return None
    passing.sort(key=lambda r:(r["complexity"]["adapter_macs_per_timestep"],r["complexity"]["continuous_dof"],r["complexity"]["enabled_ports"],FAMILY_ORDER[r["family"]],r["scope"]))
    r=passing[0];return {"config_id":r["config_id"],"family":r["family"],"scope":r["scope"],"ports":r["ports"],"adapter_macs_per_timestep":r["complexity"]["adapter_macs_per_timestep"],"continuous_dof":r["complexity"]["continuous_dof"],"selection_metrics":r[track]}


def _canonical_anchors():
    source=read_json(HERE/"raw/source_classes.json");anchors=[]
    for c in source["classes"]:
        by={}
        for r in c["compatible_occurrences"]:by.setdefault(r["family"],[]).append(r)
        for fam,rows in sorted(by.items()):
            rows.sort(key=lambda r:(float(r.get("confirmation_class_distance",1e30)),r["engine"],r["organism_id"],r["occurrence_id"]))
            if len(rows)>=2:anchors.append({"class_id":c["class_id"],"family":fam,"canonical_occurrence":rows[0],"selection_rule":"confirmation-compatible; min frozen class distance; deterministic engine/organism tie-break","frozen_before_test":True})
    payload={"version":"V837al","anchors":anchors};payload["sha256"]=sha256_json(payload);write_json(HERE/"raw/canonical_anchors.json",payload);return payload


def freeze_selected():
    loc=read_json(HERE/"raw/scope_selection_results.json");global_w=_winner(loc["results"],"global_track");causal_w=_winner(loc["results"],"causal_track");payload={"version":"V837al","frozen_before_align_test":True,"global_track":global_w,"causal_track":causal_w}
    payload["selected_interface_sha256"]=sha256_json(payload);write_json(HERE/"raw/selected_interface_configs.json",payload);write_json(HERE/"diagnostics/selected_interface.json",payload);_canonical_anchors();return payload

if __name__=="__main__":freeze_selected()
