from __future__ import annotations

from experiments.v837_primitive_invention.v837ak.utils import HERE,write_json


def freeze_probe_partition()->dict:
    discovery=list(range(20000,20064));confirmation=list(range(20064,20128));d1=list(range(20000,20032));d2=list(range(20032,20064));c1=list(range(20064,20096));c2=list(range(20096,20128));fresh=set(range(90000,90500))
    payload={"version":"V837ak","discovery_probe":discovery,"D1":d1,"D2":d2,"confirmation_probe":confirmation,"C1":c1,"C2":c2,"discovery_count":64,"confirmation_count":64,"union_count":128,"disjoint":set(discovery).isdisjoint(confirmation),"fresh_audit_intersection":sorted((set(discovery)|set(confirmation))&fresh),"candidate_creation_from_confirmation":False,"fresh_audit_consumed":False}
    if not payload["disjoint"] or payload["union_count"]!=128 or payload["fresh_audit_intersection"]:raise RuntimeError("V837AK_PROBE_PARTITION_INVALID")
    write_json(HERE/"diagnostics/probe_partition.json",payload);return payload
