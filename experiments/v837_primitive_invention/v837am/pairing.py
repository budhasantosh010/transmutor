from __future__ import annotations

from .authorization import SOURCE, assert_authorized
from .utils import HERE, ROOT, read_json, sha256_json, write_json


def freeze_pairs():
    auth=assert_authorized(); src=read_json(ROOT/SOURCE["v837al_pairs"]); pairs=[p for p in src["pairs"] if p.get("primary_causal")]
    if len(pairs)!=38: raise RuntimeError("V837AM_PAIR_POWER_DRIFT")
    payload={"version":"V837am","source":"V837al frozen pairs; unchanged","source_frozen_pairs_sha256":src["frozen_pairs_sha256"],"pair_count":38,"primary_causal_class":auth["primary_causal_class"],"pairs":pairs}
    payload["frozen_pairs_sha256"]=sha256_json(payload);write_json(HERE/"raw/frozen_pairs.json",payload)
    power={"version":"V837am","independent_recipients":38,"minimum_possible_p":2.0**-38,"strong_claim_powered":True,"p_threshold":0.01}
    write_json(HERE/"diagnostics/pair_power.json",power);return payload

if __name__=="__main__": freeze_pairs()
