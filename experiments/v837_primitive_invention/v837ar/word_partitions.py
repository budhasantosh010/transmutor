from __future__ import annotations
from .utils import HERE,sha256_json,write_json,read_json

PARTITIONS={
"iterative_state":{"FIT_WORDS":["single_early","single_middle","single_late","two_adjacent"],"SELECT_WORDS":["two_separated","two_opposite_sign"],"META_WORDS":["three_seen_magnitudes"],"FINAL_UNSEEN_WORDS":["three_novel_order","four_step","four_alternating_sign","novel_time_locations","midpoint_magnitude"]},
"delayed_recall":{"FIT_WORDS":["delay_4","delay_6","delay_8"],"SELECT_WORDS":["delay_5","delay_7"],"META_WORDS":["delay_9"],"FINAL_UNSEEN_WORDS":["delay_10","delay_11","delay_12"]},
"conditional_routing":{"FIT_WORDS":["control_only","payload_A_only","payload_B_only","control_x_A","control_x_B"],"SELECT_WORDS":["A_x_B","opposite_sign_pairs"],"META_WORDS":["control_plus_payload_held_sign"],"FINAL_UNSEEN_WORDS":["control_A_B_triple","unseen_known_magnitude_combo","midpoint_payload_magnitude","swapped_order"]}}

def freeze_word_partitions():
    out={"version":"V837ar","partitions":PARTITIONS,"partition_sha256":sha256_json(PARTITIONS),"opened_final_before_ir_freeze":False}
    write_json(HERE/"raw/frozen_operator_word_partitions.json",out);return out

def assert_disjoint(payload=None):
    p=(payload or read_json(HERE/"raw/frozen_operator_word_partitions.json"))["partitions"]
    for fam,d in p.items():
        seen=set()
        for role,words in d.items():
            if seen.intersection(words):raise RuntimeError(f"V837AR_WORD_LEAK:{fam}:{role}")
            seen.update(words)
    return True
