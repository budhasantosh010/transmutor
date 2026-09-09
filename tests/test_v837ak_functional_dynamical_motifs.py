from __future__ import annotations

import itertools
import json
import math
import unittest
from pathlib import Path

from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.candidate_discovery import _rank_key
from experiments.v837_primitive_invention.v837ak.ported_primitive import calibration_occurrences
from experiments.v837_primitive_invention.v837ak.probe_partition import freeze_probe_partition
from experiments.v837_primitive_invention.v837ak.structural_signatures import structural_class_id, structural_signature
from experiments.v837_primitive_invention.v837ak.utils import HERE, DIRECTED, RANDOM, jaccard, sha256_json

ROOT = Path(__file__).resolve().parents[1]


class TestV837akFrozenProgram(unittest.TestCase):
    def test_authorization_is_exact(self):
        auth = assert_v837ak_authorized()
        self.assertTrue(auth["authorized"])
        self.assertEqual(auth["v837aj_diagnosis"], "RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT")
        self.assertFalse(auth["fresh_audit_consumed"])
        self.assertFalse(auth["v838_started"])

    def test_probe_partition_is_frozen_and_disjoint(self):
        p = freeze_probe_partition()
        self.assertEqual(p["discovery_probe"], list(range(20000, 20064)))
        self.assertEqual(p["confirmation_probe"], list(range(20064, 20128)))
        self.assertEqual(p["D1"], list(range(20000, 20032)))
        self.assertEqual(p["D2"], list(range(20032, 20064)))
        self.assertEqual(p["C1"], list(range(20064, 20096)))
        self.assertEqual(p["C2"], list(range(20096, 20128)))
        self.assertTrue(p["disjoint"])
        self.assertFalse(p["candidate_creation_from_confirmation"])

    def test_frozen_gate_science_locks(self):
        gate = json.loads((HERE / "frozen_primitive_discovery_gate.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["starting_sha"], "4e989e474b4d93cd04246219457990746015c33f")
        self.assertEqual(gate["source_population_count"], 50)
        self.assertEqual(gate["competent_count"], 40)
        self.assertEqual(gate["incompetent_count"], 10)
        self.assertFalse(gate["archive_promotion"])
        self.assertEqual(gate["primitives_promoted"], 0)
        self.assertFalse(gate["fresh_audit_consumed"])
        self.assertFalse(gate["v838_started"])


class TestV837akStructuralIdentity(unittest.TestCase):
    def test_signature_ignores_weights_and_metadata(self):
        topology = {
            "edges": [
                {"src": 0, "dst": 2, "recurrent": False, "weight": 0.2},
                {"src": 2, "dst": 2, "recurrent": True, "weight": -4.0},
                {"src": 3, "dst": 2, "recurrent": True, "weight": 7.0},
            ],
            "family": "ignored",
        }
        changed = json.loads(json.dumps(topology))
        for e in changed["edges"]:
            e["weight"] *= 1000
        changed["family"] = "also_ignored"
        a = structural_signature(topology, [0, 2])
        b = structural_signature(changed, [0, 2])
        self.assertEqual(a, b)
        self.assertEqual(structural_class_id(a), structural_class_id(b))

    def test_signature_preserves_same_step_vs_recurrent(self):
        topology = {"edges": [
            {"src": 0, "dst": 1, "recurrent": False},
            {"src": 1, "dst": 0, "recurrent": True},
        ]}
        sig = structural_signature(topology, [0, 1])
        self.assertEqual(sig["internal_same_step_edges"], [[0, 1]])
        self.assertEqual(sig["internal_recurrent_edges"], [[1, 0]])

    def test_all_subset_count_is_1023(self):
        total = sum(math.comb(10, k) for k in range(1, 11))
        self.assertEqual(total, 1023)


class TestV837akSelectionAndReplay(unittest.TestCase):
    def test_candidate_rank_is_deterministic(self):
        a = {"support": 8, "engine_counts": {DIRECTED: 3, RANDOM: 3}, "family_counts": {"a": 3}, "discovery_stability": 0.5, "size": 3, "class_id": "a"}
        b = {"support": 7, "engine_counts": {DIRECTED: 3, RANDOM: 3}, "family_counts": {"a": 3}, "discovery_stability": 0.9, "size": 1, "class_id": "b"}
        self.assertLess(_rank_key(a), _rank_key(b))

    def test_jaccard_dedup_threshold_behavior(self):
        self.assertEqual(jaccard(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)

    def test_replay_calibration_spans_sizes_and_engines(self):
        rows = calibration_occurrences()
        self.assertEqual(len(rows), 20)
        self.assertEqual(sorted({r["size"] for r in rows}), list(range(1, 11)))
        self.assertEqual({r["engine"] for r in rows}, {DIRECTED, RANDOM})


class TestV837akCanonicalArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recon = json.loads((HERE / "raw/reconstruction_results.json").read_text(encoding="utf-8"))
        cls.census = json.loads((HERE / "raw/subset_census_summary.json").read_text(encoding="utf-8"))
        cls.frozen = json.loads((HERE / "raw/frozen_candidate_classes.json").read_text(encoding="utf-8"))

    def test_reconstruction_population_exact(self):
        self.assertTrue(self.recon["complete"])
        self.assertEqual(self.recon["organisms_reconstructed"], 50)
        self.assertEqual(self.recon["competent"], 40)
        self.assertEqual(self.recon["incompetent"], 10)
        self.assertTrue(all(r["equivalent"] for r in self.recon["rows"]))

    def test_census_exact(self):
        self.assertEqual(self.census["total_occurrences"], 51150)
        self.assertEqual(self.census["subsets_per_organism"], 1023)
        self.assertEqual(self.census["whole_system_occurrences"], 50)
        self.assertEqual(self.census["size_distribution"], {str(k): 50 * math.comb(10, k) for k in range(1, 11)})

    def test_frozen_candidate_hash_and_caps(self):
        core = {k: v for k, v in self.frozen.items() if k != "frozen_candidate_classes_sha256"}
        self.assertEqual(sha256_json(core), self.frozen["frozen_candidate_classes_sha256"])
        self.assertLessEqual(len(self.frozen["classes"]), 12)
        self.assertLessEqual(sum(c["stream"] == "S" for c in self.frozen["classes"]), 6)
        self.assertLessEqual(sum(c["stream"] == "D" for c in self.frozen["classes"]), 6)


if __name__ == "__main__":
    unittest.main()
