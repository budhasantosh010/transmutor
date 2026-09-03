from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_active_research.py"
spec = importlib.util.spec_from_file_location("validate_active_research", VALIDATOR_PATH)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArchiveIntegrityTests(unittest.TestCase):
    def test_preserved_archive_and_registry_hashes(self) -> None:
        manifest = json.loads(
            (ROOT / "experiments" / "v836_recovery" / "integrity_manifest.json").read_text(encoding="utf-8")
        )
        for group in ("preserved_files", "preserved_registry_files"):
            for relative, expected in manifest[group].items():
                self.assertEqual(sha256_file(ROOT / relative), expected, relative)

    def test_historical_v836_remains_pass(self) -> None:
        historical = json.loads(
            (ROOT / "archive" / "preserved_artifacts" / "transmutor_experiments_v836plus" / "v836_results.json").read_text(encoding="utf-8")
        )
        self.assertIs(historical["V836_PASS"], True)


class ReproductionGuardTests(unittest.TestCase):
    def test_missing_source_is_recorded_not_fabricated(self) -> None:
        reproduction = json.loads(
            (ROOT / "experiments" / "v836_recovery" / "v836_reproduction_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(reproduction["reproduction_classification"], "CANNOT_REPRODUCE_MISSING_SOURCE")
        self.assertIsNone(reproduction["reproduction_metric"])

    def test_v836_repair_remains_blocked_even_if_independent_v837_is_authorized(self) -> None:
        proposal = json.loads(
            (ROOT / "experiments" / "v836_recovery" / "PROPOSED_NEXT_EXPERIMENT.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            proposal["status"],
            {"NOT_RUN_BLOCKED_BY_V836_REPRODUCTION", "SUPERSEDED_BY_INDEPENDENT_V837_LINEAGE_AUTHORIZATION"},
        )
        self.assertEqual(
            json.loads((ROOT / "experiments/v837_primitive_invention/lineage_status.json").read_text(encoding="utf-8"))["historical_boundary"]["v837_relation"],
            "independent_post_v836_lineage",
        )
        self.assertFalse((ROOT / "experiments/v836_recovery/variants/v836b").exists())
        self.assertFalse((ROOT / "experiments/v836_recovery/variants/v836c").exists())


class SeedAndGateTests(unittest.TestCase):
    def test_future_audit_seed_ranges_do_not_overlap(self) -> None:
        proposal = json.loads(
            (ROOT / "experiments" / "v836_recovery" / "PROPOSED_NEXT_EXPERIMENT.json").read_text(encoding="utf-8")
        )
        validator.assert_seed_ranges_disjoint(proposal["seed_ranges"])

    def test_seed_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validator.assert_seed_ranges_disjoint({"development": [1, 10], "fresh_audit": [10, 20]})

    def test_pass_gate_mutation_is_rejected(self) -> None:
        config = {
            "pass_gate_frozen_before_run": True,
            "pass_gate": {"metric": "accuracy", "min": 0.9},
            "seed_ranges": {"development": [1, 10], "fresh_audit": [20, 30]},
        }
        result = {
            "version": "TEST",
            "parent": "TEST0",
            "research_question": "q",
            "hypothesis": "h",
            "single_change": "one",
            "baselines": {},
            "pass_gate": copy.deepcopy(config["pass_gate"]),
            "development_result": {},
            "fresh_audit_result": {},
            "resource_accounting": {},
            "pass": True,
            "failure_classification": [],
            "caveats": [],
            "next_question": "n",
        }
        validator.validate_result_schema(result, config=config)
        result["pass_gate"]["min"] = 0.8
        with self.assertRaises(ValueError):
            validator.validate_result_schema(result, config=config)


class ResultSchemaTests(unittest.TestCase):
    def base_result(self) -> dict:
        return {
            "version": "TEST",
            "parent": "TEST0",
            "research_question": "q",
            "hypothesis": "h",
            "single_change": "one",
            "baselines": {},
            "pass_gate": {},
            "development_result": {},
            "fresh_audit_result": {},
            "resource_accounting": {},
            "pass": True,
            "failure_classification": [],
            "caveats": [],
            "next_question": "n",
        }

    def test_resource_accounting_is_required(self) -> None:
        result = self.base_result()
        del result["resource_accounting"]
        with self.assertRaises(ValueError):
            validator.validate_result_schema(result)

    def test_failed_result_requires_failure_classification(self) -> None:
        result = self.base_result()
        result["pass"] = False
        with self.assertRaises(ValueError):
            validator.validate_result_schema(result)
        result["failure_classification"] = ["SEARCH_FAILURE"]
        validator.validate_result_schema(result)


class FrontierMatrixTests(unittest.TestCase):
    def test_frontier_covers_every_numeric_version_450_through_836(self) -> None:
        matrix = json.loads((ROOT / "experiments" / "post_v836_frontier.json").read_text(encoding="utf-8"))
        numbers = {int(record["number"]) for record in matrix["records"]}
        self.assertEqual(numbers, set(range(450, 837)))
        self.assertEqual(len(matrix["records"]), 428)


if __name__ == "__main__":
    unittest.main()
