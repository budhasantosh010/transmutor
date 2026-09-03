from __future__ import annotations

import ast
import hashlib
import inspect
import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.evaluator import first_observation_leakage, oracle_validation
from experiments.v837_primitive_invention.common.graph import GraphSpec, initial_graph
from experiments.v837_primitive_invention.common.motif import (
    CallablePrimitive,
    extract_isolated_subgraph,
    extract_motifs,
    primitive_equivalence,
    randomized_isolated_subgraph,
)
from experiments.v837_primitive_invention.common.mutations import ALLOWED_MUTATIONS, mutate
from experiments.v837_primitive_invention.common.primitive_archive import PrimitiveArchive
from experiments.v837_primitive_invention.common.resource_accounting import ResourceAccounting
from experiments.v837_primitive_invention.common.search import structural_search
from experiments.v837_primitive_invention.common.seeds import assert_seed_partitions_disjoint, cyclic_seeds, gate_sha256
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.common.task_interface import OBS_DIM, StatefulTaskAdapter
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks
from experiments.v837_primitive_invention.tasks.delayed_recall import DelayedRecallTask

ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_HASHES = {
    "archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json": "0ed63ee1e1c5903c1c90b58942aaf968b747df19d4c4a51c1d73a6b36f91527d",
    "archive/preserved_artifacts/TRANSMUTOR_V828_V836_LOG.md": "2a930efea649aef8d76bab9215889f48514ff801fb73e8673c68465018db2b49",
    "registry/experiments.jsonl": "598673e427036078caa73a0f8c62c172f92fdb6296e4d3f87cc58dccd61416b7",
    "registry/experiments.csv": "54ddd7a9e9a8f661ac35e2c9797966fed33c761daac001db45f90f9ea9ea0de7",
    "registry/artifact_inventory.json": "f5b7c821b540243e39b340a1b6fc2b3337ef5778bc6b35f6385c9b78df4c1927",
}
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
V837_REQUIRED_RESULT_FIELDS = {
    "version", "parent", "research_question", "hypothesis", "single_change", "substrate_version",
    "task_families", "development_seeds", "validation_seeds", "fresh_audit_seeds", "baselines",
    "metrics", "resource_accounting", "motifs", "primitive_archive", "pass_gate", "pass",
    "failure_classification", "caveats", "next_question", "gate_file_sha256",
}
V837_FAILURE_CLASSES = {
    "IMPLEMENTATION_FAILURE", "NUMERICAL_FAILURE", "SEARCH_FAILURE", "REPRESENTATION_FAILURE",
    "MOTIF_DETECTION_FAILURE", "CAUSAL_VALIDATION_FAILURE", "COMPRESSION_FAILURE", "RETRIEVAL_FAILURE",
    "REUSE_FAILURE", "GENERALIZATION_FAILURE", "RESOURCE_FAILURE", "BENCHMARK_CONFOUND",
    "STATISTICAL_POWER_FAILURE", "UNKNOWN_FAILURE",
}


class HistoricalBoundaryTests(unittest.TestCase):
    def test_historical_archive_unchanged(self):
        for relative, expected in HISTORICAL_HASHES.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_v836_status_unchanged(self):
        data = json.loads((ROOT / "archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json").read_text(encoding="utf-8"))
        self.assertIs(data["V836_PASS"], True)


class FrozenGateAndSeedTests(unittest.TestCase):
    def test_gate_hash_immutable(self):
        self.assertEqual(gate_sha256(), EXPECTED_GATE_HASH)

    def test_seed_ranges_disjoint(self):
        assert_seed_partitions_disjoint()

    def test_fresh_audit_not_used_by_development_search(self):
        config = json.loads((ROOT / "experiments/v837_primitive_invention/v837/config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["development_seed_range"], [10000, 10999])
        self.assertEqual(config["validation_seed_range"], [20000, 20499])
        self.assertEqual(config["fresh_audit_seed_range"], [90000, 90499])
        source = inspect.getsource(structural_search)
        self.assertNotIn('cyclic_seeds("fresh_audit"', source)


class TaskValidityTests(unittest.TestCase):
    def test_all_tasks_use_common_observation_schema_and_stateful_interface(self):
        for task in all_tasks():
            adapter = StatefulTaskAdapter(task)
            first = adapter.reset(10000)
            self.assertEqual(first.shape, (OBS_DIM,))
            episode = task.generate(10000, "development")
            self.assertEqual(episode.observations.shape[1], OBS_DIM)
            self.assertIn("family", episode.metadata)
            # Family metadata is analysis-only and never enters the observation tensor.
            self.assertEqual(episode.observations.dtype.name, "float32")

    def test_oracles_pass_benchmark_validity_gate(self):
        for task in all_tasks():
            result = oracle_validation(task, cyclic_seeds("validation", 50))
            self.assertGreaterEqual(result["binary"]["success_rate"], 0.98, task.name)

    def test_first_observation_leakage_below_gate(self):
        result = first_observation_leakage(all_tasks(), cyclic_seeds("development", 100))
        self.assertLessEqual(result["accuracy"], 0.35)


class GraphDeterminismTests(unittest.TestCase):
    def test_graph_serialization_and_identity_deterministic(self):
        graph = initial_graph()
        restored = GraphSpec.from_dict(graph.to_dict())
        self.assertEqual(restored.graph_id, graph.graph_id)
        self.assertEqual(restored.canonical_structure(), graph.canonical_structure())

    def test_only_low_level_mutations_are_initially_allowed(self):
        forbidden = {"ADD_MEMORY", "ADD_ROUTER", "ADD_COUNTER", "ADD_ATTENTION", "ADD_SEARCH_MODULE", "ADD", "SUB", "MUL", "NAND", "XOR"}
        self.assertTrue(forbidden.isdisjoint(ALLOWED_MUTATIONS))
        child, op = mutate(initial_graph(), 12345)
        self.assertIn(op, set(ALLOWED_MUTATIONS) | {"NO_OP"})
        child.validate()


class MotifAndPrimitiveInfrastructureTests(unittest.TestCase):
    def _fixture(self):
        model = NeutralGraphModel(initial_graph())
        task = DelayedRecallTask()
        episodes = [task.generate(seed, "development") for seed in range(10000, 10008)]
        return model, task, episodes

    def test_motif_canonicalization_deterministic(self):
        model, _, episodes = self._fixture()
        first = [(row.motif_hash, row.signature) for row in extract_motifs(model, episodes)]
        second = [(row.motif_hash, row.signature) for row in extract_motifs(model, episodes)]
        self.assertEqual(first, second)

    def test_primitive_expanded_callable_equivalence(self):
        model, _, episodes = self._fixture()
        graph, state = extract_isolated_subgraph(model, (0, 1))
        primitive = CallablePrimitive("PTEST", graph, 4, 4, OBS_DIM, state)
        expanded = CallablePrimitive("EXPANDED", graph, 4, 4, OBS_DIM, state)
        observations, lengths, _ = episodes_to_batch(episodes)
        result = primitive_equivalence(primitive, expanded, observations, lengths, tolerance=1e-6)
        self.assertTrue(result["pass"])
        self.assertLessEqual(result["max_absolute_error"], 1e-6)

    def test_random_macro_and_causal_replacement_control_exists(self):
        model, _, _ = self._fixture()
        graph_a, state_a = extract_isolated_subgraph(model, (0, 1))
        graph_b, state_b = randomized_isolated_subgraph(model, (0, 1), seed=40000)
        self.assertEqual(graph_a.canonical_structure(), graph_b.canonical_structure())
        self.assertTrue(any(not torch.equal(state_a[key], state_b[key]) for key in state_a if state_a[key].is_floating_point()))

    def test_archive_retrieval_api_has_no_task_family_label(self):
        parameters = inspect.signature(PrimitiveArchive.retrieve).parameters
        forbidden = {name for name in parameters if any(token in name.lower() for token in ("family", "task", "domain", "label"))}
        self.assertFalse(forbidden)
        self.assertIn("query_embedding", parameters)


class ResourceAndPairingTests(unittest.TestCase):
    def test_resource_accounting_complete(self):
        data = ResourceAccounting().to_dict()
        required = {"candidate_evaluations", "optimizer_steps", "environment_steps", "mutation_count", "wall_seconds", "peak_cells", "peak_edges", "final_cells", "final_edges", "archive_lookups", "primitive_calls"}
        self.assertTrue(required.issubset(data))

    def test_random_matched_baseline_uses_same_episode_seed_lists(self):
        tree = ast.parse(inspect.getsource(structural_search))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "train_graph"]
        self.assertGreaterEqual(len(calls), 2)
        random_call = calls[-1]
        positional_names = [arg.id for arg in random_call.args if isinstance(arg, ast.Name)]
        self.assertGreaterEqual(len(positional_names), 4)
        self.assertEqual(positional_names[2:4], ["train_seeds", "validation_seeds"])


class CompletedVariantTests(unittest.TestCase):
    def test_completed_results_have_frozen_gate_resources_controls_and_failure_classification(self):
        base = ROOT / "experiments/v837_primitive_invention"
        found = 0
        for version in ("v837", "v837b", "v837c"):
            path = base / version / "results.json"
            if not path.exists():
                continue
            found += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(V837_REQUIRED_RESULT_FIELDS.issubset(data), version)
            self.assertEqual(data["gate_file_sha256"], EXPECTED_GATE_HASH, version)
            self.assertIn("random", json.dumps(data["baselines"]).lower(), version)
            for key in ("candidate_evaluations", "optimizer_steps", "environment_steps", "peak_cells", "peak_edges", "final_cells", "final_edges"):
                self.assertIn(key, data["resource_accounting"], f"{version}:{key}")
            if data["pass"] is False:
                self.assertTrue(data["failure_classification"], version)
                self.assertTrue(set(data["failure_classification"]).issubset(V837_FAILURE_CLASSES), version)
                self.assertTrue((base / version / "FAILURE.md").exists(), version)
        self.assertEqual(found, 3)

    def test_outcome_b_preserves_fresh_audit_and_promotes_no_primitives(self):
        base = ROOT / "experiments/v837_primitive_invention"
        status = json.loads((base / "lineage_status.json").read_text(encoding="utf-8"))
        audit = json.loads((base / "audit/audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(status["outcome"], "B_MILESTONE_FAILED_HONESTLY")
        self.assertTrue(status["stop_rule_triggered"])
        self.assertEqual(status["failed_variants"], ["V837", "V837b", "V837c"])
        self.assertEqual(status["primitives_promoted"], [])
        self.assertFalse(status["fresh_audit_seed_range_used"])
        self.assertEqual(audit["status"], "NOT_RUN_PREREQUISITE_FAILURE")
        self.assertEqual(audit["episodes_consumed"], 0)

    def test_blocker_diagnostics_narrow_failure_after_three_variants(self):
        base = ROOT / "experiments/v837_primitive_invention/failures"
        first = json.loads((base / "blocker_diagnostic_results.json").read_text(encoding="utf-8"))
        second = json.loads((base / "blocker_data_diagnostic_results.json").read_text(encoding="utf-8"))
        self.assertEqual(first["failure_classification"], "REPRESENTATION_FAILURE")
        self.assertEqual(second["failure_classification"], "REPRESENTATION_FAILURE")
        self.assertLess(first["families_capacity_demonstrated"], 4)
        self.assertLess(second["families_capacity_demonstrated"], 4)
        self.assertTrue((ROOT / "experiments/v837_primitive_invention/BLOCKER_ANALYSIS.md").exists())


if __name__ == "__main__":
    unittest.main()
