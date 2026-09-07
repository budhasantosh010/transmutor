from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.v837_primitive_invention.v837aa.candidate_law_alignment import (
    IDENTITY_PERMUTATION_INDEX,
    SIGNED_PERMUTATIONS,
    align_laws,
    block_diagonal_transform,
    classify_candidate_law,
    complete_linkage_assignments,
    functional_metric_matrices,
    functional_outputs,
    pairwise_same_matrix,
    transform_global_matrix,
    transform_law,
    validate_signed_permutations,
)
from experiments.v837_primitive_invention.v837aa.run_candidate_law_audit import (
    CONFIG,
    assert_fresh_audit_state,
    verify_frozen_hashes,
)

ROOT = Path(__file__).resolve().parents[1]
GATE = json.loads((ROOT / "experiments/v837_primitive_invention/v837aa/frozen_candidate_law_gate.json").read_text())


def random_law(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "Ws": rng.normal(size=(4, 4)),
        "Wm": rng.normal(size=(4, 4)),
        "Wx": rng.normal(size=(4, 6)),
        "b": rng.normal(size=(4,)),
        "Wo": rng.normal(size=(4, 4)),
    }


def probes(seed: int = 7, count: int = 1024) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "s": rng.uniform(-1, 1, size=(count, 4)),
        "m": rng.uniform(-1, 1, size=(count, 4)),
        "x": rng.uniform(-1, 1, size=(count, 6)),
    }


def compact(metrics: dict) -> dict:
    return {
        "median_pairwise_cosine": metrics["median_pairwise_cosine"],
        "median_pairwise_normalized_rmse": metrics["median_pairwise_normalized_rmse"],
    }


def bad_metric() -> dict:
    return {"median_pairwise_cosine": 0.0, "median_pairwise_normalized_rmse": 0.7}


def fit_evidence(raw: dict, aligned: dict) -> dict:
    return {
        "trained_raw_synthetic": raw,
        "trained_aligned_synthetic": aligned,
        "trained_raw_empirical": raw,
        "trained_aligned_empirical": aligned,
        "initial_raw_synthetic": bad_metric(),
        "initial_aligned_synthetic": bad_metric(),
        "initial_raw_empirical": bad_metric(),
        "initial_aligned_empirical": bad_metric(),
    }


class V837aaCandidateLawTests(unittest.TestCase):
    def test_signed_permutation_enumeration_exact_384_unique_orthogonal(self):
        validate_signed_permutations()
        self.assertEqual(len(SIGNED_PERMUTATIONS), 384)
        keys = {tuple(int(v) for v in row.matrix.reshape(-1)) for row in SIGNED_PERMUTATIONS}
        self.assertEqual(len(keys), 384)
        identity = np.eye(4, dtype=np.float32)
        for row in SIGNED_PERMUTATIONS:
            self.assertTrue(np.array_equal(row.matrix.T @ row.matrix, identity))
            self.assertTrue(np.all(np.isin(row.matrix, [-1, 0, 1])))

    def test_candidate_equivariance_under_signed_permutation(self):
        law = random_law(1)
        rng = np.random.default_rng(2)
        for row in SIGNED_PERMUTATIONS[::37]:
            p = row.matrix.astype(np.float64)
            transformed = transform_law(law, p)
            s = rng.normal(size=(31, 4)); m = rng.normal(size=(31, 4)); x = rng.normal(size=(31, 6))
            raw = np.tanh(s @ law["Ws"].T + m @ law["Wm"].T + x @ law["Wx"].T + law["b"])
            z = s @ p.T
            expected = raw @ p.T
            actual = np.tanh(z @ transformed["Ws"].T + m @ transformed["Wm"].T + x @ transformed["Wx"].T + transformed["b"])
            self.assertLessEqual(float(np.max(np.abs(expected - actual))), 1e-6)

    def test_output_map_equivariance_uses_wo_p_transpose(self):
        law = random_law(3); rng = np.random.default_rng(4); s = rng.normal(size=(41,4))
        for row in SIGNED_PERMUTATIONS[::53]:
            p = row.matrix.astype(np.float64); transformed = transform_law(law, p); z = s @ p.T
            original = s @ law["Wo"].T
            canonical = z @ transformed["Wo"].T
            self.assertLessEqual(float(np.max(np.abs(original - canonical))), 1e-6)

    def test_global_40d_block_transform_equivariance(self):
        rng = np.random.default_rng(5); a = rng.normal(size=(40,40)); s = rng.normal(size=(40,))
        indices = [SIGNED_PERMUTATIONS[(i * 31) % 384].index for i in range(10)]
        p = block_diagonal_transform(indices); transformed = transform_global_matrix(a, indices); z = p @ s
        self.assertLessEqual(float(np.max(np.abs(transformed @ z - p @ (a @ s)))), 1e-8)

    def test_deterministic_alignment_same_input_same_result(self):
        laws = [random_law(100 + i) for i in range(10)]
        a = align_laws(laws); b = align_laws(laws)
        self.assertEqual(a["medoid_cell"], b["medoid_cell"])
        self.assertEqual(a["assignments"], b["assignments"])
        self.assertEqual(a["convergence_iteration"], b["convergence_iteration"])
        self.assertEqual(a["centroid"], b["centroid"])

    def test_identity_case_classifies_direct_common_basis(self):
        law = random_law(11); laws = [law for _ in range(10)]; metric = compact(functional_metric_matrices(functional_outputs(laws, probes(), view="core")))
        clustering = {str(k): {"passes": False} for k in range(2,6)}; clustering["relative_basis_stable"] = True
        result = classify_candidate_law([fit_evidence(metric, metric) for _ in range(25)], clustering, GATE["classification"])
        self.assertEqual(result["candidate_law_diagnosis"], "DIRECT_COMMON_BASIS")
        self.assertEqual(result["recommended_next_axis"], "V837ab_DIRECT_SHARED_CANDIDATE_CORE")

    def test_permuted_law_recovers_and_classifies_common_up_to_signed_permutation(self):
        base = random_law(12)
        chosen = [SIGNED_PERMUTATIONS[(17 * i + 3) % 384].matrix.astype(np.float64) for i in range(10)]
        laws = [transform_law(base, p.T) for p in chosen]
        alignment = align_laws(laws)
        aligned = [transform_law(law, SIGNED_PERMUTATIONS[idx].matrix) for law, idx in zip(laws, alignment["assignments"])]
        p = probes(13, 2048)
        raw = compact(functional_metric_matrices(functional_outputs(laws, p, view="core")))
        recovered = compact(functional_metric_matrices(functional_outputs(aligned, p, view="core")))
        self.assertGreaterEqual(recovered["median_pairwise_cosine"], 0.999999)
        self.assertLessEqual(recovered["median_pairwise_normalized_rmse"], 1e-6)
        self.assertGreater(raw["median_pairwise_normalized_rmse"], 0.20)
        clustering = {str(k): {"passes": False} for k in range(2,6)}; clustering["relative_basis_stable"] = False
        result = classify_candidate_law([fit_evidence(raw, recovered) for _ in range(25)], clustering, GATE["classification"])
        self.assertEqual(result["candidate_law_diagnosis"], "COMMON_LAW_UP_TO_SIGNED_PERMUTATION")
        self.assertEqual(result["recommended_next_axis"], "V837ab_COORDINATE_ADAPTIVE_SHARED_CORE_REQUIRED")

    def test_diverse_laws_do_not_false_classify_as_common_after_alignment(self):
        laws = [random_law(2000 + i) for i in range(10)]; alignment = align_laws(laws)
        aligned = [transform_law(law, SIGNED_PERMUTATIONS[idx].matrix) for law, idx in zip(laws, alignment["assignments"])]
        p = probes(99, 2048)
        raw = compact(functional_metric_matrices(functional_outputs(laws, p, view="core")))
        recovered = compact(functional_metric_matrices(functional_outputs(aligned, p, view="core")))
        clustering = {str(k): {"passes": False} for k in range(2,6)}; clustering["relative_basis_stable"] = False
        result = classify_candidate_law([fit_evidence(raw, recovered) for _ in range(25)], clustering, GATE["classification"])
        self.assertFalse(result["direct_common_basis"])
        self.assertFalse(result["common_law_after_signed_permutation"])
        self.assertEqual(result["candidate_law_diagnosis"], "GENUINELY_DIVERSE_CANDIDATE_LAWS")

    def test_small_type_complete_linkage_recovers_known_clusters(self):
        distance = np.full((10,10), 1.0, dtype=np.float64); np.fill_diagonal(distance, 0.0)
        for group in (range(0,5), range(5,10)):
            for i in group:
                for j in group:
                    distance[i,j] = 0.01 if i != j else 0.0
        assignments = complete_linkage_assignments(distance, 2)
        same = pairwise_same_matrix(assignments)
        self.assertTrue(np.all(same[:5,:5] == 1)); self.assertTrue(np.all(same[5:,5:] == 1)); self.assertTrue(np.all(same[:5,5:] == 0))
        mediocre = {"median_pairwise_cosine": 0.80, "median_pairwise_normalized_rmse": 0.30}
        clustering = {str(k): {"passes": k == 2} for k in range(2,6)}; clustering["relative_basis_stable"] = False
        result = classify_candidate_law([fit_evidence(mediocre, mediocre) for _ in range(25)], clustering, GATE["classification"])
        self.assertEqual(result["candidate_law_diagnosis"], "STABLE_SMALL_TYPE_VOCABULARY")
        self.assertEqual(result["selected_type_count"], 2)
        self.assertEqual(result["recommended_next_axis"], "V837ab_GROUP_SHARED_CANDIDATE_TYPES")

    def test_fresh_audit_protection_rejects_nonzero_consumption(self):
        assert_fresh_audit_state({"episodes_consumed": 0})
        with self.assertRaises(RuntimeError):
            assert_fresh_audit_state({"episodes_consumed": 1})

    def test_parent_hash_protection_rejects_altered_fixture(self):
        first_path = next(iter(CONFIG["frozen_parent_git_blob_sha256"]))
        expected = CONFIG["frozen_parent_git_blob_sha256"]
        def provider(path: str) -> str:
            return "0" * 64 if path == first_path else expected[path]
        with self.assertRaises(RuntimeError):
            verify_frozen_hashes(provider)

    def test_identity_signed_permutation_index_is_exact_identity(self):
        self.assertTrue(np.array_equal(SIGNED_PERMUTATIONS[IDENTITY_PERMUTATION_INDEX].matrix, np.eye(4, dtype=np.float32)))

    def test_v837aa_does_not_authorize_architecture_changes(self):
        self.assertTrue(CONFIG["audit_only"])
        self.assertFalse(CONFIG["architecture_modified"])
        self.assertEqual(CONFIG["rerun_conditions"], ["Y3_global_control_rank4_candidate"])
        self.assertFalse(CONFIG["structural_search"])
        self.assertFalse(CONFIG["primitive_mining"])
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        self.assertFalse(CONFIG["v837ab_implemented"])
        self.assertFalse(CONFIG["v838_started"])


if __name__ == "__main__":
    unittest.main()
