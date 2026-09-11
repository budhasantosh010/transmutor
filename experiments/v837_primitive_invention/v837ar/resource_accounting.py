from __future__ import annotations

from .utils import HERE, read_json, write_json


def _nested_prediction_rows(path: str) -> int:
    p = HERE / path
    if not p.is_file():
        return 0
    payload = read_json(p)
    total = 0
    for family in payload.get("families", {}).values():
        for organism in family.get("rows", []):
            total += len(organism.get("rows", []))
    return total


def _meta_rows() -> int:
    p = HERE / "raw/meta_confirmation.json"
    if not p.is_file():
        return 0
    payload = read_json(p)
    return sum(len(d.get("metrics", {}).get("rows", [])) for d in payload.get("families", {}).values())


def _composition_prediction_rows() -> int:
    p = HERE / "raw/composition_results.json"
    if not p.is_file():
        return 0
    payload = read_json(p)
    return sum(
        int(family.get("C2_metrics", {}).get("source_prediction_rows", 0))
        for family in payload.get("families", {}).values()
    )


def _composition_ir_word_evaluations() -> int:
    p = HERE / "raw/composition_results.json"
    if not p.is_file():
        return 0
    payload = read_json(p)
    return sum(
        int(family.get("metrics", {}).get("tested_words", 0))
        for family in payload.get("families", {}).values()
    )


def _candidate_count() -> int:
    total = 0
    for rel in ("raw/iterative_ir_fits.json", "raw/routing_ir_fits.json", "raw/memory_ir_fits.json"):
        p = HERE / rel
        if p.is_file():
            total += len(read_json(p).get("fits", {}))
    return total


def _brent_iterations() -> int:
    p = HERE / "raw/memory_ir_fits.json"
    if not p.is_file():
        return 0
    fits = read_json(p).get("fits", {})
    return sum(int(d.get("brent_iterations", 0)) for d in fits.values())


def _svd_count() -> int:
    count = 0
    if (HERE / "raw/predictive_hankel.json").is_file():
        count += len(read_json(HERE / "raw/predictive_hankel.json").get("families", {}))
    if (HERE / "raw/predictive_realization_fallback.json").is_file():
        count += sum(1 for d in read_json(HERE / "raw/predictive_realization_fallback.json").get("families", {}).values() if d.get("required"))
    return count


def compute_resource_accounting(stage_times=None):
    refs = read_json(HERE / "raw/reference_response_tables.json") if (HERE / "raw/reference_response_tables.json").is_file() else {"families": {}}
    final_rows = _nested_prediction_rows("raw/final_unseen_word_results.json")
    heldout_rows = _nested_prediction_rows("raw/reused_aq_heldout_results.json")
    meta_rows = _meta_rows()
    composition_rows = _composition_prediction_rows()
    stages = dict(stage_times or {})
    payload = {
        "version": "V837ar",
        "accounting_semantics": "Counts are exact artifact/evaluated-episode counts where reconstructible; low-level batched torch forward calls are not inferred from artifact rows.",
        "response_tensor_rows_read": sum(d.get("raw_row_count", 0) for d in refs.get("families", {}).values()),
        "final_unseen_evaluated_predictions": final_rows,
        "reused_heldout_evaluated_predictions": heldout_rows,
        "meta_evaluated_predictions": meta_rows,
        "composition_evaluated_predictions": composition_rows,
        "composition_ir_word_evaluations": _composition_ir_word_evaluations(),
        "total_postfit_evaluated_predictions": final_rows + heldout_rows + meta_rows + composition_rows,
        "program_ir_candidate_fits": _candidate_count(),
        "ridge_lambda": 1e-6,
        "memory_lambda_grid_points_per_rank1_candidate": 2049,
        "rank1_memory_grid_candidates": 2 if (HERE / "raw/memory_ir_fits.json").is_file() else 0,
        "scalar_lambda_grid_evaluations": 4098 if (HERE / "raw/memory_ir_fits.json").is_file() else 0,
        "brent_refinement_iterations": _brent_iterations(),
        "svd_decompositions": _svd_count(),
        "organism_forward_calls": "batched and not reconstructed from artifact rows",
        "source_model_inference_only": True,
        "source_training_performed": False,
        "stage_wall_seconds": stages,
        "accounted_stage_wall_seconds": float(sum(stages.values())),
        "gpu_training_seconds": 0.0,
        "new_source_model_fits": 0,
        "source_optimizer_steps": 0,
        "source_training_examples": 0,
        "source_architecture_changes": 0,
        "fresh_audit_episodes": 0,
        "primitive_archive_population": False,
        "primitives_promoted": 0,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics/resource_accounting.json", payload)
    return payload
