# V837 Causal Operator Canonicalization and Program IR Report

## 1. Starting state
V837ar starts from `287d37743751554c68742bc275b558d7f9bb1f95` and preserves V837aq and earlier scientific blobs.

## 2. What V837aq established
Three coordinate-free operators were confirmed: conditional routing, delayed recall, iterative state. Routing required second-order response; recall had rank-1 predictive response; only iterative state closed broadly under the AQ discovery composition gate.

## 3. What V837aq composition failure does and does not mean
It rejects broad fine factorization, not operator existence. V837ar therefore tests canonical executable programs at the minimum causal granularity.

## 4. Program IR definition
A semantic interface + causal operator + behavioral state/history requirement + composition law, with zero AF1D hidden-state coordinates.

## 5. Canonical semantic response basis
Hash: `f2e27c55056295956988c8b38e049335b62fe073465a4edd6fda4c954bac311c`.

## 6. Raw response-tensor baseline
Used only as a non-archiveable empirical fidelity/compression reference.

## 7. Operator-word holdout design
FIT/SELECT/META/FINAL_UNSEEN structural word partitions were frozen before fitting; the separate composition seed partition is 11320–11383.

## 8. Iterative-state reality gate
Selected: `I2_AFFINE_UPDATE`; pass=True.

## 9. Iterative canonical IR
{"a": 0.6144277411259843, "b": 0.37089636590751696, "c": -0.0017115126857833584, "initial_state": 0.0}

## 10. Routing operator granularity
Selected model: `R1_TWO_STAGE`; granularity: `TWO_STAGE`; diagnosis: `ROUTING_TWO_STAGE_PRIMITIVE_SUFFICIENT`.

## 11. Routing second-order canonicalization
First-order NRMSE=None; bilinear NRMSE=None; third order used=False.

## 12. Delayed-recall granularity
Selected: `M2_RANK1_MEMORY`.

## 13. Predictive rank-1 memory realization
Rank=1; lambda=1.0142603857277603.

## 14. Hankel response analysis and AR13 fallback
Ranks: {"conditional_routing": {"rank90": 2, "rank95": 2, "rank99": 3, "rank999": 3}, "delayed_recall": {"rank90": 1, "rank95": 1, "rank99": 1, "rank999": 1}, "iterative_state": {"rank90": 3, "rank95": 3, "rank99": 3, "rank999": 5}}
Fallback: {"conditional_routing": {"compact_predictive_realization": false, "frozen_ir_present": false, "hankel_rank99": 3, "reason": "NO_FROZEN_PROGRAM_IR"}, "delayed_recall": {"compact_predictive_realization": true, "frozen_ir_present": true, "granularity": "PREDICTIVE_RANK1_MEMORY", "hankel_rank99": 1, "ir_predictive_state_dimension": 1, "operator_class": "LINEAR_PHASE_MACHINE_R1", "rank1_memory_realization": true}, "iterative_state": {"compact_predictive_realization": false, "frozen_ir_present": false, "hankel_rank99": 3, "reason": "NO_FROZEN_PROGRAM_IR"}}

## 15. Shared versus organism-specific fits
{"conditional_routing": {"available": false, "canonical_agreement_pass": false, "reason": "NO_FROZEN_PROGRAM_IR"}, "delayed_recall": {"available": true, "canonical_agreement_pass": true, "discovery_unseen_gate": {"both_engines": true, "organisms": 6, "pass": true, "pass_fraction": 1.0, "passing": 6, "passing_engines": ["DIRECTED_STRUCTURAL_SEARCH", "RANDOM_STRUCTURAL_SAMPLER"], "required": 4}, "grammar": "M2_RANK1_MEMORY", "granularity": "PREDICTIVE_RANK1_MEMORY", "individual_coefficients_used_for_prediction": false, "median_normalized_coefficient_deviation": 0.04938207663088902, "organisms": 6, "p90_deviation": 0.06691247912361795, "reused_heldout_gate": {"both_engines": true, "organisms": 4, "pass": true, "pass_fraction": 0.75, "passing": 3, "passing_engines": ["DIRECTED_STRUCTURAL_SEARCH", "RANDOM_STRUCTURAL_SAMPLER"], "required": 3}, "shared_fit_primary": true, "shared_ir_sha256": "ab42c3fa4b2c4dead03304184050882cec53725770c7576d308f1d818521478f", "sign_agreement": 1.0}, "iterative_state": {"available": false, "canonical_agreement_pass": false, "reason": "NO_FROZEN_PROGRAM_IR"}}

## 16. Program IR compression
{"conditional_routing": {"archiveable": false, "byte_compression_ratio": null, "compactness_class": "NO_IR", "execution_cost": {"nonlinear_functions": 0, "scalar_additions": 0, "scalar_multiplies": 0, "state_scalars": 0, "temporary_scalars": 0}, "ir_bytes": 0, "ir_parameters": 0, "raw_bytes": 1208757, "raw_numeric_scalars": 19200, "raw_rows": 1920, "reference_response_table_archiveable": false, "scalar_compression_ratio": null}, "delayed_recall": {"archiveable": true, "byte_compression_ratio": 2042.1779475982532, "compactness_class": "TINY_IR", "execution_cost": {"nonlinear_functions": 0, "scalar_additions": 1, "scalar_multiplies": 2, "state_scalars": 1, "temporary_scalars": 1}, "ir_bytes": 916, "ir_parameters": 5, "raw_bytes": 1870635, "raw_numeric_scalars": 30060, "raw_rows": 3006, "reference_response_table_archiveable": false, "scalar_compression_ratio": 6012.0}, "iterative_state": {"archiveable": false, "byte_compression_ratio": null, "compactness_class": "NO_IR", "execution_cost": {"nonlinear_functions": 0, "scalar_additions": 0, "scalar_multiplies": 0, "state_scalars": 0, "temporary_scalars": 0}, "ir_bytes": 0, "ir_parameters": 0, "raw_bytes": 3491427, "raw_numeric_scalars": 61512, "raw_rows": 5592, "reference_response_table_archiveable": false, "scalar_compression_ratio": null}}

## 17. META confirmation
{"conditional_routing": false, "delayed_recall": true, "iterative_state": false}

## 18. Frozen Program IRs
SHA-256: `02f88c6d7bd0ce389f7b5a70351708b3104106d20029a39427a37f6f5f48da4e`.

## 19. Unseen operator-word generalization
{"conditional_routing": {"pass": false}, "delayed_recall": {"both_engines": true, "organisms": 6, "pass": true, "pass_fraction": 1.0, "passing": 6, "passing_engines": ["DIRECTED_STRUCTURAL_SEARCH", "RANDOM_STRUCTURAL_SAMPLER"], "required": 4}, "iterative_state": {"pass": false}}

## 20. Reused AQ-heldout organism evaluation
{"conditional_routing": {"pass": false}, "delayed_recall": {"both_engines": true, "organisms": 4, "pass": true, "pass_fraction": 0.75, "passing": 3, "passing_engines": ["DIRECTED_STRUCTURAL_SEARCH", "RANDOM_STRUCTURAL_SAMPLER"], "required": 3}, "iterative_state": {"pass": false}}

## 21. Cross-organism canonicality
Shared IR predictive performance is primary; individual coefficients are diagnostic only. Evidence-complete families: ['delayed_recall'].

## 22. Composition hierarchy C1/C2/C3
{"conditional_routing": {"C1_repeated_operator": false, "C2_family_program_words": false, "C3_suboperator_factorization": false, "minimum_closed_granularity": "", "reason": "NULL_AT_PROGRAM_IR_FREEZE"}, "delayed_recall": {"C1_repeated_operator": true, "C2_family_program_words": true, "C2_metrics": {"gate": {"both_engines": true, "organisms": 6, "pass": true, "pass_fraction": 0.8333333333333334, "passing": 5, "passing_engines": ["DIRECTED_STRUCTURAL_SEARCH", "RANDOM_STRUCTURAL_SAMPLER"], "required": 4}, "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344140001, "n": 384, "oracle_response_rms": 1.0, "pearson": 0.9991202027753383, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258257, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.02117969181499772, "zero_effect_median_abs": 0.0}, "organism_relative": {"direction_agreement": 0.9973958333333334, "gain_ratio": 0.9936649872668303, "n": 384, "oracle_response_rms": 0.9926951786209736, "pearson": 0.9933830136212854, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258257, "response_mae_normalized": 0.01633712799440146, "response_nrmse": 0.056836752085576245, "zero_effect_median_abs": 0.0}, "pass": true, "per_organism": [{"engine": "DIRECTED_STRUCTURAL_SEARCH", "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344139873, "n": 64, "oracle_response_rms": 1.0, "pearson": 0.9991202027753382, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.021179691814997723, "zero_effect_median_abs": 0.0}, "organism_id": "0be47e156b37323e010a65e3ec288c8610fdc49e5c6a097d092ff6056d898280", "organism_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9976523272689077, "n": 64, "oracle_response_rms": 0.994417431040505, "pearson": 0.999165064112442, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013155861463772962, "response_nrmse": 0.020352067865519862, "zero_effect_median_abs": 0.0}, "pass": true}, {"engine": "RANDOM_STRUCTURAL_SAMPLER", "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344139873, "n": 64, "oracle_response_rms": 1.0, "pearson": 0.9991202027753382, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.021179691814997723, "zero_effect_median_abs": 0.0}, "organism_id": "3edfaa94e8425d5f8bd03e35bc36933a2df2bb4069adc3f2a005e01528edb656", "organism_relative": {"direction_agreement": 1.0, "gain_ratio": 1.0046662641959623, "n": 64, "oracle_response_rms": 0.9872021799339334, "pearson": 0.9988776780381353, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.015668864648196536, "response_nrmse": 0.02353963290289799, "zero_effect_median_abs": 0.0}, "pass": true}, {"engine": "RANDOM_STRUCTURAL_SAMPLER", "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344139873, "n": 64, "oracle_response_rms": 1.0, "pearson": 0.9991202027753382, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.021179691814997723, "zero_effect_median_abs": 0.0}, "organism_id": "495486448fb57696fad60ce0396d83adc234902a069c75608a667e5cef5a6d45", "organism_relative": {"direction_agreement": 0.984375, "gain_ratio": 0.9681992679987466, "n": 64, "oracle_response_rms": 0.9896966057948172, "pearson": 0.9649604730399058, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.029810295043922835, "response_nrmse": 0.13103244703700967, "zero_effect_median_abs": 0.0}, "pass": false}, {"engine": "DIRECTED_STRUCTURAL_SEARCH", "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344139873, "n": 64, "oracle_response_rms": 1.0, "pearson": 0.9991202027753382, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.021179691814997723, "zero_effect_median_abs": 0.0}, "organism_id": "54bfdc6b05dc7c3b0877d16fb3c6df646172b1b1605e7782f551cbee092632f8", "organism_relative": {"direction_agreement": 1.0, "gain_ratio": 1.0001961227728422, "n": 64, "oracle_response_rms": 0.9917676938146858, "pearson": 0.9990397198094435, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.014178810153493752, "response_nrmse": 0.021741744165006096, "zero_effect_median_abs": 0.0}, "pass": true}, {"engine": "DIRECTED_STRUCTURAL_SEARCH", "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344139873, "n": 64, "oracle_response_rms": 1.0, "pearson": 0.9991202027753382, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.021179691814997723, "zero_effect_median_abs": 0.0}, "organism_id": "659a9f4309ec4b9d669fed38344db99d60816be900ad76a6737106e01e2523bf", "organism_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9958155024618307, "n": 64, "oracle_response_rms": 0.9963160068596917, "pearson": 0.9992218056598271, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.012701078054070801, "response_nrmse": 0.019631904533518575, "zero_effect_median_abs": 0.0}, "pass": true}, {"engine": "RANDOM_STRUCTURAL_SAMPLER", "oracle_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9920427344139873, "n": 64, "oracle_response_rms": 1.0, "pearson": 0.9991202027753382, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.013429019978492246, "response_nrmse": 0.021179691814997723, "zero_effect_median_abs": 0.0}, "organism_id": "8f8da93ac7053b65798cd4c3b78b56cd573d1397f711cdc5f955ca7436484b53", "organism_relative": {"direction_agreement": 1.0, "gain_ratio": 0.9953967503138906, "n": 64, "oracle_response_rms": 0.9967346784271777, "pearson": 0.9992204595678073, "perturbed_task_success": 1.0, "predicted_response_rms": 0.9929147930258259, "response_mae_normalized": 0.012507858602951893, "response_nrmse": 0.01966113045481106, "zero_effect_median_abs": 0.0}, "pass": true}], "source_prediction_rows": 384, "word_count": 64}, "C3_suboperator_factorization": true, "metrics": {"C1_exact_residual_max_abs": 6.661338147750939e-16, "C1_pass": true, "C3_factorization_residual_max_abs": 0.0, "C3_pass": true, "tested_words": 64}, "minimum_closed_granularity": "PREDICTIVE_RANK1_MEMORY"}, "iterative_state": {"C1_repeated_operator": false, "C2_family_program_words": false, "C3_suboperator_factorization": false, "minimum_closed_granularity": "", "reason": "NULL_AT_PROGRAM_IR_FREEZE"}}

## 23. Variable-composition negative control
{"generic_ir_falsely_universal": false, "new_variable_composition_primitive_fit": false, "routing_shape_compatible_probe": [], "version": "V837ar"}

## 24. IR execution/storage economics
{"accounted_stage_wall_seconds": 21.459634999991977, "accounting_semantics": "Counts are exact artifact/evaluated-episode counts where reconstructible; low-level batched torch forward calls are not inferred from artifact rows.", "brent_refinement_iterations": 16, "composition_evaluated_predictions": 384, "composition_ir_word_evaluations": 64, "final_unseen_evaluated_predictions": 1152, "fresh_audit_episodes": 0, "gpu_training_seconds": 0.0, "memory_lambda_grid_points_per_rank1_candidate": 2049, "meta_evaluated_predictions": 1088, "new_source_model_fits": 0, "organism_forward_calls": "batched and not reconstructed from artifact rows", "primitive_archive_population": false, "primitives_promoted": 0, "program_ir_candidate_fits": 13, "rank1_memory_grid_candidates": 2, "response_tensor_rows_read": 10518, "reused_heldout_evaluated_predictions": 768, "ridge_lambda": 1e-06, "scalar_lambda_grid_evaluations": 4098, "source_architecture_changes": 0, "source_model_inference_only": true, "source_optimizer_steps": 0, "source_training_examples": 0, "source_training_performed": false, "stage_wall_seconds": {"agreement": 0.024941899999248562, "basis": 0.38012309999976424, "composition": 0.49313379999875906, "freeze": 0.01566359999924316, "heldout": 0.5473614000002271, "iterative-reality": 3.873588700000255, "memory": 9.692111799999111, "meta": 3.5302169999995385, "minimality": 0.011531699998158729, "predictive": 0.24094110000078217, "predictive-realization": 0.0029394999983196612, "routing": 1.0702578999989782, "select": 0.03509289999783505, "source": 0.8057298999992781, "unseen": 0.7324977000025683, "words": 0.0035029999999096617}, "svd_decompositions": 3, "total_postfit_evaluated_predictions": 3392, "v838_started": false, "version": "V837ar"}

## 25. Failure analysis
Entries: 24 scientific=21 engineering=0 invalid=3.

## 26. Correct primitive granularity
{"conditional_routing": "", "delayed_recall": "PREDICTIVE_RANK1_MEMORY", "iterative_state": ""}

## 27. Archive authorization
Authorized next: True; families: ['delayed_recall']. PrimitiveArchive remains unpopulated in V837ar.

## 28. Strongest scientific claim
Within the tested AF1D population, frozen shared semantic Program IRs are canonical only for: delayed_recall. V837aq operator existence for other families remains intact, but their stronger Program IR claim is not established.

## 29. Remaining uncertainty
Generalization beyond the tested AF1D population and structural operator-word support remains unestablished. Predictive-response fallback objects are diagnostic and cannot rescue a failed executable Program IR claim.

## 30. Next single program
`V837as_PROGRAM_IR_OPERATOR_ARCHIVE`.
