# V837 Program-Level Causal Operator Localization Report

## 1. Mission
V837aq tested whether reusable computation is better represented as a coordinate-free causal response operator over trajectories than as a shared neural-state coordinate.

## 2. V837ap forensic boundary
V837ap remains `DECODABLE_LOW_DIMENSIONAL_STATE_NOT_CAUSALLY_CLOSED`. It did not test quotient/commutativity/heldout because zero candidates reached those stages; engineering-invalid chart rows remain engineering-invalid rather than scientific negatives.

## 3. Frozen ontology
Primary evidence uses natural task/environment semantic interventions. Arbitrary hidden-state SET, cross-organism state alignment, and q-vector alignment are excluded.

## 4. AQ2 reality gate
Iterative-state reality gate: 6/6 organisms PASS; kill switch=False.

## 5. Coordinate-free operator discovery
{
  "conditional_routing": {
    "competent_minus_incompetent_pass_rate": 1.0,
    "cross_organism": {
      "median_prediction_std_normalized": 0.013586251093054062,
      "organisms": 5,
      "p90_pairwise_disagreement_normalized": 0.07063128873705864,
      "pass": true,
      "q_alignment_used": false,
      "state_alignment_used": false
    },
    "first_order_gate": {
      "both_engines": true,
      "organisms": 5,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 5,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 3
    },
    "historical_incompetent_available": 1,
    "historical_incompetent_pass_rate": 0.0,
    "incompetent_specificity_pass": true,
    "operator_order": 2,
    "pass": true,
    "pass_before_interaction": true
  },
  "delayed_recall": {
    "competent_minus_incompetent_pass_rate": null,
    "cross_organism": {
      "median_prediction_std_normalized": 0.00023867162325606588,
      "organisms": 6,
      "p90_pairwise_disagreement_normalized": 0.004423296451568604,
      "pass": true,
      "q_alignment_used": false,
      "state_alignment_used": false
    },
    "first_order_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "historical_incompetent_available": 0,
    "historical_incompetent_pass_rate": null,
    "incompetent_specificity_pass": true,
    "operator_order": 1,
    "pass": true,
    "pass_before_interaction": true
  },
  "iterative_state": {
    "competent_minus_incompetent_pass_rate": null,
    "cross_organism": {
      "median_prediction_std_normalized": 0.002830232142043442,
      "organisms": 6,
      "p90_pairwise_disagreement_normalized": 0.010308516025543214,
      "pass": true,
      "q_alignment_used": false,
      "state_alignment_used": false
    },
    "first_order_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "historical_incompetent_available": 0,
    "historical_incompetent_pass_rate": null,
    "incompetent_specificity_pass": true,
    "operator_order": 1,
    "pass": true,
    "pass_before_interaction": true
  },
  "variable_composition": {
    "competent_minus_incompetent_pass_rate": 0.0,
    "cross_organism": {
      "median_prediction_std_normalized": null,
      "organisms": 0,
      "p90_pairwise_disagreement_normalized": null,
      "pass": false
    },
    "first_order_gate": {
      "both_engines": false,
      "organisms": 4,
      "pass": false,
      "pass_fraction": 0.0,
      "passing": 0,
      "passing_engines": [],
      "required": 3
    },
    "historical_incompetent_available": 2,
    "historical_incompetent_pass_rate": 0.0,
    "incompetent_specificity_pass": false,
    "operator_order": 1,
    "pass": false,
    "pass_before_interaction": false
  }
}

## 6. Operator orders
{
  "conditional_routing": 2,
  "delayed_recall": 1,
  "iterative_state": 1,
  "variable_composition": 1
}
Routing requires a second-order interaction term; recall and iterative state remain first order. Variable composition fails the frozen single-intervention operator gate and is null.

## 7. Cross-organism equivalence
{
  "conditional_routing": {
    "median_prediction_std_normalized": 0.013586251093054062,
    "organisms": 5,
    "p90_pairwise_disagreement_normalized": 0.07063128873705864,
    "pass": true,
    "q_alignment_used": false,
    "state_alignment_used": false
  },
  "delayed_recall": {
    "median_prediction_std_normalized": 0.00023867162325606588,
    "organisms": 6,
    "p90_pairwise_disagreement_normalized": 0.004423296451568604,
    "pass": true,
    "q_alignment_used": false,
    "state_alignment_used": false
  },
  "iterative_state": {
    "median_prediction_std_normalized": 0.002830232142043442,
    "organisms": 6,
    "p90_pairwise_disagreement_normalized": 0.010308516025543214,
    "pass": true,
    "q_alignment_used": false,
    "state_alignment_used": false
  },
  "variable_composition": {
    "median_prediction_std_normalized": null,
    "organisms": 0,
    "p90_pairwise_disagreement_normalized": null,
    "pass": false
  }
}

## 8. Negative controls
Time/magnitude shuffles, wrong phase, wrong-family templates, endpoint-only predictors, matched low-order regression, and family-compatible historically incompetent organisms were evaluated. Search-engine identity was never treated as a positive/negative label.

## 9. Spatiotemporal localization
{
  "conditional_routing": {
    "compact_support_gate": {
      "both_engines": false,
      "organisms": 5,
      "pass": false,
      "pass_fraction": 0.0,
      "passing": 0,
      "passing_engines": [],
      "required": 3
    },
    "median_winner_units": null,
    "pass": false
  },
  "delayed_recall": {
    "compact_support_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 0.6666666666666666,
      "passing": 4,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "median_winner_units": 4.0,
    "pass": true
  },
  "iterative_state": {
    "compact_support_gate": {
      "both_engines": false,
      "organisms": 6,
      "pass": false,
      "pass_fraction": 0.0,
      "passing": 0,
      "passing_engines": [],
      "required": 4
    },
    "median_winner_units": null,
    "pass": false
  }
}
Delayed recall reaches 4/6 compact-support passes in discovery but only 3/6 on META; compact support is therefore suggestive, not a robust final family-level claim. Routing and iterative state remain organism-scale distributed within the tested <=8-unit support family.

## 10. Composition
{
  "conditional_routing": {
    "gate": {
      "both_engines": true,
      "organisms": 5,
      "pass": false,
      "pass_fraction": 0.4,
      "passing": 2,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 3
    },
    "operator_word": "SELECT o LOAD o CONTROL",
    "pass": false
  },
  "delayed_recall": {
    "gate": {
      "both_engines": false,
      "organisms": 6,
      "pass": false,
      "pass_fraction": 0.16666666666666666,
      "passing": 1,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH"
      ],
      "required": 4
    },
    "operator_word": "READ o HOLD^n o WRITE",
    "pass": false
  },
  "iterative_state": {
    "gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "operator_word": "U(x2) o U(x1)",
    "pass": true
  }
}
Only iterative state passes the frozen discovery composition family gate. Later META/heldout success cannot rescue routing/recall discovery failures.

## 11. Predictive causal-state diagnostic
{
  "conditional_routing": {
    "required": true,
    "rank99": 4,
    "compact": true
  },
  "delayed_recall": {
    "required": true,
    "rank99": 1,
    "compact": true
  },
  "iterative_state": {
    "required": false,
    "rank99": 3,
    "compact": true
  }
}
The predictive-response diagnostic is non-gating and cannot rescue composition failure.

## 12. META confirmation
{
  "conditional_routing": {
    "compact_localization_meta_gate": {
      "both_engines": false,
      "organisms": 5,
      "pass": false,
      "pass_fraction": 0.0,
      "passing": 0,
      "passing_engines": [],
      "required": 3
    },
    "discovery_composition_pass": false,
    "operator_confirmed": true,
    "operator_meta_gate": {
      "both_engines": true,
      "organisms": 5,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 5,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 3
    },
    "operator_order": 2,
    "pass": true,
    "program_closed": false,
    "program_meta_gate": {
      "both_engines": true,
      "organisms": 5,
      "pass": true,
      "pass_fraction": 0.6,
      "passing": 3,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 3
    }
  },
  "delayed_recall": {
    "compact_localization_meta_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": false,
      "pass_fraction": 0.5,
      "passing": 3,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "discovery_composition_pass": false,
    "operator_confirmed": true,
    "operator_meta_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "operator_order": 1,
    "pass": true,
    "program_closed": false,
    "program_meta_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": false,
      "pass_fraction": 0.5,
      "passing": 3,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    }
  },
  "iterative_state": {
    "compact_localization_meta_gate": {
      "both_engines": false,
      "organisms": 6,
      "pass": false,
      "pass_fraction": 0.0,
      "passing": 0,
      "passing_engines": [],
      "required": 4
    },
    "discovery_composition_pass": true,
    "operator_confirmed": true,
    "operator_meta_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "operator_order": 1,
    "pass": true,
    "program_closed": true,
    "program_meta_gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    }
  }
}

## 13. Frozen operator contracts
Pre-heldout contract SHA-256: `461faadf9cb9cc1262c9a288236c08875ae5212685bb8dd06e6a4ed92768d9af`.

## 14. Held-out organisms
{
  "conditional_routing": {
    "cross_organism": {
      "max_response_nrmse": 0.047105949614518826,
      "median_response_nrmse": 0.044186067821901764,
      "organisms": 4,
      "pass": true,
      "q_alignment_used": false,
      "state_alignment_used": false
    },
    "operator_gate": {
      "directed": 2,
      "organisms": 4,
      "pass": true,
      "passing": 4,
      "random": 2,
      "required": 3
    },
    "operator_pass": true,
    "pass": true,
    "program_gate": {
      "directed": 2,
      "organisms": 4,
      "pass": true,
      "passing": 4,
      "random": 2,
      "required": 3
    },
    "program_pass": true
  },
  "delayed_recall": {
    "cross_organism": {
      "max_response_nrmse": 0.07043822906392773,
      "median_response_nrmse": 0.04322598378837575,
      "organisms": 4,
      "pass": true,
      "q_alignment_used": false,
      "state_alignment_used": false
    },
    "operator_gate": {
      "directed": 2,
      "organisms": 4,
      "pass": true,
      "passing": 4,
      "random": 2,
      "required": 3
    },
    "operator_pass": true,
    "pass": true,
    "program_gate": {
      "directed": 2,
      "organisms": 4,
      "pass": true,
      "passing": 3,
      "random": 1,
      "required": 3
    },
    "program_pass": true
  },
  "iterative_state": {
    "cross_organism": {
      "max_response_nrmse": 0.019573509774187745,
      "median_response_nrmse": 0.019142373367980005,
      "organisms": 4,
      "pass": true,
      "q_alignment_used": false,
      "state_alignment_used": false
    },
    "operator_gate": {
      "directed": 2,
      "organisms": 4,
      "pass": true,
      "passing": 4,
      "random": 2,
      "required": 3
    },
    "operator_pass": true,
    "pass": true,
    "program_gate": {
      "directed": 2,
      "organisms": 4,
      "pass": true,
      "passing": 4,
      "random": 2,
      "required": 3
    },
    "program_pass": true
  }
}
All three frozen operator families confirm on held-out organisms with no operator refit.

## 15. Reused historical validation
{
  "conditional_routing": {
    "gate": {
      "both_engines": true,
      "organisms": 5,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 5,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 3
    },
    "label": "REUSED_HISTORICAL_EPISODE_VALIDATION",
    "pass": true
  },
  "delayed_recall": {
    "gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "label": "REUSED_HISTORICAL_EPISODE_VALIDATION",
    "pass": true
  },
  "iterative_state": {
    "gate": {
      "both_engines": true,
      "organisms": 6,
      "pass": true,
      "pass_fraction": 1.0,
      "passing": 6,
      "passing_engines": [
        "DIRECTED_STRUCTURAL_SEARCH",
        "RANDOM_STRUCTURAL_SAMPLER"
      ],
      "required": 4
    },
    "label": "REUSED_HISTORICAL_EPISODE_VALIDATION",
    "pass": true
  }
}
This panel is descriptive/post-freeze and cannot upgrade a failed discovery or META gate.

## 16. Final diagnosis
`CAUSAL_OPERATOR_FOUND_NOT_COMPOSITIONALLY_CLOSED`

## 17. Strongest claim
Three powered families exhibit oracle-relative, coordinate-free causal response operators that reproduce across independent organisms and held-out organisms; only iterative state satisfies the frozen discovery+META+heldout composition closure, so broad program-level compositional closure is not established.

## 18. Resource accounting
{
  "version": "V837aq",
  "all_accounted": true,
  "new_source_model_fits": 0,
  "source_optimizer_steps": 0,
  "source_training_examples": 0,
  "source_architecture_changes": 0,
  "gpu_seconds": 0.0,
  "reality_response_rows": 5712,
  "coordinate_free_response_tensor_rows": 15638,
  "operator_discovery_organism_rows": 21,
  "incompetent_control_rows": 3,
  "localization_singleton_masks": 2108,
  "localization_prefix_masks": 63,
  "localization_random_control_masks": 1008,
  "composition_organism_rows": 17,
  "meta_organism_rows": 17,
  "heldout_organism_rows": 12,
  "hidden_state_set_primary_evidence": false,
  "cross_organism_state_alignment": false,
  "cross_organism_q_alignment": false,
  "fresh_audit_episodes": 0,
  "primitives_promoted": 0,
  "primitive_archive_population": false,
  "v838_started": false,
  "note": "Counts are exact artifact/evaluation-row counts; low-level framework forward-call batching is intentionally not reconstructed from wall-clock logs."
}

## 19. Failure memory
V837aq failure-memory entries: 39. Scientific and engineering failures are typed separately.

## 20. Protected locks
Fresh audit 90000-90499 remains unused. PrimitiveArchive remains blocked. Primitives promoted = 0. V838 was not started.

## 21. Next program
`V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR`
Its first gate must resolve the routing/recall composition boundary before any archive population.

## 22. Figures
- `experiments/v837_primitive_invention/v837aq/plots/aq2_reality_nrmse.png`
- `experiments/v837_primitive_invention/v837aq/plots/compact_localization_pass_fraction.png`
- `experiments/v837_primitive_invention/v837aq/plots/composition_pass_fraction.png`
- `experiments/v837_primitive_invention/v837aq/plots/control_margin_by_family.png`
- `experiments/v837_primitive_invention/v837aq/plots/cross_organism_disagreement.png`
- `experiments/v837_primitive_invention/v837aq/plots/evidence_ladder.png`
- `experiments/v837_primitive_invention/v837aq/plots/heldout_operator_pass.png`
- `experiments/v837_primitive_invention/v837aq/plots/historical_robustness.png`
- `experiments/v837_primitive_invention/v837aq/plots/localization_support_size.png`
- `experiments/v837_primitive_invention/v837aq/plots/meta_operator_vs_program.png`
- `experiments/v837_primitive_invention/v837aq/plots/operator_order.png`
- `experiments/v837_primitive_invention/v837aq/plots/operator_pass_fraction.png`
- `experiments/v837_primitive_invention/v837aq/plots/operator_response_nrmse.png`
- `experiments/v837_primitive_invention/v837aq/plots/predictive_hankel_rank.png`

