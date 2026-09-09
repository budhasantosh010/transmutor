# V837 Structural Search Recovery Report

## 1. Why structural search reopened

V837ai confirmed AF1D representation adequacy at 4x and explicitly authorized structural-search recovery.

## 2. Why old V837 search was confounded

Historical search failure coincided with an inadequate substrate. V837aj fixes the substrate first.

## 3. Frozen AF1D mechanism

Exact AF1D: ten 4D cells, rank-4 candidate coupling, one global joint input+state scalar carry controller, and ten independent 6->6 candidate projections.

## 4. Searchable structural axis

Only message-edge existence and SAME_STEP/RECURRENT timing are searchable; cell count and all learned mechanisms remain frozen.

## 5. Validation-leakage correction

Search fitness never reads seeds 20000-20127. Final validation becomes accessible only after champion topology freeze (plus the isolated AJ0 anchor reproduction control).

## 6. Initialization-confound correction

All common non-edge parameters are bit-identical within paired topology evaluations; common semantic edges receive path-independent initialization.

## 7. Calibration panel

The fixed task-independent panel contains 12 topologies.

## 8. Fidelity ladder

F_LEGACY, F0, F1, F2, F3 are compared against F4 target ranking with two initialization replicates per topology/family/fidelity.

## 9. Proxy rank results

{
  "F0": {
    "families": {
      "conditional_routing": {
        "kendall_tau_b": 0.06060606060606061,
        "pairwise_order_accuracy": 0.5303030303030303,
        "spearman_rho": 0.15384615384615385,
        "top2_recall": 0.5,
        "top4_recall": 0.75
      },
      "delayed_recall": {
        "kendall_tau_b": 0.0,
        "pairwise_order_accuracy": 0.5,
        "spearman_rho": -0.027972027972027972,
        "top2_recall": 0.0,
        "top4_recall": 0.25
      },
      "iterative_state": {
        "kendall_tau_b": 0.7575757575757576,
        "pairwise_order_accuracy": 0.8787878787878788,
        "spearman_rho": 0.8881118881118882,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      },
      "partial_observation": {
        "kendall_tau_b": 0.7575757575757576,
        "pairwise_order_accuracy": 0.8787878787878788,
        "spearman_rho": 0.8951048951048951,
        "top2_recall": 0.5,
        "top4_recall": 1.0
      },
      "variable_composition": {
        "kendall_tau_b": 0.696969696969697,
        "pairwise_order_accuracy": 0.8484848484848485,
        "spearman_rho": 0.8461538461538463,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      }
    },
    "median_kendall_tau": 0.696969696969697,
    "median_pairwise_order_accuracy": 0.8484848484848485,
    "median_spearman_rho": 0.8461538461538463,
    "median_top2_recall": 0.5,
    "median_top4_recall": 0.75,
    "minimum_family_kendall_tau": 0.0,
    "minimum_family_top4_recall": 0.25,
    "passes_frozen_gate": false
  },
  "F1": {
    "families": {
      "conditional_routing": {
        "kendall_tau_b": 0.36363636363636365,
        "pairwise_order_accuracy": 0.6818181818181818,
        "spearman_rho": 0.44055944055944063,
        "top2_recall": 0.5,
        "top4_recall": 0.75
      },
      "delayed_recall": {
        "kendall_tau_b": 0.09090909090909091,
        "pairwise_order_accuracy": 0.5454545454545454,
        "spearman_rho": 0.06993006993006995,
        "top2_recall": 0.0,
        "top4_recall": 0.25
      },
      "iterative_state": {
        "kendall_tau_b": 0.9393939393939394,
        "pairwise_order_accuracy": 0.9696969696969697,
        "spearman_rho": 0.9860139860139862,
        "top2_recall": 1.0,
        "top4_recall": 1.0
      },
      "partial_observation": {
        "kendall_tau_b": 0.9393939393939394,
        "pairwise_order_accuracy": 0.9696969696969697,
        "spearman_rho": 0.9790209790209792,
        "top2_recall": 0.5,
        "top4_recall": 1.0
      },
      "variable_composition": {
        "kendall_tau_b": 0.7575757575757576,
        "pairwise_order_accuracy": 0.8787878787878788,
        "spearman_rho": 0.8881118881118882,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      }
    },
    "median_kendall_tau": 0.7575757575757576,
    "median_pairwise_order_accuracy": 0.8787878787878788,
    "median_spearman_rho": 0.8881118881118882,
    "median_top2_recall": 0.5,
    "median_top4_recall": 0.75,
    "minimum_family_kendall_tau": 0.09090909090909091,
    "minimum_family_top4_recall": 0.25,
    "passes_frozen_gate": false
  },
  "F2": {
    "families": {
      "conditional_routing": {
        "kendall_tau_b": 0.5151515151515151,
        "pairwise_order_accuracy": 0.7575757575757576,
        "spearman_rho": 0.6293706293706295,
        "top2_recall": 1.0,
        "top4_recall": 0.5
      },
      "delayed_recall": {
        "kendall_tau_b": -0.030303030303030304,
        "pairwise_order_accuracy": 0.48484848484848486,
        "spearman_rho": -0.034965034965034975,
        "top2_recall": 0.0,
        "top4_recall": 0.25
      },
      "iterative_state": {
        "kendall_tau_b": 0.9696969696969697,
        "pairwise_order_accuracy": 0.9848484848484849,
        "spearman_rho": 0.9930069930069931,
        "top2_recall": 1.0,
        "top4_recall": 1.0
      },
      "partial_observation": {
        "kendall_tau_b": 0.8484848484848485,
        "pairwise_order_accuracy": 0.9242424242424242,
        "spearman_rho": 0.9580419580419581,
        "top2_recall": 0.5,
        "top4_recall": 1.0
      },
      "variable_composition": {
        "kendall_tau_b": 0.7878787878787878,
        "pairwise_order_accuracy": 0.8939393939393939,
        "spearman_rho": 0.9020979020979022,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      }
    },
    "median_kendall_tau": 0.7878787878787878,
    "median_pairwise_order_accuracy": 0.8939393939393939,
    "median_spearman_rho": 0.9020979020979022,
    "median_top2_recall": 1.0,
    "median_top4_recall": 0.75,
    "minimum_family_kendall_tau": -0.030303030303030304,
    "minimum_family_top4_recall": 0.25,
    "passes_frozen_gate": false
  },
  "F3": {
    "families": {
      "conditional_routing": {
        "kendall_tau_b": 0.9393939393939394,
        "pairwise_order_accuracy": 0.9696969696969697,
        "spearman_rho": 0.9860139860139862,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      },
      "delayed_recall": {
        "kendall_tau_b": 0.6666666666666666,
        "pairwise_order_accuracy": 0.8333333333333334,
        "spearman_rho": 0.8041958041958043,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      },
      "iterative_state": {
        "kendall_tau_b": 0.9696969696969697,
        "pairwise_order_accuracy": 0.9848484848484849,
        "spearman_rho": 0.9930069930069931,
        "top2_recall": 1.0,
        "top4_recall": 1.0
      },
      "partial_observation": {
        "kendall_tau_b": 0.9393939393939394,
        "pairwise_order_accuracy": 0.9696969696969697,
        "spearman_rho": 0.9860139860139862,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      },
      "variable_composition": {
        "kendall_tau_b": 0.9090909090909091,
        "pairwise_order_accuracy": 0.9545454545454546,
        "spearman_rho": 0.9720279720279721,
        "top2_recall": 1.0,
        "top4_recall": 1.0
      }
    },
    "median_kendall_tau": 0.9393939393939394,
    "median_pairwise_order_accuracy": 0.9696969696969697,
    "median_spearman_rho": 0.9860139860139862,
    "median_top2_recall": 1.0,
    "median_top4_recall": 0.75,
    "minimum_family_kendall_tau": 0.6666666666666666,
    "minimum_family_top4_recall": 0.75,
    "passes_frozen_gate": true
  },
  "F_LEGACY": {
    "families": {
      "conditional_routing": {
        "kendall_tau_b": 0.3333333333333333,
        "pairwise_order_accuracy": 0.6666666666666666,
        "spearman_rho": 0.39860139860139865,
        "top2_recall": 0.5,
        "top4_recall": 0.75
      },
      "delayed_recall": {
        "kendall_tau_b": -0.09090909090909091,
        "pairwise_order_accuracy": 0.45454545454545453,
        "spearman_rho": -0.16083916083916086,
        "top2_recall": 0.0,
        "top4_recall": 0.25
      },
      "iterative_state": {
        "kendall_tau_b": 0.5151515151515151,
        "pairwise_order_accuracy": 0.7575757575757576,
        "spearman_rho": 0.6993006993006995,
        "top2_recall": 0.5,
        "top4_recall": 0.75
      },
      "partial_observation": {
        "kendall_tau_b": 0.8484848484848485,
        "pairwise_order_accuracy": 0.9242424242424242,
        "spearman_rho": 0.9510489510489512,
        "top2_recall": 1.0,
        "top4_recall": 0.75
      },
      "variable_composition": {
        "kendall_tau_b": 0.7575757575757576,
        "pairwise_order_accuracy": 0.8787878787878788,
        "spearman_rho": 0.8951048951048951,
        "top2_recall": 0.5,
        "top4_recall": 0.75
      }
    },
    "median_kendall_tau": 0.5151515151515151,
    "median_pairwise_order_accuracy": 0.7575757575757576,
    "median_spearman_rho": 0.6993006993006995,
    "median_top2_recall": 0.5,
    "median_top4_recall": 0.75,
    "minimum_family_kendall_tau": -0.09090909090909091,
    "minimum_family_top4_recall": 0.25,
    "passes_frozen_gate": false
  }
}

## 10. Selected search fidelity

F3

## 11. Constructive search design

Anchor-free (mu+lambda) search starts from the 19-edge minimal topology and evaluates exactly 64 unique candidates/run.

## 12. Equal-budget random design

Random sampling matches directed candidate total-edge and recurrent-edge counts slot by slot, with identical initialization slots.

## 13. Directed search results

{
  "families": {
    "conditional_routing": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 5,
      "development_scores": [
        1.0,
        1.0,
        0.998046875,
        0.998046875,
        1.0
      ],
      "validation_scores": [
        0.9765625,
        0.8984375,
        0.8984375,
        0.9375,
        1.0
      ],
      "median_development": 1.0,
      "median_validation": 0.9375,
      "pass": true
    },
    "delayed_recall": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 5,
      "development_scores": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ],
      "validation_scores": [
        0.984375,
        0.9921875,
        0.984375,
        0.96875,
        0.9921875
      ],
      "median_development": 1.0,
      "median_validation": 0.984375,
      "pass": true
    },
    "iterative_state": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 5,
      "development_scores": [
        0.998046875,
        0.99609375,
        0.99609375,
        0.99609375,
        0.99609375
      ],
      "validation_scores": [
        0.9921875,
        0.9921875,
        0.9921875,
        0.9921875,
        1.0
      ],
      "median_development": 0.99609375,
      "median_validation": 0.9921875,
      "pass": true
    },
    "partial_observation": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 1,
      "development_scores": [
        0.970703125,
        0.9453125,
        0.94140625,
        0.98046875,
        0.9765625
      ],
      "validation_scores": [
        0.8203125,
        0.828125,
        0.875,
        0.8125,
        0.7734375
      ],
      "median_development": 0.970703125,
      "median_validation": 0.8203125,
      "pass": false
    },
    "variable_composition": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 4,
      "development_scores": [
        0.998046875,
        0.9921875,
        0.986328125,
        0.98046875,
        0.998046875
      ],
      "validation_scores": [
        0.9609375,
        0.8984375,
        0.8125,
        0.875,
        0.9609375
      ],
      "median_development": 0.9921875,
      "median_validation": 0.8984375,
      "pass": true
    }
  },
  "families_passing": 4,
  "structurally_competent": true
}

## 14. Random results

{
  "families": {
    "conditional_routing": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 4,
      "development_scores": [
        1.0,
        0.990234375,
        0.9921875,
        1.0,
        1.0
      ],
      "validation_scores": [
        0.9765625,
        0.796875,
        0.9453125,
        0.9765625,
        0.984375
      ],
      "median_development": 1.0,
      "median_validation": 0.9765625,
      "pass": true
    },
    "delayed_recall": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 5,
      "development_scores": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ],
      "validation_scores": [
        0.9765625,
        0.984375,
        1.0,
        0.9921875,
        1.0
      ],
      "median_development": 1.0,
      "median_validation": 0.9921875,
      "pass": true
    },
    "iterative_state": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 5,
      "development_scores": [
        1.0,
        0.99609375,
        0.998046875,
        0.99609375,
        0.998046875
      ],
      "validation_scores": [
        1.0,
        1.0,
        0.9921875,
        1.0,
        0.9921875
      ],
      "median_development": 0.998046875,
      "median_validation": 1.0,
      "pass": true
    },
    "partial_observation": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 2,
      "development_scores": [
        0.97265625,
        0.9765625,
        0.94921875,
        0.98828125,
        0.9921875
      ],
      "validation_scores": [
        0.8515625,
        0.796875,
        0.859375,
        0.828125,
        0.765625
      ],
      "median_development": 0.9765625,
      "median_validation": 0.828125,
      "pass": false
    },
    "variable_composition": {
      "runs": 5,
      "competent_required": 3,
      "competent_hits": 4,
      "development_scores": [
        0.9921875,
        0.9921875,
        0.994140625,
        0.990234375,
        0.99609375
      ],
      "validation_scores": [
        0.921875,
        0.90625,
        0.828125,
        0.8828125,
        0.9453125
      ],
      "median_development": 0.9921875,
      "median_validation": 0.90625,
      "pass": true
    }
  },
  "families_passing": 4,
  "structurally_competent": true
}

## 15. Finalized validation results

[
  {
    "family": "conditional_routing",
    "run_index": 0,
    "search_validation": 0.9765625,
    "random_validation": 0.9765625,
    "validation_delta": 0.0,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "conditional_routing",
    "run_index": 1,
    "search_validation": 0.8984375,
    "random_validation": 0.796875,
    "validation_delta": 0.1015625,
    "search_development": 1.0,
    "random_development": 0.990234375,
    "development_delta": 0.009765625,
    "search_competent": true,
    "random_competent": false
  },
  {
    "family": "conditional_routing",
    "run_index": 2,
    "search_validation": 0.8984375,
    "random_validation": 0.9453125,
    "validation_delta": -0.046875,
    "search_development": 0.998046875,
    "random_development": 0.9921875,
    "development_delta": 0.005859375,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "conditional_routing",
    "run_index": 3,
    "search_validation": 0.9375,
    "random_validation": 0.9765625,
    "validation_delta": -0.0390625,
    "search_development": 0.998046875,
    "random_development": 1.0,
    "development_delta": -0.001953125,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "conditional_routing",
    "run_index": 4,
    "search_validation": 1.0,
    "random_validation": 0.984375,
    "validation_delta": 0.015625,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "delayed_recall",
    "run_index": 0,
    "search_validation": 0.984375,
    "random_validation": 0.9765625,
    "validation_delta": 0.0078125,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "delayed_recall",
    "run_index": 1,
    "search_validation": 0.9921875,
    "random_validation": 0.984375,
    "validation_delta": 0.0078125,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "delayed_recall",
    "run_index": 2,
    "search_validation": 0.984375,
    "random_validation": 1.0,
    "validation_delta": -0.015625,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "delayed_recall",
    "run_index": 3,
    "search_validation": 0.96875,
    "random_validation": 0.9921875,
    "validation_delta": -0.0234375,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "delayed_recall",
    "run_index": 4,
    "search_validation": 0.9921875,
    "random_validation": 1.0,
    "validation_delta": -0.0078125,
    "search_development": 1.0,
    "random_development": 1.0,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "iterative_state",
    "run_index": 0,
    "search_validation": 0.9921875,
    "random_validation": 1.0,
    "validation_delta": -0.0078125,
    "search_development": 0.998046875,
    "random_development": 1.0,
    "development_delta": -0.001953125,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "iterative_state",
    "run_index": 1,
    "search_validation": 0.9921875,
    "random_validation": 1.0,
    "validation_delta": -0.0078125,
    "search_development": 0.99609375,
    "random_development": 0.99609375,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "iterative_state",
    "run_index": 2,
    "search_validation": 0.9921875,
    "random_validation": 0.9921875,
    "validation_delta": 0.0,
    "search_development": 0.99609375,
    "random_development": 0.998046875,
    "development_delta": -0.001953125,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "iterative_state",
    "run_index": 3,
    "search_validation": 0.9921875,
    "random_validation": 1.0,
    "validation_delta": -0.0078125,
    "search_development": 0.99609375,
    "random_development": 0.99609375,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "iterative_state",
    "run_index": 4,
    "search_validation": 1.0,
    "random_validation": 0.9921875,
    "validation_delta": 0.0078125,
    "search_development": 0.99609375,
    "random_development": 0.998046875,
    "development_delta": -0.001953125,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "partial_observation",
    "run_index": 0,
    "search_validation": 0.8203125,
    "random_validation": 0.8515625,
    "validation_delta": -0.03125,
    "search_development": 0.970703125,
    "random_development": 0.97265625,
    "development_delta": -0.001953125,
    "search_competent": false,
    "random_competent": true
  },
  {
    "family": "partial_observation",
    "run_index": 1,
    "search_validation": 0.828125,
    "random_validation": 0.796875,
    "validation_delta": 0.03125,
    "search_development": 0.9453125,
    "random_development": 0.9765625,
    "development_delta": -0.03125,
    "search_competent": false,
    "random_competent": false
  },
  {
    "family": "partial_observation",
    "run_index": 2,
    "search_validation": 0.875,
    "random_validation": 0.859375,
    "validation_delta": 0.015625,
    "search_development": 0.94140625,
    "random_development": 0.94921875,
    "development_delta": -0.0078125,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "partial_observation",
    "run_index": 3,
    "search_validation": 0.8125,
    "random_validation": 0.828125,
    "validation_delta": -0.015625,
    "search_development": 0.98046875,
    "random_development": 0.98828125,
    "development_delta": -0.0078125,
    "search_competent": false,
    "random_competent": false
  },
  {
    "family": "partial_observation",
    "run_index": 4,
    "search_validation": 0.7734375,
    "random_validation": 0.765625,
    "validation_delta": 0.0078125,
    "search_development": 0.9765625,
    "random_development": 0.9921875,
    "development_delta": -0.015625,
    "search_competent": false,
    "random_competent": false
  },
  {
    "family": "variable_composition",
    "run_index": 0,
    "search_validation": 0.9609375,
    "random_validation": 0.921875,
    "validation_delta": 0.0390625,
    "search_development": 0.998046875,
    "random_development": 0.9921875,
    "development_delta": 0.005859375,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "variable_composition",
    "run_index": 1,
    "search_validation": 0.8984375,
    "random_validation": 0.90625,
    "validation_delta": -0.0078125,
    "search_development": 0.9921875,
    "random_development": 0.9921875,
    "development_delta": 0.0,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "variable_composition",
    "run_index": 2,
    "search_validation": 0.8125,
    "random_validation": 0.828125,
    "validation_delta": -0.015625,
    "search_development": 0.986328125,
    "random_development": 0.994140625,
    "development_delta": -0.0078125,
    "search_competent": false,
    "random_competent": false
  },
  {
    "family": "variable_composition",
    "run_index": 3,
    "search_validation": 0.875,
    "random_validation": 0.8828125,
    "validation_delta": -0.0078125,
    "search_development": 0.98046875,
    "random_development": 0.990234375,
    "development_delta": -0.009765625,
    "search_competent": true,
    "random_competent": true
  },
  {
    "family": "variable_composition",
    "run_index": 4,
    "search_validation": 0.9609375,
    "random_validation": 0.9453125,
    "validation_delta": 0.015625,
    "search_development": 0.998046875,
    "random_development": 0.99609375,
    "development_delta": 0.001953125,
    "search_competent": true,
    "random_competent": true
  }
]

## 16. Competent hit rates

{
  "directed": 0.8,
  "random": 0.8
}

## 17. Search-vs-random statistics

{
  "permutations": 100000,
  "observed_mean_delta": 0.000625,
  "observed_median_delta": -0.0078125,
  "one_sided_search_gt_random_p": 0.4817451825481745,
  "two_sided_p": 0.9618303816961831,
  "rng_namespace": "v837aj-paired-permutation"
}

## 18. Structural diversity

{
  "directed": {
    "champions": 25,
    "unique_topologies": 25,
    "median_edge_jaccard": 0.6111111111111112,
    "median_same_step_jaccard": 0.5555555555555556,
    "median_recurrent_jaccard": 0.6363636363636364,
    "median_degree_cosine": 0.8714363172483459,
    "median_topology_edit_distance": 8.0
  },
  "random": {
    "champions": 25,
    "unique_topologies": 25,
    "median_edge_jaccard": 0.0625,
    "median_same_step_jaccard": 0.07142857142857142,
    "median_recurrent_jaccard": 0.05263157894736842,
    "median_degree_cosine": 0.6723994935181141,
    "median_topology_edit_distance": 30.0
  }
}

## 19. Message dependence

{
  "directed": {
    "competent_champions": 20,
    "median_success_drop": 0.19921875,
    "median_abs_prediction_change": 0.12395379319787025
  },
  "random": {
    "competent_champions": 20,
    "median_success_drop": 0.140625,
    "median_abs_prediction_change": 0.10005846247076988
  },
  "fixed_af1d_anchor_4x": {
    "mean_abs_prediction_delta_median": 0.23562179505825043,
    "success_drop_median": 0.453125
  }
}

## 20. Compute/resource comparison

{
  "anchor": {
    "optimizer_steps": 4800,
    "processed_examples": 2457600,
    "forward_calls": 4850,
    "backward_calls": 4800,
    "environment_interactions": 111225,
    "cpu_seconds": 1679.0,
    "wall_seconds": 2077.4327273002127,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 0,
    "fits": 25
  },
  "calibration": {
    "optimizer_steps": 63360,
    "processed_examples": 19399680,
    "forward_calls": 64800,
    "backward_calls": 63360,
    "environment_interactions": 1575240,
    "cpu_seconds": 18770.5625,
    "wall_seconds": 29326.050889701,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 182052881664,
    "fits": 720
  },
  "proxy_directed": {
    "optimizer_steps": 230400,
    "processed_examples": 88473600,
    "forward_calls": 233600,
    "backward_calls": 230400,
    "environment_interactions": 5696000,
    "cpu_seconds": 59760.78125,
    "wall_seconds": 152083.23325730092,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 783356501376,
    "fits": 1600
  },
  "proxy_random": {
    "optimizer_steps": 230400,
    "processed_examples": 88473600,
    "forward_calls": 233600,
    "backward_calls": 230400,
    "environment_interactions": 5696000,
    "cpu_seconds": 53415.203125,
    "wall_seconds": 62209.91579289839,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 783356501376,
    "fits": 1600
  },
  "final_directed": {
    "optimizer_steps": 4800,
    "processed_examples": 2457600,
    "forward_calls": 4875,
    "backward_calls": 4800,
    "environment_interactions": 111225,
    "cpu_seconds": 1121.015625,
    "wall_seconds": 1346.0956716,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 21723213312,
    "fits": 25
  },
  "final_random": {
    "optimizer_steps": 4800,
    "processed_examples": 2457600,
    "forward_calls": 4875,
    "backward_calls": 4800,
    "environment_interactions": 111225,
    "cpu_seconds": 1206.3125,
    "wall_seconds": 1390.3697025999973,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 21744378624,
    "fits": 25
  },
  "primary_proxy_directed": {
    "optimizer_steps": 230400,
    "processed_examples": 88473600,
    "forward_calls": 233600,
    "backward_calls": 230400,
    "environment_interactions": 5696000,
    "cpu_seconds": 59760.78125,
    "wall_seconds": 152083.23325730092,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 783356501376,
    "fits": 1600
  },
  "primary_proxy_random": {
    "optimizer_steps": 230400,
    "processed_examples": 88473600,
    "forward_calls": 233600,
    "backward_calls": 230400,
    "environment_interactions": 5696000,
    "cpu_seconds": 53415.203125,
    "wall_seconds": 62209.91579289839,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 783356501376,
    "fits": 1600
  },
  "primary_final_directed": {
    "optimizer_steps": 4800,
    "processed_examples": 2457600,
    "forward_calls": 4875,
    "backward_calls": 4800,
    "environment_interactions": 111225,
    "cpu_seconds": 1121.015625,
    "wall_seconds": 1346.0956716,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 21723213312,
    "fits": 25
  },
  "primary_final_random": {
    "optimizer_steps": 4800,
    "processed_examples": 2457600,
    "forward_calls": 4875,
    "backward_calls": 4800,
    "environment_interactions": 111225,
    "cpu_seconds": 1206.3125,
    "wall_seconds": 1390.3697025999973,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 21744378624,
    "fits": 25
  },
  "robustness_proxy_directed": {
    "optimizer_steps": 0,
    "processed_examples": 0,
    "forward_calls": 0,
    "backward_calls": 0,
    "environment_interactions": 0,
    "cpu_seconds": 0,
    "wall_seconds": 0,
    "gpu_seconds": 0,
    "modeled_active_mac_volume": 0,
    "fits": 0
  },
  "robustness_proxy_random": {
    "optimizer_steps": 0,
    "processed_examples": 0,
    "forward_calls": 0,
    "backward_calls": 0,
    "environment_interactions": 0,
    "cpu_seconds": 0,
    "wall_seconds": 0,
    "gpu_seconds": 0,
    "modeled_active_mac_volume": 0,
    "fits": 0
  },
  "robustness_final_directed": {
    "optimizer_steps": 0,
    "processed_examples": 0,
    "forward_calls": 0,
    "backward_calls": 0,
    "environment_interactions": 0,
    "cpu_seconds": 0,
    "wall_seconds": 0,
    "gpu_seconds": 0,
    "modeled_active_mac_volume": 0,
    "fits": 0
  },
  "robustness_final_random": {
    "optimizer_steps": 0,
    "processed_examples": 0,
    "forward_calls": 0,
    "backward_calls": 0,
    "environment_interactions": 0,
    "cpu_seconds": 0,
    "wall_seconds": 0,
    "gpu_seconds": 0,
    "modeled_active_mac_volume": 0,
    "fits": 0
  },
  "primary_stage_b": {
    "optimizer_steps": 470400,
    "processed_examples": 181862400,
    "forward_calls": 476950,
    "backward_calls": 470400,
    "environment_interactions": 11614450,
    "cpu_seconds": 115503.3125,
    "wall_seconds": 217029.61442439933,
    "gpu_seconds": 0.0,
    "modeled_active_mac_volume": 1610180594688,
    "fits": 3250
  },
  "robustness_extension": {
    "optimizer_steps": 0,
    "processed_examples": 0,
    "forward_calls": 0,
    "backward_calls": 0,
    "environment_interactions": 0,
    "cpu_seconds": 0,
    "wall_seconds": 0,
    "gpu_seconds": 0,
    "modeled_active_mac_volume": 0,
    "fits": 0
  },
  "candidate_evaluations": {
    "calibration": 720,
    "directed": 1600,
    "random": 1600
  },
  "union_unique_task_episodes": 3200
}

## 21. Capability / structure / compute scoreboard

{
  "run": true,
  "fixed_af1d_anchor": {
    "engine": "FIXED_AF1D_ANCHOR",
    "families": [
      {
        "family": "conditional_routing",
        "replicates": 5,
        "median_final_validation_success": 0.9375
      },
      {
        "family": "delayed_recall",
        "replicates": 5,
        "median_final_validation_success": 0.984375
      },
      {
        "family": "iterative_state",
        "replicates": 5,
        "median_final_validation_success": 1.0
      },
      {
        "family": "partial_observation",
        "replicates": 5,
        "median_final_validation_success": 0.8125
      },
      {
        "family": "variable_composition",
        "replicates": 5,
        "median_final_validation_success": 0.875
      }
    ],
    "median_final_validation_success": 0.9375,
    "message_ablation_success_drop": 0.453125,
    "message_ablation_scope": "V837ai 4x global median; family-specific anchor ablation was not rerun in V837aj",
    "edge_count": 55,
    "active_parameters": 1643,
    "modeled_macs_per_timestep": 1426,
    "search_evaluations_required": null,
    "search_evaluations_required_semantics": "fixed predeclared AF1D anchor; no structural-search evaluations required"
  },
  "directed": {
    "engine": "DIRECTED_STRUCTURAL_SEARCH",
    "champions": 25,
    "competent_champions": 20,
    "median_final_validation_success": 0.9609375,
    "median_message_ablation_success_drop": 0.1796875,
    "median_edge_count": 17.0,
    "median_active_parameters": 1605.0,
    "median_modeled_macs_per_timestep": 1274.0,
    "median_search_evaluations_required": 50.0,
    "rows": [
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "conditional_routing",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.9765625,
        "message_ablation_success_drop": 0.203125,
        "edge_count": 20,
        "active_parameters": 1608,
        "modeled_macs_per_timestep": 1286,
        "search_evaluations_required": 58
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "conditional_routing",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 0.8984375,
        "message_ablation_success_drop": 0.3671875,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 49
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "conditional_routing",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.8984375,
        "message_ablation_success_drop": 0.5234375,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 20
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "conditional_routing",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.9375,
        "message_ablation_success_drop": 0.1796875,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 20
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "conditional_routing",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.4921875,
        "edge_count": 22,
        "active_parameters": 1610,
        "modeled_macs_per_timestep": 1294,
        "search_evaluations_required": 32
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "delayed_recall",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.984375,
        "message_ablation_success_drop": 0.0234375,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 61
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "delayed_recall",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0390625,
        "edge_count": 15,
        "active_parameters": 1603,
        "modeled_macs_per_timestep": 1266,
        "search_evaluations_required": 45
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "delayed_recall",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.984375,
        "message_ablation_success_drop": 0.3671875,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 37
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "delayed_recall",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.96875,
        "message_ablation_success_drop": 0.0703125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 61
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "delayed_recall",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0859375,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 47
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "iterative_state",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0546875,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 61
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "iterative_state",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 61
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "iterative_state",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0078125,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 63
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "iterative_state",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": -0.0078125,
        "edge_count": 15,
        "active_parameters": 1603,
        "modeled_macs_per_timestep": 1266,
        "search_evaluations_required": 61
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "iterative_state",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.234375,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 62
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "partial_observation",
        "run_index": 0,
        "competent": false,
        "final_validation_success": 0.8203125,
        "message_ablation_success_drop": 0.0625,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 52
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "partial_observation",
        "run_index": 1,
        "competent": false,
        "final_validation_success": 0.828125,
        "message_ablation_success_drop": 0.1015625,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 14
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "partial_observation",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.875,
        "message_ablation_success_drop": 0.1953125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 30
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "partial_observation",
        "run_index": 3,
        "competent": false,
        "final_validation_success": 0.8125,
        "message_ablation_success_drop": 0.125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 59
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "partial_observation",
        "run_index": 4,
        "competent": false,
        "final_validation_success": 0.7734375,
        "message_ablation_success_drop": 0.078125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 60
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "variable_composition",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.9609375,
        "message_ablation_success_drop": 0.28125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 16
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "variable_composition",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 0.8984375,
        "message_ablation_success_drop": 0.484375,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 5
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "variable_composition",
        "run_index": 2,
        "competent": false,
        "final_validation_success": 0.8125,
        "message_ablation_success_drop": 0.1875,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 50
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "variable_composition",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.875,
        "message_ablation_success_drop": 0.375,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 62
      },
      {
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": "variable_composition",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 0.9609375,
        "message_ablation_success_drop": 0.34375,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 16
      }
    ]
  },
  "random": {
    "engine": "RANDOM_STRUCTURAL_SAMPLER",
    "champions": 25,
    "competent_champions": 20,
    "median_final_validation_success": 0.9765625,
    "median_message_ablation_success_drop": 0.1328125,
    "median_edge_count": 18.0,
    "median_active_parameters": 1606.0,
    "median_modeled_macs_per_timestep": 1278.0,
    "median_search_evaluations_required": 50.0,
    "rows": [
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "conditional_routing",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.9765625,
        "message_ablation_success_drop": 0.28125,
        "edge_count": 20,
        "active_parameters": 1608,
        "modeled_macs_per_timestep": 1286,
        "search_evaluations_required": 33
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "conditional_routing",
        "run_index": 1,
        "competent": false,
        "final_validation_success": 0.796875,
        "message_ablation_success_drop": 0.4609375,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 42
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "conditional_routing",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.9453125,
        "message_ablation_success_drop": 0.2578125,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 44
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "conditional_routing",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.9765625,
        "message_ablation_success_drop": 0.421875,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 58
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "conditional_routing",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 0.984375,
        "message_ablation_success_drop": 0.34375,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 22
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "delayed_recall",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.9765625,
        "message_ablation_success_drop": 0.0078125,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 40
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "delayed_recall",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 0.984375,
        "message_ablation_success_drop": 0.0390625,
        "edge_count": 14,
        "active_parameters": 1602,
        "modeled_macs_per_timestep": 1262,
        "search_evaluations_required": 61
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "delayed_recall",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.1328125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 55
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "delayed_recall",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.09375,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 43
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "delayed_recall",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.03125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 49
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "iterative_state",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.1484375,
        "edge_count": 14,
        "active_parameters": 1602,
        "modeled_macs_per_timestep": 1262,
        "search_evaluations_required": 57
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "iterative_state",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.0078125,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 61
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "iterative_state",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0234375,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 63
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "iterative_state",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 1.0,
        "message_ablation_success_drop": 0.0078125,
        "edge_count": 15,
        "active_parameters": 1603,
        "modeled_macs_per_timestep": 1266,
        "search_evaluations_required": 61
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "iterative_state",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 0.9921875,
        "message_ablation_success_drop": 0.0,
        "edge_count": 13,
        "active_parameters": 1601,
        "modeled_macs_per_timestep": 1258,
        "search_evaluations_required": 62
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "partial_observation",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.8515625,
        "message_ablation_success_drop": 0.078125,
        "edge_count": 17,
        "active_parameters": 1605,
        "modeled_macs_per_timestep": 1274,
        "search_evaluations_required": 61
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "partial_observation",
        "run_index": 1,
        "competent": false,
        "final_validation_success": 0.796875,
        "message_ablation_success_drop": 0.109375,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 33
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "partial_observation",
        "run_index": 2,
        "competent": true,
        "final_validation_success": 0.859375,
        "message_ablation_success_drop": 0.15625,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 38
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "partial_observation",
        "run_index": 3,
        "competent": false,
        "final_validation_success": 0.828125,
        "message_ablation_success_drop": 0.0546875,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 17
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "partial_observation",
        "run_index": 4,
        "competent": false,
        "final_validation_success": 0.765625,
        "message_ablation_success_drop": 0.078125,
        "edge_count": 16,
        "active_parameters": 1604,
        "modeled_macs_per_timestep": 1270,
        "search_evaluations_required": 50
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "variable_composition",
        "run_index": 0,
        "competent": true,
        "final_validation_success": 0.921875,
        "message_ablation_success_drop": 0.265625,
        "edge_count": 21,
        "active_parameters": 1609,
        "modeled_macs_per_timestep": 1290,
        "search_evaluations_required": 9
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "variable_composition",
        "run_index": 1,
        "competent": true,
        "final_validation_success": 0.90625,
        "message_ablation_success_drop": 0.3203125,
        "edge_count": 15,
        "active_parameters": 1603,
        "modeled_macs_per_timestep": 1266,
        "search_evaluations_required": 64
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "variable_composition",
        "run_index": 2,
        "competent": false,
        "final_validation_success": 0.828125,
        "message_ablation_success_drop": 0.6015625,
        "edge_count": 22,
        "active_parameters": 1610,
        "modeled_macs_per_timestep": 1294,
        "search_evaluations_required": 44
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "variable_composition",
        "run_index": 3,
        "competent": true,
        "final_validation_success": 0.8828125,
        "message_ablation_success_drop": 0.15625,
        "edge_count": 19,
        "active_parameters": 1607,
        "modeled_macs_per_timestep": 1282,
        "search_evaluations_required": 55
      },
      {
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": "variable_composition",
        "run_index": 4,
        "competent": true,
        "final_validation_success": 0.9453125,
        "message_ablation_success_drop": 0.3046875,
        "edge_count": 18,
        "active_parameters": 1606,
        "modeled_macs_per_timestep": 1278,
        "search_evaluations_required": 64
      }
    ]
  }
}

## 22. V837aj diagnosis

RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT

## 23. Primitive-mining authorization

True

## 24. Strongest scientific claim

Automated structural discovery is established on the frozen AF1D substrate. Diagnosis: RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT.

## 25. Next single program

V837ak_FUNCTIONAL_DYNAMICAL_MOTIF_DISCOVERY
