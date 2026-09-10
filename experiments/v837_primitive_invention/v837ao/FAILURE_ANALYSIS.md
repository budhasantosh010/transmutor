# V837ao Failure Analysis

## 1. Starting scientific state

V837an closed at GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN with four powered STATE40-K1 causal families and four semantic compilers.

## 2. Why q-vector alignment was rejected

V837ao never aligns q vectors or 40D microstates across organisms; all comparisons are semantic/backend-local.

## 3. Reader failures

{
  "conditional_routing": {
    "B0_GLOBAL_K1": {
      "organisms": 5,
      "reader_passes": 0,
      "algebra_passes": 5,
      "phase_pass_fraction": 0.0
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 5,
      "reader_passes": 0,
      "algebra_passes": 5,
      "phase_pass_fraction": 0.0
    },
    "B2_PHASE_K1": {
      "organisms": 5,
      "reader_passes": 0,
      "algebra_passes": 5,
      "phase_pass_fraction": 0.0
    }
  },
  "delayed_recall": {
    "B0_GLOBAL_K1": {
      "organisms": 6,
      "reader_passes": 0,
      "algebra_passes": 6,
      "phase_pass_fraction": 0.2777777777777778
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 6,
      "reader_passes": 0,
      "algebra_passes": 6,
      "phase_pass_fraction": 0.2777777777777778
    },
    "B2_PHASE_K1": {
      "organisms": 6,
      "reader_passes": 0,
      "algebra_passes": 6,
      "phase_pass_fraction": 0.2222222222222222
    }
  },
  "iterative_state": {
    "B0_GLOBAL_K1": {
      "organisms": 6,
      "reader_passes": 0,
      "algebra_passes": 6,
      "phase_pass_fraction": 0.6666666666666666
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 6,
      "reader_passes": 0,
      "algebra_passes": 6,
      "phase_pass_fraction": 0.6666666666666666
    },
    "B2_PHASE_K1": {
      "organisms": 6,
      "reader_passes": 0,
      "algebra_passes": 0,
      "phase_pass_fraction": 0.0
    }
  },
  "variable_composition": {
    "B0_GLOBAL_K1": {
      "organisms": 4,
      "reader_passes": 0,
      "algebra_passes": 4,
      "phase_pass_fraction": 0.0
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 4,
      "reader_passes": 0,
      "algebra_passes": 4,
      "phase_pass_fraction": 0.0
    },
    "B2_PHASE_K1": {
      "organisms": 4,
      "reader_passes": 0,
      "algebra_passes": 4,
      "phase_pass_fraction": 0.0
    }
  }
}

## 4. Writer/gauge failures

Setter algebra itself was generally well-defined where a reader existed; the scientific bottleneck was absolute semantic readout/realization, not gradient optimization.

## 5. Absolute-setpoint failures

{
  "conditional_routing": {
    "B0_GLOBAL_K1": {
      "organisms": 5,
      "passes": 0,
      "mean_organism_median_recovery": 0.07161298497743465,
      "median_organism_median_recovery": 0.0665423876607783
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 5,
      "passes": 0,
      "mean_organism_median_recovery": 0.05640990619372547,
      "median_organism_median_recovery": 0.05371951900969707
    },
    "B2_PHASE_K1": {
      "organisms": 5,
      "passes": 0,
      "mean_organism_median_recovery": 0.2920902267703749,
      "median_organism_median_recovery": 0.28782003051085536
    }
  },
  "delayed_recall": {
    "B0_GLOBAL_K1": {
      "organisms": 6,
      "passes": 0,
      "mean_organism_median_recovery": 0.9960386841196472,
      "median_organism_median_recovery": 0.9966003667161238
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 6,
      "passes": 0,
      "mean_organism_median_recovery": 0.9859199409292844,
      "median_organism_median_recovery": 0.9966391335159311
    },
    "B2_PHASE_K1": {
      "organisms": 6,
      "passes": 3,
      "mean_organism_median_recovery": 0.996949458128848,
      "median_organism_median_recovery": 0.9971162013053483
    }
  },
  "iterative_state": {
    "B0_GLOBAL_K1": {
      "organisms": 6,
      "passes": 5,
      "mean_organism_median_recovery": 0.8280113605515718,
      "median_organism_median_recovery": 0.8310355084162877
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 6,
      "passes": 6,
      "mean_organism_median_recovery": 0.8777208796997322,
      "median_organism_median_recovery": 0.875757420292999
    },
    "B2_PHASE_K1": {
      "organisms": 6,
      "passes": 0,
      "mean_organism_median_recovery": null,
      "median_organism_median_recovery": null
    }
  },
  "variable_composition": {
    "B0_GLOBAL_K1": {
      "organisms": 4,
      "passes": 0,
      "mean_organism_median_recovery": 0.5250006520958294,
      "median_organism_median_recovery": 0.5342272110842734
    },
    "B1_PHASE_GAUGE_K1": {
      "organisms": 4,
      "passes": 0,
      "mean_organism_median_recovery": 0.49844439420573894,
      "median_organism_median_recovery": 0.49821188333731836
    },
    "B2_PHASE_K1": {
      "organisms": 4,
      "passes": 0,
      "mean_organism_median_recovery": 0.4660863290097318,
      "median_organism_median_recovery": 0.4645887398839437
    }
  }
}

## 6. Random/sham-control failures

Canonical setters were compared with 32 deterministic Haar directions and shuffled semantics. Control separation was not sufficient to rescue a backend whose reader/SET gate failed.

## 7. Phase-realization failures

B0, then B1, then B2 were evaluated in the frozen order. No family reached the discovery family gate.

## 8. Quotient/residual failures

Not scientifically evaluated after the mandatory reader+absolute-SET precursor failed; no result is fabricated.

## 9. Dynamical-commutativity failures

Not scientifically evaluated after the mandatory reader+absolute-SET precursor failed; no result is fabricated.

## 10. Heldout-backend failures

No heldout backend was fit because no family candidate survived META/freeze.

## 11. Calibration-budget failures

N=1..64 remained unspent because fitting a null candidate would violate the frozen protocol.

## 12. Cross-organism disagreement

Not evaluated as a canonical agreement claim because no heldout canonical backend existed.

## 13. OOD failures

OOD was measured during absolute SET; individual failures are preserved in raw results/failure memory.

## 14. Law-recovery diagnostic failures

Audit was not run without a canonical trajectory candidate.

## 15. Engineering failures

No retained engineering failure changed the scientific result. Early source-contract assertion was corrected before scientific execution and no scientific artifact was produced from the invalid check.

## 16. Hypotheses definitively ruled out

Within frozen B0/B1/B2 linear K1 backends, the four powered families do not establish an absolute canonical scalar coordinate.

## 17. Hypotheses only provisionally ruled out

Nonlinear readers/writers, higher-dimensional projected causal states, or different absolute-coordinate constructions remain open.

## 18. Remaining uncertainty

Whether V837an causal directions embed in a nonlinear or higher-dimensional canonical state representation.

## 19. Exact next justified experiments

V837ap_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE; do not repeat B0/B1/B2 unchanged.

## Failure ledger summary

Searchable entries: 285
