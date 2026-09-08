# V837 AF1D Sample-Efficiency Report

## 1. Why architecture localization stopped

V837af restored neutral representation adequacy with AF1D at 4/5. V837ai therefore freezes architecture completely and changes only unique development data.

## 2. Frozen AF1D architecture

Exact imported `CandidateInputFactorizationY3`, condition `AF1D_deshared_candidate_input_factorization`: 1,643 active parameters, 420 candidate-projection parameters, and 1,206 recurrent/controller/projection MACs per timestep.

## 3. Historical V837l calibration

Historical GRU families passing: 1x=2/5, 2x=3/5, 4x=5/5. Historical neutral: 1/5, 1/5, 2/5. Historical residual RNN: 2/5, 2/5, 3/5. These rows were imported, not rerun.

## 4. Exact 1x/2x/4x definitions

1x=128 development episodes/family, 2x=256, 4x=512; validation is fixed at 128/family. Development seeds are exact nested prefixes beginning at 10000; validation is 20000..20127.

## 5. Nested-data proof

AI1 is a strict subset of AI2, AI2 is a strict subset of AI4, with no development/validation overlap or duplicate seeds. Union unique family/seed episodes remain 3,200.

## 6. Initialization pairing proof

For every family x replicate, AI1/AI2 and deterministic reconstruction of historical AI4 have identical trainable-tensor SHA-256 fingerprints and exact parameter-by-parameter equality under the frozen V837af seed laws. All ten projections begin bit-identical within a fit and are untied for optimization.

## 7. Reused 4x anchor provenance

Source SHA `1e2be191b39f2c624c043d47ce11fc602b95be1e`; exact V837af AF1D rows reused: 25. Reanalysis independently reproduces 4/5 and the accepted medians.

## 8. AI1 1x results

| family | development median | validation median | pass |
|---|---:|---:|---|
| delayed_recall | 1.000000 | 0.867188 | PASS |
| conditional_routing | 1.000000 | 0.406250 | FAIL |
| iterative_state | 1.000000 | 0.984375 | PASS |
| variable_composition | 1.000000 | 0.703125 | FAIL |
| partial_observation | 1.000000 | 0.734375 | FAIL |

Families passing: **2/5**

## 9. AI2 2x results

| family | development median | validation median | pass |
|---|---:|---:|---|
| delayed_recall | 0.988281 | 0.929688 | PASS |
| conditional_routing | 1.000000 | 0.554688 | FAIL |
| iterative_state | 0.996094 | 0.992188 | PASS |
| variable_composition | 1.000000 | 0.789062 | FAIL |
| partial_observation | 1.000000 | 0.789062 | FAIL |

Families passing: **2/5**

## 10. AI4 4x results

| family | development median | validation median | pass |
|---|---:|---:|---|
| delayed_recall | 1.000000 | 0.984375 | PASS |
| conditional_routing | 1.000000 | 0.937500 | PASS |
| iterative_state | 0.996094 | 1.000000 | PASS |
| variable_composition | 0.984375 | 0.875000 | PASS |
| partial_observation | 0.974609 | 0.812500 | FAIL |

Families passing: **4/5**

## 11. Family-specific scaling trajectories

- delayed_recall: 1x 0.867188 (P) -> 2x 0.929688 (P) -> 4x 0.984375 (P)
- conditional_routing: 1x 0.406250 (F) -> 2x 0.554688 (F) -> 4x 0.937500 (P)
- iterative_state: 1x 0.984375 (P) -> 2x 0.992188 (P) -> 4x 1.000000 (P)
- variable_composition: 1x 0.703125 (F) -> 2x 0.789062 (F) -> 4x 0.875000 (P)
- partial_observation: 1x 0.734375 (F) -> 2x 0.789062 (F) -> 4x 0.812500 (F)

## 12. Replicate stability

Five replicate validation scores, min/max/std, and >=0.85 counts are stored in `diagnostics/replicate_stability.json`; the historical median gate remains authoritative.

## 13. Projection specialization vs data

Final median pairwise projection distances: 1x=1.105330, 2x=1.174932, 4x=1.180436.

## 14. Message/controller diagnostics

Median message-ablation success drops: 1x=0.117188, 2x=0.195312, 4x=0.453125. Controller statistics are recorded for newly executed 1x/2x. The accepted 4x artifact did not store controller-gate summaries, and V837ai does not rerun 4x solely for a descriptive metric.

## 15. Minimum tested sufficient multiplier

`4`

## 16. Historical GRU comparison

AF1D pass-count curve: 2/5 -> 2/5 -> 4/5. GRU: 2/5 -> 3/5 -> 5/5.

## 17. Parameter-count caveat

AF1D has 1,643 active parameters versus 875 for the historical GRU (1.878x). V837ai therefore establishes unique-data threshold behavior, not parameter-matched superiority.

## 18. Processed-example comparison

The historical protocol uses the entire development set at each of 192 optimizer steps: 24,576 examples/fit at 1x, 49,152 at 2x, and 98,304 at 4x. Increasing unique data therefore also increases example-presentations.

## 19. Modeled recurrent-MAC comparison

Modeled volumes use actual active training sequence timesteps times matrix MACs/timestep. They are model-derived compute proxies, not measured energy; CPU/wall time is separately measured for the new runs.

## 20. Sample-efficiency diagnosis

`AF1D_REQUIRES_4X_UNIQUE_DATA`

## 21. Structural-search authorization

Structural-search recovery authorized: True. Recommended data multiplier: 4x. Next program: `V837aj_STRUCTURAL_SEARCH_RECOVERY`; not implemented here.

## 22. Fresh-audit/mining/V838 status

Fresh audit consumed: 0. Primitive mining: BLOCKED. V837ag/V837ah: NOT RUN. V838: NOT STARTED.

## 23. Strongest scientific claim

AF1D solves the neutral representation problem but does not reduce the minimum tested unique-development multiplier relative to the historical GRU calibration. Both first satisfy the representation gate at 4x under their respective tested architectures.
