# V837 Candidate Recurrent Geometry Report

## 1. Current V837 causal frontier

The program starts after V837ab showed that reference-side input factorization changes optimization without adding function-class capacity, and V837ac showed that the minimal authorized controller-input factorization transfer leaves Y3 at 3/5. The remaining question tested here is candidate recurrent geometry.

## 2. Why V837q did not answer this question

V837q changed state ownership (10x4 -> 5x8 -> 2x20 -> 1x40) while each path still consumed a fixed 4D projected recurrent view and produced a 4D candidate. V837ad instead keeps one recurrent tensor and directly masks the learned candidate hidden matrix `W_hn`, so the manipulated object is the affine recurrent mixing before tanh.

## 3. Why tanh call count is not the variable

Elementwise tanh commutes with concatenation. The causal variable is not one tanh versus ten tanhs; it is which prior-state coordinates each learned candidate coordinate may linearly integrate before tanh.

## 4. H13 anchor

AD0 reproduces exact historical T2 with zero family-median validation drift:

| family | median validation |
| --- | ---: |
| conditional routing | 0.875000 |
| delayed recall | 0.9765625 |
| iterative state | 1.000000 |
| partial observation | 0.875000 |
| variable composition | 0.8203125 |

AD0 = 4/5, nominal params 875, active params 602, candidate recurrent weights 169, active reference MACs 530/timestep.

## 5. H40 dense width gate

AD1 changes only hidden width from 13 to 40 while preserving the T2 projection, no-reset recurrence, scalarized dynamic update, optimizer, data, and initialization rule.

| family | median validation |
| --- | ---: |
| conditional routing | 0.9921875 |
| delayed recall | 1.000000 |
| iterative state | 0.9921875 |
| partial observation | 0.859375 |
| variable composition | 0.8203125 |

AD1 = 4/5, nominal params 5843, active params 3923, candidate recurrent weights 1600, active MACs 3716/timestep.

## 6. Width-gate decision

`WIDTH_GATE_PASS`. Dense H40 preserves >=4/5 competence under the frozen 4x/192-step regime, so candidate geometry can be interpreted without a hidden-width confound.

## 7. H40 paired initialization

For every family x replicate, AD1/AD2/AD3/AD4/AD4S-S0 and all predeclared robustness masks receive bit-identical raw input projection, `W_ih`, raw `W_hh`, biases, and readout tensors. Only the fixed nontrainable candidate mask differs. Mask rescaling is disabled.

## 8. 2x20 geometry

AD2 keeps 800 active candidate recurrent weights and reaches 4/5:

routing 1.000000; recall 0.9921875; iterative 0.9921875; partial 0.875000; composition 0.8203125.

Active params 3123; active MACs 2916/timestep.

## 9. 5x8 geometry

AD3 keeps 320 active candidate recurrent weights and reaches 4/5:

routing 0.968750; recall 0.9921875; iterative 0.9921875; partial 0.890625; composition 0.843750.

Active params 2643; active MACs 2436/timestep.

## 10. 10x4 geometry

AD4 keeps exactly 160 block-local candidate recurrent weights and still reaches 4/5:

routing 0.875000; recall 0.9765625; iterative 0.984375; partial 0.890625; composition 0.828125.

Active params 2483; active MACs 2276/timestep. Cross-block candidate recurrent energy is exactly zero by construction.

## 11. Degree-4 global sparse geometry

AD4S-S0 also uses exactly 160 active candidate recurrent weights and exactly the same active compute as AD4, but distributes four recurrent inputs per output globally with no same-4D-block edge. It reaches 4/5:

routing 0.921875; recall 0.984375; iterative 0.984375; partial 0.890625; composition 0.8359375.

## 12. Sparse topology robustness

Not run. The frozen trigger requires AD4 <4/5 and AD4S-S0 >=4/5. AD4 itself is already 4/5, so S1-S4 are scientifically unnecessary and physically blocked. No topology-robustness expenditure was made.

## 13. State/candidate effective rank

Median final effective ranks (state / candidate):

- AD0: 11 / 11
- AD1: 15 / 16
- AD2: 13 / 14
- AD3: 11 / 13
- AD4: 12 / 14
- AD4S-S0: 13 / 14

The 10x4 local candidate geometry does not collapse the 40D recurrent system to a 4D effective state. Median participation ratios remain descriptive rather than gating quantities.

## 14. Candidate Jacobian diagnostics

Median sampled candidate Jacobian effective rank is 13 for AD0 and 40 for every H40 condition, including AD4 and AD4S-S0. Median spectral norms are approximately 1.826 dense H40, 1.884 for 2x20, 1.787 for 5x8, 1.413 for 10x4, and 1.198 for degree-4 sparse. These diagnostics are descriptive only.

## 15. Update-gate compensation

Median carry fractions are:

- AD1 dense: 0.4143
- AD2 2x20: 0.4084
- AD3 5x8: 0.4419
- AD4 10x4: 0.4117
- AD4S-S0: 0.4297

No condition approaches gate saturation; near-zero and near-one fractions are zero. The dense update recurrent slice remains fully active in every H40 condition.

## 16. Compute efficiency

| condition | active candidate weights | active params | active MACs/timestep | families |
| --- | ---: | ---: | ---: | ---: |
| AD0 H13 dense | 169 | 602 | 530 | 4/5 |
| AD1 H40 dense | 1600 | 3923 | 3716 | 4/5 |
| AD2 2x20 | 800 | 3123 | 2916 | 4/5 |
| AD3 5x8 | 320 | 2643 | 2436 | 4/5 |
| AD4 10x4 | 160 | 2483 | 2276 | 4/5 |
| AD4S-S0 degree4 global | 160 | 2483 | 2276 | 4/5 |

Masked candidate gradients are exactly zero before optimizer weight-decay effects in every masked condition. Total V837ad work: 150 fits, 28,800 optimizer steps, 14,745,600 processed training examples, 3,200 unique task episodes, 31,050 forward calls, 28,800 backward calls, 1,383.65625 worker CPU seconds, 2,148.0203952004667 worker-summed wall seconds, GPU 0.

## 17. V837ad diagnosis

`TEN_BY_FOUR_CANDIDATE_GEOMETRY_SUFFICIENT_IN_REFERENCE`.

Safe claim: narrow independent 4D candidate recurrent blocks are sufficient for the calibrated successful reference when total recurrent width, global scalarized update, shared input pathway, and reference organization are preserved. Candidate recurrent geometry therefore cannot by itself explain the neutral substrate's remaining 3/5 result.

## 18. V837ae authorization

`authorized_v837ae_mode = null`. V837ae is not created or run because the required robust sparse-over-local advantage does not exist: AD4 local 10x4 already passes 4/5. Neutral Y3 remains untouched. Sample-efficiency retest, structural search, and primitive mining remain blocked; fresh-audit consumption is zero; large persistent storage is not tested; V838 is not started.

The next single variable is `GRAPH_MESSAGE_OUTPUT_INTERFACE_ORGANIZATION`.
