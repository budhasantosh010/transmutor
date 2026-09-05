# V837 Global Recurrent Coupling Report

## 1. Why global coupling became the next hypothesis
V837q showed that moving the same 40 recurrent dimensions from ten private states to one shared state did not improve the neutral substrate: every state-ownership condition remained 2/5, while the GRU reference remained 5/5. V837r therefore isolated a different property: whether the local-cell substrate lacks direct cross-dimensional recurrent influence even when state capacity, graph, messages, optimizer, data, activation, update law and readout are fixed.

## 2. Why shared state was insufficient
Shared ownership changed where state lived but did not create a learned recurrent transform from every latent dimension to every other latent dimension. V837r kept the historical 10 x 4 local state and added only direct recurrent cross-cell coupling.

## 3. Dense vanilla RNN warning
The V837q 40D dense vanilla RNN control was already 2/5. Therefore V837r used a strict stop rule and treated density as a mechanism test inside the neutral substrate, not as a license to scale model size.

## 4. Local recurrent parameterization
Historical recurrence has ten independent 4 x 4 local transforms, 856 trainable parameters, 40 total recurrent state dimensions, 55 fixed graph edges, 4D messages and a 40-wide readout. R0 reproduces this baseline exactly.

## 5. Global coupling formulation
For the snapshotted previous state S_t = concat(s_1...s_10), the added branch computes a cross-cell recurrent term and partitions it back into ten 4D contributions before the historical tanh candidate update. Low-rank variants use U V^T factors; the fixed cross-block mask then zeros every 4 x 4 self block. The configured factor rank is therefore a parameterization rank; after masking, the realized matrix can have higher algebraic rank.

## 6. Cross-block mask
All primary global conditions zero the ten local diagonal 4 x 4 blocks. This prevents the new branch from simply duplicating historical local recurrence. R5 uses a 40 x 40 trainable matrix with only cross-block entries active.

## 7. Low-rank variants
The predefined ranks were 1, 2, 4 and 8. Rank1/rank8 were run only after the screen found a 3/5 rank4 result. No ranks beyond the specification were explored.

## 8. Parameter-matched controls
Each global condition has an exact trainable-parameter matched local-only recurrent branch. None of these controls can read another cell state. Every matched local control remained 2/5, while rank4 and rank8 global coupling reached 3/5.

## 9. Baseline reproduction
R0 reproduced the V837q Q0 medians with zero drift on all five families: routing 0.5000, recall 0.9375, iterative 0.9844, partial observation 0.7734, composition 0.7812. Families passing: 2/5.

## 10. Per-family results

| Condition | Routing | Recall | Iterative | Partial | Composition | Families | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0_local | 0.5000 | 0.9375 | 0.9844 | 0.7734 | 0.7812 | 2/5 | 856 |
| R1_rank1 | 0.5547 | 0.9766 | 0.9922 | 0.8047 | 0.8203 | 2/5 | 936 |
| R2_rank2 | 0.5781 | 0.9688 | 1.0000 | 0.7969 | 0.8359 | 2/5 | 1016 |
| R3_rank4 | 0.5547 | 0.9766 | 0.9844 | 0.7969 | 0.8828 | 3/5 | 1176 |
| R4_rank8 | 0.4531 | 0.9922 | 0.9844 | 0.7734 | 0.8750 | 3/5 | 1496 |
| R5_dense_cross_block | 0.4219 | 0.9766 | 0.9844 | 0.7500 | 0.7656 | 2/5 | 2456 |

Matched local controls:

| Control | Routing | Recall | Iterative | Partial | Composition | Families | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1_rank1_local | 0.5156 | 0.9453 | 1.0000 | 0.7734 | 0.8203 | 2/5 | 936 |
| C2_rank2_local | 0.5078 | 0.9609 | 0.9922 | 0.7734 | 0.8359 | 2/5 | 1016 |
| C3_rank4_local | 0.5234 | 0.9297 | 0.9844 | 0.8047 | 0.8047 | 2/5 | 1176 |
| C4_rank8_local | 0.4531 | 0.9688 | 0.9922 | 0.7266 | 0.8281 | 2/5 | 1496 |
| C5_dense_budget_local | 0.4219 | 0.9688 | 1.0000 | 0.7891 | 0.8438 | 2/5 | 2456 |

## 11. Coupling-rank curve
Families passing across primary conditions: R0 2/5, rank1 2/5, rank2 2/5, rank4 3/5, rank8 3/5, dense cross-block 2/5. The benefit is non-monotonic: moderate factor ranks recover variable composition, while unrestricted cross-block density does not.

## 12. Coupling utilization
For R3 rank4, the mean across family medians of global/local recurrent norm ratio is about 1.56 and global/message ratio about 2.12. The branch is therefore actively used rather than trained to zero. Rank8 is stronger still (about 1.93 global/local), yet remains 3/5.

## 13. Effective learned rank
Because the block-diagonal mask is applied after the low-rank factor product, it can increase algebraic matrix rank. R3 has configured factor rank 4 but median realized masked-matrix rank 40 across families. Spectral norm averages about 3.98 across family medians. These diagnostics should not be interpreted as proof that a strict rank-4 matrix is sufficient; they show a rank-4 factorized parameterization plus deterministic cross-block masking.

## 14. Gradient diagnostics
R0 mean cell-gradient cosine across family medians is about 0.176. R3 falls to about 0.070 and rank8 to about 0.028. Gradient-norm variance also decreases substantially. The partial capability gain is therefore not explained by higher pathway gradient alignment; global coupling changes credit distribution but does not recover broad competence.

## 15. Message dependence
R0 mean message-success-drop across family medians is about 0.267; R3 is about 0.284, rank8 about 0.219 and dense about 0.203. Direct recurrent coupling does not cleanly replace explicit messages. The no-message follow-up remained blocked because no global condition reached >=4/5.

## 16. Cross-cell causal influence
R0 has zero direct cross-cell recurrent influence by construction. For R3, zeroing one source cell only in the global recurrent branch changes other-cell next-state values by about 0.120 and final predictions by about 0.137 on average across family medians. The learned branch is functionally cross-cell, not merely nominal.

## 17. Compute and energy estimates

| Condition | Recurrent MACs/timestep | Approx recurrent FLOPs/timestep | Families |
|---|---:|---:|---:|
| R0_local | 160 | 320 | 2/5 |
| R1_rank1 | 320 | 640 | 2/5 |
| R2_rank2 | 480 | 960 | 2/5 |
| R3_rank4 | 800 | 1600 | 3/5 |
| R4_rank8 | 1440 | 2880 | 3/5 |
| R5_dense_cross_block | 1600 | 3200 | 2/5 |

The historical local recurrence is 160 MACs/timestep. R3 rank4 is approximately 800 implemented recurrent MACs/timestep; dense cross-block is 1600. Since neither restores adequacy, no dense mechanism is retained as an architectural winner.

## 18. Strongest supported V837r diagnosis
**GLOBAL_COUPLING_PARTIAL_BENEFIT.** Rank4 and rank8 reach 3/5 while exact matched local-capacity controls stay 2/5. The benefit is specific enough to justify the predefined interaction test, but global coupling alone does not reach representation adequacy.

## 19. Authorized V837s interaction
V837r machine-readable decision state authorized exactly one 2 x 2 interaction with the frozen V837p dynamic scalar mechanism. No mechanism redesign was allowed. The conditions were S0 local/no modulation, S1 local/dynamic scalar, S2 rank4/no modulation, S3 rank4/dynamic scalar, plus S3C rank4 with the parameter-matched dynamic additive control.

| V837s condition | Routing | Recall | Iterative | Partial | Composition | Families | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0_local_no_modulation | 0.5000 | 0.9375 | 0.9844 | 0.7734 | 0.7812 | 2/5 | 856 |
| S1_local_dynamic_scalar | 0.8203 | 0.9453 | 0.9922 | 0.8516 | 0.7734 | 3/5 | 1006 |
| S2_rank4_no_modulation | 0.5547 | 0.9766 | 0.9844 | 0.7969 | 0.8828 | 3/5 | 1176 |
| S3C_rank4_matched_dynamic_additive | 0.8125 | 0.9766 | 1.0000 | 0.8203 | 0.9609 | 3/5 | 1326 |
| S3_rank4_dynamic_scalar | 0.7031 | 0.9844 | 1.0000 | 0.7969 | 0.8828 | 3/5 | 1326 |

S3 remains 3/5. It improves routing relative to S2 but loses the partial-observation pass seen in S1; the interaction does not cross the 4/5 gate. The matched additive S3C also reaches 3/5 and outperforms the true scalar modulator on routing and composition in this batch. Multiplicative specificity is therefore not established.

## 20. Representation adequacy
**FAIL.** No V837r or V837s condition reaches the calibrated >=4/5 fixed-topology representation-adequacy criterion.

## 21. Sample-efficiency status
Blocked. The 1x/2x retest remains prohibited until a fixed architecture first reaches >=4/5 at 4x unique data.

## 22. Structural-search status
Blocked. No evolutionary/structural search was run in V837r or V837s.

## 23. Primitive-mining status
Blocked. Primitive mining remains false; fresh-audit episodes consumed remain 0; primitives promoted remain 0; V838 was not started.

## 24. Next single variable
The next isolated variable is **dynamic vector-valued state modulation**. V837p/V837s tested only one scalar adaptive coefficient per cell, while the successful GRU reference uses dimension-wise adaptive control. Global density, state sharing and scalar modulation have now each been localized as insufficient alone, and the scalar x rank4 interaction is also insufficient. Any next test must keep the established 4x regime and science locks and must not reopen structural search or primitive mining.

## Resource accounting
V837r: 275 fits, 52800 optimizer steps, 27033600 examples, 1223475 environment interactions, 61475 forward calls, 14330.484 worker CPU seconds, 15364.706 summed worker wall seconds, GPU 0.
V837s: 125 fits, 24000 optimizer steps, 12288000 examples, 556125 environment interactions, 26875 forward calls, 8242.641 worker CPU seconds, 10239.576 summed worker wall seconds, GPU 0.
Combined V837r+s: 400 fits, 76800 optimizer steps, 39321600 examples, 1779600 environment interactions, 88350 forward calls, 22573.125 worker CPU seconds, 25604.282 summed worker wall seconds, GPU 0.
