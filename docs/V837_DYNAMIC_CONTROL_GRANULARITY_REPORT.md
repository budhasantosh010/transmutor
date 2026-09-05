# V837 Dynamic Control Granularity Report

## 1. Current causal frontier

V837l through V837s established that the calibrated GRU reference reaches 5/5 at the frozen 4× data regime while the neutral substrate remains 2–3/5 under transport, scalar dynamic modulation, state sharing, global recurrent coupling, and the rank-4-coupling × scalar-modulation interaction. V837t therefore returned to the successful reference before transferring any larger controller.

## 2. Why vector granularity became the next hypothesis

The successful GRU emits 13-dimensional update/reset gates, whereas the transferred V837p mechanism emits one scalar per neutral cell and broadcasts it over four state dimensions. The unresolved causal question was whether the GRU actually needs dimension-specific gate values.

## 3. Missing reference-side scalarization experiment

V837t removed only gate output heterogeneity. The original vector gate networks, parameter tensors, input/state dependence, optimizer, seeds, and training budget were retained. The experiment therefore isolates dynamic-control granularity rather than controller capacity.

## 4. Exact GRU semantics

The experiment-local model preserves the frozen explicit PyTorch-compatible GRU equations, including reset-after-hidden-transform semantics: `n = tanh(i_n + r * h_n)` and `h_next = (1-z) * n + z * h`.

## 5. Scalarization definition

For an already-sigmoided gate vector `g`, V837t computes `mean(g, hidden_dim, keepdim=True)` and broadcasts that mean back to the full hidden width. This preserves the gate's post-sigmoid mean while making inter-dimensional output variance exactly zero.

## 6. Positive controls

T0 full-vector GRU, T1 vector-update/no-reset, and T3 no-update/vector-reset all reproduced at 5/5. The positive-control guard therefore passed and scalarized conditions are interpretable.

## 7. Dynamic update scalarization

T2 scalarized-update/no-reset reached **4/5**. Validation medians were routing 0.8750, recall 0.9766, iterative 1.0000, partial observation 0.8750, and composition 0.8203. Therefore dimension-specific update-gate values are not required for representation adequacy in the successful reference.

## 8. Dynamic reset scalarization

T4 no-update/scalarized-reset reached **3/5**. Validation medians were routing 0.8438, recall 0.9922, iterative 0.9766, partial observation 0.8828, and composition 0.8125. Scalarized reset alone was insufficient.

## 9. Dual scalarized pathways

T5 dual-scalarized reached **3/5** with routing 0.8359, recall 0.9844, iterative 0.9922, partial observation 0.8984, and composition 0.7734. Two independent scalar dynamic pathways therefore did not outperform the sufficient single scalarized-update pathway.

## 10. Per-family effects

| Condition | Routing | Recall | Iterative | Partial | Composition | Passing |
|---|---:|---:|---:|---:|---:|---:|
| T0 full vector | 0.9375 | 0.9844 | 1.0000 | 0.8906 | 0.8828 | 5/5 |
| T1 vector update/no reset | 0.9062 | 0.9922 | 1.0000 | 0.8906 | 0.8594 | 5/5 |
| T2 scalarized update/no reset | 0.8750 | 0.9766 | 1.0000 | 0.8750 | 0.8203 | 4/5 |
| T3 no update/vector reset | 0.8594 | 0.9844 | 0.9844 | 0.8828 | 0.9453 | 5/5 |
| T4 no update/scalarized reset | 0.8438 | 0.9922 | 0.9766 | 0.8828 | 0.8125 | 3/5 |
| T5 dual scalarized | 0.8359 | 0.9844 | 0.9922 | 0.8984 | 0.7734 | 3/5 |

## 11. Gate temporal dynamics

Median gate temporal variance remained nonzero for scalarized dynamic pathways: T2 update 0.0140, T4 reset 0.0139, and T5 update/reset 0.0151/0.00565. Scalarization removed dimension-specific values, not temporal adaptivity.

## 12. Gate interdimension heterogeneity

Vector anchors learned substantial gate heterogeneity: median inter-dimensional variance was 0.0410/0.0347 for T0 update/reset, 0.0392 for T1 update, and 0.0450 for T3 reset. Scalarized gate outputs had exactly zero inter-dimensional variance by construction. Since T2 still reached 4/5, observed vector heterogeneity is not itself proof of necessity.

## 13. Counterfactual flattening

Inference-only flattening of vector-trained gates caused nontrivial drops without retraining. T1 update flattening had median validation delta 0.1250; T3 reset flattening 0.1016; T0 update/reset/both had medians 0.0859/0.0156/0.1172. These counterfactuals show trained vector models use heterogeneity, but retrained T2 demonstrates the reference can reorganize and remain adequate with a scalar dynamic update. Retrained causal intervention therefore takes precedence over the no-retraining counterfactual for necessity.

## 14. Reference-side diagnosis

**DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED.** T2 reaches the frozen >=4/5 representation criterion, so vector-valued neutral transfer is not licensed by the reference evidence.

## 15. Authorized neutral follow-up

The V837t decision file machine-authorized exactly one V837u mode: **DYNAMIC_SCALAR_CARRY**. No vector controller, reset-order transfer, global coupling, state sharing, additional data, or search was allowed.

## 16. Neutral mechanism equations

The selected V837u branch retains the frozen neutral candidate computation and the frozen V837p scalar controller, but moves that scalar to the state update: `s_next = g*s_prev + (1-g)*candidate`. This directly tests adaptive preserve-vs-replace carry rather than candidate-state access.

## 17. Matched controls

V837u reran historical U0 and frozen V837p scalar-candidate U1, tested U2 dynamic scalar carry, and included U2C with the same controller/parameter budget but dynamic candidate scaling rather than old-state carry. U2 and U2C retain the same controller parameter count.

## 18. Neutral results

| Condition | Routing | Recall | Iterative | Partial | Composition | Passing |
|---|---:|---:|---:|---:|---:|---:|
| U0 historical direct | 0.4922 | 0.9062 | 0.9844 | 0.8125 | 0.7891 | 2/5 |
| U1 V837p scalar candidate | 0.8125 | 0.9531 | 0.9922 | 0.8672 | 0.7812 | 3/5 |
| U2 dynamic scalar carry | 0.6406 | 0.9141 | 0.9922 | 0.8047 | 0.7969 | 2/5 |
| U2C same-controller scale control | 0.7188 | 0.9453 | 0.9844 | 0.8594 | 0.8125 | 3/5 |

The adaptive-carry transfer did not reproduce the reference-side scalarized-update success. Its same-controller scaling control performed better in family count.

## 19. Representation adequacy

**FAIL.** The selected U2 mechanism reaches 2/5, below the frozen >=4/5 gate.

## 20. Data-efficiency status

Sample-efficiency testing remains **BLOCKED** because no neutral V837u representation reached >=4/5 at 4×. Unique data remained fixed at 512 development + 128 validation seeds per family across five families = **3,200 unique family/seed episodes**, reused across all paired conditions/replicates.

## 21. Compute-efficiency implications

V837t used 150 fits, 28,800 optimizer steps, and 14,745,600 processed training examples. V837u used 100 fits, 19,200 optimizer steps, and 9,830,400 processed training examples. Combined: 250 fits, 48,000 optimizer steps, 24,576,000 processed examples, 7,223.0 CPU seconds and 0 GPU seconds. The scalar-carry controller uses the low-bandwidth scalar mechanism rather than a vector controller, but it did not earn retention because capability fell to 2/5.

## 22. Structural-search status

**BLOCKED.** Fixed-topology representation adequacy has not been restored.

## 23. Primitive-mining status

**BLOCKED.** Structural-search competence has not been reopened, and no primitives were promoted.

## 24. Fresh-audit status

Fresh-audit seeds 90000–90499 remain untouched. Fresh-audit episodes consumed: **0**. V838 remains **NOT STARTED**.

## 25. Strongest scientific claim

The successful GRU does **not** require dimension-specific dynamic update-gate values: a post-sigmoid mean-and-broadcast update gate still reaches 4/5. However, transferring the corresponding low-bandwidth idea as a scalar preserve-vs-replace carry operation into the neutral 10×4 graph is insufficient (2/5). Therefore the missing neutral property is not justified as vector gate granularity, and scalar carry by itself is not the transferable principle.

## 26. Next single variable

The next experiment should localize **which remaining GRU update-pathway semantic property makes scalarized update control effective in the dense reference but not in the neutral cell graph**. It should not default to vector gating. A subsequent specification should isolate one remaining difference at a time (for example controller conditioning/location relative to candidate/state transition), while keeping global coupling, shared state, extra data, structural search, primitive mining, and V838 locked until evidence explicitly authorizes them.

## Long-term scoreboards

Capability, unique-data accounting, and compute accounting are recorded. Architectural reuse, persistent-memory scalability, continual learning, semantic organization, sparse activation, catastrophic forgetting, and hardware-energy behavior remain **not tested at the current V837 stage** rather than being inferred from recurrent state.
