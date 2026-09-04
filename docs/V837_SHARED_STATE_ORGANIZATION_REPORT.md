# V837 Shared-State Organization Report

## 1. Current V837 frontier

V837q begins from the closed V837p frontier. The calibrated 4× unique-data regime is known to be learnable because the small GRU reference reaches 5/5, while the historical neutral high-capacity graph reaches 2/5. V837m showed that stable linear state transport was insufficient. V837o localized dynamic state modulation as necessary inside the successful dense GRU reference, but V837p showed that transferring one generic scalar dynamic modulator into the decomposed neutral substrate only improved it to 3/5 and did not restore representation adequacy.

The remaining hypothesis tested here is whether recurrent-state fragmentation itself is the main scaffold bottleneck.

## 2. Why state fragmentation became the next hypothesis

The historical neutral substrate gives each of ten cells a private four-dimensional recurrent state. Cross-cell influence therefore has to pass through local transforms, explicit graph messages, aggregation, and another local update. A dense recurrent reference instead exposes one hidden vector to its recurrent transform. V837q therefore changes only recurrent-state ownership while keeping the neutral cell transform, graph, message interface, readout width, task suite, data regime, optimizer regime, and total recurrent dimensionality fixed.

## 3. Historical local-state substrate

The Q0 baseline is the exact historical high-capacity neutral graph used by the calibrated V837l 4× condition:

- 10 cells
- 4 private recurrent dimensions per cell
- 40 total recurrent scalar dimensions
- 55 internal graph edges
- historical direct tanh update
- broadcast raw input
- 856 trainable parameters
- 40-dimensional final readout input

Q0 reproduced the preserved V837l validation medians with **zero absolute drift on all five families** and remained **2/5**. This cleared the baseline-compatibility gate before the shared-state conditions were interpreted.

## 4. State-layout abstraction

V837q introduces a dedicated `StateLayoutSpec` and `SharedStateNeutralGraphModel` inside the V837q experiment rather than changing historical `NeutralGraphModel` behavior. The four primary layouts are:

| Condition | State groups | Group dimensions | Total recurrent dimensions |
| --- | ---: | --- | ---: |
| Q0 local | 10 | 10 × 4 | 40 |
| Q1 group-shared | 5 | 5 × 8 | 40 |
| Q2 group-shared | 2 | 2 × 20 | 40 |
| Q3 fully shared | 1 | 1 × 40 | 40 |

Cell-to-group assignments are deterministic, contiguous, fixed before training, and task-independent. Shared layouts use fixed deterministic projection buffers for cell-local four-dimensional views and transpose writeback. The projections are not trainable parameters.

## 5. Parameter/state-capacity matching

All four Transmutor-side primary conditions contain exactly:

- **40 recurrent state dimensions**
- **856 trainable parameters**
- **40 readout inputs**
- the same graph cells, edges, raw-input semantics, optimizer, task generators, paired seeds, and 192-step training budget

The state-sharing manipulation therefore does not increase trainable capacity.

Reference controls are intentionally diagnostic rather than parameter-matched Transmutor candidates:

- QR1 dense vanilla RNN, 40 hidden dimensions: **1,921 parameters**
- QR2 successful GRU reference, 13 hidden dimensions: **875 parameters**

## 6. Q0 local baseline

Validation medians:

| Family | Validation median | Capacity |
| --- | ---: | --- |
| Conditional routing | 0.5000 | FAIL |
| Delayed recall | 0.9375 | PASS |
| Iterative state | 0.9844 | PASS |
| Partial observation | 0.7734 | FAIL |
| Variable composition | 0.7812 | FAIL |

**Families passing: 2/5.**

This exactly reproduces the preserved calibrated neutral baseline.

## 7. Q1 group-shared 5×8

Validation medians:

| Family | Validation median | Capacity |
| --- | ---: | --- |
| Conditional routing | 0.4297 | FAIL |
| Delayed recall | 0.9141 | PASS |
| Iterative state | 0.9922 | PASS |
| Partial observation | 0.7891 | FAIL |
| Variable composition | 0.7891 | FAIL |

**Families passing: 2/5.**

Reducing ten private states to five shared groups did not increase the number of competent families.

## 8. Q2 group-shared 2×20

Validation medians:

| Family | Validation median | Capacity |
| --- | ---: | --- |
| Conditional routing | 0.3906 | FAIL |
| Delayed recall | 0.9609 | PASS |
| Iterative state | 0.9922 | PASS |
| Partial observation | 0.8359 | FAIL |
| Variable composition | 0.7891 | FAIL |

**Families passing: 2/5.**

The stronger sharing regime improves recall and partial-observation scores relative to Q0, but routing degrades and the representation-adequacy gate remains unmet.

## 9. Q3 fully shared 1×40

Validation medians:

| Family | Validation median | Capacity |
| --- | ---: | --- |
| Conditional routing | 0.3750 | FAIL |
| Delayed recall | 0.9609 | PASS |
| Iterative state | 0.9766 | PASS |
| Partial observation | 0.8438 | FAIL |
| Variable composition | 0.8359 | FAIL |

**Families passing: 2/5.**

The fully shared 40-dimensional recurrent state does not restore neutral-substrate competence. Mean family validation median changes only from about **0.7953** at Q0 to **0.7984** at Q3, far below the predeclared meaningful-partial-benefit criterion and with no increase in families passing.

## 10. Dense RNN / GRU controls

### QR1 dense vanilla RNN, 40D

Validation medians:

| Family | Validation median | Capacity |
| --- | ---: | --- |
| Conditional routing | 0.7969 | FAIL |
| Delayed recall | 0.9688 | PASS |
| Iterative state | 1.0000 | PASS |
| Partial observation | 0.7656 | FAIL |
| Variable composition | 0.8203 | FAIL |

**Families passing: 2/5.**

A globally dense 40-dimensional tanh recurrent state without adaptive gating also fails the 4/5 gate. This is consistent with the earlier V837o finding that successful dense-reference learning still depends on dynamic state modulation.

### QR2 GRU positive reference

Validation medians:

| Family | Validation median | Capacity |
| --- | ---: | --- |
| Conditional routing | 0.9375 | PASS |
| Delayed recall | 0.9844 | PASS |
| Iterative state | 1.0000 | PASS |
| Partial observation | 0.8906 | PASS |
| Variable composition | 0.8828 | PASS |

**Families passing: 5/5.**

The calibrated benchmark remains learnable in this exact data regime.

## 11. Per-family results and state-sharing curve

The number of competent families is flat across the primary sharing axis:

```text
state groups:       10 → 5 → 2 → 1
families passing:    2 → 2 → 2 → 2
mean family median: .7953 → .7828 → .7938 → .7984
```

Paired comparisons against Q0 show mixed family-specific shifts rather than a monotonic capability gain. Q3 improves partial observation by a paired mean of about +0.050 and composition by about +0.050, but routing declines by about -0.094. Q2 likewise improves recall while substantially reducing routing. The trade-offs do not yield a net competence recovery.

## 12. State-rank diagnostics

Sharing changes the geometry of the learned state but does not improve the competence gate. Median effective rank, summarized across family medians, decreases as state becomes more shared:

| Condition | Approx. median effective rank |
| --- | ---: |
| Q0 local | 9.22 |
| Q1 5 groups | 6.58 |
| Q2 2 groups | 4.53 |
| Q3 shared | 3.40 |

The fully shared state does not exploit a broader effective latent subspace; in these trained models it becomes more correlated and lower-rank. This is descriptive evidence, not a pass criterion, but it argues against the idea that merely pooling ownership unlocks a richer shared workspace.

## 13. Gradient-alignment diagnostics

Gradient alignment is defined for pathways that write to a common state group. It therefore has no direct Q0 local analogue. Across the shared layouts, median family-level pathway-gradient alignment decreases as sharing becomes stronger:

- Q1: approximately 0.231
- Q2: approximately 0.081
- Q3: approximately 0.024

The observed optimization geometry therefore does not support a simple story in which stronger sharing improves competence by producing increasingly coordinated pathway credit assignment. The strongest-sharing condition actually shows the weakest alignment under this diagnostic.

## 14. Message-dependence diagnostics

Median family-level validation success drop under explicit message ablation is approximately:

- Q0: 0.188
- Q1: 0.211
- Q2: 0.398
- Q3: 0.414

Rather than making graph messages redundant, stronger state sharing is associated with greater message dependence on this diagnostic. Because Q3 itself fails representation adequacy, the conditional Q3 no-message experiment was **not run**; running it would violate the predeclared branch guard.

## 15. Projection sensitivity

Projection-sensitivity reruns were conditional on Q3 reaching the representation-adequacy gate. Q3 remained 2/5, so the five-projection-seed sensitivity study was **not run**. This avoids spending additional runs on a mechanism that failed its primary gate.

The primary fixed projections were deterministic, task-independent, norm-tested, non-trainable, and serialized through the state-layout specification.

## 16. Cross-cell influence

The intervention diagnostic confirms that shared layouts do create immediate cross-path influence through common state ownership. Approximate median family-level prediction deltas after zeroing a pathway contribution are:

- Q0: 0.142
- Q1: 0.109
- Q2: 0.154
- Q3: 0.133

Thus the manipulation is functionally active: shared ownership changes state coupling and intervention pathways. Its failure to improve the 4/5 gate is therefore not explained by a completely inert implementation.

## 17. Strongest supported diagnosis

**STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED**

Progressively replacing ten private recurrent state objects with five, two, and finally one shared state object—while preserving 40 recurrent dimensions and the same 856 trainable parameters—does not increase neutral-substrate competence beyond 2/5.

The experiment therefore closes the specific hypothesis that state ownership/fragmentation by itself is the main remaining bottleneck.

This does **not** show that modularity or shared workspaces never matter. It shows only that, under the calibrated V837 synthetic suite and this fixed neutral computation, moving the same recurrent capacity into shared state objects is insufficient.

## 18. Representation adequacy

**FAIL.**

No Q0–Q3 condition reaches the required >=4/5 fixed-topology representation-adequacy gate. The best primary layouts remain 2/5.

## 19. Sample-efficiency status

**Blocked.**

The 1×/2×/4× sample-efficiency retest may open only after a recovered neutral representation reaches >=4/5 at 4× unique data. V837q does not meet that prerequisite.

## 20. Structural-search and primitive-mining status

- Structural search: **BLOCKED**
- Primitive mining: **BLOCKED**
- Fresh-audit episodes consumed: **0**
- Primitives promoted: **0**
- V838: **NOT STARTED**

No V837r/V837s downstream experiment was executed automatically.

## 21. Next single variable

The next scientifically isolated difference is **global cross-dimensional recurrent coupling**, not state ownership.

A dense RNN or GRU uses a recurrent matrix that can directly transform all hidden dimensions into all hidden dimensions. The neutral graph instead retains many small per-cell recurrent transforms plus message bottlenecks. V837q shows that merely making the state object shared does not remove this parameterization/coupling difference.

A future experiment should therefore keep the 40-dimensional state organization fixed while changing only whether the recurrent transform is local/block-structured or globally cross-dimensional. It should not add a GRU gate, dynamic modulation, attention, structural search, motif mining, or fresh-audit data at the same time.
