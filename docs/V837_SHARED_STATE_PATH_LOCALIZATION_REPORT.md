# V837 Shared-State-Path Localization Report

## 1. V837n recap

V837n established that the successful calibrated GRU does not depend on either named GRU mechanism individually. Under the 4x unique-data regime, the full GRU, no-update GRU, and no-reset GRU each reached 5/5 families, while removing both update and reset paths reduced competence to 3/5. That result localized redundancy/complementarity but left one confound: whether temporal/state-conditioned adaptation itself mattered, or whether multiple learned static recurrent pathways were enough.

## 2. Remaining confound

V837o asks: **is dynamic adaptive state control necessary, or are complementary static recurrent pathways sufficient?** The experiment stays entirely inside the already-successful dense reference. No Transmutor cell mechanism is transferred until the factorial result selects a mechanism family.

## 3. Factorial design

The explicit V837n GRU implementation is reused unchanged. Two factors are manipulated while retaining the same 4x unique-data regime, 192 AdamW steps, development seeds 10000-10511, validation seeds 20000-20127, five task families, and five paired model replicates per condition.

Update-path levels: dynamic, static vector, static scalar, off. Reset-path levels: dynamic, static vector, static scalar where used, off. Disabled mechanism tensors remain nominally registered so mechanism removal is not silently conflated with deleting the outer model scaffold.

## 4. Positive control

G0 full dynamic GRU reproduced the successful reference at **5/5**. Validation medians were:

| Family | G0 validation median |
| --- | ---: |
| Conditional routing | 0.9375 |
| Delayed recall | 0.9844 |
| Iterative state | 1.0000 |
| Partial observation | 0.8906 |
| Variable composition | 0.8828 |

The positive-control compatibility check passed, so the factorial conditions are interpretable.

## 5. Dynamic single-path conditions

Both dynamic single-path anchors retained full competence:

- G1 dynamic update / no reset: **5/5**.
- G2 no update / dynamic reset: **5/5**.

This reproduces the central V837n redundancy result inside one matched factorial batch.

## 6. Static single-path conditions

Replacing the surviving dynamic pathway with a learned time-independent vector did not preserve full competence:

- G3 static update vector / no reset: **3/5**.
- G4 no update / static reset vector: **3/5**.

The major remaining weakness was conditional routing, with validation medians 0.5938 and 0.6094 respectively. Static dimension-wise state flow therefore did not reproduce either successful dynamic single-path reference.

## 7. Static dual-path condition

The critical G5 condition used both a learned static update vector and a learned static reset vector. It reached only **3/5**, with conditional-routing validation median 0.5625 and variable-composition median 0.7891.

Therefore two complementary learned static recurrent pathways are **not sufficient** to explain the 5/5 GRU result.

## 8. Scalar/vector comparisons

The remaining static combinations also stayed at 3/5:

| Condition | Families passing |
| --- | ---: |
| G0 full dynamic | 5/5 |
| G1 dynamic update / no reset | 5/5 |
| G2 no update / dynamic reset | 5/5 |
| G3 static update vector / no reset | 3/5 |
| G4 no update / static reset vector | 3/5 |
| G5 static update vector / static reset vector | 3/5 |
| G6 static update scalar / static reset vector | 3/5 |
| G7 static update vector / static reset scalar | 3/5 |
| G8 static update scalar / static reset scalar | 3/5 |
| G9 no update / no reset | 3/5 |

Neither vector heterogeneity nor two static scalar pathways recovered the dynamic reference capability.

## 9. Factorial interaction effects

In the dynamic-versus-off 2x2 comparison, conditional routing showed the largest useful main effects: update main effect +0.1523, reset main effect +0.1055, with interaction -0.1484. Variable composition showed reset main effect +0.0664 and interaction -0.0859. The static-vector-versus-off factorial did not show a comparable recovery; its routing update and reset main effects were negative relative to the dense both-off control.

The scientific use of these effects is mechanistic localization rather than p-value optimization: dynamic state-conditioned pathways restore capability that static replacements do not.

## 10. Static parameter diagnostics

The static-vector conditions did learn non-zero inter-dimension variance, so the negative result cannot be reduced to vectors remaining mathematically identical to scalars. Typical learned inter-dimension variances were on the order of 1e-3. Counterfactual flattening and vector-distribution diagnostics are preserved in `v837o/diagnostics/` and `v837o/plots/` as descriptive evidence.

## 11. Strongest supported shared property

V837o closes with:

**DYNAMIC_STATE_MODULATION_REQUIRED**

The strongest supported statement is narrower than "GRU gates are required." Either dynamic update or dynamic reset can support 5/5 after retraining, but replacing the available dynamic path or paths with learned static vectors/scalars leaves the reference at 3/5. The shared property supported by the factorial is therefore **state/input-conditioned temporal modulation**, not a specific GRU gate name or location.

## 12. Chosen neutral follow-up

Because V837o clearly selected a dynamic mechanism family, exactly one minimal neutral transfer was authorized as V837p. It adds one generic scalar coefficient per cell and time step:

`g_t = sigmoid(u_s^T s_t + u_m^T m_t + u_x^T x_t + b)`

The new mechanism modulates recurrent-state access before the historical candidate projection. It does not add a GRU carry path, reset gate, attention, router, state-size change, message-size change, or topology change.

An exactly parameter-matched dynamic additive control uses the same coefficient-network parameter budget but adds its dynamic signal without multiplicatively modulating recurrent-state access.

## 13. Neutral follow-up result

V837p results under the same 4x regime:

| Neutral condition | Parameters | Families passing |
| --- | ---: | ---: |
| Historical direct | 856 | 2/5 |
| V837g scalar persistence | 866 | 2/5 |
| Dynamic scalar state modulation | 1006 | **3/5** |
| Parameter-matched dynamic additive | 1006 | **3/5** |

Dynamic scalar state-modulation validation medians were:

| Family | Validation median |
| --- | ---: |
| Conditional routing | 0.8125 |
| Delayed recall | 0.9531 |
| Iterative state | 0.9922 |
| Partial observation | 0.8672 |
| Variable composition | 0.7812 |

The learned scalar was genuinely dynamic: median mean coefficient about 0.516, median coefficient standard deviation about 0.166, and median temporal variance about 0.0179. Nevertheless the representation-adequacy gate remained unmet. The exactly parameter-matched dynamic additive control also reached 3/5, so the observed improvement cannot establish multiplicative recurrent-state modulation as sufficient or uniquely responsible.

V837p therefore closes as:

**SHARED_PROPERTY_TRANSFER_FAILURE**

## 14. Representation adequacy

Representation adequacy requires at least 4/5 families at 4x unique data, the regime where the small GRU reference is known to reach 5/5. V837p reached 3/5. Therefore representation adequacy is **not restored**.

## 15. Sample-efficiency status

The 1x/2x/4x sample-efficiency retest is not run because its prerequisite failed. No recovered neutral architecture reached the 4x representation-adequacy gate. The historical calibration remains the relevant comparison: GRU reaches 5/5 at 4x while the neutral lineage remains below the adequacy gate.

## 16. Structural-search status

Full structural search remains **blocked**. The fixed high-capacity topology has not demonstrated representation adequacy for the transferred neutral mechanism, so reopening the original V837 search would confound representation failure with search behavior.

## 17. Primitive-mining status

Primitive mining remains **blocked**. Fresh-audit seeds 90000-90499 remain unused and primitives promoted remain 0.

## 18. Remaining scaffold

The reference-side result is now sharper: temporal/state-conditioned modulation matters, but a single generic scalar modulation path transplanted into the decomposed neutral substrate is insufficient. The next isolated question should therefore move away from copying named GRU gates and test a deeper organizational difference: **dense shared hidden-state organization versus decomposed cell-local recurrent state**. That variable can be studied without reopening structural search, motif mining, or fresh audit.

## Resource accounting

V837o used 250 model fits, 48,000 optimizer steps, 24,576,000 processed examples, and 1,112,250 environment interactions. V837p used 100 model fits, 19,200 optimizer steps, 9,830,400 processed examples, and 444,900 environment interactions. Combined: 350 fits, 67,200 optimizer steps, 34,406,400 processed examples, 1,557,150 environment interactions, 75,675 forward calls, 7,624.47 summed worker CPU seconds, and no GPU time.
