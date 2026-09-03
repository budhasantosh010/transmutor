# V837 GRU Mechanism Localization Report

## 1. Why 4x GRU success changed the diagnosis

V837j showed that the parameter-matched learned references did not resolve benchmark learnability at the original 1x data regime. V837k showed that increasing optimizer steps alone did not rescue the GRU. V837l then changed only unique development data and the same 875-parameter GRU moved from 2/5 at 1x, to 3/5 at 2x, to 5/5 at 4x, while the neutral high-capacity substrate reached only 2/5 at 4x. This established benchmark learnability under a calibrated recurrent-learning regime and strengthened the representation gap, while retaining sample efficiency as a real confound for the original V837 regime.

V837m then showed that stable linear state transport did not repair the neutral cell. The next justified question was therefore not which GRU feature to copy, but which property of the successful GRU was actually necessary.

## 2. Why mechanism localization precedes more neutral-cell design

A GRU contains multiple interacting properties: dense hidden organization, candidate-state conditioning, adaptive preserve-vs-update control, dimension-wise coefficients, and multiplicative state modulation. Copying the whole cell would destroy the scientific isolation built through V837d–m. V837n therefore stayed entirely on the successful learned-reference side and used retrained ablations under the exact 4x regime where the positive control works.

No V837o/p/q neutral mechanism was allowed to start until this reference-side localization closed.

## 3. Explicit GRU equations

V837n implemented the PyTorch `GRUCell` convention used by V837j/l explicitly rather than relying on an opaque recurrent primitive:

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_t + b_hr)
z_t = sigmoid(W_iz x_t + b_iz + W_hz h_t + b_hz)
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_t + b_hn))
h_(t+1) = (1-z_t) * n_t + z_t * h_t
```

The existing learned-reference input projection and readout were preserved. With the same initialization seed, every shared tensor initializes identically to the V837j/l framework GRU. A direct forward-equivalence test gives maximum absolute output error below `1e-6` (observed approximately `2.3e-8`).

The full explicit reference contains exactly **875 parameters**, matching the successful V837l GRU.

## 4. Positive-control reproduction

The explicit GRU was trained under exactly the successful V837l 4x regime:

- 512 unique development episodes, seeds 10000–10511;
- 128 validation episodes, seeds 20000–20127;
- AdamW;
- 192 optimizer steps;
- learning rate 0.005;
- weight decay 0.0001;
- gradient clipping 5.0;
- hidden size 13;
- five paired initialization replicates per family using the existing `v837j-primary-init` namespace.

The full explicit GRU reproduced **5/5 families** and passed the frozen compatibility check against V837l. Therefore the ablation experiment is interpretable.

Full-GRU validation medians:

| Family | Median validation | Capacity result |
| --- | ---: | --- |
| conditional routing | 0.9375 | PASS |
| delayed recall | 0.9844 | PASS |
| iterative state | 1.0000 | PASS |
| partial observation | 0.8906 | PASS |
| variable composition | 0.8828 | PASS |

## 5. Update ablations

Three update/carry controls were retrained while preserving all other conditions.

| Condition | Nominal params | Active params | Families passing |
| --- | ---: | ---: | ---: |
| full GRU | 875 | 875 | 5/5 |
| static update vector | 888 | 615 | 4/5 |
| static update scalar | 876 | 603 | 4/5 |
| no update / forced overwrite | 875 | 602 | 5/5 |

The decisive result is the forced-overwrite control. Setting `z_t = 0`, so `h_next = candidate`, did **not** destroy competence. It retained 5/5 aggregate family competence. Therefore adaptive carry/update control is not individually necessary for the successful-reference result.

The static vector and static scalar controls also remain at 4/5, showing that dynamic update trajectories are not uniquely required when dynamic candidate conditioning remains available.

## 6. Reset/candidate-conditioning ablations

Reset-side controls produced the complementary result.

| Condition | Nominal params | Active params | Families passing |
| --- | ---: | ---: | ---: |
| full GRU | 875 | 875 | 5/5 |
| no reset (`r_t = 1`) | 875 | 602 | 5/5 |
| static reset vector | 888 | 615 | 5/5 |

Removing dynamic reset modulation does not destroy competence. Therefore reset/candidate-state conditioning is also not individually necessary when dynamic update control remains available.

## 7. Coupling controls

The double ablation removes both adaptive mechanisms:

```text
r_t = 1
z_t = 0
h_(t+1) = tanh(W_in x_t + b_in + W_hn h_t + b_hn)
```

It keeps the same outer input projection/readout and retains the disabled GRU tensors nominally, but only 329 parameters are functionally active.

Result: **3/5 families**.

| Family | Double-ablation validation median | Capacity result |
| --- | ---: | --- |
| conditional routing | 0.6797 | FAIL |
| delayed recall | 0.9844 | PASS |
| iterative state | 0.9922 | PASS |
| partial observation | 0.8906 | PASS |
| variable composition | 0.8359 | FAIL |

This is not the strong N4 pattern of a collapse to <=2/5, so V837n does not claim that the two named GRU gates are jointly indispensable in a strict sense. It does show that removing both adaptive state-control paths at once drops the model below the 4/5 adequacy gate, whereas retaining either one is sufficient for 5/5 in the strongest retrained ablations.

The frozen diagnosis is therefore **MECHANISM_REDUNDANCY_OR_COMPLEMENTARITY**.

## 8. Per-family results

Validation medians across the seven conditions:

| Condition | Routing | Recall | Iterative | Partial obs. | Composition | Passing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full GRU | 0.9375 | 0.9844 | 1.0000 | 0.8906 | 0.8828 | 5/5 |
| static update vector | 0.8125 | 0.9453 | 0.9922 | 0.8906 | 0.8750 | 4/5 |
| static update scalar | 0.8047 | 0.9609 | 0.9922 | 0.8906 | 0.8828 | 4/5 |
| no update | 0.8594 | 0.9844 | 0.9844 | 0.8828 | 0.9453 | 5/5 |
| no reset | 0.9062 | 0.9922 | 1.0000 | 0.8906 | 0.8594 | 5/5 |
| static reset vector | 0.8984 | 0.9766 | 0.9922 | 0.8984 | 0.8516 | 5/5 |
| no update + no reset | 0.6797 | 0.9844 | 0.9922 | 0.8906 | 0.8359 | 3/5 |

The double ablation primarily exposes conditional routing and variable composition. Delayed recall and iterative state remain strong, and partial observation remains above the criterion.

## 9. Gate dynamics

The trained full GRU does use dynamic coefficients descriptively.

Aggregate full-GRU statistics across 25 trained family/replicate models:

- update coefficient mean: approximately **0.408**;
- update coefficient median: approximately **0.379**;
- update temporal variance median: approximately **0.0193**;
- update entropy median: approximately **0.564**;
- reset coefficient mean: approximately **0.593**;
- reset coefficient median: approximately **0.625**;
- reset temporal variance median: approximately **0.0163**;
- reset entropy median: approximately **0.556**;
- hidden-state autocorrelation median: approximately **0.735**;
- hidden-state norm median: approximately **1.54**;
- state saturation median: approximately **0.0032**.

Counterfactual replay on trained full-GRU models shows that execution-time gate trajectories matter to the already-trained solution. Replacing or time-shuffling update trajectories causes large drops on routing, recall, and composition; reset replay also hurts routing and composition. For example, median validation under update-time shuffling falls to roughly 0.523 on routing and 0.633 on composition.

This replay result must not be confused with retrained necessity. Retraining can compensate for losing either update or reset dynamics individually. That distinction is one of V837n's most important findings.

## 10. Strongest causal mechanism diagnosis

The strongest defensible causal claim is:

> **No individual named GRU gate mechanism explains the 5/5 learned-reference capability. Adaptive update/carry can be removed with 5/5 retained, and adaptive reset/candidate conditioning can also be removed with 5/5 retained. Removing both adaptive paths together drops the dense recurrent reference below the 4/5 adequacy gate to 3/5, indicating redundancy or complementarity among adaptive state-control paths rather than necessity of either GRU gate in isolation.**

This is Outcome C of the program: the research should not transfer either GRU gate directly into Transmutor based on V837n.

## 11. Minimal neutral mechanism selected

**None selected.**

V837o adaptive scalar update is not justified because update control is not individually necessary. V837q candidate conditioning is not justified because reset conditioning is not individually necessary. V837p is also not justified: static vector persistence is not uniquely required, and even the static scalar update condition remains at 4/5 while dynamic reset is present.

The next research variable should stay on the successful-reference side and isolate the **smallest location-agnostic adaptive multiplicative state-modulation property** shared by the redundant solutions, or test whether the remaining gap is better explained by dense hidden-state organization/optimization geometry.

## 12. Neutral follow-up results

No V837o, V837p, V837q, or V837r neutral transfer was executed. Creating one despite the V837n evidence would violate the program's central rule: interrogate the successful reference before transferring a mechanism.

## 13. Representation adequacy

No new neutral cell has reached the >=4/5 representation-adequacy gate in this phase. Therefore:

```text
REPRESENTATION_ADEQUACY = NOT_RESTORED_IN_NEUTRAL_SUBSTRATE
FULL_STRUCTURAL_SEARCH = BLOCKED
```

V837n itself is a diagnostic PASS because the successful reference was reproduced and the ablations meaningfully ruled out both named GRU mechanisms as individually necessary.

## 14. Sample efficiency

No recovered neutral architecture exists, so V837s was not run. The previously established V837l calibration remains the current sample-efficiency evidence:

- GRU: 2/5 at 1x, 3/5 at 2x, 5/5 at 4x;
- neutral high-capacity baseline: 2/5 at 4x.

The sample-efficiency question for a recovered neutral cell therefore remains open.

## 15. Remaining unresolved scaffold

The remaining difference is narrower than before V837n. It is not cleanly attributable to GRU update/carry control or reset/candidate conditioning individually. Candidate next explanations include:

1. a location-agnostic adaptive multiplicative modulation property, where either one of multiple placements can supply the necessary flexibility;
2. dense hidden-state organization and shared recurrent parameterization;
3. optimization geometry induced by the dense recurrent reference;
4. an interaction between dense organization and adaptive modulation.

The next experiment should change only one of these properties. Structural search, primitive mining, and fresh audit remain locked.

## Resource accounting and locks

V837n used 175 CPU model fits, 33,600 optimizer steps, 17,203,200 processed examples, 778,575 unique environment interactions, and 37,900 recorded forward calls. No GPU was used.

Fresh-audit episodes consumed: **0**.

Primitives promoted: **0**.

Primitive mining: **blocked**.

Full structural search: **blocked**.

V838: **not started**.
