# V837 Representation Recovery Report

## 1. V837 failure recap

The completed V837 lineage (`V837`, `V837b`, `V837c`) failed the original neutral-substrate competence gate and isolated `REPRESENTATION_FAILURE` as the strongest blocker. This recovery program began from main `1280b2ad30af3976633f7052943650c769d2e508` and did not modify historical V836 or V837 result artifacts.

## 2. Why representation was isolated

Structural-search difficulty had already been substantially reduced by a fixed high-capacity 10-cell/55-edge diagnostic graph with all continuous parameters trainable. Even then, competence generalized reliably only on a minority of the required task families. The recovery program therefore tested minimal task-independent representation properties before spending more search budget.

## 3. Historical substrate

Historical cells used state/message dimension 4 and the update

`candidate = tanh(Ws*s + Wm*m + Wx*x + b)`

`state_next = candidate`

with the complete raw observation visible to every cell. No named high-level operator, task-family label, router, attention mechanism, memory gate, or primitive was provided.

## 4. V837d — sparse raw-input access

### Single change

Raw observation accessibility changed from all-to-all broadcast to deterministic fixed sparse masks. Cell law, dimensions, graph topology, optimizer, learning rate, training steps, task generators, seeds and frozen capacity criterion remained unchanged.

### Baseline compatibility

The refactored broadcast condition reproduced all five preserved blocker-diagnostic comparison scores exactly: absolute difference `0.0` for conditional routing, delayed recall, iterative state, partial observation and variable composition.

### Density sweep

Development-only densities were 12.5%, 25% and 50%, with eight independent mask seeds per density. The frozen aggregate selection rule selected 50% density.

Selected fixed-sparse validation medians:

| Family | Validation median | Capacity |
|---|---:|---|
| conditional routing | 0.4141 | FAIL |
| delayed recall | 0.6641 | FAIL |
| iterative state | 0.9766 | PASS |
| partial observation | 0.6914 | FAIL |
| variable composition | 0.5273 | FAIL |

Families passing: **1/5**.

### Controls and diagnostics

Degree-preserving shuffled sparse masks also passed 1/5 families. The no-message sparse control also passed 1/5. Sparse access increased generic message dependency relative to broadcast, but did not restore general competence. Median pairwise state correlation and tanh saturation did not reveal a simple correlation/saturation repair story.

**V837d result: FAIL — `INPUT_ACCESS_FAILURE`, `MESSAGE_MEDIATION_FAILURE`.**

The evidence did not justify V837e evolvable input edges or V837f mediated ingress, so both were skipped.

## 5. V837e

**NOT RUN — not scientifically justified by V837d.** Fixed sparsity did not establish enough broad directional value to justify evolving input edges under additional search.

## 6. V837f

**NOT RUN — not scientifically justified by V837d.** The message-mediation diagnostics did not establish sparse ingress as the missing capacity property.

## 7. V837g — generic state persistence

### Single change

The historical overwrite update was replaced with one learned task-independent scalar coefficient per ordinary cell:

`alpha = sigmoid(a)`

`state_next = (1-alpha)*state + alpha*candidate`

No input-conditioned gate was added. Broadcast input access and the remaining historical diagnostic conditions were kept fixed.

Learned alpha summary across trained cells: mean **0.5503**, median **0.5481**, p10 **0.4923**, p90 **0.6062**.

Validation medians:

| Family | Validation median | Capacity |
|---|---:|---|
| conditional routing | 0.4102 | FAIL |
| delayed recall | 0.7422 | FAIL |
| iterative state | 0.9922 | PASS |
| partial observation | 0.6875 | FAIL |
| variable composition | 0.5547 | FAIL |

Families passing: **1/5**.

Parameter count increased from 856 to 866 (+1.17%).

**V837g result: FAIL — `STATE_UPDATE_FAILURE`, `CAPACITY_WITHOUT_GENERALIZATION`.**

## 8. V837h — low-rank multiplicative interaction

### Single change

The candidate interaction basis was changed to a rank-2 generic multiplicative branch. A parameter-matched additive branch was implemented as the required confound control. Both conditions had exactly **1,096 parameters**, versus 856 historically (+28.04%). Cell and graph sizes remained fixed.

### Parameter-matched additive validation medians

| Family | Median | Capacity |
|---|---:|---|
| conditional routing | 0.3867 | FAIL |
| delayed recall | 0.6602 | FAIL |
| iterative state | 0.9766 | PASS |
| partial observation | 0.7148 | FAIL |
| variable composition | 0.5234 | FAIL |

### Rank-2 multiplicative validation medians

| Family | Median | Capacity |
|---|---:|---|
| conditional routing | 0.4648 | FAIL |
| delayed recall | 0.4961 | FAIL |
| iterative state | 0.9648 | PASS |
| partial observation | 0.6797 | FAIL |
| variable composition | 0.5234 | FAIL |

The multiplicative basis improved routing relative to the matched additive branch, but strongly hurt delayed recall and did not increase the number of capable families.

**V837h result: FAIL — `INTERACTION_BASIS_FAILURE`, `CAPACITY_WITHOUT_GENERALIZATION`.**

## 9. Interaction experiments conclusion

A small multiplicative interaction basis changes the family trade-off but is not a general competence repair. Because the additive and multiplicative conditions are exactly parameter matched, the V837h outcome cannot be explained simply by the extra parameter count.

## 10. All failures

| Version | Single representation change | Families passing | Result |
|---|---|---:|---|
| V837d | broadcast → fixed sparse raw-input access | 1/5 | FAIL |
| V837g | direct overwrite → learned scalar state-update coefficient | 1/5 | FAIL |
| V837h | additive candidate → rank-2 multiplicative interaction | 1/5 | FAIL |

Three scientifically distinct representation failures trigger the recovery stop rule. V837i combined representation was therefore **not** run.

## 11. Resource cost

Representation recovery used **360 model/candidate evaluations**, **69,120 optimizer steps**, **645,408 environment interactions**, **8,847,360 examples processed**, and **69,840 forward calls**. Summed worker wall time was about **17,867.9 seconds** and summed CPU time about **14,965.8 seconds**.

The optimizer-step cost was about **5.59%** and environment-interaction cost about **5.35%** of the original V837 research reference, satisfying the cheap-diagnostic-before-full-search discipline.

## 12. Strongest supported diagnosis

The tested task-independent tanh continuous-cell family remains inadequate for the required cross-family competence under the examined representation changes. Sparse input access, simple learned state persistence and rank-2 multiplicative interaction each change behavior, but none repairs general capacity under the frozen criterion.

This is not evidence that all neutral continuous-cell substrates are impossible. It is evidence against continuing to invest in this specific family through incremental stacking without a new falsifiable cell-law hypothesis.

## 13. Final recovered substrate

**None.** No representation candidate reached the `>=4/5` high-capacity recovery prerequisite, so full structural search was not reopened.

## 14. What remains human-supplied

The low-level cell update family itself is still human-specified. The representation-recovery evidence moves this scaffold to the foreground: before asking the system to invent reusable higher-level primitives, the supplied cell family must first be capable of representing the required behaviors robustly.

## 15. Whether primitive mining may reopen

**NO.**

- neutral substrate competence: FAIL
- primitive mining allowed: false
- primitives promoted during recovery: 0
- fresh-audit episodes consumed: 0
- V838 started: no

The next experiment must test one fundamentally different generic cell-update property or cell family. It must not combine the failed V837d/V837g/V837h features merely to force a pass, and it must not resume primitive mining until a recovered substrate passes the original full V837 competence gate.
