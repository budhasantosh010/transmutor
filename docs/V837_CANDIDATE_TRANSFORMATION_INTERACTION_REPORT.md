# V837 Candidate Transformation Interaction and Stage-Organization Report

## 1. Current frontier

The program starts from the closed V837v/V837w/V837x frontier. V837x established that the V837w-authorized joint global scalar carry controller improves the neutral substrate to 3/5 but does not restore representation adequacy. V837r independently established that rank-4 cross-cell candidate/recurrent coupling reaches 3/5 with a complementary family signature.

## 2. Why candidate organization is next

The strongest causal opportunity was the untested interaction between two independently useful partial mechanisms: one low-bandwidth global temporal control scalar and rank-4 cross-cell candidate integration. The program deliberately did not redesign either mechanism.

## 3. Complementary family signatures

Historical global control (Y1/X2) strongly improves routing; historical rank-4 candidate coupling (Y2/R3) strongly improves composition. Both preserve recall and iterative-state competence. V837y tests whether those gains can coexist.

## 4. Factorial design

Five frozen conditions were run on the same data and paired initialization regime:

- Y0: controller OFF, candidate coupling OFF
- Y1: controller ON, candidate coupling OFF
- Y2: controller OFF, rank-4 cross-block candidate coupling ON
- Y3: controller ON, rank-4 cross-block candidate coupling ON
- Y3C: controller ON, exact rank-4 parameter-matched local-only candidate branch

## 5. Frozen global scalar controller

The controller is exactly the V837x joint input+state scalar mechanism. At the start of timestep t it reads the concatenated previous 40D state and current 6D observation and computes one sigmoid scalar. It has 47 parameters and approximately 46 MACs/timestep. It never reads messages, candidates, partially updated current states, or task labels.

## 6. Frozen rank4 candidate mechanism

The candidate branch reuses V837r rank-4 factorization, initialization, scaling, and cross-block mask. It contributes inside the candidate tanh. The matched-local branch uses the exact V837r local-capacity control and cannot read another cell's state.

## 7. Y0 baseline

Y0 reproduced the historical neutral anchor exactly at the family-median level and passed 2/5 families.

| Family | Validation median |
| --- | ---: |
| conditional_routing | 0.492188 |
| delayed_recall | 0.906250 |
| iterative_state | 0.984375 |
| partial_observation | 0.812500 |
| variable_composition | 0.789062 |

Parameters: 856. Recurrent/controller MACs: 160.

## 8. Y1 global control

Y1 reproduced V837x X2 exactly and passed 3/5.

| Family | Validation median |
| --- | ---: |
| conditional_routing | 0.859375 |
| delayed_recall | 0.929688 |
| iterative_state | 0.992188 |
| partial_observation | 0.820312 |
| variable_composition | 0.804688 |

Parameters: 903; controller parameters: 47; recurrent/controller MACs: 206.

## 9. Y2 candidate coupling

Y2 remained compatible with V837r R3 and passed 3/5. Maximum historical-median drift was only 0.0391, below the frozen drift stop rule.

| Family | Validation median |
| --- | ---: |
| conditional_routing | 0.593750 |
| delayed_recall | 0.976562 |
| iterative_state | 0.992188 |
| partial_observation | 0.781250 |
| variable_composition | 0.875000 |

Parameters: 1,176. Recurrent/controller MACs: 800.

## 10. Y3 interaction

Y3 combined only the frozen global scalar controller and frozen rank-4 candidate branch. It passed 3/5, not the required 4/5.

| Family | Validation median |
| --- | ---: |
| conditional_routing | 0.843750 |
| delayed_recall | 0.953125 |
| iterative_state | 0.992188 |
| partial_observation | 0.773438 |
| variable_composition | 0.851562 |

Parameters: 1,223. Recurrent/controller MACs: 846.

## 11. Y3C matched capacity control

Y3C kept the same global controller and exactly matched Y3's parameter count with V837r's local-only rank-4 capacity branch. It reached only 2/5.

| Family | Validation median |
| --- | ---: |
| conditional_routing | 0.726562 |
| delayed_recall | 0.960938 |
| iterative_state | 1.000000 |
| partial_observation | 0.804688 |
| variable_composition | 0.820312 |

Parameters: 1,223. Recurrent/controller MACs: 526. The lower compute arises because the matched branch is local even though the parameter count exactly matches Y3.

## 12. Per-family results

Y3 retained strong routing (0.84375) and strong composition (0.85156) simultaneously, but partial observation remained weak (0.77344). Recall and iterative state remained strong. The combination therefore produced the hypothesized qualitative complementarity without crossing the 4/5 capacity gate.

## 13. Interaction effects

The frozen factorial statistic is `AB = Y3 - Y2 - Y1 + Y0` computed per paired family/replicate. Overall AB mean was -0.03656 and median -0.02344. Global-control main effect A had mean +0.05234; rank-4 main effect B had mean +0.01984. Interaction magnitude is descriptive only and was never used as a pass criterion.

Per-family median AB was approximately -0.05469 for routing, -0.03906 for recall, -0.00781 for iterative state, -0.08594 for partial observation, and +0.00781 for composition.

## 14. Routing/composition complementarity

Relative to Y0, Y1 improves routing by +0.36719 median validation and Y2 improves composition by +0.08594. Y3 preserves routing near the Y1 level (-0.01563 versus Y1) and preserves a high composition score (0.85156 versus Y2's 0.875). This is real complementarity, but it is insufficient because partial observation remains below the family gate and the total remains 3/5.

## 15. Candidate branch utilization

The Y3 rank-4 branch was strongly active rather than collapsing toward zero. Median rank-4 candidate-term norm was 2.8243; median local recurrent norm was 1.6516; median global/local norm ratio was 1.8243; median global/message ratio was 1.7157; median global/input ratio was 1.6046. Cross-block energy was nonzero with median 48.2006, off-diagonal fraction 1.0, and median spectral norm 4.6018.

The configured factor rank remained 4. The masked realized matrix is generally full numerical rank after block masking; the recorded median realized effective matrix rank was 40. This diagnostic does not change the configured low-rank factorization semantics.

## 16. Controller behavior

Y3's global carry fraction had mean 0.3953 and median 0.4098; rewrite fraction median was 0.5902. Median gate temporal variance was 0.00663, median p10 0.3030, median p90 0.5314, and median near-zero/near-one fractions were both 0.

For comparison, Y1's carry median was 0.3704 with median temporal variance 0.01592. Candidate mixing therefore changed the learned control regime descriptively but did not remove dynamic temporal control.

## 17. Message dependence

Message ablation remained consequential. Mean validation-success drop was 0.2578 for Y0, 0.3359 for Y1, 0.2253 for Y2, 0.2359 for Y3, and 0.2644 for Y3C. The interaction did not make the historical graph-message system unnecessary.

## 18. Causal interventions

In Y3, zeroing one cell only as a source to the rank-4 candidate branch—while preserving its local recurrence, ordinary messages, and local candidate pathway—changed other-cell candidates (median mean absolute delta 0.09786), other-cell next states (0.08523), and final outputs (0.09914). The cross-cell branch therefore has genuine causal use under the global controller.

Controller interventions at g=0, 0.5, 1, and the learned mean gate were stored per family/replicate. They are inference-only diagnostics and do not alter the primary training result.

## 19. Compute efficiency

| Condition | Families | Params | Controller params | Rank4/local-extra params | Recurrent/controller MACs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Y0 | 2/5 | 856 | 0 | 0 | 160 |
| Y1 | 3/5 | 903 | 47 | 0 | 206 |
| Y2 | 3/5 | 1,176 | 0 | 320 | 800 |
| Y3 | 3/5 | 1,223 | 47 | 320 | 846 |
| Y3C | 2/5 | 1,223 | 47 | 320 | 526 |

Y3 gains one family over Y0 at +686 recurrent/controller MACs/timestep, but still fails adequacy. Compute was never used to relax the competence gate.

## 20. Representation adequacy

V837y representation adequacy: **FAIL**. No condition reached the frozen >=4/5 criterion.

## 21. V837z authorization

Because anchors reproduced, V837y completed below 4/5, and Y3 was machine-selected as the best parent, V837y wrote `v837z_allowed=true` and selected `Y3_global_control_rank4_candidate`. No V837z files existed before this authorization.

V837y diagnosis: `GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT`.

## 22. Sample-efficiency status

BLOCKED. Representation adequacy was not restored by V837y.

## 23. Structural-search status

BLOCKED.

## 24. Primitive-mining status

BLOCKED.

## 25. Fresh-audit status

Fresh-audit episodes consumed: 0. Reserved seeds 90000–90499 remain unused.

## 26. Strongest claim

Joint global temporal control and rank-4 cross-cell candidate integration are both causally active and preserve complementary routing/composition competence, while an exact matched local-capacity branch performs worse; however, their combination remains insufficient to restore fixed-topology representation adequacy under the frozen 4× data regime.

## 27. Next variable

V837y itself authorized only candidate-stage synchronization, not a new candidate parameterization or rank sweep.

## 28. Candidate-stage hypothesis

V837z asked whether the neutral substrate's within-timestep sequential cascade of candidate computations was the remaining blocker compared with the successful GRU's single candidate stage.

## 29. Historical mixed-stage semantics

Z0 preserved Y3 exactly. Non-recurrent edges from already-computed earlier cells may consume same-timestep outputs; recurrent or not-yet-available sources use previous outputs.

## 30. Fully synchronous semantics

Z1 kept all 55 edges but forced every edge to read only the previous-output snapshot. Controller and rank-4 branch read the same previous-state snapshot as before. All candidate states are computed without current-timestep output leakage and committed together.

## 31. Stage-depth diagnostics

Historical Z0 induced effective candidate depths `[1,2,3,4,5,6,7,8,9,10]`: minimum 1, median 5.5, maximum 10. Z1 forced every cell to depth 1.

The explicit leakage regression test confirms that perturbing an earlier cell's current output can alter a later cell's same-timestep candidate in Z0, whereas Z1 prevents that change until the following timestep.

## 32. Z0 vs Z1 results

Z0 reproduced Y3 exactly at every family median and passed 3/5:

- routing 0.843750
- recall 0.953125
- iterative 0.992188
- partial 0.773438
- composition 0.851562

Z1 fell to 2/5:

- routing 0.773438
- recall 0.960938
- iterative 1.000000
- partial 0.789062
- composition 0.812500

Mean family validation median fell from 0.88281 to 0.86719. Mean state-change synchrony diagnostic changed from 0.5342 to 0.5202. Mean message-ablation success drop changed from 0.2359 to 0.2553.

## 33. Final candidate-organization diagnosis

V837z diagnosis: `HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL`.

Final representation adequacy remains **FAIL**. Candidate-organization improvisation is now stopped. Sample-efficiency testing, structural search, and primitive mining remain blocked; fresh-audit consumption remains 0; primitives promoted remain 0; V838 remains NOT STARTED.

The separate blocker analysis selects exactly one later variable without implementing it: **candidate parameter sharing — one shared candidate parameterization versus independently parameterized per-cell candidate transformations**.
