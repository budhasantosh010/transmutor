# V837 Input Organization Transfer Report

## 1. Authorization

V837ac exists only because V837ab closed as `SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT` and machine-authorized exactly `TRAINABLE_CONTROLLER_INPUT_FACTORIZATION`.

The frozen neutral parent is V837y Y3: local 10×4 state, rank-4 cross-cell candidate coupling, one joint global scalar carry controller, historical within-step message cascade, and the original 55-edge graph.

## 2. Transfer conditions

- **AC0**: exact historical Y3 parent.
- **AC1**: one shared trainable 6→6 projection used only by the global scalar controller input path. Candidate cells continue to receive raw historical input.
- **AC1F**: exact folded direct control derived from the same AC1 initialization, with no runtime projection.

Because there is only one controller consumer, a de-shared controller projection is scientifically meaningless and was not fabricated.

## 3. Step-zero equivalence

AC1 and AC1F were checked across the paired family/replicate initialization set. Candidate input terms, controller input term, global gate, candidate states, next states and predictions matched with maximum absolute error `7.897615432739258e-07`, below the frozen `1e-6` tolerance.

## 4. Parent reproduction

AC0 reproduced Y3 exactly, with zero family-median drift:

| Family | AC0 validation median |
| --- | ---: |
| conditional routing | 0.843750 |
| delayed recall | 0.953125 |
| iterative state | 0.992188 |
| partial observation | 0.773438 |
| variable composition | 0.851562 |

Families passing: **3/5**.

## 5. Neutral transfer results

| Condition | Routing | Recall | Iterative | Partial | Composition | Families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AC0 Y3 parent | 0.843750 | 0.953125 | 0.992188 | 0.773438 | 0.851562 | **3/5** |
| AC1 controller-input factorization | 0.835938 | 0.968750 | 1.000000 | 0.773438 | 0.882812 | **3/5** |
| AC1F folded control | 0.843750 | 0.953125 | 0.992188 | 0.773438 | 0.851562 | **3/5** |

AC1 improves recall by `+0.015625`, iterative state by `+0.0078125`, and composition by `+0.03125`, but routing falls by `-0.0078125` and partial observation is unchanged. The frozen >=4/5 representation gate therefore remains unmet.

## 6. Projection dynamics

AC1 adds 42 trainable coefficients and 36 projection MACs/timestep. Its recurrent/controller/projection accounting is 882 MACs/timestep versus 846 for AC0/AC1F.

Across the 25 AC1 fits, median projection dynamics are:

| Step | Projection Frobenius | Condition number | Projection drift | Bias drift | Effective controller-input norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.382789 | 17.778775 | 0.000000 | 0.000000 | 0.181192 |
| 24 | 1.431854 | 19.123291 | 0.301799 | 0.102098 | 0.397445 |
| 48 | 1.477881 | 13.854500 | 0.411333 | 0.194677 | 0.585812 |
| 96 | 1.503954 | 15.733896 | 0.513250 | 0.286375 | 0.710854 |
| 144 | 1.498060 | 20.933256 | 0.544866 | 0.329529 | 0.700201 |
| 192 | 1.513768 | 15.141847 | 0.520241 | 0.343583 | 0.688409 |

The controller factorization is actively used and substantially changes the effective controller input map, but that optimization benefit is not enough to restore representation adequacy.

## 7. Partial-observation and message diagnostics

Median diagnostics at convergence:

| Condition | Input-term norm | Input temporal variance | Candidate Jacobian norm approx | Controller input norm | Message success drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| AC0 | 0.515099 | 0.084862 | 0.803384 | 0.286732 | 0.257813 |
| AC1 | 0.505114 | 0.084992 | 0.781053 | 0.688409 | 0.296875 |
| AC1F | 0.515099 | 0.084862 | 0.803384 | 0.286732 | 0.257813 |

The transfer changes controller sensitivity and slightly increases message dependence, but does not improve the partial-observation family at all.

## 8. Diagnosis

`INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT`

Representation adequacy remains **FAIL** at 3/5. Sample-efficiency retesting, structural search and primitive mining remain blocked. Fresh-audit consumption is zero; no primitives were promoted; V838 was not started.

## 9. Data and compute

V837ac used 75 fits, 14,400 optimizer steps and 7,372,800 processed training examples, reusing the same 3,200 unique family/seed episodes. GPU time was zero.

Combined V837ab+V837ac used 225 fits, 43,200 optimizer steps and 22,118,400 processed training examples, still with only 3,200 unique family/seed episodes.

## 10. Strongest claim

Input factorization is a genuine optimization property of the successful T2 reference: a same-function folded model loses one family, while either candidate-only or update-only trainable factorization recovers 4/5. However, transferring the smallest supported property—controller-input factorization—into the best neutral Y3 substrate does not repair its remaining representation deficit. The input-organization axis is therefore closed for this substrate.
