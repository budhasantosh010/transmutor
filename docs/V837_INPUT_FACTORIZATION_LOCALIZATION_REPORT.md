# V837 Input Factorization Localization Report

## 1. Why this axis was tested

V837aa closed candidate-law sharing as `GENUINELY_DIVERSE_CANDIDATE_LAWS` and recommended input organization as the next exact reference-side difference. The successful T2 reference contains a learned 6→6 input projection, but every downstream active input path is linear until the GRU nonlinearities. Therefore the projection is algebraically foldable and cannot enlarge the representable function class by itself.

## 2. Exact function-class folding proof

For `p = A x + a` and downstream `y = W p + b`, define `W_eff = W A` and `b_eff = b + W a`. Then `W(Ax+a)+b = W_eff x + b_eff` exactly.

The full trained T2 model was folded across the complete GRU input matrix, including reset/update/candidate slices. Maximum float32 trace discrepancy was `1.1920928955078125e-07`, below the frozen `1e-6` tolerance. Prediction, hidden states, candidates, raw update vectors, scalarized updates, and raw reset values all passed.

## 3. Matched effective initialization

AB0–AB4 are derived from one deterministic T2 source initialization per family/replicate. AB1/AB2/AB3 fold the relevant paths exactly. AB4 executes the same shared projection but freezes it. All AB0–AB4 candidate/update/state/prediction computations match at optimizer step 0 within `1e-6`. AB5 alone uses the ordinary direct GRU input initialization distribution.

## 4. V837ab results

| Condition | Routing | Recall | Iterative | Partial | Composition | Families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AB0 exact factorized T2 | 0.875000 | 0.976562 | 1.000000 | 0.875000 | 0.820312 | **4/5** |
| AB1 fully folded equivalent | 0.734375 | 0.945312 | 0.992188 | 0.875000 | 0.835938 | **3/5** |
| AB2 candidate factorized / update folded | 0.882812 | 0.968750 | 0.992188 | 0.882812 | 0.820312 | **4/5** |
| AB3 candidate folded / update factorized | 0.695312 | 0.945312 | 0.992188 | 0.867188 | 0.851562 | **4/5** |
| AB4 frozen shared projection | 0.656250 | 0.937500 | 0.976562 | 0.875000 | 0.765625 | **2/5** |
| AB5 naive projection-free | 0.734375 | 0.953125 | 1.000000 | 0.875000 | 0.835938 | **3/5** |

AB0 reproduced historical T2 at exactly 4/5 with zero family-median drift.

## 5. Decisive AB0 vs AB1 comparison

AB0 and AB1 begin from the same effective candidate and update input functions. AB0 reaches 4/5 while AB1 reaches 3/5. Therefore the factorized parameterization matters during optimization even though it does not expand the function class.

## 6. Candidate/update pathway localization

AB2 and AB3 each independently reach 4/5. This means either candidate-path or update-path trainable factorization can be sufficient on the T2 reference. Because the controller/update path is the smaller semantic intervention for neutral transfer, the frozen machine diagnosis is:

`SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT`

Authorized V837ac mode:

`TRAINABLE_CONTROLLER_INPUT_FACTORIZATION`

## 7. Frozen preconditioning and naive direct controls

AB4 falls to 2/5, so simply retaining the original projection as a fixed preconditioner is insufficient. AB5 reaches 3/5, so ordinary direct initialization is also insufficient. The benefit therefore cannot be reduced to function-class capacity, a frozen coordinate transform, or the naive direct initialization distribution.

## 8. Optimization geometry

At step 192, median AB0 projection singular values are:

`[1.357914, 0.984115, 0.712372, 0.511908, 0.303852, 0.062910]`

AB0 median projection Frobenius norm is `1.851700` and median condition number is `21.687610`. For AB2 they are `1.884930` and `15.847712`; for AB3 `2.349130` and `31.732334`. AB4 remains exactly at its initialization and has median Frobenius norm `1.382789`, condition number `17.778775`.

The effective maps diverge substantially from their identical step-zero starting functions. For AB0 at step 192, median effective-map distance from initialization is `2.387543` for candidate and `2.309445` for update. The fully folded AB1 reaches `1.893775` and `2.394567`, showing that the two parameterizations follow different optimization trajectories despite exact initial functional equivalence.

## 9. Projection gradient decomposition

On frozen development batches for 25 trained AB0 fits:

- median candidate-path projection-gradient norm: `0.0008313`
- median update-path projection-gradient norm: `0.0002251`
- median full projection-gradient norm: `0.0008786`
- median candidate/update gradient cosine: `-0.001270`
- median additive residual relative error: `6.83e-7`

The full gradient numerically equals candidate + update contributions. Candidate/update signals are approximately orthogonal or mixed rather than strongly cooperative.

## 10. Data and compute

V837ab used exactly 150 fits, 28,800 optimizer steps and 14,745,600 processed training examples. The same 3,200 unique family/seed episodes were reused across all conditions and replicates. Fresh-audit data remained unused and GPU time was zero.

## 11. V837ab conclusion

The T2 input projection is **not** required for representational capacity, because it is exactly foldable, but a trainable linear factorization on either one active input pathway can be sufficient to preserve the 4/5 reference result under matched step-zero function. The smallest supported reference property for neutral transfer is controller-input factorization.
