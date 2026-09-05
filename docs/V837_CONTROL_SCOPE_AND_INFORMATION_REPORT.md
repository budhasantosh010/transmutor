# V837 Control Scope and Controller-Information Localization Report

## 1. Current V837 frontier

This program continues the frozen V837t/V837u frontier. V837t established that dimension-specific dynamic update values are not required in the successful reference: T2 scalarized update/no-reset reaches 4/5. V837u then showed that ten independent local scalar carry controllers in the neutral substrate reach only 2/5.

## 2. Why scalar carry transfer failed

V837u left two confounded differences between successful T2 and failed U2: T2 has one scalar controlling the entire recurrent state, and that scalar can depend on the whole recurrent hidden state; U2 has one scalar per cell and each controller sees only local state/message/input. V837v and V837w isolate those properties separately.

## 3. Successful T2 versus failed U2

```text
T2: one shared scalar, whole recurrent-state information, 4/5
U2: ten local scalars, local information only,             2/5
```

## 4. Control scope vs controller information scope

**Control scope** asks how many independent temporal decisions are emitted. **Controller information scope** asks what information is available when computing a decision. V837v changes only the first. V837w changes only the second inside the frozen successful T2 reference. V837x transfers only the information source authorized by V837w.

## 5. Why primary V837v avoids gate pooling

V837v never computes `mean(g_1...g_10)` as its primary mechanism. Pooling would simultaneously broaden output scope and aggregate information across cells. Instead each domain uses one fixed earliest local source controller and broadcasts that one controller's scalar to its domain.

## 6. V0 — 10-domain baseline

V0 exactly reproduces V837u U2: **2/5**. Validation medians match with zero drift:

| Family | Median |
| --- | ---: |
| conditional routing | 0.640625 |
| delayed recall | 0.914063 |
| iterative state | 0.992188 |
| partial observation | 0.804688 |
| variable composition | 0.796875 |

Active controllers: 10; active controller parameters: 150; controller MACs/timestep: 140; total recurrent/controller MACs: 300.

## 7. V1 — 5-domain synchronization

V1 remains **2/5**. Validation medians: routing 0.648438, recall 0.929688, iterative 0.992188, partial 0.812500, composition 0.789063. Active controllers: 5; controller parameters: 75; controller MACs: 70; total MACs: 230.

## 8. V2 — 2-domain synchronization

V2 remains **2/5**. Validation medians: routing 0.617188, recall 0.945313, iterative 0.992188, partial 0.804688, composition 0.789063. Active controllers: 2; controller parameters: 30; controller MACs: 28; total MACs: 188.

## 9. V3 — global source broadcast

V3 remains **2/5**. Validation medians: routing 0.601563, recall 0.929688, iterative 0.992188, partial 0.828125, composition 0.796875. Active controllers: 1; controller parameters: 15; controller MACs: 14; total MACs: 174.

V3 is one **local** controller whose output is globally broadcast. It is not a global observer.

## 10. Per-family results

| Condition | Routing | Recall | Iterative | Partial | Composition | Passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 10 domains | 0.640625 | 0.914063 | 0.992188 | 0.804688 | 0.796875 | 2/5 |
| V1 5 domains | 0.648438 | 0.929688 | 0.992188 | 0.812500 | 0.789063 | 2/5 |
| V2 2 domains | 0.617188 | 0.945313 | 0.992188 | 0.804688 | 0.789063 | 2/5 |
| V3 1 domain | 0.601563 | 0.929688 | 0.992188 | 0.828125 | 0.796875 | 2/5 |

## 11. State synchrony

Cross-cell state-change correlation rises monotonically as control domains are coarsened:

```text
V0 0.3711 median
V1 0.4091
V2 0.5132
V3 0.5908
```

Thus shared control measurably synchronizes update timing, but that synchronization does not improve the representation gate.

## 12. Message dependence

Median validation performance drop under message ablation:

```text
V0 0.273438
V1 0.328125
V2 0.273438
V3 0.218750
```

Global broadcast slightly lowers message dependence at V3, but not enough to restore competence.

## 13. Compute efficiency

Capability stays fixed at 2/5 while active controller cost falls from 140 to 14 MACs/timestep and active controller parameters fall from 150 to 15. Broader control scope therefore improves controller efficiency but not capability in this substrate.

## 14. V837v diagnosis

**`CONTROL_SCOPE_ALONE_INSUFFICIENT`**.

No 10→5→2→1 domain scale reaches the 4/5 representation gate. V837w is therefore authorized.

## 15. T2 update-logit decomposition

V837w decomposes the successful T2 update logit into:

```text
L_input = W_iz x_projected
L_state = W_hz h_previous
B       = b_iz + b_hz
```

The exact W0 anchor uses the frozen fused T2 arithmetic for numerical identity, while the decomposition is exposed separately for source ablation. Scalarization remains after sigmoid.

## 16. Joint controller

W0 input+state exactly reproduces T2 at **4/5**, with zero per-family median drift.

Validation medians: routing 0.875000, recall 0.976563, iterative 1.000000, partial 0.875000, composition 0.820313.

## 17. Input-only controller

W1 reaches **3/5**: routing 0.632813, recall 0.976563, iterative 0.992188, partial 0.890625, composition 0.812500.

## 18. State-only controller

W2 reaches **3/5**: routing 0.781250, recall 0.976563, iterative 1.000000, partial 0.882813, composition 0.828125.

## 19. Bias-only controller

W3 reaches **3/5**: routing 0.593750, recall 0.960938, iterative 0.992188, partial 0.890625, composition 0.812500. Gate temporal variance is exactly zero.

## 20. Reference information-source diagnosis

**`JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED`**.

Neither input-only nor state-only reaches 4/5. Bias-only also fails. Only joint dynamic input+state information preserves successful T2 competence.

W0 median logit diagnostics:

```text
input logit norm   1.481156
state logit norm   1.985084
input/total ratio  0.426372
state/total ratio  0.636338
gate temporal var  0.013993
```

V837x is authorized only as `JOINT_INPUT_STATE_GLOBAL_SCALAR`.

## 21. Authorized global neutral controller

V837x keeps the 10×4 neutral substrate fixed and adds one global scalar controller that reads the concatenated 40D **previous** neutral state plus the current 6D observation.

## 22. Controller equation

```text
S_t = [s_0,t ; ... ; s_9,t]

g_t = sigmoid(w_s^T S_t + w_x^T x_t + b)
```

Parameters: 40 + 6 + 1 = **47**. Approximate controller MACs: **46**. `g_t` is computed exactly once at the start of each timestep and then frozen while all cells execute.

## 23. Global carry result

X2 global scalar carry reaches **3/5**.

| Family | Validation median |
| --- | ---: |
| conditional routing | 0.859375 |
| delayed recall | 0.929688 |
| iterative state | 0.992188 |
| partial observation | 0.820313 |
| variable composition | 0.804688 |

This is a real improvement over X1 local scalar carry at 2/5, but it remains below the 4/5 gate.

## 24. Matched control

X2C uses the same 47-parameter global controller but no old-state carry (`state_next = g_t * candidate`). It also reaches **3/5**: routing 0.804688, recall 0.960938, iterative 0.984375, partial 0.867188, composition 0.742188.

Because X2 and X2C both reach 3/5, adaptive carry specificity is not established.

## 25. Representation adequacy

**FAIL.** Best neutral result: 3/5.

V837x diagnosis: **`GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT`**.

## 26. Capability / data / compute comparison

| Condition | Passes | Unique episodes | Params | Controller params | Controller MACs | Total neutral MACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X0 historical direct | 2/5 | 3,200 | 856 | 0 | 0 | 160 |
| X1 local scalar carry | 2/5 | 3,200 | 1,006 | 150 | 140 | 300 |
| X2 global scalar carry | 3/5 | 3,200 | 903 | 47 | 46 | 206 |
| X2C global scaling control | 3/5 | 3,200 | 903 | 47 | 46 | 206 |

The best new mechanism improves capability while using fewer controller parameters and MACs than ten local carry controllers.

## 27. Sample-efficiency status

**BLOCKED.** Fixed-topology neutral representation never reaches 4/5.

## 28. Structural-search status

**BLOCKED.** No fixed-topology representation pass occurred.

## 29. Primitive-mining status

**BLOCKED.** Neither fixed-topology competence nor structural-search competence is available.

## 30. Fresh-audit status

Fresh-audit episodes consumed: **0**. Reserved seeds 90000–90499 remain unused.

## 31. Strongest scientific claim

The strongest justified claim is:

> Broad temporal coordination is helpful but insufficient. A global scalar signal conditioned jointly on current input and the complete previous neutral state improves the distributed neutral substrate from 2/5 to 3/5, yet fails to reproduce the 4/5 competence of the successful scalarized GRU. Therefore neither scalar output scope nor scalar controller information scope alone explains the remaining representation gap.

This does **not** justify vector gates, global recurrence, larger controllers, more state, more cells, or more data.

## 32. Next single variable

The next single variable is **candidate transformation organization**: one shared/dense candidate transformation versus ten local candidate transformations, while keeping the V837x joint global scalar controller fixed. Input-projection placement must remain fixed during that first test.

## Program resource accounting

Each stage uses the same 3,200 unique `(family, seed)` episodes; they are not re-counted across conditions, replicates, or variants.

```text
V837v  100 fits  19,200 optimizer steps   9,830,400 processed examples
V837w  100 fits  19,200 optimizer steps   9,830,400 processed examples
V837x  100 fits  19,200 optimizer steps   9,830,400 processed examples

Program 300 fits  57,600 optimizer steps  29,491,200 processed examples
Unique seed-defined episodes: 3,200
GPU seconds: 0
```

CPU and wall accounting are stored in the machine-readable program resource files.

## Final locks

```text
representation adequacy       FAIL
sample-efficiency retest      BLOCKED
structural search             BLOCKED
primitive mining              BLOCKED
fresh audit consumed          0
primitives promoted           0
large persistent storage      NOT TESTED
V838                          NOT STARTED
```
