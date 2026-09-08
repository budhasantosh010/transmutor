# V837 Candidate-Input Sibling Closure Report

## 1. Program question

V837af closes the remaining non-nested input-factorization sibling left open by V837ab/V837ac: does candidate-side trainable input factorization restore neutral Y3 representation adequacy, and does the projection need to be shared across candidate consumers?

The parent is exact `Y3_global_control_rank4_candidate`. Graph topology, historical mixed same-step/recurrent message schedule, rank-4 candidate coupling, global scalar carry controller, local 10x4 state layout, output transforms, readout, data, optimizer, seeds, and 192-step budget are frozen.

## 2. Conditions

- `AF0_y3_parent`: exact Y3 anchor.
- `AF1_shared_candidate_input_factorization`: one trainable 6x6+bias projection shared by all ten candidate consumers.
- `AF1F_folded_candidate_input_control`: algebraic fold of the same AF1 initialization into the direct candidate maps; no runtime projection.
- `AF1D_deshared_candidate_input_factorization`: ten independent 6x6+bias projections, initialized bit-identically to AF1 and untied thereafter.

The ordinary AF1 projection initialization uses the frozen T2/V837ab/V837ac 6x6 Linear initialization law. AF1F is derived exactly from AF1, and AF1D clones the same initial projection ten ways. AF0 alone is the historical-Y3 reproduction anchor.

## 3. Step-zero equivalence

AF1, AF1F and AF1D pass the frozen `1e-6` step-zero equivalence gate. The maximum recorded error across the paired diagnostic replay is `1.791228837477732e-07`.

This establishes that the shared, folded, and de-shared transfer conditions begin from the same effective candidate-input function before optimization while differing in parameterization and tying.

## 4. AF0 parent reproduction

AF0 reproduces the committed Y3 family medians exactly with zero drift:

| family | validation median |
| --- | ---: |
| conditional routing | 0.843750 |
| delayed recall | 0.953125 |
| iterative state | 0.9921875 |
| partial observation | 0.7734375 |
| variable composition | 0.8515625 |

AF0 = 3/5.

## 5. AF1 shared factorization

| family | validation median |
| --- | ---: |
| conditional routing | 0.9921875 |
| delayed recall | 0.9921875 |
| iterative state | 1.000000 |
| partial observation | 0.7890625 |
| variable composition | 0.8359375 |

AF1 = 3/5. Parameter count 1265; projection parameters 42; total recurrent/controller/projection MACs 882/timestep.

## 6. AF1F folded control

| family | validation median |
| --- | ---: |
| conditional routing | 0.906250 |
| delayed recall | 0.984375 |
| iterative state | 1.000000 |
| partial observation | 0.796875 |
| variable composition | 0.843750 |

AF1F = 3/5. Parameter count 1223; runtime projection MACs 0; total recurrent/controller MACs 846/timestep.

The folded condition therefore does not recover adequacy by effective mapping alone.

## 7. AF1D de-shared factorization

| family | validation median |
| --- | ---: |
| conditional routing | 0.937500 |
| delayed recall | 0.984375 |
| iterative state | 1.000000 |
| partial observation | 0.812500 |
| variable composition | 0.875000 |

AF1D = **4/5**. Parameter count 1643; projection parameters 420; projection MACs 360/timestep; total recurrent/controller/projection MACs 1206/timestep.

This is the first neutral condition in the post-Y3 localization chain to restore the frozen representation-adequacy gate.

## 8. Projection specialization

The ten AF1D projections are exactly identical at optimizer step 0, with median pairwise cosine 1.0 and median pairwise distance 0.0. They then specialize during training.

At step 192:

- median pairwise projection cosine: `0.7738974690437317`
- median pairwise projection distance: `1.1804355382919312`
- median mean projection Frobenius norm: `1.64255690574646`
- median mean drift from the common initialization: `0.860247278213501`

The shared AF1 projection also moves substantially, reaching median weight drift `1.010270118713379` by step 192, but shared tying remains insufficient at 3/5.

## 9. Message dependence and partial-observation diagnostics

Median message-ablation success drops are:

- AF0: 0.2578125
- AF1: 0.390625
- AF1F: 0.3515625
- AF1D: 0.453125

AF1D therefore relies more strongly on the historical message pathway while restoring routing/recall/iterative/composition competence. Partial observation remains the one failing family at 0.8125, so the result is 4/5 rather than 5/5.

## 10. Frozen diagnosis

`DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT`

Qualifier: `SHARED_INPUT_BASIS_HARMFUL`.

Safe causal claim: candidate-side trainable input factorization can restore neutral representation adequacy, but forcing all ten candidate consumers to share one trainable input basis is insufficient. Independent candidate-input bases, initialized identically and allowed to specialize, are sufficient under the frozen 4x/192-step regime.

## 11. Hard stop

Representation adequacy is now **PASS 4/5**.

Therefore the architecture-localization hard stop fires immediately:

- V837ag is **not authorized and not created**.
- V837ah is **not authorized and not created**.
- V837ae remains absent.
- structural search remains blocked in this program.
- primitive mining remains blocked.
- fresh-audit consumption remains zero.
- V838 is not started.

The next authorized experiment class is sample-efficiency characterization of the successful AF1D neutral condition at 1x/2x/4x unique data, not another architecture-localization stage.

## 12. Resource accounting

Accepted V837af scientific matrix:

```text
model fits                    100
optimizer steps               19,200
processed training examples   9,830,400
unique seed-defined episodes  3,200
environment interactions      444,900
forward calls                 20,800
backward calls                19,200
CPU seconds (worker sum)      6,820.5
wall seconds (worker sum)     8,689.619448099285
GPU seconds                   0
```

The same 3,200 frozen family/seed episodes are reused across all conditions and replicates. An earlier pre-correction transfer attempt was discarded before scientific acceptance and is not part of these accepted results or accounting.
