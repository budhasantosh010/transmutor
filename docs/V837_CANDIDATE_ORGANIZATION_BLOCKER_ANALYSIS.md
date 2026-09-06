# V837 Candidate Organization Blocker Analysis

## Trigger

V837y failed the frozen representation gate with `GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT` (Y3 = 3/5), and the machine-authorized V837z stage test also failed: Z0 = 3/5, Z1 = 2/5, diagnosis `HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL`.

No additional candidate-organization experiment is authorized by this program.

## Successful reference versus best final neutral

The successful reference is V837t `T2_scalarized_update_no_reset` (4/5). The best final neutral is V837y Y3 / V837z Z0 (identical forward/training semantics and exactly reproduced medians), which remains 3/5.

| Property | Successful T2 scalarized GRU | Best final neutral Y3/Z0 |
| --- | --- | --- |
| State organization | one dense 13D recurrent state | ten local 4D states, total 40D |
| Candidate transform width | one 13D candidate transform | ten 4D candidate blocks plus rank-4 cross-cell term |
| Candidate parameter sharing | one global candidate parameterization | independently parameterized per-cell candidate transforms plus shared low-rank exchange |
| Candidate recurrence | dense hidden-to-candidate transform over the 13D state | local 4x4 recurrence plus masked rank-4 cross-cell candidate contribution |
| Candidate stage depth | one candidate stage per timestep | historical mixed cascade, effective depths 1..10 |
| Message schedule | no explicit graph-message cascade | 55 historical graph edges; earlier non-recurrent sources may use same-step outputs |
| Input projection | single shared/current-input pathway into the GRU candidate | ten historical per-cell 6→4 input transforms |
| Controller information | current input + complete previous 13D state | current input + complete previous 40D state |
| Controller scope | one scalarized update gate broadcast across 13 dims | one global scalar carry gate broadcast across all ten cells / 40 dims |
| Carry equation | `g*state + (1-g)*candidate` | same preserve-versus-replace convention |
| Activation | tanh candidate | tanh candidate |
| Readout | reference recurrent-state readout | tanh readout over concatenated 40D neutral state |
| Parameters | 875 nominal, 602 active in T2 | 1,223 trainable/active |
| Recurrent/controller MACs | reference GRU active matrix compute; not directly normalized by the neutral accounting convention | 846/timestep under the frozen neutral recurrent/controller accounting |
| Unique data | 3,200 family/seed episodes | 3,200 family/seed episodes |
| Optimizer | AdamW, 192 steps, frozen 4× regime | AdamW, 192 steps, same frozen 4× regime |

## What V837y ruled out

Adding the two independently useful partial mechanisms together did not restore representation adequacy. Y3 retained strong routing and composition simultaneously but remained 3/5:

- routing: 0.84375
- recall: 0.953125
- iterative: 0.9921875
- partial observation: 0.7734375
- composition: 0.8515625

The exact matched local-capacity control Y3C reached only 2/5. This supports real causal use of cross-cell candidate organization relative to matched local capacity, but cross-cell rank-4 organization is still not sufficient.

The Y3 rank-4 branch did not train to zero. Median global/local recurrent norm ratio was 1.8243, median global/message ratio was 1.7157, and intervention on one cell's rank-4 source contribution changed other-cell candidates (median mean absolute delta 0.0979), other-cell next states (0.0852), and the final output (0.0991).

## What V837z ruled out

Making the candidate stage fully synchronous was harmful rather than reparative. Z0 exactly reproduced Y3 at 3/5; Z1 fell to 2/5. Historical effective candidate depth ranged from 1 to 10 (median 5.5), while Z1 forced every cell to depth 1. Therefore the remaining deficit is not explained by the existence of the within-step candidate cascade; under the current substrate that cascade is beneficial.

## Remaining exact differences

Still unresolved, one variable at a time:

1. one shared candidate parameterization versus independently parameterized per-cell candidate transforms;
2. one shared input projection before recurrence versus per-cell input transforms;
3. dense 13D candidate state versus partitioned 10×4 candidate blocks;
4. one candidate nonlinearity versus multiple independent nonlinear blocks;
5. recurrent interaction with the output/readout pathway.

## Next single variable

**Selected later variable: candidate parameter sharing — one shared candidate parameterization versus independently parameterized per-cell candidate transformations, while preserving the 10×4 state layout, the frozen Y3 global controller, rank-4 exchange, historical message schedule, input placement, data, and optimizer.**

This is documentation only. It is not implemented in V837y/V837z.

## Locks

- representation adequacy: FAIL
- sample-efficiency retest: BLOCKED
- structural search: BLOCKED
- primitive mining: BLOCKED
- fresh-audit episodes consumed: 0
- primitives promoted: 0
- V838: NOT STARTED
