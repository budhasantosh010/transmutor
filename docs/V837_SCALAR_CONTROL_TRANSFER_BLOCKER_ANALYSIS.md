# V837 Scalar-Control Transfer Blocker Analysis

## Trigger

The mandatory scalar-control sequence is closed:

- V837v: `CONTROL_SCOPE_ALONE_INSUFFICIENT` — 2/5 at 10, 5, 2, and 1 local-source control domains.
- V837w: `JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED` — exact T2 W0 4/5; input-only 3/5; state-only 3/5; bias-only 3/5.
- V837x: `GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT` — X2 joint global scalar carry 3/5; matched X2C global dynamic scaling control 3/5.

Therefore scalar-control experiments stop. No wider scalar MLP, vector gate, global recurrence, extra cells/state, or extra data is authorized.

## Successful T2 GRU versus best failed neutral global-control model

| Property | Successful T2 scalarized GRU | Best neutral X2 global scalar carry |
| --- | --- | --- |
| Families passing | **4/5** | **3/5** |
| State dimensionality | 13 | 40 total |
| State organization | one dense recurrent vector | ten local 4D cell states |
| Candidate transformation | one shared dense 13D GRU candidate transform | ten independent local 4D candidate transforms |
| Input projection | learned shared 6→6 projection before recurrent cell | per-cell 6→4 input transforms |
| Recurrent mixing | dense 13→13 hidden candidate transform | local 4→4 recurrent transforms; cross-cell information only through graph messages |
| Control source | current projected input + full previous 13D state | current raw visible input + concatenated previous 40D neutral state |
| Control scope | one scalar broadcast to all 13 hidden dimensions | one scalar broadcast to all ten cells / 40 state dimensions |
| Control timing | once per timestep from previous state + current input | once per timestep from previous states + current input |
| Carry equation | `z*h + (1-z)*candidate` | `g*S_i + (1-g)*candidate_i` |
| Activation | tanh candidate, sigmoid update | tanh candidate, sigmoid global scalar |
| Readout | linear from 13D final state, then tanh | linear from concatenated 40D final state, then tanh |
| Nominal parameters | 875 | 903 |
| Active parameters | 602 | 903 |
| Global-controller parameters | update pathway contained in GRU tensors | 47 |
| Global-controller MACs | dense GRU update pathway | 46 |
| Neutral recurrent/controller MACs | not directly comparable to neutral accounting | 206/timestep under frozen neutral accounting |
| Unique seed-defined episodes | 3,200 | 3,200 |
| Optimizer | AdamW, 192 steps | AdamW, 192 steps |
| Fresh audit | unused | unused |

## What has already been matched

The remaining deficit cannot be attributed to any of the following without contradicting completed experiments:

1. **Vector-valued dynamic control.** T2 succeeds with one scalar update value broadcast across all 13 dimensions.
2. **Control output scope.** V837v globally broadcasting one local scalar still gives 2/5.
3. **Controller information source.** V837w shows T2 needs joint input+state information; V837x transfers exactly that class of observer.
4. **Controller timing.** X2 computes once from previous state plus current observation before any same-timestep cell update.
5. **Adaptive preserve-vs-replace carry alone.** X2 improves to 3/5 but does not pass; X2C also reaches 3/5, so carry specificity is not established.
6. **More training data or optimizer steps.** Both sides use the same frozen 4× unique-data regime and 192-step optimizer budget.

## Remaining exact structural differences

The most important unmatched structural difference is now the **candidate transformation organization**:

```text
T2 GRU
one dense shared candidate transform
across one 13D recurrent state

versus

X2 neutral substrate
10 separate local candidate transforms
across 10×4D states
with sparse/message-mediated cross-cell interaction
```

A secondary difference is where the input projection occurs: T2 uses one shared 6→6 projection before the recurrent candidate/update paths, while the neutral substrate applies separate 6→4 transforms inside each cell.

## Next single variable

**Candidate transformation organization** is the next variable.

The next experiment, if authorized in a later program, should hold the V837x joint global scalar controller fixed and vary only whether candidate construction remains ten local transforms or is produced by one shared/dense candidate transform over the recurrent substrate. Input-projection placement must remain unchanged during that first test so it does not become a second simultaneous variable.

## Locked state

- Representation adequacy: **FAIL**.
- Sample-efficiency retest: **BLOCKED**.
- Structural search: **BLOCKED**.
- Primitive mining: **BLOCKED**.
- Fresh-audit episodes consumed: **0**.
- Primitives promoted: **0**.
- Large persistent storage tested: **false**.
- V838: **NOT STARTED**.
