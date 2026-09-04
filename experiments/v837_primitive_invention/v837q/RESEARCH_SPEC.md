# V837q Research Specification

## Question

Does reducing recurrent-state fragmentation restore fixed-topology neutral-substrate representation adequacy under the calibrated 4× unique-data regime?

## Frozen frontier

- V837l: GRU 5/5 at 4× unique data; neutral 2/5.
- V837m: linear transport 2/5; matched additive 1/5.
- V837n/o: successful dense reference requires at least one dynamic state-modulation path, but no named GRU gate is individually necessary.
- V837p: one generic dynamic scalar modulator transfers only to 3/5, matched dynamic additive also 3/5.

The next isolated variable is therefore **state organization**, not another update mechanism.

## Single change

All primary Q conditions preserve 40 recurrent scalar dimensions, the same 10-cell high-capacity graph, the same historical cell parameters and direct tanh update law, the same 4D message transforms, the same 40-wide readout, and the same training/data/gate regime. Only ownership of the 40 recurrent dimensions changes.

| Condition | State owners | Dimensions per owner | Total recurrent dims |
| --- | ---: | ---: | ---: |
| Q0_local_10x4 | 10 | 4 | 40 |
| Q1_group5_5x8 | 5 | 8 | 40 |
| Q2_group2_2x20 | 2 | 20 | 40 |
| Q3_shared_1x40 | 1 | 40 | 40 |

## Shared-state semantics

For every group state `H_g,t`, all member cells read from the same snapshotted state tensor. Cell `i` receives a fixed task-independent 4D view:

`r_i = P_i H_g,t`

where `P_i` is deterministic, has orthonormal rows, is not trainable, and is reconstructed from the frozen projection seed.

The historical cell computation remains:

`candidate_i = tanh(W_s,i r_i + W_m,i message_i + W_x,i x_t + b_i)`

Each candidate contributes back through the fixed transpose:

`contribution_i = candidate_i P_i`

Group proposals are the normalized sum of member contributions. The predeclared normalization is:

`sqrt(group_dim / (member_count * local_view_dim))`

which equals 1.0 for every primary V837q layout because `group_dim = member_count × 4`. This preserves expected per-coordinate frame energy without family-specific tuning.

The group state is committed simultaneously after all pathway candidates have been computed. No cell can read a same-timestep group-state write from another cell. Historical non-recurrent message ordering is retained via sequential 4D candidate outputs; recurrent state ownership remains snapshotted until group commit.

## Parameter control

Q0–Q3 all reuse the same trainable historical parameters. Fixed projections are buffers and do not count as trainable parameters. The 40-input readout is unchanged. Any >10% trainable-parameter drift is a hard implementation failure.

## References

- QR1: task-independent vanilla tanh RNN with hidden size 40. This tests whether a dense globally coupled 40D tanh state is learnable under the same regime.
- QR2: the calibrated hidden-size-13 GRU positive control (875 parameters).

Neither reference is a Transmutor primitive or structural-search candidate.

## Training/data lock

- 512 development episodes: seeds 10000–10511
- 128 validation episodes: seeds 20000–20127
- AdamW
- 192 optimizer steps
- learning rate 0.005
- weight decay 0.0001
- gradient clipping 5.0
- 5 paired initialization replicates
- initialization namespace `v837j-primary-init`
- original V837 family criterion: development >=0.90 and validation >=0.85; representation adequacy >=4/5 families

Fresh-audit seeds 90000–90499 remain forbidden.

## Baseline gate

Q0 must reproduce the preserved V837l 4× neutral baseline. If absolute median validation drift exceeds 0.10 on at least two families, classify `BASELINE_DRIFT` and stop before Q1–Q3 interpretation.

## Diagnostics

For each trained primary condition record:

- validation/development statistics and learning curves
- state norm, covariance effective rank, participation ratio, state correlation
- pathway parameter-gradient norm and within-group gradient cosine alignment
- message-ablation effect
- cell/path contribution intervention effect
- trainable parameter count, fixed projection elements, parameter bytes
- full resource accounting

Conditional controls are only allowed after Q3 completes:

- Q3-NM if Q3 itself reaches >=4/5
- five deterministic projection seeds if Q3 reaches >=4/5
- no grouping evolution or downstream state×modulation factorial in V837q

## Diagnoses

- `STATE_FRAGMENTATION_CRITICAL`: at least one shared condition reaches >=4/5 while Q0 remains near baseline.
- `STATE_SHARING_PARTIAL_BENEFIT`: sharing materially improves competence but no shared condition reaches >=4/5.
- `INTERMEDIATE_MODULARITY_OPTIMAL`: a partially shared condition clearly beats both local and fully shared conditions.
- `STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED`: state sharing does not materially improve the neutral model.

## Locks

`fresh_audit_consumed = false`

`primitive_mining_allowed = false`

`structural_search_allowed = false`

`dynamic_modulation_allowed = false`

`primitives_promoted = 0`

V838 is not started by this experiment.
