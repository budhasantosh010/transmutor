# V837s frozen research specification

## Parent authorization

V837s may exist and run only if `v837r/diagnostics/decision_state.json` records all of:

- `v837r_complete = true`
- `diagnosis = GLOBAL_COUPLING_PARTIAL_BENEFIT`
- `best_condition = R3_rank4`
- `interaction_followup_allowed = true`

The parent decision is frozen evidence and must not be reinterpreted informally.

## Single interaction question

V837p showed local dynamic scalar recurrent-state modulation = 3/5. V837r showed rank-4 cross-block recurrent coupling = 3/5 with an exact local-only control = 2/5. V837s tests only whether these two independently partial mechanisms interact.

Factor A: no global coupling vs rank-4 cross-block-only coupling.

Factor B: no dynamic modulation vs the exact V837p `dynamic_scalar_candidate` mechanism.

A fifth condition, `S3C`, uses the exact V837p parameter-matched dynamic additive control under rank-4 coupling.

## Preserved semantics

- 10 cells × 4 local recurrent dimensions = 40 total state dimensions.
- Historical 55-edge message graph remains active.
- Historical direct tanh update remains active.
- The global term reads the snapshotted previous 40D state only.
- Cross-block diagonal 4×4 blocks are zeroed exactly.
- The dynamic scalar is computed from the same local previous state, message and raw input as V837p.
- In `dynamic_scalar_candidate`, the scalar multiplies only local recurrent-state access before the historical local recurrent matrix. It does not gate the global coupling term.
- In `dynamic_scalar_matched_additive`, recurrent-state access is unmodulated and the same scalar is instead added to the candidate preactivation.
- Same-step global recurrent leakage is forbidden.

## Frozen training regime

AdamW, 192 steps, learning rate 0.005, weight decay 0.0001, gradient clip 5.0, 512 development episodes (10000–10511), 128 validation episodes (20000–20127), five paired replicates, CPU only.

## Frozen outcomes

- `GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INTERACTION`: S3 reaches >=4/5 while S1 and S2 each remain below 4/5; if S3C remains below 4/5 this additionally establishes multiplicative specificity.
- `INTERACTION_RECOVERY_WITHOUT_MULTIPLICATIVE_SPECIFICITY`: S3 reaches >=4/5 but S3C also reaches >=4/5.
- `GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT`: S3 remains below 4/5.

Representation adequacy is the existing V837 >=4/5 gate. No sample-efficiency retest, structural search, primitive mining, V837t or V838 is permitted before this result is analyzed.
