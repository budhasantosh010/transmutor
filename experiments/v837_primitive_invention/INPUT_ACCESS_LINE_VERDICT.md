# INPUT ACCESS LINE VERDICT

## Scope

This line tested whether the V837 representation failure was primarily caused by broadcasting the complete raw observation to every cell.

## Broadcast control

The refactored `broadcast` mode reproduced the preserved high-capacity blocker diagnostic exactly on all five historical first-three validation scores. No family drifted at all, so sparse-input conclusions are not explained by an implementation change.

## Fixed sparse access — V837d

Development-only densities: 12.5%, 25%, 50%, with eight independently seeded masks per density. The nominal 12.5% condition had an effective density of 16.67% because every cell and every input dimension had to remain connected.

The frozen aggregate selection rule chose 50% density. It still passed only 1/5 capacity families:

- conditional routing: validation median 0.4141 — FAIL
- delayed recall: 0.6641 — FAIL
- iterative state: 0.9766 — PASS
- partial observation: 0.6914 — FAIL
- variable composition: 0.5273 — FAIL

Sparse access increased generic message dependence relative to historical broadcast, but this did not translate into general competence.

## Shuffled sparse control

A degree-preserving shuffled sparse control also passed only 1/5 families. This rules out the claim that the selected exact sparse mask revealed a broadly sufficient hidden topology.

## No-message control

With the selected sparse input mask retained but cell-to-cell messages disabled, only 1/5 families passed. Some family scores moved in both directions, which means input restriction has regularization effects, but the sparse condition did not establish message-mediated computation as the missing general representation property.

## Evolvable access / ingress

V837e and V837f were **not run**. The fixed-sparse line did not meet the specified directional evidence threshold needed to justify spending additional search on evolving input edges or adding mediated ingress.

## Verdict

`INPUT_ACCESS_NOT_SUFFICIENT_TO_EXPLAIN_REPRESENTATION_FAILURE`

Input accessibility changes computation, but the evidence does not support continuing the input-access branch as the primary repair. The scientifically justified next isolated variable was the generic cell update law.
