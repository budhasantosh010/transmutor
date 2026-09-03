# V837 REPRESENTATION BLOCKER ANALYSIS

## Stop-rule trigger

Three scientifically distinct representation variants failed the unchanged V837 high-capacity competence prerequisite:

1. **V837d — input accessibility:** historical broadcast → deterministic fixed sparse input access.
2. **V837g — state update:** overwrite-style tanh state → one learned scalar state-update coefficient per cell.
3. **V837h — interaction basis:** additive tanh candidate → rank-2 low-rank multiplicative interaction, tested against an exactly parameter-matched additive control.

No variant reached the required `>=4/5` capable task families. The representation-recovery stop rule therefore applies. No combined representation was tested.

## WHAT failed?

The tested generic continuous-cell family did not provide task-independent generalizable capacity across at least four of the five frozen task families.

## WHERE did it fail?

Across all three recovery variants, conditional routing, delayed recall, partial observation, and variable composition remained below the frozen capacity criterion in aggregate. Iterative state remained the only consistently capable family.

## WHEN did it fail?

Failure persisted after the original V837/V837b/V837c lineage had already isolated representation as the strongest blocker, and then persisted through V837d, V837g, and V837h recovery diagnostics.

## WHY is this believed to be a representation-family blocker?

The original structural-search difficulty was substantially removed by using the same fixed 10-cell/55-edge task-independent high-capacity graph. V837d changed only raw-input access and did not recover competence. V837g added only a generic learned persistence coefficient and did not recover competence. V837h changed only the continuous interaction basis; its multiplicative branch and parameter-matched additive control both passed only 1/5 families.

Because these controlled changes target three different representation properties while structural topology, tasks, data splits, optimization budget, and historical capacity gate remain fixed, the evidence no longer supports spending more budget on structural-search breadth or on primitive mining under the same cell family.

## HOW was the failure reproduced?

The historical broadcast refactor exactly reproduced all five preserved blocker-diagnostic comparison scores. The recovery variants then used the same ablation seed region, same 10-cell/55-edge topology, same full-AdamW training schedule, same task generators and same capacity criterion.

## WHAT was tried?

### V837d — sparse input access

Selected 50% fixed sparse access passed 1/5 families. Generic message dependence increased, but general competence did not.

### V837g — learned state persistence

One learned scalar update coefficient per cell converged around mean 0.5503 / median 0.5481. Aggregate validation medians were routing 0.4102, recall 0.7422, iterative 0.9922, partial observation 0.6875, composition 0.5547. Only iterative state passed.

### V837h — rank-2 multiplicative interaction

The multiplicative condition had exactly the same 1,096 trainable parameters as its additive control. Validation medians were routing 0.4648, recall 0.4961, iterative 0.9648, partial observation 0.6797, composition 0.5234. Only iterative state passed. The additive control also passed only iterative state.

## WHAT alternatives were ruled out?

- **Refactor drift:** ruled out by exact broadcast reproduction.
- **Broken benchmark:** the preceding V837 oracle validity tests remain 100% on all five families.
- **Task-label leakage:** the preceding first-observation leakage test remains below its frozen threshold.
- **Raw broadcast alone:** not sufficient; sparse access altered message dependence without restoring capacity.
- **Lack of simple state persistence alone:** not sufficient; V837g failed.
- **Parameter count alone in V837h:** controlled by an exactly parameter-matched additive branch.
- **Structural-search breadth alone:** already weakened by the fixed high-capacity diagnostic used throughout recovery.

## WHAT remains unknown?

The experiments do not establish that every possible neutral continuous-cell substrate is inadequate. They show only that this tested tanh-centered cell family, including the isolated sparse-access, learned-leaky-state, and rank-2 multiplicative variants, is not yet a competent substrate for the required cross-family program.

## Current one-line fix / hypothesis

**Replace or fundamentally revise the generic cell update family as the next single representation variable; do not stack the failed recovery features, resume motif mining, consume fresh-audit seeds, or spend more full structural-search budget first.**
