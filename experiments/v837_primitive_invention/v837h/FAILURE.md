# V837h FAILURE

## WHAT failed?

The rank-2 low-rank multiplicative interaction basis did not restore the frozen `>=4/5` high-capacity competence prerequisite. It passed only iterative state, exactly one of five families.

## WHERE?

Multiplicative validation medians were conditional routing 0.4648, delayed recall 0.4961, iterative state 0.9648, partial observation 0.6797, and variable composition 0.5234. The exactly parameter-matched additive control also passed only iterative state.

## WHEN?

After V837d fixed-sparse access failed and V837g learned state persistence failed. V837h was the third scientifically distinct representation recovery attempt.

## WHY is this suspected?

The multiplicative basis improved routing compared with the parameter-matched additive control, but hurt delayed recall and did not increase the number of capable families. This shows a family-specific interaction trade-off rather than a general representational recovery.

## HOW was it reproduced?

Eight paired training replicates per family were run for both the rank-2 multiplicative branch and an exactly parameter-matched additive branch using the same high-capacity graph, task seeds, optimizer, training schedule and frozen capacity criterion.

## WHAT evidence supports the diagnosis?

Both V837h conditions had exactly 1,096 trainable parameters. Therefore the comparison isolates interaction form from trainable parameter count. Neither condition exceeded 1/5 aggregate capable families.

## WHAT alternatives were ruled out?

- simple extra-parameter benefit: ruled out by exact parameter matching.
- one lucky multiplicative seed: eight independent paired replicates per family were used.
- a hidden graph-capacity increase: cell count and internal edge count stayed fixed.
- fresh-audit adaptation: none; fresh-audit episodes consumed = 0.
- primitive assistance: none; primitives promoted = 0.

## WHAT single change is proposed next?

None inside this incremental recovery sequence. V837d + V837g + V837h trigger the three-variant representation stop rule. Close the line and formulate a new falsifiable experiment around a more fundamental change to the generic cell update family rather than stacking these failed features.

Failure classification: `INTERACTION_BASIS_FAILURE`, `CAPACITY_WITHOUT_GENERALIZATION`.
