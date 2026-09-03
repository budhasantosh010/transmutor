# V837g FAILURE

## WHAT failed?

The generic learned state-update coefficient did not restore the frozen `>=4/5` high-capacity competence prerequisite. Only iterative state passed.

## WHERE?

Aggregate validation medians were conditional routing 0.4102, delayed recall 0.7422, iterative state 0.9922, partial observation 0.6875, and variable composition 0.5547. Routing, recall, partial observation, and composition remained below the unchanged capacity criterion.

## WHEN?

After V837d showed that input-access sparsity changed message dependence but did not recover competence. V837g therefore tested the next isolated representation property under historical broadcast input.

## WHY is this suspected?

Learned coefficients moved away from the historical alpha=1 overwrite behavior (mean 0.5503, median 0.5481), proving that optimization used the new degree of freedom. Despite this, the number of capable families remained 1/5. Simple generic state persistence is therefore not sufficient to explain the missing cross-family capacity.

## HOW was it reproduced?

Eight independently seeded training replicates were run per task family using the same 10-cell/55-edge task-independent high-capacity topology, ablation seed region, full-AdamW schedule, task generators and frozen capacity criterion used by the representation diagnostics.

## WHAT evidence supports the diagnosis?

The learned coefficient adds only one scalar per cell, increasing parameter count from 856 to 866 (+1.17%). It changed the state law without altering graph topology, task distributions, input-access mode or training budget. The four historically weak families still failed.

## WHAT alternatives were ruled out?

- alpha remained fixed at 1: ruled out; learned values converged around 0.55.
- large parameter-count confound: ruled out by the minimal +10 parameters.
- fresh-audit tuning: none; fresh-audit episodes consumed = 0.
- primitive assistance: none; primitives promoted = 0.

## WHAT single change is proposed next?

Test one rank-2 generic multiplicative interaction basis with an exactly parameter-matched additive control (V837h). Do not combine it with sparse input access or learned persistence.

Failure classification: `STATE_UPDATE_FAILURE`, `CAPACITY_WITHOUT_GENERALIZATION`.
