# V836 Recovery Context

## Provenance rule

This file is a reconstruction from preserved evidence only. No historical artifact was edited. The preserved archive is authoritative. Where the archive does not contain enough information, this document says so instead of guessing.

## Causal sequence

### V833 — hand-compiled second-generation DOT2

**Hypothesis.** A second-generation abstraction, `DOT2 = ADD(MUL(a,b), MUL(c,d))`, can shorten search on larger dot-product-like expressions.

**Method.** Compare a MUL-only library against a library containing the hand-compiled DOT2 abstraction on 12 DOT4 transfer repetitions.

**Result.** Mixed/FAIL overall. MUL-only solved 12/12 with median 2,593 evaluations. DOT2 solved 8/12; successful runs were much cheaper (median 796.5 evaluations in the preserved JSON), but four runs failed under the recorded budget.

**Lesson.** A useful new primitive can reduce search cost while simultaneously increasing branching/reliability risk. More abstraction is not automatically safer search.

### V833B — protected old/new search portfolio

**Hypothesis.** Keep the old and new vocabularies as protected alternatives rather than immediately replacing the old vocabulary.

**Method.** A 50/50 portfolio over MUL-only and DOT2 search.

**Result.** PASS. 12/12 portfolio success; median portfolio compute 1,593; hard regret 2x the best of the two strategies for every preserved repetition.

**Lesson.** A new abstraction can be treated as a new search strategy while retaining the old strategy as protection against abstraction mistakes.

### V834 — automatic repeated-pattern mining

**Hypothesis.** Second-generation abstractions can be discovered from repeated solved call graphs rather than named by a human.

**Method.** Six independently solved curriculum tasks all normalized to `ADD(MUL(V,V),MUL(V,V))`; that repeated structure was automatically promoted to a generic macro and transferred to DOT4.

**Result.** PASS on speed, mixed on standalone reliability. BASE solved 10/10 at median 2,269 evaluations. AUTO_MACRO solved 7/10 at median 723 evaluations.

**Lesson.** Automatic abstraction mining worked in this controlled setting, but the promoted macro created the same branching/reliability problem seen in V833.

### V834B — mined macro behind a protected portfolio

**Hypothesis.** Protecting BASE and automatically mined macro search in parallel should recover reliability without losing all search savings.

**Method.** 50/50 BASE/AUTO_MACRO portfolio.

**Result.** PASS. BASE 10/10, AUTO_MACRO 7/10, portfolio 10/10; median portfolio compute 1,446 versus BASE 2,269; hard regret 2x.

**Lesson.** Automatic abstraction mining plus a protected old/new search portfolio is viable, but this protection cost scales with portfolio size.

### V835 — portfolio explosion

**Hypothesis/question.** Does protected-portfolio robustness remain affordable when the number of candidate primitive libraries becomes large?

**Method.** Compare 12 equally protected candidate libraries with a hand-picked 3-library portfolio over 20 tasks.

**Result.** Finding confirmed. The 12-way equal portfolio had exactly 12x median/mean/max regret. The hand-picked 3-library portfolio solved 20/20 with median regret 4.5x and mean regret about 4.97x.

**Lesson.** Portfolios are robust but do not scale for free. A hierarchy above search strategies is needed to decide which complete vocabularies deserve protection.

### V836 — library-level admission by held-out coverage

**Historical hypothesis.** The same admission principle used below the library level can recurse upward: choose a compact set of complete search vocabularies by development-set reuse/coverage value instead of protecting all libraries equally.

**Preserved method.** Twelve candidate libraries; 48 disjoint development tasks; greedy meta-selection added libraries to minimize development portfolio regret/coverage failure. K=2..5 selected portfolios were then evaluated on unseen test tasks.

**Historical expected result.** The archive does not preserve an explicit pre-run pass-gate expression or historical source code, so the exact expected threshold cannot be recovered without guessing. The preserved narrative clearly expected compact learned admission to control coverage failure and reduce portfolio regret relative to the 12-way equal portfolio.

**Actual result.** Historical status is explicitly **PASS**. K=3 selected `L9_ORXN`, `L10_IMPLX`, `L2_OR`; development mean regret 4.780343649759554 with zero unsolved; unseen-test mean regret 4.713419328623934, median 4.4375, max 9.471503698319722, zero unsolved. References: 12-way equal mean regret 12.0; hand-small3 reference mean regret 4.969338178511565.

**Important recovery problem.** V836 did not historically fail. The current recovery is blocked from satisfying the new reproducibility/completion standard because the registry references `transmutor_v828_v836_experiments.zip` (SHA-256 `dcb0bfc2e4927ee89511ae7f87701ba465f6a7193a838cdb1b2dc0e1c122f953`), but that source archive is not present in the migrated repository or migration ZIP. Only the log and result JSON are preserved. Therefore the original generator, seeds, exact search implementation, budgets, and boolean pass-gate logic cannot be executed exactly.

**Fresh-audit caveat.** The result JSON contains `best_K_on_unseen_test_for_diagnostic: 3` after reporting unseen-test results for K=2..5. Under the new rules, those unseen-test comparisons cannot be treated as a pristine one-shot fresh audit for the final K=3 choice. This does not rewrite historical V836 from PASS to FAIL; it means historical PASS is not yet closed under the stricter post-migration completion gates.

## Evidence paths

- `archive/preserved_artifacts/TRANSMUTOR_V828_V836_LOG.md`
- `archive/preserved_artifacts/transmutor_experiments_v833plus/v833_results.json`
- `archive/preserved_artifacts/transmutor_experiments_v833bplus/v833b_results.json`
- `archive/preserved_artifacts/transmutor_experiments_v834plus/v834_results.json`
- `archive/preserved_artifacts/transmutor_experiments_v834bplus/v834b_results.json`
- `archive/preserved_artifacts/transmutor_experiments_v835plus/v835_results.json`
- `archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json`
- `registry/experiments.jsonl`
- `registry/artifact_inventory.json`
