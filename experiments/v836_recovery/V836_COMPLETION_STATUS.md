# V836 completion status under the post-migration rules

Historical V836 remains **PASS**. It is not rewritten.

The stricter recovery program is currently **BLOCKED before repair** because the historical executable source archive is missing. There is therefore no evidence-based V836b repair to run: inventing a failure and then fixing it would violate the historical-integrity rule.

## Gates

- **Gate A — Historical integrity:** PASS so far. Preserved V833–V836 hashes are pinned in `integrity_manifest.json`.
- **Gate B — Baseline reproduced:** BLOCKED — `CANNOT_REPRODUCE_MISSING_SOURCE`.
- **Gate C — Root cause isolated:** NOT APPLICABLE to historical V836 because it is recorded PASS; no new failure may be invented. Any stronger-standard deficiency must first be reproduced from exact source.
- **Gate D — Repair passes original gate:** NOT RUN. Exact pre-run gate logic is not preserved.
- **Gate E — Fresh audit:** NOT SATISFIED by preserved evidence. K=2..5 were all evaluated on the unseen-test set and the JSON records `best_K_on_unseen_test_for_diagnostic=3`; there is no separate pristine audit seed region in the preserved artifact.
- **Gate F — Negative control:** NOT PRESERVED for V836.
- **Gate G — Cost accounted:** PARTIAL. Portfolio regret is preserved; discovery wall time, candidate evaluations, fits, memory, and seed-level resource accounting are not.
- **Gate H — Failure archive retained:** PASS as a repository-integrity property; no historical failure was removed.

## Scientific consequence

The V450–V836 frontier audit can proceed because it is an evidence-classification task over preserved artifacts. A new post-V836 experiment must **not** be executed as though V836 were fully closed under the new standard. The next experiment may be specified and prioritized, but execution is blocked until either the exact V828–V836 source archive is recovered or an explicitly new, separately versioned benchmark is created without pretending to reproduce V836.
