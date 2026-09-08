# V837aj Failure Log

This log records implementation/runtime failures encountered during V837aj. It excludes expected scientific negative results, which belong in the experiment decision artifacts.

## F-001 — Calibration panel duplicate topology
- **What failed:** Preflight `test_calibration_panel_has_12_topologies` and `test_panel_task_independent` failed because P9 had the same `topology_id` as P0.
- **Where:** `experiments/v837_primitive_invention/v837aj/topology.py`, calibration-panel endpoint rewiring.
- **When:** Before any V837aj scientific fit.
- **Why:** The deterministic 15% endpoint-rewiring routine could remove an edge and select the same endpoint again, producing no structural change for P9.
- **How detected:** Dedicated 79-test V837aj preflight suite raised `calibration panel contains duplicate topologies`; direct ID inspection showed `P0 == P9`.
- **What was tried:** Inspected panel IDs/counts and changed endpoint rewiring to choose a legal absent edge distinct from the removed edge.
- **Impact:** None on scientific results; caught before anchor/calibration execution.
- **One-line solution:** Require rewiring replacements to be legal, absent, and structurally different from the removed edge.

## F-002 — Fidelity runner lost partial work on process interruption
- **What failed:** An F1 calibration process exited before the original runner wrote its final 120-row cache file.
- **Where:** `experiments/v837_primitive_invention/v837aj/run_fidelity_calibration.py`, `run_fidelity`.
- **When:** During Stage-A F1 calibration.
- **Why:** The original runner buffered all rows in memory and only persisted them after all workers completed; an external process interruption could therefore discard already-finished rows.
- **How detected:** Process exited nonzero; `raw/fidelity_runs.json` initially contained only F_LEGACY/F0 despite completed F1 stdout rows.
- **What was tried:** Hardened `run_fidelity` to persist the cache after every completed fit and to resume only missing `(panel_id, family, replicate)` rows.
- **Impact:** Some F1 work had to be recomputed once; no thresholds/seeds/model/data/fitness changed.
- **One-line solution:** Checkpoint each completed calibration fit immediately and resume by exact candidate key.

## F-003 — Harness process/session interruption during Stage A
- **What failed:** A tool read returned `Session terminated` while F2 was running.
- **Where:** Harness background-process session, not experiment code.
- **When:** During Stage-A F2 calibration.
- **Why:** External harness/session interruption; the underlying process remained listed as running.
- **How detected:** JSON-RPC `Session terminated`, followed by `list_processes` showing the F2 process still active.
- **What was tried:** Reattached to the existing process and verified the per-fit cache continued increasing.
- **Impact:** No scientific work lost after F-002 hardening.
- **One-line solution:** Reattach to the live process and rely on per-fit persistent cache rather than restarting completed work.

## F-004 — Complete-cache invocation reported nonzero harness status
- **What failed:** A repeated F1 invocation printed `F1: cache complete; not rerunning` but the harness process was later marked stopped/nonzero.
- **Where:** Harness process lifecycle around a completed-cache invocation.
- **When:** After F1 had already reached 120/120 cached rows.
- **Why:** Process-wrapper/session behavior after the cache-complete path; the scientific artifact itself was complete and readable.
- **How detected:** Direct cache inspection showed exactly 120 unique F1 rows and aggregate counts `F_LEGACY=120, F0=120, F1=120`.
- **What was tried:** Verified artifact completeness directly instead of rerunning F1.
- **Impact:** None; no rerun performed.
- **One-line solution:** Treat the hash-valid 120-row cache as source of truth and do not rerun completed fidelity work.
