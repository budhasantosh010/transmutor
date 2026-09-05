# Live Repository Verification Index

This file is the compact entry point for independent review of the live public repository `budhasantosh010/transmutor`. The current verification package indexes the V837 representation-recovery lineage through the V837t successful-reference dynamic-granularity study and its single machine-authorized V837u neutral follow-up. Use `git rev-parse HEAD` for the exact live commit; `verification/live_repo_manifest.json` and `verification/active_research_sha256.txt` provide machine-readable artifact and canonical Git-blob SHA coverage.

## Historical boundary

- V836 historical status: **PASS**; exact historical source reproduction remains separately documented.
- V837/V837b/V837c historical results remain immutable.
- V837d/V837g/V837h representation-recovery failures remain frozen.
- V837j–V837m calibration/cell-law results remain frozen.
- V837n/V837o freeze the successful-reference mechanism evidence.
- V837p is the frozen scalar-modulator neutral transfer.
- V837q freezes state-ownership localization.
- V837r/V837s freeze global-coupling localization and its authorized scalar-modulation interaction.
- V837t interrogates gate-output granularity inside the successful GRU before any larger neutral controller is transferred.
- V837u exists only because the V837t decision state authorized exactly `DYNAMIC_SCALAR_CARRY`; no fallback mechanism was run.

## Current V837t/V837u evidence

| Artifact | Repository path |
| --- | --- |
| V837t config | `experiments/v837_primitive_invention/v837t/config.json` |
| V837t frozen gate | `experiments/v837_primitive_invention/v837t/frozen_dynamic_granularity_gate.json` |
| V837t GRU implementation | `experiments/v837_primitive_invention/v837t/gru_dynamic_granularity.py` |
| V837t runner | `experiments/v837_primitive_invention/v837t/run_dynamic_granularity.py` |
| V837t analyzer | `experiments/v837_primitive_invention/v837t/analyze_results.py` |
| V837t results | `experiments/v837_primitive_invention/v837t/results.json` |
| V837t decision state | `experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json` |
| V837t raw anchor runs | `experiments/v837_primitive_invention/v837t/raw/anchor_runs.json` |
| V837t raw scalarized runs | `experiments/v837_primitive_invention/v837t/raw/scalarized_runs.json` |
| V837u config | `experiments/v837_primitive_invention/v837u/config.json` |
| V837u frozen authorization gate | `experiments/v837_primitive_invention/v837u/frozen_neutral_followup_gate.json` |
| V837u dynamic-control implementation | `experiments/v837_primitive_invention/v837u/dynamic_control.py` |
| V837u runner | `experiments/v837_primitive_invention/v837u/run_neutral_followup.py` |
| V837u analyzer | `experiments/v837_primitive_invention/v837u/analyze_results.py` |
| V837u results | `experiments/v837_primitive_invention/v837u/results.json` |
| V837u decision state | `experiments/v837_primitive_invention/v837u/diagnostics/decision_state.json` |
| V837u raw runs | `experiments/v837_primitive_invention/v837u/raw/runs.json` |
| Program status | `experiments/v837_primitive_invention/dynamic_control_granularity_status.json` |
| Program resource accounting | `experiments/v837_primitive_invention/dynamic_control_granularity_resource_accounting.json` |
| Final report | `docs/V837_DYNAMIC_CONTROL_GRANULARITY_REPORT.md` |
| V837t validator | `scripts/validate_v837_dynamic_control_granularity.py` |
| V837u validator | `scripts/validate_v837_neutral_dynamic_followup.py` |
| Active validator | `scripts/validate_active_research.py` |
| Reproduction dispatcher | `scripts/reproduce_v837_recovery.py` |
| Machine-readable live manifest | `verification/live_repo_manifest.json` |
| Canonical active-research SHA list | `verification/active_research_sha256.txt` |

All earlier evidence remains indexed through `verification/live_repo_manifest.json`; this document emphasizes the current frontier rather than duplicating every historical path.

## Current causal sequence

```text
V837l
GRU @ 4x unique data       5/5
neutral                    2/5
        ↓
V837m transport            2/5
        ↓
V837n/V837o successful-GRU localization
dynamic update/no reset    5/5
no update/dynamic reset    5/5
all-static combinations    3/5
        ↓
DYNAMIC_STATE_MODULATION_REQUIRED
        ↓
V837p scalar neutral modulation        3/5
V837q state sharing                    2/5
V837r rank4/rank8 coupling             3/5
V837s rank4 x scalar modulation        3/5
        ↓
representation adequacy still FAIL
        ↓
V837t reference-side gate scalarization
T0 full vector GRU                     5/5
T1 vector update / no reset            5/5
T2 scalarized update / no reset        4/5
T3 no update / vector reset            5/5
T4 no update / scalarized reset        3/5
T5 dual scalarized                     3/5
        ↓
DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED
        ↓
Machine authorization: DYNAMIC_SCALAR_CARRY only
        ↓
V837u neutral follow-up
U0 historical direct                   2/5
U1 frozen V837p scalar candidate       3/5
U2 dynamic scalar carry                2/5
U2C same-controller scaling control    3/5
        ↓
DYNAMIC_SCALAR_CARRY_INSUFFICIENT
```

The key new result is causal: successful reference computation does not require dimension-specific dynamic **update** values, because a retrained post-sigmoid mean-and-broadcast update gate remains adequate at 4/5. The no-retraining flattening counterfactuals show that already-trained vector models use gate heterogeneity, but retraining proves that heterogeneity is not necessary for successful reference computation. Consequently V837t did not authorize vector neutral control. It authorized only scalar adaptive carry, and V837u showed that this transfer is insufficient in the neutral substrate.

## Unique data and resources

- Development seed-defined episodes/family: **512**.
- Validation seed-defined episodes/family: **128**.
- Task families: **5**.
- Total unique `(family, seed)` episodes: **3,200**.
- The same seed-defined episodes are paired and reused across conditions/replicates; repeated processing is not counted as new unique data.
- V837t: 150 fits, 28,800 optimizer steps, 14,745,600 processed examples.
- V837u: 100 fits, 19,200 optimizer steps, 9,830,400 processed examples.
- Combined: 250 fits, 48,000 optimizer steps, 24,576,000 processed examples, 0 GPU seconds.

## Locked scientific state

- Representation adequacy: **FAIL**.
- Sample-efficiency retest: **BLOCKED**.
- Structural search: **BLOCKED**.
- Primitive mining: **BLOCKED**.
- Fresh-audit episodes consumed: **0**.
- Primitives promoted: **0**.
- Large persistent storage tested: **false**.
- V838: **NOT STARTED**.
- No vector-neutral controller was run because the V837t decision state did not authorize it.
- No post-transform reset transfer, global-coupling combination, shared-state variant, extra-data run, hidden-size increase, or architecture search was run in V837u.

## Current scientific frontier

**Dynamic vector granularity is not established as a required property. Scalar adaptive carry is also insufficient in the neutral substrate. The next experiment must localize the remaining semantic difference between the successful scalarized GRU update pathway and the neutral scalar-carry transfer, one variable at a time.**

## Historical SHA anchors

The established historical scientific anchors remain unchanged. Canonical committed Git-blob SHA-256 coverage is maintained in `verification/active_research_sha256.txt` and `verification/live_repo_manifest.json`.

- Frozen V837 gate: `a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`
- Frozen capacity criterion: `7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa`

## Fast verification

```text
python scripts/verify_live_repo.py
python scripts/validate_active_research.py
python scripts/validate_registry.py
python -m unittest discover -s tests
python scripts/reproduce_v837_recovery.py --variant v837t
python scripts/reproduce_v837_recovery.py --variant v837u
```

The reproduction dispatcher remains dry-run by default. V837u execution is machine-guarded by the committed V837t decision state.
