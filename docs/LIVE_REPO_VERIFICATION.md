# Live Repository Verification Index

This is the compact independent-review entry point for `budhasantosh010/transmutor`. The active scientific frontier now extends through V837v control-scope localization, V837w successful-reference controller-information localization, and the single machine-authorized V837x global-scalar neutral transfer. Canonical committed Git-blob SHA-256 coverage is stored in `verification/active_research_sha256.txt`; the machine-readable artifact index is `verification/live_repo_manifest.json`.

## Historical boundary

- V836 historical status remains **PASS**.
- V837/V837b/V837c historical artifacts remain immutable.
- V837d/V837g/V837h representation-recovery failures remain frozen.
- V837j–V837m calibration and cell-law results remain frozen.
- V837n/V837o freeze successful-GRU mechanism evidence.
- V837p freezes scalar neutral modulation transfer.
- V837q freezes state-ownership localization.
- V837r/V837s freeze global-coupling localization and interaction results.
- V837t freezes dynamic gate-granularity localization.
- V837u freezes local scalar-carry transfer failure.
- V837v changes only control-domain output scope with fixed local source controllers; no gate pooling is permitted.
- V837w is reference-only and exists only because V837v failed representation adequacy.
- V837x exists only because V837w authorized exactly `JOINT_INPUT_STATE_GLOBAL_SCALAR`.
- V838 has not started.

## Current causal sequence

```text
V837t successful reference
T0 full vector GRU                         5/5
T1 vector update / no reset                5/5
T2 scalarized update / no reset            4/5
T3 no update / vector reset                5/5
T4 no update / scalarized reset            3/5
T5 dual scalarized                         3/5
        ↓
DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED
        ↓
V837u neutral local scalar carry
U0 historical direct                       2/5
U1 scalar candidate modulation             3/5
U2 dynamic scalar carry                    2/5
U2C same-controller scaling control        3/5
        ↓
DYNAMIC_SCALAR_CARRY_INSUFFICIENT
        ↓
V837v control scope only
V0 10 local control domains                2/5
V1 5 shared domains                        2/5
V2 2 shared domains                        2/5
V3 1 globally broadcast LOCAL source       2/5
        ↓
CONTROL_SCOPE_ALONE_INSUFFICIENT
        ↓
V837w successful-reference information source
W0 joint input + state                     4/5
W1 input only                              3/5
W2 state only                              3/5
W3 bias only                               3/5
        ↓
JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED
        ↓
Machine authorization:
JOINT_INPUT_STATE_GLOBAL_SCALAR only
        ↓
V837x neutral transfer
X0 historical direct                       2/5
X1 local scalar carry                      2/5
X2 joint global scalar carry               3/5
X2C same global controller / no carry      3/5
        ↓
GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT
        ↓
scalar-control hard stop
next single variable:
CANDIDATE TRANSFORMATION ORGANIZATION
```

## Current V837v evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837v/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837v/frozen_control_scope_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837v/control_scope.py` |
| Runner | `experiments/v837_primitive_invention/v837v/run_control_scope.py` |
| Analyzer | `experiments/v837_primitive_invention/v837v/analyze_results.py` |
| Raw V0 | `experiments/v837_primitive_invention/v837v/raw/v0_runs.json` |
| Raw shared-scope runs | `experiments/v837_primitive_invention/v837v/raw/shared_scope_runs.json` |
| Results | `experiments/v837_primitive_invention/v837v/results.json` |
| Decision | `experiments/v837_primitive_invention/v837v/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_control_scope.py` |
| Tests | `tests/test_v837v_control_scope.py` |

V0 reproduces historical V837u U2 with zero validation-median drift in all five families. V3 uses one local cell-0 controller broadcast across all ten cells; it does not pool controllers or read global state.

## Current V837w evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837w/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837w/frozen_controller_information_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837w/gru_controller_information.py` |
| Runner | `experiments/v837_primitive_invention/v837w/run_controller_information.py` |
| Analyzer | `experiments/v837_primitive_invention/v837w/analyze_results.py` |
| Raw runs | `experiments/v837_primitive_invention/v837w/raw/runs.json` |
| Results | `experiments/v837_primitive_invention/v837w/results.json` |
| Decision | `experiments/v837_primitive_invention/v837w/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_controller_information.py` |
| Tests | `tests/test_v837w_controller_information.py` |

W0 is numerically anchored to the exact fused T2 computation and reproduces all T2 family medians with zero drift. Dynamic input-only, dynamic state-only, and static bias-only each fall to 3/5, so only the joint source is authorized for transfer.

## Current V837x evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837x/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837x/frozen_global_scalar_controller_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837x/global_scalar_control.py` |
| Runner | `experiments/v837_primitive_invention/v837x/run_global_scalar_control.py` |
| Analyzer | `experiments/v837_primitive_invention/v837x/analyze_results.py` |
| Raw runs | `experiments/v837_primitive_invention/v837x/raw/runs.json` |
| Results | `experiments/v837_primitive_invention/v837x/results.json` |
| Decision | `experiments/v837_primitive_invention/v837x/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_global_scalar_control.py` |
| Tests | `tests/test_v837x_global_scalar_control.py` |

The X2 controller is exactly one scalar:

```text
S_t = concat(previous cell states) ∈ R^40
g_t = sigmoid(w_s^T S_t + w_x^T x_t + b)
```

It has 47 parameters and about 46 controller MACs/timestep. It is computed once before cell execution and never reads partially updated same-timestep states, messages, outputs, or candidates.

## Program reports and accounting

- Final report: `docs/V837_CONTROL_SCOPE_AND_INFORMATION_REPORT.md`
- Scalar-control blocker comparison: `docs/V837_SCALAR_CONTROL_TRANSFER_BLOCKER_ANALYSIS.md`
- Combined resource accounting: `experiments/v837_primitive_invention/control_scope_program_resource_accounting.json`
- Program status: `experiments/v837_primitive_invention/control_scope_program_status.json`

Resource totals for V837v+V837w+V837x:

```text
model fits                    300
optimizer steps               57,600
processed examples            29,491,200
unique seed-defined episodes  3,200
environment interactions      1,334,700
forward calls                 64,775
CPU seconds                   11,564.984375
GPU seconds                   0
```

The 3,200 unique family/seed episodes are reused across conditions, replicates, and variants; repeated processing is not counted as new unique data.

## Locked scientific state

```text
representation adequacy       FAIL
sample-efficiency retest      BLOCKED
structural search             BLOCKED
primitive mining              BLOCKED
fresh-audit episodes consumed 0
primitives promoted           0
large persistent storage      NOT TESTED
V838                          NOT STARTED
```

Reserved fresh-audit seeds 90000–90499 remain unused.

## Strongest current claim

A low-bandwidth global control plane helps but is not sufficient. The smallest reference-justified global scalar observer—joint current input plus the complete previous neutral state—improves the neutral substrate from 2/5 to 3/5, yet does not recover the 4/5 scalarized-GRU reference. Output scope and controller-information scope have therefore both been localized and are not, by themselves, the missing representation property.

The next single variable is **candidate transformation organization**: one shared/dense candidate transformation versus ten local candidate transformations, while keeping the V837x joint global scalar controller fixed and keeping input-projection placement unchanged.

## Fast verification

```text
python scripts/verify_live_repo.py
python scripts/validate_active_research.py
python scripts/validate_registry.py
python -m unittest discover -s tests
python scripts/reproduce_v837_recovery.py --variant v837v
python scripts/reproduce_v837_recovery.py --variant v837w
python scripts/reproduce_v837_recovery.py --variant v837x
```

The reproduction dispatcher remains dry-run by default and machine-enforces the V837v→V837w→V837x authorization tree.

## Historical SHA anchors

- Frozen V837 gate: `a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`
- Frozen capacity criterion: `7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa`

Use `git rev-parse HEAD` for the final live repository SHA. The verification manifest records the V837x scientific closure commit separately from later verification-integration commits.
