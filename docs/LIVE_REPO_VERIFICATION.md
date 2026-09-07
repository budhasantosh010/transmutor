# Live Repository Verification Index

This is the compact independent-review entry point for `budhasantosh010/transmutor`. The active scientific frontier now extends through V837ab reference-side input-factorization localization and the exactly authorized V837ac minimal neutral controller-input transfer. Canonical committed Git-blob SHA-256 coverage is stored in `verification/active_research_sha256.txt`; the machine-readable artifact index is `verification/live_repo_manifest.json`.

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
- V837y tests exactly the frozen V837x global scalar controller × frozen V837r rank-4 candidate branch, plus the mandatory matched-local capacity control.
- V837z exists only because V837y failed representation adequacy and machine-selected Y3; it changes only historical mixed-stage message timing versus fully synchronous previous-output-only timing.
- V837aa is diagnosis-only: it reruns only the frozen Y3 parent, audits local candidate-law similarity under exact signed-permutation symmetries, and does not implement sharing or any new architecture.
- V837ab proves the T2 6→6 input projection is exactly linearly foldable and then localizes which trainable factorized input pathway is optimization-relevant under matched step-zero functions.
- V837ac exists only because V837ab authorized `TRAINABLE_CONTROLLER_INPUT_FACTORIZATION`; it transfers only that property into the frozen Y3 neutral parent and includes the exact folded control.
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
V837y candidate interaction
Y0 historical                               2/5
Y1 global scalar control                    3/5
Y2 rank4 candidate coupling                 3/5
Y3 global control + rank4 candidate         3/5
Y3C global control + matched local capacity 2/5
        ↓
GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT
        ↓
Machine authorization: Y3 parent only
        ↓
V837z candidate-stage organization
Z0 historical mixed-stage                   3/5
Z1 fully synchronous previous-output-only   2/5
        ↓
HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL
        ↓
candidate-organization hard stop
        ↓
V837aa diagnosis-only candidate-law audit
Y3 parent reproduction                         PASS / exact 3/5
raw common-law threshold fits                  0/25
signed-permutation common-law threshold fits   0/25
stable 2-5 cell-type vocabulary                NONE
gradient compatibility                         MIXED
        ↓
GENUINELY_DIVERSE_CANDIDATE_LAWS
        ↓
V837ab reference input-factorization localization
AB0 exact factorized T2                         4/5
AB1 same-function fully folded                 3/5
AB2 candidate factorized / update folded       4/5
AB3 candidate folded / update factorized       4/5
AB4 frozen shared projection                   2/5
AB5 naive direct                               3/5
        ↓
SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT
        ↓
Machine authorization:
TRAINABLE_CONTROLLER_INPUT_FACTORIZATION only
        ↓
V837ac neutral minimal transfer
AC0 exact Y3 parent                            3/5
AC1 controller-input factorization             3/5
AC1F folded control                            3/5
        ↓
INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT
        ↓
input axis closed
next documented single variable:
CANDIDATE_STATE_GEOMETRY_LOCALIZATION
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

## Current V837y evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837y/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837y/frozen_candidate_interaction_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837y/candidate_interaction.py` |
| Runner | `experiments/v837_primitive_invention/v837y/run_candidate_interaction.py` |
| Analyzer | `experiments/v837_primitive_invention/v837y/analyze_results.py` |
| Raw anchors | `experiments/v837_primitive_invention/v837y/raw/anchor_runs.json` |
| Raw interaction | `experiments/v837_primitive_invention/v837y/raw/interaction_runs.json` |
| Results | `experiments/v837_primitive_invention/v837y/results.json` |
| Decision | `experiments/v837_primitive_invention/v837y/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_candidate_interaction.py` |
| Tests | `tests/test_v837y_candidate_interaction.py` |

Y0 and Y1 reproduce their V837x anchors exactly. Y2 remains compatible with V837r R3. Y3 reaches 3/5 while Y3C reaches 2/5. Cross-cell interventions confirm genuine causal use of the rank-4 branch, but the combination does not restore the frozen >=4/5 representation gate.

## Current V837z evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837z/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837z/frozen_candidate_stage_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837z/candidate_stage.py` |
| Runner | `experiments/v837_primitive_invention/v837z/run_candidate_stage.py` |
| Analyzer | `experiments/v837_primitive_invention/v837z/analyze_results.py` |
| Z0 raw | `experiments/v837_primitive_invention/v837z/raw/z0_runs.json` |
| Z1 raw | `experiments/v837_primitive_invention/v837z/raw/z1_runs.json` |
| Results | `experiments/v837_primitive_invention/v837z/results.json` |
| Decision | `experiments/v837_primitive_invention/v837z/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_candidate_stage.py` |
| Tests | `tests/test_v837z_candidate_stage.py` |

Z0 reproduces Y3 with zero family-median drift. Z1 preserves all 55 graph edges and every parameterized mechanism but forces all messages to read previous outputs. It falls from 3/5 to 2/5, while effective candidate depth changes from historical 1..10 (median 5.5) to uniformly 1.

## Current V837aa evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837aa/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837aa/frozen_candidate_law_gate.json` |
| Alignment implementation | `experiments/v837_primitive_invention/v837aa/candidate_law_alignment.py` |
| Runner | `experiments/v837_primitive_invention/v837aa/run_candidate_law_audit.py` |
| Analyzer | `experiments/v837_primitive_invention/v837aa/analyze_results.py` |
| Initial snapshots | `experiments/v837_primitive_invention/v837aa/raw/initial_parameter_snapshots.json` |
| Trained snapshots | `experiments/v837_primitive_invention/v837aa/raw/trained_parameter_snapshots.json` |
| Rerun rows | `experiments/v837_primitive_invention/v837aa/raw/runs.json` |
| Results | `experiments/v837_primitive_invention/v837aa/results.json` |
| Decision | `experiments/v837_primitive_invention/v837aa/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_candidate_law_alignment.py` |
| Tests | `tests/test_v837aa_candidate_law_alignment.py` |

The 25 regenerated Y3 fits reproduce the committed parent exactly and retain 3/5. The primary recurrent/message candidate core has median raw synthetic cosine/NRMSE 0.5070/0.5373 and empirical 0.4433/0.5709. Exhaustive 384-way signed-permutation alignment improves these only to 0.6121/0.4669 synthetic and 0.6489/0.4317 empirical, still far from the frozen 0.95/0.20 common-law gate. Trained aligned similarity is worse than the aligned initialization null, relative bases are unstable, no k=2..5 reusable type vocabulary passes, and aligned core gradients are mixed. Diagnosis: `GENUINELY_DIVERSE_CANDIDATE_LAWS`.

## Current V837ab evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837ab/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837ab/frozen_input_factorization_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837ab/input_factorization.py` |
| Runner | `experiments/v837_primitive_invention/v837ab/run_input_factorization.py` |
| Analyzer | `experiments/v837_primitive_invention/v837ab/analyze_results.py` |
| Results | `experiments/v837_primitive_invention/v837ab/results.json` |
| Decision | `experiments/v837_primitive_invention/v837ab/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_input_factorization.py` |
| Tests | `tests/test_v837ab_input_factorization.py` |

The full trained T2 projection folds exactly with maximum trace error `1.1920928955078125e-07`. AB0 reproduces T2 at 4/5. AB1 starts from the same effective input function but drops to 3/5, while AB2 and AB3 each recover 4/5. AB4 frozen projection is 2/5 and AB5 naive direct is 3/5. Diagnosis: `SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT`; authorized neutral transfer: `TRAINABLE_CONTROLLER_INPUT_FACTORIZATION`.

## Current V837ac evidence

| Artifact | Path |
| --- | --- |
| Config | `experiments/v837_primitive_invention/v837ac/config.json` |
| Frozen gate | `experiments/v837_primitive_invention/v837ac/frozen_input_transfer_gate.json` |
| Implementation | `experiments/v837_primitive_invention/v837ac/shared_input_transfer.py` |
| Runner | `experiments/v837_primitive_invention/v837ac/run_input_transfer.py` |
| Analyzer | `experiments/v837_primitive_invention/v837ac/analyze_results.py` |
| Results | `experiments/v837_primitive_invention/v837ac/results.json` |
| Decision | `experiments/v837_primitive_invention/v837ac/diagnostics/decision_state.json` |
| Validator | `scripts/validate_v837_input_transfer.py` |
| Tests | `tests/test_v837ac_input_transfer.py` |

AC0 reproduces Y3 exactly at 3/5. AC1 and AC1F are step-zero equivalent within `7.897615432739258e-07`. AC1 remains 3/5, with composition improving from 0.851562 to 0.882812 but partial observation unchanged at 0.773438. AC1F remains exactly at the Y3 family medians. Diagnosis: `INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT`; the input axis is closed.

## Program reports and accounting

- Candidate-interaction/stage report: `docs/V837_CANDIDATE_TRANSFORMATION_INTERACTION_REPORT.md`
- Candidate-organization blocker analysis: `docs/V837_CANDIDATE_ORGANIZATION_BLOCKER_ANALYSIS.md`
- V837y accounting: `experiments/v837_primitive_invention/v837y_resource_accounting.json`
- V837z accounting: `experiments/v837_primitive_invention/v837z_resource_accounting.json`
- Combined candidate-organization accounting: `experiments/v837_primitive_invention/candidate_organization_program_resource_accounting.json`
- Candidate-organization status: `experiments/v837_primitive_invention/candidate_organization_program_status.json`
- Candidate-law alignment audit report: `docs/V837_CANDIDATE_LAW_ALIGNMENT_AUDIT.md`
- V837aa resource accounting: `experiments/v837_primitive_invention/v837aa_resource_accounting.json`
- Candidate-law audit program accounting: `experiments/v837_primitive_invention/candidate_law_alignment_program_resource_accounting.json`
- Candidate-law audit status: `experiments/v837_primitive_invention/candidate_law_alignment_program_status.json`
- Input-factorization localization report: `docs/V837_INPUT_FACTORIZATION_LOCALIZATION_REPORT.md`
- Input-organization transfer report: `docs/V837_INPUT_ORGANIZATION_TRANSFER_REPORT.md`
- Post-input-axis blocker analysis: `docs/V837_POST_INPUT_AXIS_BLOCKER_ANALYSIS.md`
- V837ab accounting: `experiments/v837_primitive_invention/v837ab_resource_accounting.json`
- V837ac accounting: `experiments/v837_primitive_invention/v837ac_resource_accounting.json`
- Combined input-factorization accounting: `experiments/v837_primitive_invention/input_factorization_program_resource_accounting.json`
- Input-factorization program status: `experiments/v837_primitive_invention/input_factorization_program_status.json`

Resource totals for V837y+V837z:

```text
model fits                    175
optimizer steps               33,600
processed examples            17,203,200
unique seed-defined episodes  3,200
environment interactions      778,575
forward calls                 39,175
CPU seconds                   9,840.375
wall seconds (worker sum)     10,630.135340699227
GPU seconds                   0
```

The same 512 development + 128 validation episodes per family are reused across all conditions, variants, and replicates. Unique data therefore remains exactly 3,200 family/seed episodes.

V837aa adds diagnosis-only work on the same frozen Y3 data regime:

```text
model fits                    25
optimizer steps               4,800
processed training examples   2,457,600
unique task episodes          3,200
synthetic probes              102,400
empirical probes              102,400
training forward calls        5,300
diagnostic forward calls      50
diagnostic backward calls     25
CPU seconds (training workers) 1,459.734375
CPU seconds (audit process)   92.21875
GPU seconds                   0
```

Synthetic probes are deterministic non-task diagnostics; empirical probes are drawn only from frozen development trajectories. Neither adds fresh task episodes.

V837ab+V837ac combined resource totals:

```text
model fits                    225
optimizer steps               43,200
processed training examples   22,118,400
unique task episodes          3,200
environment interactions      1,001,025
forward calls                 46,575
backward calls                43,275
CPU seconds (worker sum)      5,684.40625
wall seconds (worker sum)     6,984.907963900245
GPU seconds                   0
```

The same 3,200 family/seed episodes are reused across every V837ab/V837ac condition and replicate.

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

Reserved fresh-audit seeds 90000-90499 remain unused.

## Strongest current claim

V837ab shows that T2's learned 6→6 projection is not a function-class requirement: it folds exactly into the downstream GRU input matrix. However, the matched step-zero comparison demonstrates an optimization effect: exact factorized T2 reaches 4/5 while its fully folded same-function counterpart reaches 3/5, and either candidate-only or controller/update-only trainable factorization independently recovers 4/5. Frozen preconditioning and naive direct initialization are insufficient.

V837ac then transfers the smallest supported property—trainable controller-input factorization—to the best neutral Y3 substrate. The transfer changes the learned controller input geometry and improves composition, but remains 3/5 and leaves partial observation unchanged. Therefore input organization is not the remaining representation blocker for Y3.

The next documented single variable is **candidate/state geometry localization**: one dense 13D candidate/state coordinate geometry versus the partitioned 10×4 local geometry, designed so it does not simply repeat V837q shared-state-only testing. It is not implemented in this program.

## Fast verification

```text
python scripts/verify_live_repo.py
python scripts/validate_active_research.py
python scripts/validate_registry.py
python -m unittest discover -s tests
python scripts/reproduce_v837_recovery.py --variant v837y
python scripts/reproduce_v837_recovery.py --variant v837z
python scripts/reproduce_v837_recovery.py --variant v837aa
python scripts/reproduce_v837_recovery.py --variant v837ab
python scripts/reproduce_v837_recovery.py --variant v837ac
```

The reproduction dispatcher remains dry-run by default and machine-enforces the V837x -> V837y -> V837z -> V837aa -> V837ab -> authorized V837ac frontier.

## Historical SHA anchors

- Frozen V837 gate: `a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`
- Frozen capacity criterion: `7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa`

Use `git rev-parse HEAD` for the final live repository SHA. The verification manifest records the V837ac scientific closure commit separately from later verification-integration commits.
