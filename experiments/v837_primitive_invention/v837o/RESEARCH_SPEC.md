# V837o Research Spec

## Question

Is GRU success under the calibrated 4× unique-data regime caused by dynamic adaptive state control itself, or can static differentiated recurrent pathways recover the same capability?

## Frozen parent evidence

- V837l: GRU 5/5 at 4× unique data; neutral 2/5.
- V837m: stable linear transport 2/5; matched additive 1/5.
- V837n: full 5/5; no-update 5/5; no-reset 5/5; both-off 3/5.

V837n is immutable. V837o imports its explicit GRU implementation but does not edit it.

## Controlled factors

Update pathway: dynamic, static vector, static scalar, off.

Reset/candidate pathway: dynamic, static vector, static scalar, off.

Required conditions G0–G9 are frozen in `config.json` and `frozen_factorial_gate.json`.

Static coefficients reuse retained GRU gate-bias tensor slices. Static slices are reset to logit 0.0 after common initialization, so the initial coefficient is exactly 0.5. This keeps all conditions at the same nominal 875 parameters; active parameter counts are reported separately.

## Matched regime

- CPU only.
- 512 development episodes: seeds 10000–10511.
- 128 validation episodes: seeds 20000–20127.
- 192 AdamW steps, lr 0.005, wd 0.0001.
- Gradient clip 5.0.
- Five paired initialization replicates per condition/family.
- Same five frozen task generators and capacity criterion.
- No task-family label enters the model.

## Positive-control stop

G0 must reproduce at least 4/5 and remain compatible with frozen V837n full-GRU medians. If not, stop as `IMPLEMENTATION_OR_BASELINE_DRIFT`.

If G9 unexpectedly reaches >=4/5, classify `BASELINE_DRIFT_OR_STATISTICAL_INSTABILITY` and block neutral transfer.

## Decision priority

1. If G8 passes, prefer `MULTI_PATH_RECURRENCE_SUFFICIENT`.
2. Else if G3 passes and G4 does not: `STATIC_CARRY_PATH_SUFFICIENT`.
3. Else if G4 passes and G3 does not: `STATIC_CANDIDATE_MODULATION_SUFFICIENT`.
4. Else if G5 passes while G3/G4 fail: `COMPLEMENTARY_STATIC_PATHWAYS_REQUIRED`.
5. Else if G5 passes: `DYNAMIC_GATING_NOT_REQUIRED`.
6. Else if G1/G2 pass while G3/G4/G5 fail: `DYNAMIC_STATE_MODULATION_REQUIRED`.
7. Otherwise: `DIAGNOSTIC_INCONCLUSIVE`.

Only a clear diagnosis can authorize one downstream neutral transfer family.
