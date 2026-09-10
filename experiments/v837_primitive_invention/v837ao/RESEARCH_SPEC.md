# TRANSMUTOR — V837ao LATENT PRIMITIVE CANONICALIZATION

## Frozen identity
- Repository: `budhasantosh010/transmutor`
- START SHA: `fe210ae3ee6fad90b394865ce6dd06d525e11902`
- Branch: `research/v837-latent-primitive-canonicalization`
- Predecessor diagnosis: `GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN`
- Powered V837an families: conditional routing, delayed recall, iterative state, variable composition
- V837an winner for each: `STATE40-K1` with semantic compiler
- Partial observation: unresolved/underpowered, never reinterpreted as failure

## Scientific question
Do the validated V837an `STATE40-K1` abstractions define canonical semantic state variables whose meaning, SET operation, residual equivalence class, and transition law remain stable across independently trained organisms?

## Hypotheses
H1 global canonical causal state; H2 phase-conditional canonical state; H3 causal direction but not global coordinate; H4 scalar causal but residual not dispensable; H5 canonical state exists but backend compilation does not generalize.

## Architecture/science locks
No AF1D/source-weight changes, source retraining, structure search, motif mining, routing reopening, rank-4-bus reopening, primitive promotion, fresh-audit use, or V838. New model fits/model optimizer steps/backend gradient steps are all zero. Canonical backends are deterministic linear algebra only.

## Canonical primitive
`P=(Z,U,F,G,Φ)` is substrate-independent. Each organism backend is `B_i=(E_i,W_i)`, where `E_i:S_i→Z` reads canonical state and `W_i:(S_i,Z*)→S_i'` writes it. The reusable object is P, never the organism-specific q vector or microstate.

## Canonical semantic units
- Routing: `z=control∈{-1,+1}`, output `SELECT(control,A,B)`.
- Recall: `z=remembered value∈{-1,+1}`, delay law `z_next=z`.
- Iterative: `z_next=0.65*z+0.35*x`.
- Composition: `z_next=tanh(g*z+d)`, `z∈[-1,1]`.

## Organism holdout split
For each powered family and V837aj engine separately, sort competent IDs lexicographically. Last two are BACKEND_HOLDOUT; all earlier are BACKEND_DISCOVERY. Split is frozen before canonicalization metrics and never performance-sorted.

## Episode partitions
AO_BACKEND_FIT 10000–10063; AO_BACKEND_SELECT 10064–10127; AO_QUOTIENT_FIT 10128–10191; AO_QUOTIENT_SELECT 10192–10255; AO_DYNAMICS_FIT 10256–10319; AO_DYNAMICS_SELECT 10320–10383; AO_META_CONFIRM 10384–10447; AO_HELDOUT_ORGANISM_EVAL 10448–10511; HISTORICAL_VALIDATION_ROBUSTNESS 20000–20127 (reused/descriptive only); FRESH_AUDIT 90000–90499 unused.

## Backend variants and complexity rule
B0 GLOBAL_K1: one q, writer and affine reader for all semantic-active phases. B1 PHASE_GAUGE_K1: same q, phase-specific reader scale/offset and writer scalar normalization. B2 PHASE_K1: phase-specific q/writer/reader. Hard order B0 < B1 < B2; escalation only after simpler family gate failure.

## Reader/writer
Primary reader `E(s)=a q^T s+b`, with FULL40 affine reader diagnostic only. Reader fit uses AO_BACKEND_FIT discovery organisms only. Continuous semantic range is pooled discovery P05/P95; routing/recall range is 2. Raw writer is the V837an semantic compiler direction. Gauge `g=aq`, `γ=g^T w_raw`, canonical `w=w_raw/γ`; `|γ|<1e-6` invalid. SET is `s+w(z*-E(s))`. Algebraic identity/read-after-write/idempotence/overwrite gates are frozen.

## Absolute SET
Binary targets are exactly {-1,+1}; continuous targets are pooled discovery Q10/Q30/Q50/Q70/Q90 from AO_BACKEND_FIT. Skip target/base separations below 0.10 Rz. Required per-organism SET performance: median recovery ≥0.70, direction ≥0.80, task success ≥0.75, median OOD ≤2, and ≥80% target values individually recovery ≥0.60. Canonical setter must beat 32 deterministic Haar controls and shuffled semantics by ≥0.20 recovery.

## Frozen phases
Routing POST_CONTROL/POST_PAYLOAD_A/POST_PAYLOAD_B; recall POST_WRITE/MID_DELAY/PRE_READ; iterative and composition EARLY/MIDDLE/LATE by normalized semantic-update thirds. No phase discovery.

## Quotient/residual test
With canonical writer unit-gain, `P=w g^T`, residual is `(I-P)s`. Donor states are same organism/family/phase, semantic distance ≤0.25 Rz, selected for maximum residual distance under AO_QUOTIENT_FIT. Donor is aligned to anchor semantics with SET, both are then SET to common z* and continued under anchor future context. Gates: median residual sensitivity ≤0.20, p90 ≤0.40, trajectory disagreement ≤0.10 median, task-success disagreement ≤0.10, median OOD ≤2.

## Dynamical commutativity
Primary equation: `E_i(T_i(s,u))≈F(E_i(s),u)`. Natural one-step NRMSE ≤0.10, median normalized absolute error ≤0.075, direction/sign consistency ≥0.90. Interventional one-step error ≤0.10, rollout NRMSE ≤0.15, final abstract-output recovery ≥0.80, task success ≥0.75, random-control margin ≥0.20, OOD ≤2. Horizons 1/2/4/8, with routing/recall phase machines.

## Discovery/META/freeze
Discovery organism passes only when reader, absolute SET, controls, quotient and dynamics all pass. Family requires ≥60% and both engines. Select minimum backend B0/B1/B2. META 10384–10447 is no-refit and no fallback. Surviving specs are frozen and hashed before any held-out-organism backend calibration evidence is read.

## Held-out backend compilation
Frozen algorithm only; no held-out V837an q/compiler/backend artifact reads. Calibration ladder N=1,2,4,8,16,32,64 uses deterministic AO_BACKEND_FIT prefixes. Evaluate only on AO_HELDOUT_ORGANISM_EVAL with no refit. Family pass at N requires ≥3/4 (or ceil(.75*n)) and both engines. Qualifiers: LOW_DATA ≤8, MODERATE=16, EXPENSIVE=32/64. No passing N is HELDOUT_BACKEND_GENERALIZATION_FAILURE.

## Cross-organism canonical agreement
Compare only canonical `z` for matched seeds; never 40D state. Family gate median pairwise normalized disagreement ≤0.10 and p90 ≤0.25 with both engines.

## Historical robustness/law audit
20000–20127 is `REUSED_HISTORICAL_VALIDATION`, descriptive/post-freeze only and cannot select/rescue. Law recovery is non-gating: iterative affine coefficients toward .65/.35/0; composition atanh linearization toward 1/1/0; recall fixed grammar HOLD/AFFINE/ZERO; routing fixed SELECT/always-A/always-B/control-independent mixture.

## Strong success
At least three powered families, including ≥1 discrete and ≥1 continuous, must pass discovery, META, held-out backend compilation, quotient, dynamics and cross-organism agreement. Diagnosis: `GENERAL_CANONICAL_CAUSAL_LATENT_PRIMITIVE_PATTERN`; archive authorization may become true for the next program, but V837ao itself always promotes zero primitives. All four adds `FOUR_FAMILY_CANONICAL_CAUSAL_LATENT_PRIMITIVES`; all B0 adds `GLOBAL_CANONICAL_REALIZATION`; B1/B2 adds `PHASE_CONDITIONAL_CANONICAL_REALIZATION`.

## Closeout locks
Secret scan zero; protected historical diff zero; caches gitignored/reconstructable/hash-manifested; existing live verification system updated; research branch pushed then main fast-forwarded; no force push; remote critical blobs verified.
