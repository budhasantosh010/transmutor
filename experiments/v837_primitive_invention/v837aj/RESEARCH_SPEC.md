# V837aj frozen research spec

Question: can an automated topology-only discovery process recover competent communication structures on the proven AF1D substrate, and does directed structural search add value over an equal-budget complexity-matched random sampler?

## Immutable substrate

- exact `CandidateInputFactorizationY3`
- condition `AF1D_deshared_candidate_input_factorization`
- 10 cells, 4D state/cell, 40D total recurrent state
- rank-4 cross-cell candidate coupling
- one joint input+full-state global scalar carry controller
- ten untied trainable 6→6 candidate-input projections
- historical candidate laws, input visibility, output transforms, readout
- AdamW, lr 0.005, wd 0.0001, clip 5.0

## Searchable axis

Only message edges. A SAME_STEP edge is legal only for `src < dst`; a RECURRENT edge permits all 100 ordered pairs including self loops. Maximum total message edges: 64. Cell count is fixed at 10.

## Data isolation

- search train: 10000–10383
- search selection: 10384–10511
- final validation: 20000–20127
- final validation is inaccessible during calibration/search/random sampling
- full 512-development retraining occurs only after champion topology freeze

## Stage A

Twelve task-independent topologies, two initialization replicates, F_LEGACY plus F0/F1/F2/F3/F4. F4 is the target ranking. The cheapest of F0–F3 that satisfies every predeclared correlation/order/top-k gate is selected. If none pass, Stage B is blocked and diagnosis is `SEARCH_FIDELITY_PROXY_INVALID`.

## Stage B

If authorized, directed `(μ+λ)` search uses 16 parents, 4 offspring/generation, 12 generations, exactly 64 unique candidate evaluations/run, five runs/family. Constructive search starts from the 19-edge minimal topology and may not seed the 55-edge historical anchor. Each directed slot is paired with one uniformly sampled random topology having the same total and recurrent edge counts and the same candidate initialization slot.

Champions are frozen before final validation, reinitialized from the independent `v837aj-finalize` namespace, retrained from scratch on all 512 development episodes for 192 steps, then evaluated on final validation. Five additional runs/family are allowed only if the frozen borderline trigger fires.

No primitive mining or motif promotion occurs inside V837aj.
