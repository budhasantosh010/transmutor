# V837af Frozen Research Specification

Question: does the candidate-side trainable input factorization that independently preserved T2 adequacy transfer successfully into Y3?

## Frozen parent

`Y3_global_control_rank4_candidate`: 10 cells × 4 state dimensions, 40D total state, historical 55-edge mixed same-step/recurrent graph, historical per-cell candidate equations, rank-4 cross-cell candidate coupling, one globally informed scalar carry controller, historical output transforms and 40D readout.

## Conditions

`AF0`: exact Y3.

`AF1`: one trainable shared 6×6 projection plus 6-vector bias, applied only after historical visibility masking and before each cell candidate input map. On broadcast Y3 input it is computed once per timestep.

`AF1F`: initialize AF1, then fold `W_eff = W_x A` and `b_eff = b + W_x a` into each candidate map; no runtime projection.

`AF1D`: ten independent 6×6+bias projections initialized bit-identically to AF1; no tying after initialization.

AF1/AF1F and AF1/AF1D must match projected visible inputs, candidate input terms, candidates, global gate, states, outputs, and predictions within 1e-6 before training.

## Frozen regime

512 development episodes/family, 128 validation episodes/family, seeds 10000–10511 and 20000–20127, five families, five replicates, 192 AdamW steps, lr 0.005, weight decay 0.0001, gradient clip 5.0. Exactly 3200 unique seed-defined episodes; fresh audit seeds unused.

## Decisions

- AF1F ≥4/5 → `CANDIDATE_COMPOSED_EFFECTIVE_INPUT_MAPPING_SUFFICIENT`.
- AF1 ≥4/5 and AF1D ≥4/5 with AF1F <4/5 → `CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT` + `SHAREDNESS_NOT_ESTABLISHED`.
- AF1 ≥4/5 and AF1D <4/5 with AF1F <4/5 → `SHARED_CANDIDATE_INPUT_FACTORIZATION_SPECIFICALLY_SUFFICIENT`.
- AF1 <4/5 and AF1D ≥4/5 with AF1F <4/5 → `DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT` + `SHARED_INPUT_BASIS_HARMFUL`.
- all three transfer conditions <4/5 → `CANDIDATE_INPUT_FACTORIZATION_TRANSFER_INSUFFICIENT` and authorize V837ag.

Any neutral ≥4/5 result hard-stops architecture localization. Structural search and primitive mining remain blocked in this program.
