# V837af frozen research specification

Question: does the candidate-side trainable input factorization that independently preserves T2 adequacy transfer successfully into Y3?

## Frozen parent

`Y3_global_control_rank4_candidate`: ten 4D states, historical 55-edge mixed same-step/recurrent message schedule, rank-4 cross-cell candidate coupling, one globally informed scalar carry controller, historical per-cell candidate input maps, historical output transforms and 40D readout.

## Conditions

- `AF0_y3_parent`: exact Y3.
- `AF1_shared_candidate_input_factorization`: one trainable 6×6+bias projection shared by all ten candidate consumers; projection is applied only after historical visibility.
- `AF1F_folded_candidate_input_control`: the exact AF1 initialization folded into every cell input map; no runtime projection.
- `AF1D_deshared_candidate_input_factorization`: ten independent 6×6+bias projections initialized bit-identically to AF1.

AF1 uses a frozen parent-preserving factorization initialization `A=I6, a=0`; AF1D copies that initialization into ten independent parameter sets and AF1F folds it. Thus AF1, AF1F and AF1D are exactly the same historical Y3 candidate-input function at optimizer step zero while retaining different trainable parameterizations after training begins. AF1F and AF1D must match AF1 within `1e-6` for the projection diagnostic, candidate input affine term, candidate state, global gate, next state, cell output and prediction.

## Data/training

512 development episodes/family (`10000..10511`), 128 validation/family (`20000..20127`), five families, five replicates, 192 AdamW steps, lr 0.005, weight decay 0.0001, gradient clip 5.0. Unique task episodes remain 3200.

## Decision

Any transfer condition at ≥4/5 is representation adequacy and hard-stops later architecture localization. V837ag is authorized only if AF1, AF1F and AF1D all remain <4/5. Fresh audit, V837ae, V838, structural search and primitive mining are forbidden.
