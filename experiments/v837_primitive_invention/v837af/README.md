# V837af — Candidate-side Input-Factorization Transfer Closure

V837af closes the remaining non-nested input-factorization sibling left open by V837ab/V837ac.

Parent: exact `Y3_global_control_rank4_candidate`.

Only the candidate-input parameterization may change:

- `AF0_y3_parent`: exact Y3 anchor.
- `AF1_shared_candidate_input_factorization`: one shared trainable 6→6 projection before the already-visible candidate input.
- `AF1F_folded_candidate_input_control`: exact algebraic fold of AF1 at optimizer step zero, with no runtime projection.
- `AF1D_deshared_candidate_input_factorization`: ten independent 6→6 projections, all initialized bit-identically to AF1.

The historical graph, rank-4 candidate coupling, same-step message cascade, global one-sigmoid scalar carry controller, output transforms, readout, state layout, optimizer, data, seeds, and training budget are frozen.

If any transfer reaches ≥4/5 families, architecture localization stops and sample-efficiency characterization becomes the next experiment. V837ag is authorized only if AF1, AF1F, and AF1D all remain below 4/5.
