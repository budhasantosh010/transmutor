# V837aa frozen research specification

Parent: `V837y:Y3_global_control_rank4_candidate` at the V837y/V837z closed frontier.

Question: do ten independent Y3 local cells already learn a common candidate law directly, up to exact signed/permuted tanh-compatible coordinates, as a stable 2–5-type vocabulary, or not?

Frozen execution:

- 5 families × 5 replicates = 25 Y3 refits only.
- AdamW, 192 steps, LR 0.005, weight decay 0.0001, gradient clip 5.0.
- development seeds 10000–10511; validation seeds 20000–20127.
- exactly 3,200 unique task episodes reused from V837y.
- exactly 4,096 deterministic synthetic probes and 4,096 deterministic empirical development probes per fit.
- exhaustive 384 signed-permutation alignment on `Ws,Wm,b` only.
- matched initial untrained snapshots are the alignment-search null.
- functional, gradient-direction, global-coupling, relative-basis, and deterministic k=2..5 cell-type diagnostics.

Representation adequacy remains the frozen Y3 parent result (3/5). V837aa cannot unlock structural search or primitive mining and cannot implement V837ab.
