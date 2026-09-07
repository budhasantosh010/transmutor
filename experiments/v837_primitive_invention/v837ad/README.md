# V837ad — Candidate Recurrent Geometry Localization

V837ad separates hidden width, candidate recurrent geometry, and candidate recurrent fan-in in the successful scalarized-update/no-reset T2 reference.

Execution is physically staged:

1. `AD0_H13_dense` exact T2 anchor.
2. `AD1_H40_dense` width gate.
3. Only if AD1 reaches the frozen >=4/5 representation gate: H40 candidate geometry conditions `AD2`, `AD3`, `AD4`, `AD4S_S0`.
4. Only if AD4 fails and AD4S-S0 passes: predeclared sparse topology robustness S1-S4.
5. V837ae is authorized only by robust `GLOBAL_SPARSE_CANDIDATE_GEOMETRY_SUFFICIENT`.

Only the candidate hidden recurrent slice `W_hn` is masked. Update recurrence stays dense; reset remains off; the T2 trainable 6x6 input projection and post-sigmoid scalarized update remain unchanged. Fresh-audit seeds are never consumed, structural search and primitive mining remain blocked, and V838 is not started.
