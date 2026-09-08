# V837af — Candidate-side input-factorization sibling closure

V837af closes the untested V837ab sibling before any new architecture family is opened. It uses exact historical Y3 as parent and changes only candidate-input parameterization.

Conditions: AF0 exact Y3; AF1 shared trainable 6→6 candidate-input factorization; AF1F exact folded direct control; AF1D ten independently trainable factorization copies initialized bit-identically.

All task data, graph/message schedule, rank-4 candidate coupling, global scalar controller, state layout, readout, optimizer and 192-step 4× regime remain frozen. Fresh-audit data, V837ae, V838, structural search and primitive mining are prohibited.
