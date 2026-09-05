# V837w frozen research specification

Decompose the T2 update logit as `L_input + L_state + B`, apply sigmoid, then mean-scalarize and broadcast. W0 uses input+state, W1 input only, W2 state only, W3 bias only. All conditions retain identical nominal tensors, exact T2 no-reset candidate computation, 4x unique data, 192 AdamW steps, five families and five replicates. W0 must reproduce 4/5. V837x is authorized only by the frozen decision tree and only after V837v failed.
