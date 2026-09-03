# V837g — Generic state-update coefficient

## Question
Does generic tunable state persistence provide representational capacity missing from the original overwrite-style tanh state?

## Why V837g follows V837d
V837d showed that fixed sparse raw-input access increased generic message dependence but did not restore competence and did not justify further sparsity evolution. The next isolated variable is therefore the cell state-update law.

## Single change
Historical: `s_next = tanh(Ws*s + Wm*m + Wx*x + b)`.

V837g: `candidate = tanh(...)`, `alpha = sigmoid(a)`, `s_next = (1-alpha)*s + alpha*candidate`, with one learned scalar `a` per ordinary cell.

No input-conditioned gate exists. `alpha` is called a state-update coefficient, not a memory gate.

## Frozen controls
The alpha=1 historical control is the exact V837d broadcast rerun, which reproduced the preserved blocker scores with zero drift. V837g uses the same 10-cell/55-edge graph, ablation seeds, optimizer, training budget, tasks, metrics, and original capacity criterion. Raw input access is broadcast.

## Gate
Representation recovery candidate only if at least 4/5 families satisfy the unchanged capacity criterion. This is not a primitive-invention pass and cannot reopen primitive mining by itself.
