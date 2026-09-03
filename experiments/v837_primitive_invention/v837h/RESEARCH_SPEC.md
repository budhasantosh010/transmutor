# V837h — Low-rank interaction basis

## Why this follows V837g
V837g's generic learned state-update coefficient left conditional routing and variable composition far below the unchanged capacity threshold. After V837d input access and V837g persistence both failed to recover competence, V837h tests one final distinct representational property before the three-variant recovery stop rule.

## Single change
The cell preactivation changes from the historical additive basis to a rank-2 generic interaction basis:

`u = A_s s + A_m m + A_x x`

`v = B_s s + B_m m + B_x x`

`interaction = u * v`

`candidate = tanh(C_u u + C_v v + C_i interaction + b)`

No multiplication operator is exposed to graph search. Rank is fixed at 2.

## Parameter-matched control
The additive control has exactly the same trainable parameter count and uses `C_i (u+v)` instead of `C_i (u*v)`. Thus any multiplicative advantage cannot be attributed simply to more parameters relative to that control.

Input access is historical broadcast; state update is historical direct overwrite. Tasks, seeds, optimizer, training budget, graph topology and capacity gate remain unchanged.
