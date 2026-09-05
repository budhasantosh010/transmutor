# V837u Frozen Research Specification — DYNAMIC_SCALAR_CARRY

Authorization source: `v837t/diagnostics/decision_state.json`.

Primary condition:

`candidate = tanh(W_s s + W_m m + W_x x + b)`

`g = sigmoid(u_s·s + u_m·m + u_x·x + b_g)`

`s_next = g*s + (1-g)*candidate`

The controller is exactly the scalar state/message/input-conditioned controller frozen in V837p. The candidate recurrent term itself is not gated.

Control:

`s_next = g*candidate`

This preserves controller parameters and dynamic scaling while eliminating old-state carry.

Paired context conditions rerun historical direct and V837p scalar candidate modulation. All use the frozen 4× data/training regime. Representation adequacy remains >=4/5 families.
