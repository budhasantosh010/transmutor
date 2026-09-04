# V837p Research Specification

## Question

Does the shared property localized by V837o — dynamic state modulation — repair the neutral cell without importing a GRU update/reset architecture?

## Scientific authorization

V837o result: `DYNAMIC_STATE_MODULATION_REQUIRED`.

- dynamic update + no reset: 5/5
- no update + dynamic reset: 5/5
- every static single/dual pathway: 3/5
- both off: 3/5

Therefore the smallest justified neutral transfer is one generic dynamic scalar modulator, not a GRU update gate or reset gate.

## New mechanism

For each neutral cell:

```text
g_t = sigmoid(u_s^T s_t + u_m^T m_t + u_x^T x_t + b_g)
conditioned_state = g_t * s_t
candidate = tanh(W_s conditioned_state + W_m m_t + W_x x_t + b)
s_(t+1) = candidate
```

`g_t` is one scalar per cell and time step. It has no semantic task label and no carry path.

## Exact parameter-matched control

The control uses the exact same dynamic scalar network and parameter count but cannot multiplicatively control recurrent-state access:

```text
g_t = sigmoid(u_s^T s_t + u_m^T m_t + u_x^T x_t + b_g)
candidate = tanh(W_s s_t + W_m m_t + W_x x_t + b + g_t)
s_(t+1) = candidate
```

Thus both new conditions contain 1,006 parameters. Historical direct contains 856 and the closest prior state-flow control, V837g scalar persistence, contains 866.

## Conditions

- C0 `historical_direct`
- C1 `scalar_persistence`
- C2 `dynamic_scalar_state_modulation`
- C3 `parameter_matched_dynamic_additive`

## Frozen regime

- fixed V837 high-capacity generic topology
- 512 unique development episodes, seeds 10000–10511
- 128 validation episodes, seeds 20000–20127
- AdamW, 192 steps, lr 0.005, weight decay 0.0001
- gradient clipping 5.0
- five paired initialization replicates per family
- exact historical capacity criterion

## Interpretation

C2 >=4/5 and C3 <4/5:
`GENERIC_DYNAMIC_STATE_MODULATION_SUFFICIENT`

C2 >=4/5 and C3 >=4/5:
`DYNAMIC_MODULATION_SUFFICIENT_MULTIPLICATIVE_SPECIFICITY_UNRESOLVED`

C2 <4/5:
`SHARED_PROPERTY_TRANSFER_FAILURE`

Representation adequacy is a 4×-data fixed-topology diagnostic only. Primitive mining, structural search before a representation pass, and fresh-audit seeds remain locked.
