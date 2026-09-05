# V837t Frozen Research Specification

## Question

Does successful recurrent computation require dimension-specific dynamic modulation, or can the same vector gate networks be collapsed after sigmoid to one time-varying scalar per pathway without losing representation competence?

## Single variable

`vector gate output -> post-sigmoid mean-and-broadcast output`.

No parameter tensor is removed. No gate network is replaced. Input/state dependence, optimizer, seeds, training budget, GRU equations, hidden size and readout remain fixed.

## Conditions

- T0 full vector update + vector reset
- T1 vector update + reset off
- T2 scalarized update + reset off
- T3 update off + vector reset
- T4 update off + scalarized reset
- T5 scalarized update + scalarized reset

Reset-off is exactly `r=1`; update-off is exactly `z=0`. Reset multiplication remains after the hidden recurrent transform: `r * (W_hn h + b_hn)`.

## Data and optimization

512 development seeds/family, 128 validation seeds/family, five families, 5 paired model replicates, AdamW, 192 steps, lr 0.005, wd 0.0001, grad clip 5.0. Unique family/seed episodes are 3,200 and are reused across conditions and replicates.

## Decision priority

1. T2 >=4/5 -> `DYNAMIC_SCALAR_CARRY`
2. else T4 >=4/5 -> `POST_TRANSFORM_SCALAR_MODULATION`
3. else T5 >=4/5 -> `DUAL_SCALAR_DYNAMIC_PATHWAYS`
4. else, if T0/T1/T3 >=4/5 -> `DYNAMIC_VECTOR_STATE_MODULATION`
5. positive-control failure -> no V837u authorization.

V837u may run exactly the mode written by `diagnostics/decision_state.json`.
