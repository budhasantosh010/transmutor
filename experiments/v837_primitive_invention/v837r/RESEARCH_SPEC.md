# V837r frozen research specification

## Question

Does direct global cross-dimensional recurrent coupling provide capability that the existing local-cell/message system cannot efficiently express?

## Single variable

Only the topology/bandwidth of recurrent coupling changes. The recurrent state remains local 10×4 (40 total dimensions). Dynamic modulation, shared state, interaction branches, attention, routing, memory, search, mining and audit data remain disabled.

## Primary recurrence

For previous state `S_t = concat(s_1...s_10) ∈ R^40`, a global branch produces `G_t`. Its ten 4D slices are added to the historical preactivation:

`candidate_i = tanh(W_s,i s_i + W_m,i m_i + W_x,i x + g_i + b_i)`.

The primary global matrix has every same-cell 4×4 diagonal block zero. Thus the added branch contains cross-cell recurrence only; historical local recurrence remains in `W_s,i`.

Low-rank matrices are parameterized as `U V^T` and then cross-block masked. The configured factorization rank and the learned masked-matrix effective rank are both reported because masking can increase algebraic rank. Dense coupling is a 40×40 trainable matrix with the same cross-block mask.

## Parameter controls

For a global low-rank factorization rank `r`, the added parameter budget is `80r`. The exact matched local control gives each of ten cells a local factorized 4→r→4 recurrent branch, also `80r` parameters total. The dense global condition adds 1600 nominal parameters; its matched local control uses per-cell 4→20→4 factorized recurrent branches, exactly 1600 parameters total. Controls participate in computation but cannot see another cell's state.

## Execution order and stop rule

1. R0 historical local baseline.
2. Screen R2 rank2, R3 rank4, R5 dense, each with its exact matched local control.
3. If every screened global condition remains ≤2/5 and none beats its matched local control by ≥0.05 mean family validation median, stop. R1/rank1 and R4/rank8 are not run.
4. If a screened condition reaches ≥3/5 or clears the predeclared specificity delta, run rank1/rank8 and their controls to localize the curve.
5. V837s is not automatically executed. Its decision-state permission requires the frozen interaction guard.

## Data and gates

Use the existing 4× unique-development calibration: 512 development episodes, 128 validation episodes, AdamW 192 steps, lr 0.005, weight decay 0.0001, gradient clip 5.0, five paired replicates, unchanged task generators/seeds and unchanged V837 capacity criterion. Representation adequacy remains ≥4/5 families.

## Compute accounting

Historical local recurrent matrix MACs are 160/timestep. Requested coupling-core estimates are 80r for low-rank and 1600 for dense. Because exact cross-block exclusion requires removing same-cell contributions, the implementation separately reports an approximate actual masked low-rank cost of 160r. Dense cross-block has 1440 active matrix entries. Both the requested architectural core estimate and implemented masked estimate are retained rather than hiding this distinction.

## Scientific locks

Fresh audit 90000–90499 remains untouched. Structural search and primitive mining remain false. V838 remains unstarted. Historical V837 through V837q scientific artifacts are immutable.
