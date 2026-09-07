# V837ad frozen research specification

## Question

Does successful scalarized recurrent computation require candidate-state integration across a sufficiently broad recurrent coordinate geometry, or can candidate recurrence be partitioned into narrow local blocks?

## Frozen order of causal localization

1. Total recurrent width: H13 dense versus H40 dense.
2. Candidate recurrent geometry at fixed H40: dense, 2x20, 5x8, 10x4.
3. Candidate recurrent fan-in/locality control: 10x4 local degree 4 versus globally distributed degree 4.

No block experiment is interpretable or executable unless H40 dense reaches >=4/5.

## Frozen reference semantics

- exact V837t T2 scalarized-update/no-reset reference
- trainable shared 6x6 input projection
- reset fixed to one and excluded from candidate modulation
- dynamic update sigmoid vector, mean, broadcast
- dense update hidden recurrence in every condition
- only candidate hidden recurrent slice `W_hn` receives a fixed nontrainable mask
- H40 raw tensors are bit-identical across paired family/replicate conditions before masking
- no mask rescaling

## Data and optimization

- 512 development episodes/family
- 128 validation episodes/family
- five families
- 3,200 unique family/seed episodes total
- AdamW, 192 steps, lr 0.005, weight decay 0.0001, clip 5.0
- five replicates
- exact historical task seed ranges and initialization namespace

## Sparse topologies

Five task-independent deterministic 4-regular masks S0-S4 are predeclared in `config.json`. Each must contain exactly 160 candidate recurrent weights, exactly four inputs/output and four outputs/input, no same-4D-block edge, and a strongly connected directed graph.

Only S0 runs initially. S1-S4 run only if AD4 fails and S0 passes.

## Locks

No fresh audit, structural search, primitive mining, V838, persistent-memory claim, post-hoc block sizes, mask scaling, extra data, or longer optimization is authorized by this program.
