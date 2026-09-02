# Transmutor Research Log — V0 through V8

## Core hypothesis
Can useful computation organize itself from small adaptive primitives under finite resources,
rather than requiring humans to specify the final computational architecture?

## Results so far

### V0 — topology can emerge
A population-level evolutionary search grew and pruned a graph that solved AND/OR/XOR/XNOR.
Best graph: 3 hidden cells, 13 connections, 100% accuracy.

### V1 — lifetime learning causes interference
One fixed system learned while its environment changed.
Final: AND 100%, OR 100%, XOR 75%, XNOR 50%.
Main lesson: online learning can overwrite old abilities.

### V2 — state is not automatically memory
Plain recurrent tanh state failed delayed recall at about chance.

### V2b — controlled retention works
Adding learnable retention/write gates produced 100% delayed recall.
Strongest memory cell correlated 0.9963 with the remembered bit.

### V3 — blind growth is not enough
The system grew from 2 to 8 cells on XNOR and still stayed at 75%.
More capacity did not produce the missing computation.

### V3b — directed structural repair helps
Candidate structural repairs were evaluated by actual error improvement.
XOR: 1 -> 2 cells, 75% -> 100%.
XNOR: 1 -> 2 cells, 75% -> 100%.
Easy OR later pruned 2 -> 1 while retaining 100%.
Caveat: current-task pruning destroyed older skills.

### V4 — computation can be priced
With no compute penalty the model used 7.92/12 equivalent cells.
With penalty lambda=0.1 it used 1.28/12 and stayed at 100%.
It spent more computation on XOR/XNOR than AND/OR.
Caveat: this is a toy activation-cost proxy.

### V5 — generic selective memory collapsed
A generic write gate chose the degenerate solution "write nothing".
Recall stayed at chance.
This is an important failure: memory economics can collapse without a route to relevance.

### V5b — addressed selective memory works
Once an explicit key-match relevance signal was available:
- accuracy: 100%
- effective writes: ~1.20 / 20 items
- relevant write gate: 0.848
- distractor write gate: 0.018
Caveat: discovering the addressing mechanism remains unsolved.

### V6 — exact symmetric backprop was unnecessary on XOR
Across 30 seeds:
- exact backprop success: 30/30
- random feedback alignment success: 30/30
Median first-perfect step:
- backprop: 36.5
- random feedback: 33.5
Caveat: tiny task and still uses a broadcast output error.

### V7 — a learning-rule search can rediscover a useful rule
Search over four coefficients:
- random-feedback error
- Hebbian term
- weight decay
- sign-feedback term

Held-out mean accuracy:
- discovered rule: 99.38%
- hand-designed DFA: 99.38%
- pure Hebbian: 62.5%

Interpretation:
The search infrastructure works, but it did not discover a clearly better principle.

### V8 — primitive composition recovers compact exact programs
Genetic programming over:
add, subtract, multiply, max, min, negate, tanh

Discovered:
- AND  = min(x1, x2)
- OR   = max(x1, x2)
- XOR  = -(x1) * x2
- XNOR = x1 * x2

All achieved 100%.
Caveat: the primitive vocabulary was supplied by us.

## Current strongest lessons

1. More cells are not automatically more intelligence.
2. State is not automatically useful memory.
3. Memory needs retention/write control.
4. Structural growth needs credit assignment.
5. Finite-compute pressure can create selective computation.
6. Memory pressure can create degenerate collapse if relevance is unavailable.
7. Exact backprop symmetry is not necessary on at least simple tasks.
8. Search can recover useful learning rules and programs, but so far it mostly rediscovers known principles.
9. The key unsolved step is still open-ended discovery:
   - discovering relevance/addressing
   - protecting old skills while restructuring
   - discovering the update rule itself
   - discovering useful primitives not already supplied

## Scientific status
These are toy experiments. They do NOT establish AGI, ASI, or a successor to Transformers.
Their purpose is to convert the hypothesis into falsifiable subproblems and learn from both success and failure.
