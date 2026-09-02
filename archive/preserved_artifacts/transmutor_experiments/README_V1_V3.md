# Transmutor experiments — V1 to V3

## V1 — Lifetime learning
One fixed 7→8→1 network remained alive while the task stream changed:
AND → OR → XOR → XNOR → AND → XOR.

Final result:
- AND: 100%
- OR: 100%
- XOR: 75%
- XNOR: 50%

Interpretation:
The system can update during its lifetime, but shared-weight learning causes interference and forgetting.

## V2 — Plain internal state
A vanilla recurrent tanh cell system was tested on 14-step delayed recall.

Result:
- Stateless: ~50%
- Vanilla recurrent state: ~50%

Interpretation:
Simply adding recurrent state did not preserve information.

## V2b — Gated memory
The same delayed-recall task was tested with cells that learned retention and write gates.

Result:
- Stateless: 49.83%
- Gated stateful cells: 100%
- Strongest memory-cell correlation with remembered bit: 0.9963
- Mean learned retain gate: ~0.915

Interpretation:
State needs a mechanism that controls what is preserved and what is overwritten.

## V3 — Structural plasticity during one lifetime
Task stream:
OR → XOR → AND → XNOR → OR

The same system could add/prune hidden computational cells.

Observed structural events:
- XOR: 1 → 2 cells; XOR reached 100%
- XNOR: repeated growth 2 → 8 cells, but remained stuck at 75%
- Final OR: pruning 8 → 7 → 6 while retaining 100%

Interpretation:
Growth/pruning works, but simply adding capacity is not enough to discover the right computation.
The next architecture needs smarter structural credit assignment: what should grow, where should it connect, and why?

## Scientific status
These are toy experiments, not evidence for AGI or a new computing paradigm.
Their value is that they turn the hypothesis into measurable failures/successes that can guide the next versions.
