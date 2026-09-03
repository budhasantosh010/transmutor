# V837d — Fixed Sparse Raw-Input Access

## Question
Can the existing neutral `tanh` cell family recover generalizable high-capacity competence when the only representation change is which raw observation dimensions each ordinary cell may directly access?

## Why this follows V837c
V837/V837b/V837c failed the full competence prerequisite. A strong fixed-topology diagnostic then showed that the current substrate could fit development data but did not generalize across at least four families. The first isolated representation hypothesis is that broadcasting the complete raw observation to every cell encourages correlated shortcut fitting and reduces pressure for message-mediated computation.

## Single change
Historical: every cell receives every raw observation dimension.

V837d: a deterministic graph-level binary mask selects raw dimensions visible to each cell. The masked vector remains full-sized, so `W_x` dimensionality and parameter count are unchanged.

No cell class, gate, router, attention mechanism, memory operator, task label, optimizer, task generator, success metric, high-capacity topology, state/message dimension or training budget changes.

## Conditions
- B0 broadcast: all raw dimensions visible to all cells.
- B1 fixed sparse: requested densities 12.5%, 25%, 50%, eight task-independent mask seeds per density.
- B2 shuffled sparse: degree-preserving rewiring of the selected sparse masks.
- B3 no-message: selected sparse mask retained, while all cell-to-cell message terms are forced to zero during training and evaluation; recurrent state remains active.

## Frozen selection
Select density by (1) highest median number of families satisfying the historical capacity criterion per replicate, then (2) highest worst-family median validation success, then (3) lowest requested density.

A family satisfies the aggregate density screen iff its median development success is >=0.90 and median validation success is >=0.85 across the eight paired replicates. Representation recovery requires >=4/5 families.

## Historical gate source
The original V837 gate file remains `experiments/v837_primitive_invention/frozen_gates.json`, SHA-256 `a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`.

The high-capacity criterion is the exact blocker criterion: development >=0.90 and validation >=0.85, with >=4/5 families required. Its canonical fingerprint is `7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa`.

## Fresh-audit guard
Seeds 90000–90499 remain prohibited. Primitive promotion and scientific motif mining remain blocked regardless of the V837d capacity result. A successful capacity screen only justifies rerunning full neutral structural search with a frozen recovered representation.
