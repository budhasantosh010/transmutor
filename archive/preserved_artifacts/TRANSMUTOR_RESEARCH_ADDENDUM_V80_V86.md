# Transmutor Research Addendum — V80 through V86

This phase moved the minimal staged-search principle into unrelated domains and tried to falsify it.

The key result was not "adaptive cascades always work." They do not.

The experiments narrowed the principle substantially.

---

# V80 — Cross-domain test: obstacle-grid planning

Search portfolio:

- cheap: Greedy Best-First Search
- repair: Weighted A*
- global: Dijkstra

Escalation used path cost relative to Manhattan distance.

Held-out 20x20 grids:

- GBFS:
  - mean path / optimal = 1.1127
  - mean expansions = 51.1

- Weighted A*:
  - mean path / optimal = 1.0473
  - mean expansions = 53.5

- Dijkstra:
  - optimal
  - mean expansions = 298.1

- adaptive cascade:
  - optimal
  - mean expansions = 260.6
  - repair used 78.6%
  - global used 55.0%

The cascade transferred, but saving was modest because more than half the problems still required global search.

---

# V81 — Exact certificate with no tuned threshold

Cheap GBFS was accepted only when:

    path_cost == Manhattan lower bound

On a unit-cost 4-neighbor grid this is a mathematical certificate of optimality.

Across 18x18, 24x24 and 30x30 grids:

- every final path was exactly optimal
- cheap search could certify only ~11.9% overall
- total staged expansions were ~14.2% WORSE than always running A*

Why:

the cheap search was rarely enough, and its work was mostly extra overhead.

This falsified the idea that a safe cheap-first stage is automatically useful.

---

# V82 — Reuse cheap search as an upper bound

GBFS first produced a feasible path.

Then bounded A* used that path cost as an incumbent upper bound and expanded only nodes whose admissible lower bound could improve it.

All results remained exact.

But:

- GBFS + bounded A* was still ~12.6% slower overall than ordinary A*

Thus even reuse did not help enough in this planning setting.

A* itself was already efficient enough that the extra incumbent-finding stage cost more than it saved.

---

# V83 — Exact break-even law for staged search

Define:

- E = cost of always running expensive search
- C = cheap-stage cost
- s = probability cheap stage is enough
- q = expensive-stage cost on failures divided by E

Normalize:

    c = C / E

Expected staged cost ratio:

    R = c + (1-s)q

Staged search saves cost iff:

    c + (1-s)q < 1

Special case: expensive search restarts from scratch, q = 1:

    c < s

This is an exact accounting identity under the additive expected-cost model.

Empirical check:

V77 sparse search:
- effective c ≈ 0.1087
- s ≈ 0.9786
- observed cost ratio ≈ 0.1301
- condition predicts saving

V81 planning:
- effective c ≈ 0.2611
- s ≈ 0.1189
- observed ratio ≈ 1.1421
- condition predicts failure

The same law explains both.

---

# V84 — Cross-domain SAT cascade

SAT cascade:

1. unit propagation / pure literals
2. WalkSAT
3. exact DPLL

All returned assignments were verified.

Validation selected a WalkSAT budget of 0.

Held-out:

- always DPLL:
  - mean clause-scan proxy ≈ 532.5

- cascade:
  - mean ≈ 618.2
  - ~16.1% WORSE

Finish stages:

- propagation: 54.4%
- WalkSAT: 2.2%
- DPLL: 43.3%

Why did the cascade still lose even though propagation solved over half the instances?

Because DPLL ALREADY begins with propagation.

The external cascade duplicated logic already integrated into the strong solver.

This was an important structural falsification.

---

# V85 — Integrated conditional computation inside SAT search

Compared three exact SAT approaches:

- brute-force complete assignments
- conflict-pruning backtracking
- DPLL with propagation + branching only when unresolved

Mean clause scans:

- brute force: 6,221.8
- conflict-pruning backtrack: 3,150.5
- DPLL: 854.7

Relative to brute force:

- backtracking: 0.5064x
- DPLL: 0.1374x

All found valid satisfying assignments.

The computational gain exists inside the search procedure:
cheap constraint consequences reduce the space before branching.

---

# V86 — Integrated exact heuristic search in planning

Compared:

- Dijkstra
- A* with admissible Manhattan heuristic

Across six size/density settings:

- both always returned exactly the same optimal path cost
- A* reduced node expansions in every setting

Savings:

- 18x18, density .18: 22.8%
- 18x18, density .26: 37.1%
- 24x24, density .18: 24.4%
- 24x24, density .26: 42.6%
- 30x30, density .20: 32.2%
- 30x30, density .27: 49.2%

Overall expansion saving:

    ~35.1%

Thus the same pattern appears in a second exact domain:

    weak exhaustive search
        ->
    integrated lower-bound / heuristic guidance
        ->
    less computation with unchanged correctness

---

# The central hypothesis changed

The V67–V79 phase suggested:

    cheap solver
      ->
    repair
      ->
    expensive solver

V80–V86 showed that this outer cascade is NOT generally the right abstraction.

Planning and SAT both produced cases where a strong solver already integrated cheap reasoning more efficiently.

The narrower surviving architecture-level principle is:

```text
state / unresolved problem
          |
          v
derive cheap local consequences / bounds
          |
          v
which possibilities can still matter?
          |
          +---- impossible / dominated ----> do not compute
          |
          v
expand only unresolved promising possibilities
          |
          v
repeat until certificate / solution
```

This describes:

- DPLL-style propagation + branching
- A*-style lower-bound-guided expansion
- branch-and-bound
- many forms of constraint propagation
- conditional computation more broadly

---

# What is now genuinely "solved" versus unresolved

## Solved under the stated model

### 1. Exact break-even accounting law

    R = c + (1-s)q

and staged search saves expected additive cost iff:

    c + (1-s)q < 1

This is mathematics, not a benchmark-specific empirical claim.

### 2. Safe lower-bound certificate

On unit-cost 4-neighbor grids:

    feasible path cost == Manhattan distance

implies the path is optimal.

Again, mathematical.

### 3. Outer cascades are not universally efficient

We now have direct counterexamples:
- grid planning V81/V82
- SAT V84

Therefore any universal "cheap then expensive is always better" claim is false.

---

## Strong empirical result, but not universal theorem

Integrated informed search can substantially reduce work while preserving exactness.

Observed examples:
- DPLL vs brute force on toy SAT
- A* vs Dijkstra on random grids

These are instances of established classical search principles.

---

## Not established / should not be claimed

- a new general theory of intelligence
- a Transformer successor
- AGI architecture
- a novel adaptive-search principle
- learned repair as necessary
- external cascades as universally efficient

---

# Literature mapping

The surviving ideas strongly overlap existing fields:

- algorithm selection
- rational metareasoning / value of computation
- informed heuristic search
- branch-and-bound
- constraint propagation
- adaptive computation time
- conditional computation / mixture-of-experts
- learned halting / pondering

Therefore future Transmutor work must go beyond merely rediscovering conditional search allocation if it is to make a novel architectural contribution.
