# Transmutor Experimental Log — V828 to V836

## Goal
Push the architecture beyond fixed primitive libraries and fixed causal contexts.

Main questions:
1. Can primitive libraries adapt over time?
2. Should uncertain search strategies be routed or kept in portfolios?
3. Can causal contexts update online as cells drift?
4. Can a learned continuous organ become a reusable primitive?
5. Can the system discover how to compose that primitive automatically?
6. Can second-generation abstractions emerge automatically?
7. What happens when the number of abstraction/search strategies grows?

---

## V828 — Online primitive birth/retirement/reactivation
FAIL.

Stream:
A = macro-friendly
B = NAND-native
C = macro-friendly again

Simple per-macro recent-utility management:
- STATIC_FULL: 25/26 solved, 452,465 evals
- STATIC_BASE: 14/26, 1,719,883 evals
- ADAPTIVE: 23/26, 721,066 evals

Adaptive system was worse than always keeping the full macro library.

Cause:
utility estimates changed too slowly and local retirement decisions were unstable.

Conclusion:
individual macro keep/drop decisions are not enough.

---

## V828B — Library ecology
FAIL.

Maintained three search vocabularies:
BASE, XOR_LIB, FULL.

Multiplicative weighting became overconfident.
It assigned only ~8% compute to rare strategies.
A rare task requiring a neglected strategy caused huge delay.

- equal portfolio: 713,631 compute, 26/26 solved
- adaptive ecology: ~1,694,581 compute, 26/26

Conclusion:
adaptive strategy weighting can catastrophically underfund rare but necessary strategies.

---

## V828C — Exploration-floor sweep
PASS as boundary test.

Minimum share per strategy:
- 0.05 -> max regret ~20x
- 0.08 -> ~12.5x
- 0.12 -> ~8.33x
- 0.18 -> ~5.56x
- 0.24 -> ~4.17x
- 0.30 -> ~3.33x
- 1/3 equal -> exactly 3x worst-case

Equal allocation was best on this nonstationary stream.

Conclusion:
when future strategy utility is weakly predictable, enforced diversity can outperform adaptive confidence.

---

## V829 — Online context discovery under drift
Invalid/FAIL due benchmark flaw.

Cell population drifted but the useful pair kept fixed IDs.
Frozen initial contexts accidentally continued to keep the pair together.

This test was corrected in V829B.

---

## V829B — Moving useful relations + online contexts
PASS.

96 developmental epochs.
Useful pair itself moved to new cells while latent families drifted.

Results:
- FIXED_CONTEXTS: 28/96 exact
- GLOBAL: 3/96
- ONLINE_CONTEXTS: 76/96

Mean clustering agreement with current latent organization:
- fixed ARI ~0.228
- online ARI ~0.993

Online contexts contained the current useful pair together 96/96.

Conclusion:
causal contexts cannot be fixed taxonomies.
They must update with the evolving population.

---

## V830 — Continuous organ exposed only as a feature
FAIL.

Learned multiplication organ was only appended as an extra feature.

Transfer:
- helped xy+zw
- helped xy-z
- hurt xyz

Cause:
downstream network still had to rediscover multiplication between the organ output and z.

Conclusion:
a compiled organ must become a callable operation, not merely an extra feature.

---

## V830B — Recursively callable continuous primitive
PASS.

Trained neural multiplication organ:
test MSE ~6.34e-6.

Recursive reuse:
- xyz -> g(g(x,y),z)
- xy+zw -> g(x,y)+g(z,w)
- xy-z -> g(x,y)-z

Test MSE:
- xyz ~6.95e-6
- xy+zw ~1.64e-5
- xy-z ~6.10e-6

Fresh scratch MLPs required roughly:
- xyz median 150 training steps
- xy+zw 325
- xy-z 200

Compiled compositions required no task-specific training in this test.

Important caveat:
the call graph was supplied by hand here.

Conclusion:
a learned continuous organ can function as a reusable primitive.

---

## V831 — Automatic call-graph discovery
PASS 24/24.

System received:
- ADD
- SUB
- frozen learned MUL primitive

It saw only examples and searched compositions.

Results:
XYZ:
- 8/8 solved/generalized
- median 63 candidate evaluations
- discovered forms such as MUL(x0,MUL(x1,x2))

XY_PLUS_ZW:
- 8/8
- median 2,671
- discovered ADD(MUL(...),MUL(...))

XY_MINUS_Z:
- 8/8
- median 524
- discovered SUB(MUL(x0,x1),x2)

Conclusion:
the system can discover call graphs over learned continuous primitives from examples.

---

## V832 — Recursive approximation error
PASS, with warning.

Repeated products length 2..12.

Absolute MSE remained small, but normalized error grew as true products became tiny.

Balanced composition trees were more stable than long left chains.

At length 8:
- left-chain MSE ~5.68e-6
- balanced-tree MSE ~2.30e-6

At length 12:
- left normalized MSE ~1.88
- balanced ~0.48

Conclusion:
primitive quality is not enough.
Composition topology affects accumulated abstraction error.

---

## V833 — Hand-compiled second-generation DOT2
Mixed/FAIL overall.

DOT2 = ADD(MUL(a,b),MUL(c,d))

Transfer to DOT4:
MUL-only library:
- 12/12 solved
- median 2,593 evals

DOT2 library:
- 8/12 solved
- median successful cost ~797 evals

Conclusion:
second-generation abstraction can greatly shorten search, but every new primitive also increases branching and can reduce reliability.

---

## V833B — Second-generation library portfolio
PASS.

50/50 portfolio:
- restored 12/12 reliability
- median compute ~1,593 vs 2,593 MUL-only
- hard regret bound 2x oracle

Conclusion:
new abstraction should often create a new search strategy rather than replace the old vocabulary immediately.

---

## V834 — Automatic repeated-pattern mining
PASS on speed, mixed on standalone reliability.

Six independent curriculum tasks were solved from ADD/SUB/MUL.
All six normalized to the same automatically discovered structure:

ADD(MUL(V,V),MUL(V,V))

The system promoted that recurring pattern into a generic new macro without being given the name DOT2.

Transfer to DOT4:
BASE:
- 10/10
- median 2,269 evals

AUTO_MACRO:
- 7/10
- median 723 evals

Conclusion:
second-generation abstractions can emerge from repeated solved call graphs automatically.
But macro branching still creates reliability loss.

---

## V834B — Automatically mined macro behind a portfolio
PASS.

BASE success: 10/10
AUTO_MACRO standalone: 7/10
50/50 portfolio: 10/10

Median compute:
- BASE ~2,269
- portfolio ~1,446

Hard regret bound: 2x.

Conclusion:
automatic abstraction mining + protected old/new search portfolio is viable in this controlled setting.

---

## V835 — Portfolio explosion
Finding confirmed.

12 candidate primitive libraries.

Equal protected portfolio:
- median regret exactly 12x oracle
- mean 12x
- max 12x

Hand-picked 3-library portfolio:
- solved 20/20
- median regret 4.5x
- mean ~4.97x
- max ~12.76x

Conclusion:
portfolios are robust, but portfolio cost scales with the number of protected strategies.
A hierarchy above search strategies is necessary.

---

## V836 — Library-level admission by held-out coverage
PASS.

12 candidate libraries.
48 disjoint development tasks.
Greedy meta-selection chose libraries that minimized portfolio regret/coverage failure.

Unseen test tasks:

K=2:
- selected L9_ORXN, L10_IMPLX
- mean regret ~5.11x

K=3:
- added L2_OR
- mean regret ~4.71x
- median ~4.44x
- no unsolved tasks

K=4:
- mean regret ~5.63x

K=5:
- mean regret ~5.13x

The learned 3-library portfolio beat:
- full 12-way equal portfolio: 12x
- hand generic 3-way portfolio: ~4.97x mean

Conclusion:
the admission principle recurses upward:
cells, organs, primitives, and complete search vocabularies can all be selected by future reuse/coverage value.

---

# Strongest narrowed conclusions from V828-V836

1. Fixed keep/drop macro management is unstable under nonstationarity.
2. Adaptive weighting can become dangerously overconfident.
3. A protected exploration floor gives a hard tradeoff between efficiency and rare-strategy survival.
4. When future utility is poorly predictable, equal diversity can be optimal among tested policies.
5. Causal contexts must update online as the population changes.
6. Learned continuous organs can become callable reusable primitives.
7. A primitive must be callable recursively; using it only as a feature is weaker.
8. Call graphs over learned continuous primitives can be discovered from examples.
9. Approximate primitives accumulate error; abstraction-tree topology matters.
10. Repeated solved structures can be normalized and promoted into second-generation macros automatically.
11. New macros can greatly reduce successful search while simultaneously lowering reliability from branching.
12. Keeping old and new vocabularies in parallel portfolios protects against abstraction mistakes.
13. Portfolios themselves do not scale indefinitely; K equally protected strategies imply up to K-fold compute regret.
14. Search libraries therefore need admission/selection just like primitives.
15. The same developmental pattern appears recursively at multiple levels:

   create candidate
   -> sandbox / evaluate
   -> prove reuse value
   -> admit
   -> keep alternatives alive when uncertain
   -> retire or reduce protection only when evidence is strong

# Current candidate architecture

Experience
  -> evolving cells
  -> online learned contexts
  -> protected relationship-credit portfolio
  -> sandboxed candidate organs
  -> certified organs
  -> callable continuous primitives
  -> automatic repeated-pattern mining
  -> second-generation primitives
  -> competing primitive libraries
  -> library-level admission / protected portfolios
  -> progressive evidence allocation and rescue

The same control logic may recursively govern multiple abstraction levels.

# Still unresolved

- integrate all of these mechanisms in one single continuously evolving population
- discover primitives without separately pretraining a target organ such as multiplication
- scale online contexts and context-graphs to much larger populations
- avoid O(G^2) context-pair growth
- learn portfolio/library structure online without disjoint development sets
- test nonstationary macro retirement across much longer curricula
- test learned primitives outside their training domains
- move beyond synthetic Boolean / low-dimensional continuous tasks
- real hardware accounting
- compare against strong modern neural baselines on matched compute

These remain controlled synthetic experiments.
They do not establish AGI, a Transformer replacement, or a general post-Transformer architecture.
