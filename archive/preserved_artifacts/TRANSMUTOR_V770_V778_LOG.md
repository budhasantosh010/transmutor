# Transmutor Experimental Log — V770 to V778

## Goal of this batch
Resolve the V769B failure:
individually useful cells can form a collectively bad program.

The batch tests:
- interaction-aware structural credit
- local bond / organ formation
- higher-order credit
- adaptive interaction order
- hierarchical compression
- finite-budget stopping
- organ-level survival

---

## V770 — Initial pairwise interaction credit
PASS, but not diagnostic.
On the original relation task, marginal credit already ranked the two true cells first.
Pairwise coalition credit also selected them, but interaction-awareness was unnecessary.

Lesson:
a benchmark must contain misleading marginal credit before it can test coalition credit.

## V770B — Redundant-decoy credit trap
PASS, 5/5.

Environment:
- true cells T1,T2 are complementary; neither solves the task alone.
- decoys D1,D2 each look stronger individually but are redundant.

Results:
- marginal-only choice: D1,D2 in 5/5 runs
- marginal compiled accuracy: ~87.62%
- interaction-aware choice: T1,T2 in 5/5 runs
- interaction-aware compiled accuracy: 100%

Key result:
conditional/pairwise credit can distinguish complementary cooperation from redundant individual usefulness.

## V771 — Local bond / organ formation
PASS, 5/5.

Removed global pair winner selection.

Each cell pair independently maintained:
- interaction mean
- variance
- confidence interval

Bond rule:
mean interaction - 4*SE > 0

Results:
- only T1--T2 formed a confident positive bond
- D1--D2 interaction was strongly negative
- one local organ {T1,T2} formed
- compiled accuracy 100%

Key result:
the correct coalition can emerge as a local bond, without globally ranking all coalitions.

## V772 — Pairwise bonds build a three-cell organ
PASS, 5/5.

True task required T1,T2,T3 together.
Any one or two true cells were insufficient.
Three individually strong decoys were redundant.

Results:
- T1--T2, T1--T3, T2--T3 all formed positive local bonds
- true cells became one connected 3-cell organ
- decoy pair bonds were negative
- exact organ 5/5
- compiled accuracy 100%

Key result:
explicit third-order credit is not necessary when a higher-order coalition decomposes into pairwise conditional relations.

## V773 — Irreducible third-order dependency
PASS, 5/5.

Centered Rademacher perturbation diagnostic:
P(reward=1)=0.5+0.35*s0*s1*s2

In expectation:
- all first-order effects = 0
- all pair effects = 0
- true triple effect = 0.35

Results:
- confident positive pair bonds: none, 5/5
- exact true triple hyperedge {0,1,2}: 5/5
- mean true triple signal ~0.3500
- largest pair signal ~0.00051

Key result:
some dependencies are genuinely higher-order. Pairwise credit cannot universally replace higher-order interaction credit.

## V774 — Adaptive interaction order
PASS, 9/9.

Worlds:
- order 1 dependency
- order 2 dependency
- order 3 dependency

Controller:
1. test order 1
2. escalate to order 2 only if order 1 has no confident structure
3. escalate to order 3 only if lower orders fail

Results:
- exact minimal order and exact hidden structure: 9/9
- order-1 worlds used 8/92 possible hierarchy statistics (~8.7%)
- order-2 worlds used 36/92 (~39.1%)
- order-3 worlds used all 92

Key result:
interaction complexity itself can be allocated adaptively instead of always paying for the highest order.

## V775 — Hierarchical credit compression
PASS, 3/3.

24 raw cells.
Hidden six-cell structure:
- organ A=(0,1)
- organ B=(2,3)
- organ C=(4,5)
- top-level interaction among A,B,C

Flat raw sixth-order interaction space:
C(24,6)=134,596

Hierarchical route:
- raw pair stats C(24,2)=276
- compress 3 discovered pair organs
- one organ-level triple statistic
- total = 277

Results:
- exact 3 pair organs: 3/3
- exact top-level triple: 3/3
- statistic reduction: 99.794%

Key result:
when a world contains lower-level compositional structure, organ formation can turn a very high-order dependency into a small hierarchy.

## V776 — Pure six-way negative control
PASS as a negative control, 3/3.

Reward contained ONLY:
s0*s1*s2*s3*s4*s5

All lower-order terms were zero.

Results:
- no confident pair organs: 3/3
- mean six-way signal ~0.34994
- pair signals ~0.00127 noise scale
- flat sixth-order candidate count: 134,596
- expected uniform random checks without replacement: 67,298.5

Key result:
hierarchical compression is not free. If the environment provides no lower-order compositional clues, the cheap hierarchy stalls.

## V777 — Budget-aware escalation
PASS, 3/3 + 3/3.

Finite interaction-statistic budget: 5,000.

Compositional six-cell world:
- pair organ discovery + organ triple
- solved for 277 statistics, 3/3

Pure irreducible six-way world:
- order 2 spent 276, found nothing
- order 3 spent 2,024, found nothing
- total spent 2,300
- next raw order 4 would cost 10,626 > remaining budget
- controller returned UNRESOLVED_UNDER_BUDGET, 3/3

Key result:
finite-resource intelligence should sometimes stop and remain uncertain instead of exploding computation or inventing a false structure.

## V778 — Bonded organ enters local survival dynamics
PASS, 5/5.

Phase A:
- local bond formation discovers T1+T2 organ.

Phase B:
- bonded component becomes one computational unit
- unbonded decoys remain individual units
- all units use ordinary local participation credit + resource cost

Results:
- true organ survival probability ~0.986
- redundant decoys ~0.37-0.40
- noise cells ~0.044-0.047
- automatic largest-gap pruning kept only ORGAN(T1+T2), 5/5
- compiled accuracy 100%

Key result:
interaction-aware organ formation can repair the exact failure seen in V769B:
marginally useful redundant cells no longer have to be compiled together.

---

# Strongest narrowed conclusions from V770-V778

1. Marginal structural credit is insufficient in environments with complementary and redundant components.
2. Pairwise interaction credit can distinguish complementary cooperation from redundant usefulness.
3. Interaction evidence can be maintained locally by the pair and used to form computational organs.
4. Pairwise bonds can assemble larger organs when higher-order structure decomposes into pairwise relationships.
5. Some tasks contain irreducible higher-order interactions; no universal pairwise shortcut exists.
6. A controller can escalate interaction order only when lower-order evidence fails.
7. Hierarchical organ compression can reduce high-order structural search enormously when the world is compositional.
8. Pure high-order structure remains expensive; this is a genuine no-free-lunch boundary.
9. A finite-resource controller can rationally abstain instead of exceeding its search budget.
10. Once a complementary coalition is fused into an organ, ordinary local survival plus resource pressure can suppress redundant decoys and yield a compact compiled program.

These are controlled synthetic experiments, not evidence of AGI or a Transformer replacement.
