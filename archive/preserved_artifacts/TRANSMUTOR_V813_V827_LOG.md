# Transmutor Experimental Log — V813 to V827

## Goal of this batch
Stop endlessly adding mechanisms and instead narrow the smallest defensible core.

Main questions:
1. Which mechanisms are actually necessary?
2. Can causal contexts be discovered rather than hand-labeled?
3. Can the number of contexts be discovered?
4. What happens when behavioral contexts are causally wrong?
5. Can hierarchical search reduce context-pair cost?
6. Can the system invent reusable primitives/macros?
7. Should learned macros be globally active or context-gated?

---

## V813 — Minimal-core ablation
FULL:
- synergy 5/5
- coverage 5/5
- temporal 5/5
- total 15/15

NO_PORTFOLIO:
- 5/15
- synergy survived, coverage and temporal failed.

NO_PROTECTED:
- 9/15
- ordinary behavior starved / distorted causal evidence.

NO_SANDBOX:
- 10/15
- cheap top-score trust catastrophically failed synergy.

NO_WIDEN:
- 15/15
- widening was unnecessary on these easy versions.

Narrowed conclusion:
relationship-credit portfolio, protected intervention, and sandbox proof are core in this benchmark.
Adaptive widening is conditional rather than universally required.

---

## V814 — Proposal difficulty
Hard 80-cell global synergy screen.

At 240 cheap audit steps:
- top5 exact 4/20
- top15 exact 6/20
- top40 exact 9/20

At lower evidence budgets even top40 often failed.

Conclusion:
widening helps only if the cheap proposal mechanism contains meaningful signal.
When proposal evidence is nearly random, widening cannot rescue it cheaply.

---

## V815 — Context-local causal screening
Same hard 80-cell problem.

GLOBAL, 40 proposals:
- 15/30 exact

LOCAL causal contexts:
- only 8 total proposals
- 30/30 exact

The true relationship ranked #1 inside its local context in all 30 runs.

Conclusion:
separating causal contexts can improve signal far more than blindly widening a global shortlist.

---

## V816 — Learned contexts without human labels
Cells had unlabeled neutral behavioral signatures.
KMeans discovered causal contexts.

Signature noise 0.4:
- mean ARI 1.00
- GLOBAL 0/8
- RANDOM contexts 1/8
- LEARNED contexts 8/8

Noise 0.8:
- mean ARI ~0.989
- GLOBAL 1/8
- LEARNED 7/8

Noise 1.2:
- mean ARI ~0.798
- GLOBAL 0/8
- LEARNED 6/8

Conclusion:
useful causal context separation can emerge from behavior rather than explicit operator labels, but degrades as behavioral family structure becomes ambiguous.

---

## V817 — Automatic number of contexts
True latent context counts G ∈ {3,4,6}.
Learner tried k=2..8 and chose by silhouette score.

Selected exact k:
- G=3: 6/6
- G=4: 6/6
- G=6: 6/6

But relationship proposal width was fixed at top2/context:
- G=3 auto-context success 0/6
- G=4 5/6
- G=6 6/6

Conclusion:
the system can infer context count, but large contexts require more search/evidence.

---

## V817B — Auto contexts + adaptive widening
Proposal width: 2 -> 5 -> 10.

Results:
- G=3: 7/10
- G=4: 10/10
- G=6: 9/10

Conclusion:
widening helps broad contexts but does not fully compensate for insufficient causal evidence.

---

## V818 — Context-size adaptive audit effort
Causal screen effort scaled with context size/relationship count.
Proposal widening 2 -> 5 -> 10 -> 20.

Results:
- G=3: 11/12
- G=4: 11/12
- G=6: 12/12

Larger contexts received more audit samples.
Most smaller-context runs stopped at width 2.

Conclusion:
resource allocation should depend on the size/uncertainty of the local search space.

---

## V819 — Hidden relationship type + hidden context count
System was not told:
- relationship type: synergy / coverage / temporal
- context count: 3 / 4 / 6
- temporal lag

Pipeline:
neutral signatures -> auto context count -> learned contexts -> relationship-credit portfolio -> context-size audit effort -> adaptive widening -> sandbox proof.

Results:
18/18 exact.

Static worlds selected static proof route.
Temporal worlds selected temporal route.
Context counts were correctly recovered in all tested runs.

Conclusion:
multiple adaptive mechanisms can cooperate without supplying the hidden relationship type or context count.

---

## V820 — Causally wrong behavioral contexts
Negative control.

ALIGNED:
- within-context search 12/12.

MISALIGNED:
behavioral signatures clearly separated the true causal pair into different contexts.

Results:
- WITHIN_ONLY: 0/12
- WITHIN + global top40: 3/12
- context-graph top2/context-pair: 3/12

Important:
perfect clustering can still be causally wrong.
Behavioral similarity must not become a hard causal law.

---

## V820B — Adaptive context-graph search
Search every learned context-pair as its own protected causal context.
Proposal width 2 -> 5 -> 10 -> 20.

Misaligned world:
- exact 18/20
- mean protected pair proofs 54.5
- raw pair space = 1128

Conclusion:
overlapping/cross-context causal search is a strong structured fallback when learned contexts are wrong.

---

## V821 — Hierarchical context-pair ranking
64 cells, 8 contexts:
- raw pairs 2016
- context-pairs 36

Cheap single context-pair score was noisy:
- true context average rank ~13.6/36
- top8 context budget solved only 3/14
- full 36 contexts solved 8/14

Conclusion:
context-level selection has the same winner's-curse problem as cell/pair selection.

---

## V821B — Reproducible context-level evidence
Two independent screens; required relation evidence to repeat.

Still insufficient:
- true context mean rank ~11/36
- top12 solved 8/16

Conclusion:
replication helps some cases but weak hierarchical signals remain sample-limited.

---

## V822 — Context-pair sample complexity
Two independent context screens.

40 steps/screen:
- mean true context rank 6.9
- top8 recall 8/10

80 steps:
- mean rank 3.9
- top8 recall 8/10

160 steps:
- mean rank 1.5
- top8 recall 10/10
- true relation top inside its context 10/10

Conclusion:
the hierarchy was primarily evidence-starved, not fundamentally impossible.

---

## V823 — Adaptive context compute cascade
Instead of 160-step auditing all 36 context-pairs:

- stage1: 40 steps all 36
- stage2: 80 steps top16
- stage3: 160 steps top6
- sandbox top3 contexts

Results:
- 9/12 exact
- 31.9% less context-audit effort than uniform deep screening

Failure source:
true contexts sometimes pruned too early.

---

## V823C — Conditional rescue
Added deeper rescue only when base certification failed.

Results:
- 16/16 exact
- 11 base successes
- 1 rescue-A
- 4 rescue-B
- mean effort 9910 vs uniform 12000
- 17.4% mean effort reduction

Conclusion:
progressive search + conditional rescue can preserve reliability while reducing average compute.

---

# Primitive / macro branch

## V824 — First automatic macro compilation
Initial attempt failed.

Problems:
1. search incorrectly terminated when one exact expression cost produced no new semantic functions
2. blindly adding every learned macro increased branching

This failure motivated separating:
MACRO INVENTION from MACRO ADMISSION.

---

## V824B — Macro invention + reuse-based admission
Primitive vocabulary initially:
- NAND only

2-bit curriculum solved from NAND:
- AND cost 3
- OR cost 3
- XOR cost 5
- XNOR cost 5
- IMPLIES cost 2

All became candidate macros.

Held-out 3-variable compositional probe tasks measured actual future-search savings.
Scores:
- AND: harmful
- OR: useful
- XOR: strongly useful
- XNOR: strongly useful
- IMPLIES: harmful

Admitted:
- XNOR
- XOR
- OR

Transfer:
Parity3:
- NAND-only: 37,845 evals, cost 14
- admitted macros: 90 evals, cost 2
- ~99.76% search reduction

Parity4:
- NAND-only unresolved after 1,000,001 evaluations
- macros solved in 1,383 evaluations, cost 3

Parity5:
- NAND-only unresolved after 1,000,001
- macros solved in 31,866, cost 4

Conclusion:
successful programs can become genuinely useful new callable primitives, but only after demonstrating reuse value.

---

## V825 — Macros are not free
Negative control on unseen n=4 tasks.

NAND_NATIVE tasks:
- macro library slower on 10/11 jointly solved tasks
- median cost ratio = 9.28x slower
- only 1 macro-faster task

MACRO_NATIVE tasks:
- macro library faster on 11/12
- median eval ratio ~0.15x
- one task solved by macros while NAND-only hit the 400k cap

Conclusion:
a useful macro library is task/context dependent.
Globally enabling all learned primitives can create severe branching overhead.

---

## V826 / V826B — Early primitive-library routing
Tiny pilot routing by Hamming distance to target.

Budget 400:
- 16/24 correct library choices

Budget 2000:
- 17/24

Budget 10000:
- 16/24

Failures included macro-native tasks where macros eventually gave >100x search savings.

Conclusion:
early progress toward a target is not a reliable predictor of long-horizon search vocabulary.

---

## V827 — Primitive-search portfolio
Instead of routing early, run BASE and MACRO search concurrently.

50/50 compute split:
- hard regret bound = 2.0x oracle on every solved task
- mean regret = 2.0x
- median = 2.0x

Early 2000-eval router:
- mean regret ~9.18x
- median ~5.15x
- worst ~49.31x

Unequal splits reduced average cost for favorable task distributions but allowed worst-case regret up to 4x.

Conclusion:
when search-strategy routing evidence is weak, maintaining a parallel portfolio can be far safer than committing early.

---

# Strongest narrowed architecture after V813-V827

The candidate core is no longer "every mechanism we tried."

Strongly supported core roles:

1. PROTECTED INTERVENTION
   The evaluated system must not control all causal excitation.

2. RELATIONSHIP-CREDIT PORTFOLIO
   Synergy, complementary coverage, and temporal direction are distinct.

3. SANDBOX / INDEPENDENT PROOF
   Cheap proposals must not immediately alter deployed behavior.

4. LEARNED CAUSAL CONTEXTS
   Context separation can emerge from behavior and dramatically improve weak causal signal.

5. SOFT / OVERLAPPING CONTEXT ASSUMPTIONS
   Learned similarity can be causally wrong; cross-context search must remain possible.

6. ADAPTIVE EVIDENCE ALLOCATION
   Larger / noisier search spaces deserve more causal samples.

7. PROGRESSIVE SEARCH + CONDITIONAL RESCUE
   Start cheap, deepen only when proof fails, but keep a path to recover early pruning mistakes.

8. PRIMITIVE / MACRO COMPILATION
   Successful structures can become new callable primitives.

9. MACRO ADMISSION / RETIREMENT PRESSURE
   New primitives must prove reuse value because every extra primitive increases branching.

10. SEARCH-STRATEGY PORTFOLIOS
    If strategy routing evidence is unreliable, parallel competing search modes can dominate premature commitment.

Conditional rather than universal:
- shortlist widening
- one fixed context count
- one fixed exploration mixture
- one globally active primitive vocabulary

Open problems:
- make context discovery fully online inside the continuous developmental population
- allow context graph / cross-context relations at larger G without O(G^2) growth
- discover primitive signatures from lower-level continuous cells, not symbolic truth tables
- learn when to retire macros over long nonstationary curricula
- integrate macro invention with evolving cells/organs/memory rather than a separate symbolic searcher
- move beyond controlled synthetic environments
- hardware-level accounting

These experiments remain controlled synthetic evidence. They do not establish AGI or a Transformer replacement.
