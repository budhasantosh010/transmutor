# Transmutor Experimental Log — V779 to V784

## V779 — Protected interaction audit + mutable learning rules
FAIL (3/5 exact).

Cells had mutable action-learning rules:
CENTERED / ACTIVE / SIGNED / ANTI / NONE.

A protected pairwise audit estimated interaction from ordinary behavior.

Failure:
mutable rules drove action probabilities close to 0 or 1.
That starved ordinary behavior of causal excitation.
Spurious positive bonds appeared in 2/5 runs.

Mean compiled accuracy: ~98.87%, but exact organ only 3/5.

Lesson:
a protected evaluator is insufficient if the evaluated system can avoid or saturate the evaluator's interventions.

---

## V779B — Protected randomized audit probes
PASS, 5/5.

Separated behavior from audit.

Behavior:
mutable learning rules controlled ordinary action probabilities.

Audit:
25% of batches overrode cell action probabilities with protected Bernoulli(.5) assignments.
Cells could not opt out or rewrite these interventions.

Results:
- exact T1+T2 organ: 5/5
- compiled accuracy: 100%
- true pair LCB strongly positive
- redundant D1+D2 pair strongly negative
- no spurious positive bonds

Lesson:
protected causal evaluation needs protected intervention/excitation, not only protected bookkeeping.

---

## V780 — Developmental mixed-operator integration
FAIL.

Harder environment:
- true complementary cells MUL(0,1), MUL(2,3)
- misleading redundant decoys MUL(4,5), MUL(6,7)
- mixed ADD/SUB/MUL genome space
- targets absent initially
- local birth/death/mutation

Both targets were invented and became top useful cells in the completed seed.

But audit randomized half of all ~30 cells simultaneously.
The decision context saturated.
No target bond formed.

Lesson:
audit context itself matters. Too many simultaneous interventions can erase a local interaction signal.

---

## V780B — Sparse focal-pair protected audits
FAIL on first developmental seed, for a different reason.

Audit fix:
each audit sample perturbed only one randomly chosen cell pair.
All other cells were off.

This removed saturation.

However one required target genome was never generated in the developmental run.

Lesson:
the integrated problem separated into:
1. interaction auditing
2. structural coverage

Perfect auditing cannot select a genome that never exists.

---

## V780C — Static mixed-operator sparse-audit isolation
FAIL exact-organ criterion, but diagnostically useful.

Fixed 30-cell ADD/SUB/MUL populations contained both targets.

Sparse focal-pair audit:
- true target pair positive in all runs
- redundant decoy pair negative in all runs

But additional real secondary positive bonds appeared.
Connected-component merging produced oversized organs.

Lesson:
"all positive bonds in one connected component = one organ" can over-merge distinct relationships.

---

## V780D — Bond candidates compete as separate organs
PASS, 5/5.

Fix for V780C:
- every confident pair bond becomes its own 2-cell candidate organ
- do NOT merge all connected bonds immediately
- candidate organs then undergo local stochastic survival with finite resource cost

Results:
- true MUL(0,1)+MUL(2,3) organ survival p ~0.99
- secondary bond-organs fell to very low survival
- automatic largest-gap pruning kept only true organ
- exact 5/5
- compiled accuracy 100%

Lesson:
bond formation and organ consolidation are separate stages.
Local competition can decide which candidate relationships deserve to become stable macros.

---

## V781 — Structural coverage policies
PASS.

Genome space:
210 ADD/SUB/MUL + wiring genomes.
Targets absent initially.
Equal birth budget = 120.

5000 Monte Carlo curricula:

RANDOM:
- both targets found: 18.82%
- mean unique genomes tested: 108.5

FLAT NOVELTY:
- both targets: 43.60%
- unique tested: 150

FACTORIZED NOVELTY (normalized family coverage):
- both targets: 44.54%

Avoiding repeated structural trials more than doubled discovery probability.
But normalized factor novelty was only slightly better than flat novelty.

---

## V781B — Balanced operator novelty
PASS.

New rule:
give ADD, SUB, MUL roughly equal numbers of distinct trials,
without knowing which family is useful.

Same 120-birth budget:

RANDOM: 18.58%
FLAT NOVELTY: 43.40%
BALANCED OPERATOR NOVELTY: 79.52%

Balanced policy tested ~50 unique genomes per operator family.

Lesson:
diversity should exist not only over genomes but over structural categories.

---

## V782A — Early family-level signal diagnostic
Standalone candidate accuracies in developmental trap:

ADD:
- mean ~55.07%
- max ~68.76%

SUB:
- mean ~51.92%
- max ~61.97%

MUL:
- mean ~55.44%
- max ~87.55%

The final true cells are complementary/weak alone, but MUL contains a much stronger high-performing tail.

This creates a possible coarse exploration signal.

---

## V782B — Reward-adaptive structural exploration
PASS.

Start with minimum exploration in every operator family.
Then use noisy binary candidate outcomes to favor families that have already produced unusually useful candidates.
15% uniform exploration remains.

5000 curricula, same 120 births:

BALANCED:
- target-pair discovery: 79.64%

ADAPTIVE:
- target-pair discovery: 99.20%

No target operator label was provided.

Lesson:
environmental outcomes can guide structural exploration at a coarser abstraction level than exact genome search.

---

## V783 — Misleading exploration signal
PASS as a negative-control experiment.

Adversarial world:
- hidden targets still in MUL
- target MUL cells weak alone
- strongest early standalone decoys placed in ADD

Results:

BALANCED:
- targets found: 79.14%

ADAPTIVE, 15% exploration floor:
- 37.60%

ADAPTIVE, 30%:
- 40.98%

ADAPTIVE, 50%:
- 46.32%

Lesson:
reward-adaptive structural exploration can catastrophically over-exploit misleading early evidence.
A larger exploration floor helps but does not match balanced diversity.

---

## V784 — Policy-level exploration diversity
PASS, 500 Monte Carlo runs per world/q.

Two unknown worlds:
HELPFUL early clues
MISLEADING early clues

Meta-policy mixes:
- balanced structural novelty with probability q
- reward-adaptive exploration with probability 1-q

Results:

q=0.00 pure adaptive:
- helpful 99.0%
- misleading 36.4%
- worst case 36.4%

q=0.25:
- helpful 99.8%
- misleading 61.0%
- worst 61.0%

q=0.50:
- helpful 94.0%
- misleading 70.8%
- worst 70.8%

q=0.75:
- helpful 79.4%
- misleading 78.6%
- worst 78.6%

q=1.00 pure balanced:
- helpful 77.8%
- misleading 81.8%
- worst 77.8%

Best tested worst-case mixture: q=0.75.

Lesson:
diversity may be needed at the exploration-policy level itself.
Pure exploitation maximizes favorable-world performance but is fragile.
A portfolio of conservative diversity + adaptive exploitation substantially improves robustness.

---

# Strongest narrowed conclusions V779-V784

1. Protected evaluation requires protected interventions; otherwise mutable agents can starve the audit of causal evidence.
2. Causal audit interventions must be context-sized. Perturbing too many components simultaneously can saturate the system and hide local interactions.
3. Positive interaction bonds should not automatically be transitively merged into one giant organ.
4. A useful pipeline is:
   local bond evidence -> candidate organs -> resource competition -> stable macro.
5. Interaction-aware candidate-organ competition solved mixed ADD/SUB/MUL secondary-bond noise 5/5.
6. Structural coverage is now a distinct bottleneck from structural selection.
7. Novelty helps because repeated trials waste finite developmental budget.
8. Diversity across structural categories can be much more valuable than flat novelty.
9. Reward-adaptive exploration can nearly solve coverage when early clues are aligned with the final solution.
10. The same adaptive policy can fail badly under misleading clues.
11. Robust systems may need diversity across exploration policies, not one universally optimal exploration rule.

These remain controlled synthetic experiments. They do not establish AGI, a Transformer replacement, or general self-organizing intelligence.
