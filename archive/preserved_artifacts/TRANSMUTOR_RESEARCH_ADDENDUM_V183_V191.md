# Transmutor Research Addendum — V183 through V191b

This phase focused on four narrower questions:

1. Does active falsification reveal synergistic shortcuts on real data?
2. Is exact causal attribution actually necessary for repair?
3. Can a useful composed subtree be promoted into a reusable computational macro?
4. How should promoted abstractions be challenged without making falsification itself destructive?

---

# V183 — Real XOR shortcut exists, but model ignores it

Dataset:
- sklearn handwritten digits
- task: even vs odd

A hidden two-bit XOR shortcut was added to 12 auxiliary variables.
Each member had approximately zero marginal signal.

Using a strong pixel representation and ExtraTrees:

- passive XOR-pair recovery: 0%
- active intervention recovery: 8.33%
- mean pair intervention drop: ~0.00087
- full deployment accuracy after destroying XOR: ~97.53%
- stable pixels-only accuracy: ~97.63%

Conclusion:

> A shortcut's existence is not sufficient for falsification to reveal it. The trained system must actually rely on the shortcut.

This is a negative control, not a successful shortcut-discovery result.

---

# V183b — Once the model relies on XOR, intervention finds it

The stable visual channel was intentionally compressed to 6 PCA components so the XOR mechanism became attractive to the learner.

Results:

- passive pair recovery: 6.25%
- active intervention pair recovery: 100%
- mean marginal correlation of pair members: ~0.0357
- mean intervention drop: ~0.2812
- IID full-model accuracy: ~97.32%
- deployment after XOR destruction: ~69.86%
- stable PCA-only reference: ~88.82%
- pair-removal repair: ~79.11%

Conclusion:

> Synergistic features can have nearly zero marginal predictive signal yet become obvious under intervention once the learned model actually depends on them.

But removing exactly the discovered pair did not restore all stable performance.

---

# V184 — Repair scope is a separate structural decision

Candidate repairs:

- keep full representation
- remove discovered XOR pair
- drop entire auxiliary namespace

Selection:
maximize worst-case accuracy over ordinary and all-aux-randomized validation.

Results:

- pair discovery: 88.89%
- FULL selected: 0%
- PAIR_ONLY selected: 0%
- DROP_AUX selected: 100%

Deployment accuracy:

- full: ~70.96%
- pair-only: ~77.71%
- drop all auxiliary: ~89.03%
- robustly chosen: ~89.03%

Conclusion:

> Knowing which shortcut caused a failure does not imply that the narrowest causal repair is the best behavioral repair.

---

# V185 — Re-audit after every repair

Hidden shortcut system:

- one direct marginal shortcut
- one XOR shortcut pair

Procedure:

    falsify
      ↓
    remove strongest influential feature
      ↓
    retrain
      ↓
    falsify again

Results:

- direct shortcut found first: 100%
- all three causal contributors recovered: 0%
- mean removed features: 2
- initial shifted accuracy: ~28.13%
- final shifted accuracy: ~79.53%
- stable-only reference: ~88.87%

Mean intervention drop by round:

- round 1: ~0.330
- round 2: ~0.267
- round 3: ~0.014 < stop threshold

Interpretation:

The loop typically removed:

1. the direct shortcut
2. one member of the XOR pair

After one XOR member was removed, the other became behaviorally harmless.

Therefore exact causal-member recovery was unnecessary for disabling the failure mechanism.

---

# V186 — Exact repair formulation: minimum hitting set

Represent each active failure mechanism as a feature set M_i.

A repair set R disables all mechanisms iff:

    for every i:
        R ∩ M_i != empty

Therefore:

> Minimum-cardinality behavioral repair is exactly the minimum hitting-set problem.

Across 700 random small hypergraphs:

- mean union of all causal contributors: 8.23
- mean minimum repair: 2.861
- mean greedy repair: 2.903
- all-causal / minimum-repair ratio: ~3.04x
- simple greedy exactly optimal: 95.86%

Conclusion:

> "Identify every causal contributor" can be much more expensive than "find a sufficient intervention set."

Caveat:
minimum hitting set is NP-hard in general.

---

# V187 — Minimum-size repair is not minimum-cost repair

Each removable feature received a heterogeneous collateral cost.

Optimal repair became:

    minimum-weight hitting set

Results across 650 random problems:

- mean minimum-cardinality repair size: 2.789
- mean minimum-cost repair size: 2.955
- mean cost of minimum-size repair: 4.152
- true minimum cost: 2.943
- cost-aware greedy: 3.099

Important ratios:

- minimum-size repair was also cost-optimal only 35.85%
- minimum-size repair cost / optimum: ~1.613x
- cost-aware greedy cost / optimum: ~1.050x
- cost-aware greedy exactly optimal: 65.38%

Conclusion:

> The fewest architectural changes can be the wrong repair when changes have different collateral costs.

---

# V188 — Value of diagnosis before repair

Two possible failure modes H1/H2.

Available actions:

- targeted repair R1
- targeted repair R2
- universal repair U
- optional noisy diagnostic

Exact principle:

    diagnose iff

    expected posterior-optimal repair savings
        >
    diagnostic cost

Representative cases:

p=.5, diagnostic reliability=.95, cost=.2:
- act now expected cost: 2.0
- diagnose then act: 1.50
- diagnosis worthwhile

p=.5, q=.65, cost=.2:
- act now: 2.0
- diagnose: 2.2
- diagnosis not worthwhile

p=.5, q=.95, cost=.8:
- act now: 2.0
- diagnose: 2.1
- diagnosis not worthwhile

Conclusion:

> More causal understanding should be purchased only when it changes the repair decision enough to repay its evidence cost.

---

# V189 — Missing transform discovered from lower-level composition

Hidden residual:

    A sin(k x² + phase)

Instead of supplying an exponent p, the transform grammar contained:

    X
    MUL(expr, X)

Generated transforms:

    X
    MUL(X,X)
    MUL(MUL(X,X),X)

Oscillator wrapper remained supplied.

Results across 550 episodes:

- selected MUL(X,X): 549/550
- recovery: 99.82%
- mean frequency error: ~0.0311
- clean held-out MSE: ~0.000730

Conclusion:

> A useful missing subtree can emerge through composition of lower-level primitives rather than through a named high-level parameter.

Caveat:
multiplication and the sinusoidal wrapper were supplied.

---

# V190 — Promote recurring subtree into a reusable macro

Promoted:

    SQUARE(E) := MUL(E,E)

Future families:

    x²
    x⁴
    x⁸

Selection accuracy:

- x²: 93.18%
- x⁴: 96.82%
- x⁸: 100%

Description accounting:

x²:
- raw multiplication syntax: 3 nodes
- SQUARE macro syntax: 2
- 1.5x shorter

x⁴:
- raw: 7
- macro: 3
- 2.33x shorter
- raw binary-tree shapes: 5

x⁸:
- raw: 15
- macro: 4
- 3.75x shorter
- raw binary multiplication tree shapes: Catalan(7)=429

Conclusion:

> Promoted reusable subtrees can compress both description length and repeated structural search.

But this creates abstraction bias.

---

# V191 — Macro fast path becomes a trap on incompatible tasks

Future powers:

- macro-compatible: 2,4,8
- incompatible: 3,5

Policies:

- macro only
- full raw search
- macro first, then fallback if ordinary validation MSE exceeds threshold

Overall:

- macro-only accuracy: 58.11%
- full raw: 78.11%
- macro+fallback: 65.22%
- full raw search evaluations: 432
- macro+fallback: 208.56
- search reduction: ~2.07x

Problem:

Fallback triggered too rarely on odd powers:

- x³: ~15.6%
- x⁵: ~26.1%

A promoted even-power macro could imitate odd-power data well enough locally to pass ordinary adequacy.

Conclusion:

> A promoted abstraction must remain falsifiable and must not close the lower-level grammar.

But the first adequacy rule was too weak.

---

# V191b — Over-aggressive falsification also fails

A harder extrapolation range was introduced:

fit:
    x in [0.08, 0.78]

falsify:
    x in [0.82, 1.22]

This forced fallback aggressively:

- x³: 92.1%
- x⁵: 96.3%
- x⁸: 99.5%

But it also rejected useful macros and made the entire structural-identification problem harder.

Overall:

- macro-only: 34.53%
- full raw: 38.84%
- macro+fallback: 38.84%
- full raw evaluations: 464
- macro+fallback: 573.04

So the fallback recovered no accuracy advantage and cost more than full search.

Conclusion:

> "Falsify harder" is not a general solution.

A useful challenge must maximize:

    expected diagnostic information
    -------------------------------
    experiment / measurement cost

rather than merely pushing farther out of distribution.

V191 and V191b should be preserved as paired failure controls.

---

# What V183–V191 narrow down

## 1. Shortcut existence != shortcut dependence

A model should be audited for the dependencies it actually uses, not merely for potentially spurious variables present in the environment.

## 2. Marginal relevance != causal necessity

XOR members can have essentially zero marginal signal but large intervention value.

## 3. Causal attribution != sufficient repair

If mechanisms are conjunctive/interacting, breaking one necessary member can disable the entire mechanism.

## 4. Minimum behavioral repair = hitting set under the stated mechanism model

And with heterogeneous costs:

    minimum-cost repair = weighted hitting set

## 5. Exact understanding has an economic value

Diagnosis is useful only if its expected repair savings exceed diagnosis cost.

## 6. Primitive discovery can climb one compositional level

A useful subtree MUL(X,X) was selected from lower-level multiplication grammar and then reused as SQUARE.

## 7. Learned abstractions create their own failure modes

Macros accelerate compatible future search but can trap the system on incompatible tasks.

## 8. Falsification itself must be optimized

Weak criticism accepts bad abstractions.
Over-aggressive criticism destroys useful fast paths and can make inference harder.

So the next frontier is no longer simply:

    "Can the system falsify its own abstractions?"

It is:

> Can the system actively design the cheapest falsification experiment that best distinguishes the promoted abstraction from its strongest lower-level alternatives?

That connects V191 directly back to V148, V152, V154, V159, and V177.
