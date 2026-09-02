# Transmutor Research Addendum — V67 through V79

This addendum is a falsification / narrowing phase.

The objective was no longer to accumulate mechanisms. It was to attack the strongest claims and remove anything that did not survive stricter controls.

---

# V67 — Distribution-shift stress test of learned repair

A repair ranker trained on 14-feature K=3 sparse tasks was evaluated under:

- 18 features
- 22 features
- weaker correlation
- stronger correlation
- continuous coefficients
- higher noise

Results:

- in distribution: greedy 98.67%, repair 100%
- 18 features: 99.67% -> 100%
- 22 features: 99.00% -> 100%
- weaker correlation: 100% -> 100%
- stronger correlation: 87.00% -> 97.67%
- continuous coefficients: 100% -> 100%
- more noise: 99.67% -> 100%

The first important boundary appeared at extreme feature correlation.

---

# V68 — Self-supervised repair without oracle solution labels

The repair learner was retrained using only:

    observed reduction in training residual MSE

No hidden true feature set was used as a training target.

Held-out results:

- corr .90: greedy 100%, repair 100%
- corr .95: 98.75% -> 99.69%
- corr .98: 90.31% -> 100%
- 22 features corr .98: 89.38% -> 99.69%

This established that local repair ranking can be learned from observable search outcomes rather than oracle correctness labels.

Caveat:
the candidate move space and move descriptors remained manually specified.

---

# V69 — Remove oracle correctness from evaluation too

Search was judged only by prediction on unseen data.

Tasks:
- noisy sparse linear processes
- search sees train samples only
- evaluation uses held-out test samples

Results:

- greedy:
  - normalized test MSE 0.01386
  - cost 1.00

- learned repair:
  - MSE 0.01018
  - cost 1.20

- exhaustive:
  - MSE 0.00911
  - cost 12.40

- adaptive stack:
  - MSE 0.00932
  - cost 5.15
  - repair used on 42%
  - exhaustive used on 35.67%

This was substantially harsher than exact-support tests:
self-evaluation under noise required much more escalation.

---

# V70 — Transfer to larger hidden programs

The K=3 repair ranker was used without retraining on K=4 and K=5 tasks.

Results:

- K3, corr .98: 95.00% -> 100%
- K3, corr .99: 85.45% -> 99.55%
- K4, corr .98: 89.09% -> 99.09%
- K4, corr .99: 71.36% -> 95.45%
- K5, corr .98: 75.45% -> 94.55%
- K5, corr .99: 49.09% -> 83.64%

Boundary:
one learned one-swap repair no longer solved the extreme larger-program regime.

---

# V71 — Iterative learned repair

The same learned repair policy was applied repeatedly.

Results:

K4 corr .99:
- greedy 63.33%
- 1 repair round 91.67%
- 2 rounds 98.33%
- 3 rounds 99.17%

K5 corr .98:
- 75.00%
- 95.00%
- 98.33%
- 99.17%

K5 corr .99:
- 48.33%
- 74.17%
- 85.00%
- 86.67%

Multiple local repairs do compose, but the extreme K5/.99 case still remained unsolved.

---

# V72 — Is the bottleneck the ranker or the one-swap neighborhood?

Extreme K5, 25 features, corr .99.

- greedy: 50.71%
- learned local repair: 92.14%
- exhaustive one-swap hill climb: 99.29%

Therefore the main bottleneck was the learned ranking policy, not the existence of a local one-swap path.

---

# V73 / V74 — Broader repair training failed

Two attempts were made to make repair more generic.

V73:
- curriculum across K=3/4/5
- target = absolute observable residual gain

Held-out K5/.99:
- old K3 ranker, 3 rounds: 89.23%
- curriculum ranker, 3 rounds: 67.69%
- full local: 100%

V74:
- target changed to relative residual improvement

Held-out K5/.99:
- old K3 ranker, 3 rounds: 90.67%
- relative-gain ranker, 3 rounds: 63.33%
- full local: 100%

Important negative result:

> More diverse repair training did not create a more general repair learner.

This rejects a simple "train the repair policy on more failures and it will generalize" story.

---

# V75 — Causal control: learned ranking vs random and simple heuristic

700 hard tasks.
All top-4 methods received exactly the same candidate-evaluation budget.

Results with 95% Wilson intervals:

- greedy:
  - 80.29%
  - CI 77.18–83.06

- random top-4:
  - 81.71%
  - CI 78.68–84.40

- residual-correlation top-4:
  - 97.57%
  - CI 96.15–98.48

- learned top-4:
  - 97.86%
  - CI 96.49–98.70

- full local:
  - 98.57%
  - CI 97.39–99.22

The learned ranker clearly beat random, but did not clearly beat the simple residual-correlation heuristic.

This substantially weakened the "learning from failures is essential" claim.

---

# V76 — Learned repair vs residual heuristic across K and correlation

Same top-5 candidate budget.

K3 corr .98:
- heuristic 100%
- learned 100%

K3 corr .99:
- heuristic 99.23%
- learned 100%

K4 corr .98:
- heuristic 98.46%
- learned 98.46%

K4 corr .99:
- heuristic 90.77%
- learned 93.08%

K5 corr .98:
- heuristic 95.77%
- learned 95.38%

K5 corr .99:
- heuristic 83.85%
- learned 78.46%

Conclusion:

> Learned repair is not a core mechanism in this benchmark family.

A simple residual-guided ranking is at least as defensible and sometimes better.

---

# V77 — Minimal search stack, no learned repair

Architecture:

    greedy
      ->
    residual-correlation top-4 local repair if suspicious
      ->
    exhaustive fallback if still suspicious

Held-out exact-support benchmark:

- greedy:
  - 91.90%
  - cost 1.00

- always heuristic repair:
  - 98.81%
  - cost 1.20

- exhaustive:
  - 100%
  - cost 9.16

- minimal adaptive stack:
  - 100%
  - cost 1.192
  - heuristic invoked on 8.81%
  - exhaustive invoked on 2.14%

Thus the learned-repair layer can be removed without destroying the core adaptive-search effect.

---

# V78 — Minimal heuristic stack on noisy held-out prediction

No learned repair model.

Results:

- greedy:
  - held-out normalized MSE 0.01447
  - cost 1.00

- heuristic repair:
  - MSE 0.01088
  - cost 1.20

- exhaustive:
  - MSE 0.00932
  - cost 12.40

- minimal adaptive stack:
  - MSE 0.00942
  - cost 4.935
  - heuristic invoked on 43.75%
  - exhaustive invoked on 33.75%

The minimal stack nearly matched exhaustive predictive performance at ~40% of the proxy cost.

---

# V79 — Replication and feature-count shift audit

V78 thresholds were frozen.
No retuning.

Nine fresh held-out batches:
- 18 features
- 20 features
- 22 features
- mixed correlation and noise

Results:

18 features:
- MSE ratio minimal/exhaustive: 1.0091 ± 0.0129
- cost ratio: 0.433 ± 0.012

20 features:
- MSE ratio: 1.0112 ± 0.0079
- cost ratio: 0.412 ± 0.005

22 features:
- MSE ratio: 1.0050 ± 0.0071
- cost ratio: 0.390 ± 0.015

Overall:

- mean MSE ratio: 1.0084
- worst batch MSE ratio: 1.0274
- mean cost ratio: 0.411

Interpretation:

Across these shifted batches, the frozen minimal stack used about 41% of exhaustive search cost while averaging about 0.84% higher held-out MSE.

---

# What survived the falsification phase

## Surviving claim A — staged adaptive search is useful in this benchmark family

Within the tested noisy sparse-linear search family:

- cheap search handles many tasks
- residual-guided local repair closes much of the gap
- exhaustive search is useful as a fallback
- escalation based on residual diagnostics reduces average search cost substantially

The strongest replicated number is V79:

> Mean held-out MSE = 1.0084 × exhaustive while mean search-cost proxy = 0.411 × exhaustive.

This is a scoped empirical result, not a universal law.

---

## Surviving claim B — search quality can dominate solution-space capacity

Earlier V56 experiments showed:

- the correct composition existed in the feature/program library
- greedy sparse search failed
- exact combinatorial search recovered the exact compact composition

Therefore:

> "The solution exists in the hypothesis space" does not imply "the current search process can find it."

This is conceptually obvious in optimization theory, but the experiments demonstrate it directly inside this prototype architecture.

---

## Surviving claim C — local repair is valuable, but learning the repair ranker is not established as necessary

Residual-guided one-swap repair consistently improved greedy search.

The learned ranker:
- beat random rankings
- sometimes beat residual ranking
- sometimes lost to residual ranking
- failed under attempted broad curriculum training

Therefore the correct narrowed claim is:

> Local residual-guided repair is useful in this family.

Not:

> A learned repair policy is a required or general mechanism.

---

# Claims that should be rejected or kept unresolved

## Reject as core claim: learned repair policy is necessary

V75/V76 do not support necessity.
A simple residual heuristic is competitive or better.

## Unresolved: generic repair learning across changing program size

V73/V74 failed.

## Unresolved: fully online latent generative memory matches simple raw-stat replay

It does not.
V61/V61c improved it, but raw-stat replay remained stronger in the tested setup.

## Unresolved: local learning replaces backprop at scale

Toy/real-digit results are interesting but far from enough.

## Unresolved: the primitive vocabulary can discover itself

Current systems still receive human-designed primitive operations, state loops, repair neighborhoods, and search portfolios.

## Unresolved: this architecture is a successor to Transformers

Nothing in the current experiments establishes that.

---

# Current minimal research hypothesis

After removing weaker mechanisms, the strongest architecture-level idea is now much simpler:

```text
problem
   |
   v
cheap search
   |
   v
measure residual / uncertainty
   |
   +---- good enough ----> stop
   |
   v
cheap local repair
   |
   v
measure again
   |
   +---- good enough ----> stop
   |
   v
expensive global search
```

The experimentally supported principle is:

> Under finite resources, a staged search system can allocate expensive search selectively and preserve nearly all of the predictive quality of exhaustive search in the tested sparse-search domain.

Everything beyond that remains hypothesis.
