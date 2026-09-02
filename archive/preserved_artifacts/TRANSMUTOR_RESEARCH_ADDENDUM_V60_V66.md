# Transmutor Research Addendum — V60 through V66

This addendum extends the V0–V59 research log.

## V60 — Meta-search: choose how to search

Sparse symbolic-regression tasks were constructed so that:
- greedy search usually works
- some correlated tasks require exhaustive search

A meta-controller saw cheap diagnostics:
- maximum feature correlation
- condition number
- greedy residual

Results:

- always greedy:
  - exact recovery: 93.75%
  - search-cost proxy: 1.00

- always exhaustive:
  - exact recovery: 100%
  - cost: 18.20

- learned meta-search:
  - exact recovery: 100%
  - cost: 2.08
  - exhaustive invoked on only 6.25% of tasks

Lesson:
a system can preserve expensive-search accuracy while spending expensive search only where it predicts it is needed.

Caveat:
the search portfolio, diagnostics and cost proxy are human-supplied.

---

## V60b — Resource-priced search

The meta-search decision was reframed as:

    maximize accuracy - lambda * search_cost

Results:

- lambda = 0:
  - exhaustive on 100% of tasks
  - accuracy 100%
  - cost 18.20

- lambda >= 0.01 in the tested range:
  - exhaustive on 6.25%
  - accuracy 100%
  - cost 2.08

The controller's diagnostics separated the difficult cases well enough that resource pressure reduced compute without sacrificing accuracy in this benchmark.

Lesson:
search strategy can be treated as an economic/resource allocation decision.

---

## V61 — Fully online learned latent generative memory

V54's PCA basis had seen the full unlabeled training corpus.
V61 removed that advantage.

Sequential digits:
0/1 -> 2/3 -> 4/5 -> 6/7 -> 8/9

Online system:
- AE 64 -> 8 -> 64
- old memories exist only as latent class statistics
- old pseudo-examples are decoded before each new stage
- the AE is then retrained only on:
  - current raw examples
  - self-generated old pseudo-examples

Final continual accuracy:

- no replay: 50.52%
- raw isotropic generative stats: 88.22%
- fully online latent replay: 80.15%

The latent class-stat memory used only 90 floats, but the AE itself adds ~1,096 parameters.

Lesson:
fully online learned latent replay works, but representation drift / crude latent distribution causes a substantial loss.

---

## V61b — Stabilizing the latent model

Penalty:

    reconstruction
    + lambda * ||AE_now - AE_previous_stage||^2

Results:

- lambda 0: 80.44%
- 0.002: 79.78%
- 0.01: 80.89%
- 0.05: 79.67%
- 0.2: 79.00%

Simple anchoring helped only slightly.

Lesson:
latent-model drift is not the whole problem.

---

## V61c — Richer online latent memory

Same online AE, richer per-class distributions:

- scalar spread:
  - 90 class-stat floats
  - 83.56%

- diagonal Gaussian:
  - 160 floats
  - 84.67%

- two-component diagonal mixture:
  - 330 floats
  - 86.67%

Raw input-stat replay reference:
- ~88.22%

Lesson:
the representation stored for each class matters strongly; richer latent geometry recovers much of the lost continual-learning performance.

---

## V62 — Adaptive memory complexity by multimodality score

Each class could receive:
- diagonal memory
- mixture-2 memory

A KMeans separation score decided whether to spend richer memory.

Held-out result:
- all diagonal: 82.44%, 160 floats
- adaptive: 83.44%, 160 floats
- all mixture: 83.89%, 330 floats

The adaptive controller chose zero mixture classes on held-out runs.

Lesson:
the first complexity-allocation heuristic did not meaningfully identify which classes deserved richer memory.

---

## V62b — Adaptive memory complexity by local predictive value

A more principled signal was used:

1. split class latent samples into fit/local-validation
2. fit diagonal and mixture memories
3. compare held-out log likelihood
4. use mixture only if predictive improvement exceeds a threshold

Held-out:

- all diagonal:
  - 85.44%
  - 160 floats

- predictive adaptive:
  - 86.33%
  - 177 floats
  - 1 mixture class out of 10

- all mixture:
  - 85.67%
  - 330 floats

Lesson:
memory structure can be selectively allocated based on local predictive utility rather than globally making every memory rich.

Caveat:
only two held-out splits were used.

---

## V63 — Global memory-budget allocation

Each class always gets a 16-float diagonal memory.
Upgrading a class to mixture-2 costs +17 floats.

The allocator ranks classes by local held-out likelihood benefit.

Mean results:

- ≤0 mixture classes:
  - 160 floats
  - 82.67%

- ≤1:
  - 177 floats
  - 84.67%

- ≤2:
  - 194 floats
  - 83.56%

- ≤4:
  - 228 floats
  - 85.44%

- ≤6:
  - 262 floats
  - 83.78%

- ≤10:
  - ~304 floats
  - 84.22%

More memory was not monotonically better.

Lesson:
allocation quality can matter more than raw capacity.

---

## V64 — Hierarchical search: greedy -> local repair -> exhaustive

Search levels:

1. greedy
2. enumerate all one-swap repairs around greedy result
3. exhaustive fallback

Results:

- greedy:
  - 87.00%
  - cost 1.00

- always local repair:
  - 99.67%
  - cost 2.70

- exhaustive:
  - 100%
  - cost 19.20

- hierarchical residual-gated policy:
  - 100%
  - cost 1.54
  - repair on 14.00%
  - exhaustive on 1.67%

Lesson:
search can be staged so expensive reasoning is invoked only after cheaper reasoning fails.

Caveat:
the repair neighborhood was hand-designed.

---

## V65 — Learn which repair move to try

V64 enumerated ~34 one-swap candidates.

V65 learned to rank repair moves from previous failed searches.

Repair-candidate descriptors included:
- incoming feature/residual correlation
- outgoing selected-feature coefficient
- incoming/outgoing feature correlation
- current residual
- incoming feature correlation with selected set

Results:

- greedy: 90.29%

- enumerate all one-swaps:
  - 99.71%
  - 34 candidate evaluations/task

- learned top-1 repair:
  - 97.43%
  - 0.10 candidate evaluations/task average

- learned top-2:
  - 98.00%
  - 0.12

- learned top-4:
  - 99.71%
  - 0.15

- learned top-8:
  - 99.71%
  - 0.16

Lesson:
the repair policy itself can be learned from previous search failures, reducing repair exploration by over two orders of magnitude.

Caveat:
the swap action space and move descriptors are still supplied, and training labels use the known hidden feature set.

---

## V66 — Integrated adaptive search stack

Combined:

    greedy
       ->
    learned top-4 repair
       ->
    exhaustive only if residual still looks wrong

Held-out:

- greedy:
  - 91.67%
  - cost 1.000

- learned repair always:
  - 100%
  - cost 1.200

- exhaustive:
  - 100%
  - cost 19.200

- integrated adaptive stack:
  - 100%
  - cost 1.104
  - repair invoked on 8.57%
  - exhaustive invoked on only 0.48%

This is the strongest search-efficiency result so far.

---

# New architectural lesson

The research path now separates several different layers:

```text
problem
   |
   v
cheap attempt
   |
   v
self-evaluate
   |
   +---- sufficient ----> stop
   |
   v
learned repair policy
   |
   v
self-evaluate
   |
   +---- sufficient ----> stop
   |
   v
expensive exhaustive search
```

This suggests "reasoning depth" is not one scalar.

A more general system may need:

- a portfolio of search processes
- learned repair operators/policies
- self-evaluation after each stage
- resource-sensitive escalation
- memory of previous search failures
- learning about how to search from those failures

The key emerging hypothesis is:

> Intelligence may require meta-computation: mechanisms that allocate and improve the search procedures used to construct other computation.

This remains a toy result, not evidence of AGI or a post-Transformer architecture.
