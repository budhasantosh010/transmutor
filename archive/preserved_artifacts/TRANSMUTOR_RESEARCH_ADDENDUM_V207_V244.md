# Transmutor Research Addendum — V207 through V244

This phase moved one level above V192–V206.

The central question became:

> Can the system learn not only a structure, but how to generate structural challengers, how to synthesize new challenger operations from lower-level primitives, how to promote recurrent constructions into reusable macros, how to compose those macros, and how to allocate search depth under finite resources?

The most important discipline in this phase was to distinguish:

- a true scientific result,
- a failed hypothesis,
- an invalid comparison later repaired,
- and an implementation timeout that produced no scientific result.

---

# V207 — Learn which challenger generator to try

Supplied challenger families:

- TREND
- BUMP
- HARMONIC
- FRACTIONAL

Meta-input:

- cheap residual summary features from the incumbent

Supervised meta-target:

- hidden failure family

Held-out:

- hidden-family classification: 66.9%
- top-1 recovery of actual best challenger: 60.6%
- top-2: 80.8%
- all four: 100%

Search:

- top-2 evaluates 2 families instead of 4

Conclusion:

> Structural-search effort can be allocated from residual signatures instead of treating all mutation operators equally.

Caveat:

The operator vocabulary and supervised failure-family labels were supplied.

---

# V208 — Remove failure-family labels

Meta-training receives no hidden-family label.

Instead it observes, for each past episode:

    residual context
    +
    actual validation gain from each tried operator

One utility regressor per operator predicts future usefulness.

Results:

- top-1 best operator recovery: 48.78%
- top-2: 71.0%
- top-1 captures ~75.19% of exhaustive-best gain
- top-2 captures ~87.88%

Conclusion:

> Challenger-generator allocation can be learned from operator success rather than semantic failure labels.

Caveat:

Development still evaluates all operators on past episodes.

---

# V209 — Online partial-feedback challenger allocation

Contextual bandit:

- sees residual context
- chooses one operator
- receives reward only for the chosen operator

Distribution shifts:

Phase A:
- TREND/BUMP common

Phase B:
- HARMONIC/FRACTIONAL common

Phase C:
- roughly uniform

Results over 2400 episodes:

- LinUCB best-operator recovery: 56.5%
- random: 24.75%
- LinUCB mean fractional error reduction: 0.609
- random: 0.375
- oracle audit: 0.704

Conclusion:

> Meta-search can improve online under partial feedback and changing operator frequencies.

---

# V210 — Resource-aware challenger allocation

Synthetic operator costs:

- TREND: 1.0
- HARMONIC: 1.5
- BUMP: 3.0
- FRACTIONAL: 4.0

Raw-reward learner:

- mean raw reward: 0.6131
- mean operator cost: 2.9356
- mean net value: 0.4810

Cost-aware learner:

- raw reward: 0.6109
- operator cost: 2.7402
- net value: 0.4876

Conclusion:

> Meta-search should optimize marginal improvement minus structural-search cost, not raw improvement alone.

---

# V211 — Birth of a new challenger operator from lower-level grammar

Initial named vocabulary:

- TREND
- BUMP
- HARMONIC
- FRACTIONAL

Unseen mechanism:

    STEP = a * INDICATOR(x > c)

The generic fallback grammar already contained threshold atoms.

If threshold atoms repeatedly became adequate generic-search winners:

    promote STEP into named challenger vocabulary

Results:

- promotion occurred
- 8 adequate wins required
- promotion happened after the 55th STEP episode
- mean candidate evaluations on STEP worlds:
  - before promotion: ~96.28
  - after: ~70.34

Conclusion:

> Recurrent successful lower-level constructions can be compiled into reusable challenger operators.

Important caveat:

The threshold primitive itself was supplied. The system promoted a recurring construction; it did not invent comparison semantics from nothing.

---

# V212 / V212b — New operator birth does not automatically create good routing

V212 inserted the promoted STEP operator into a contextual bandit.

Results:

Birth policy:

- STEP mean structural evaluations: 94.82

Closed vocabulary:

- 107.40

So the new arm helped somewhat.

But:

- post-promotion fallback remained ~62%
- direct STEP choice only ~42%

V212b warm-started the new arm from its discovery contexts.

Cold:

- direct STEP: 44.37%
- evaluations: 94.65

Warm:

- direct STEP: 45.55%
- evaluations: 94.39

Conclusion:

> Creating an operator is not equivalent to learning where it applies.

Warm-start from sparse discovery successes was not enough.

---

# V213 / V214 — Applicability learning is bottlenecked by sparse or lossy context

V213:

Train a STEP applicability gate using labels:

    STEP_ATOM was adequate generic-fallback winner

No hidden STEP label.

Positive examples were only ~2.72%.

Results:

- direct STEP routing on STEP worlds: ~0.68%
- almost no search reduction

Negative result.

V214:

Deliberately explore the newborn STEP operator on mixed contexts.

Applicability label:

    STEP beats best old operator by >=3% of incumbent error

Results:

- gate AUC: ~0.685
- direct STEP routing: 2.59%
- only modest search reduction

Conclusion:

> Operator applicability could not be recovered well from the supplied summary features.

This motivated V215.

---

# V215 — What the meta-controller can see matters

Compare:

1. 12 hand-designed residual summary features
2. full ordered residual vector

Applicability AUC:

- summaries: 0.707
- raw residual: 0.992 on training construction used for the comparison

Held-out STEP structural evaluations:

- no gate: 80.72
- summary gate: 77.36
- raw-residual gate: 62.39

STEP direct routing:

- summary: 4.32%
- raw residual: 19.46%

Conclusion:

> Meta-reasoning can fail because the controller's observation representation discards the information required for structural decisions.

---

# V216 — Compress the rich meta-context after finding it

PCA learned on unlabeled residual vectors.

Held-out applicability:

- raw 32-d residual AUC: 0.876
- best <=12 dimensional PCA representation:
  - 4 dimensions
  - AUC: 0.890
  - ~64.75% variance retained
  - 8x dimensional reduction

Conclusion:

> A rich meta-sensor may be needed first; it can then be compressed.

---

# V217 — Multiple operator births do not happen just because mechanisms are real

Generic grammar contained:

- threshold atoms
- hinge atoms

Unseen worlds:

- STEP
- HINGE

Promotion required 8 adequate generic-fallback wins.

Results:

- STEP promoted at stream episode 198
- HINGE never reached promotion threshold
- STEP wins: 8
- HINGE wins: 5

At first this looked like a possible failure.

V218/V218b audited why.

---

# V218 — Invalid specialist comparison

Initial specialist audit froze the incumbent's shared theta before fitting STEP/HINGE specialists.

It suggested specialists were worse than old vocabulary.

This comparison was invalid because the new structure and old shared parameters had not been jointly re-optimized.

V218 is retained as an implementation/scientific-design failure.

---

# V218b — Joint refitting gives the fair marginal value

All families jointly re-optimized shared theta and their own structure.

STEP:

- old vocabulary adequacy: 70.8%
- STEP specialist: 82.0%
- mean marginal gain: +0.0276 incumbent-error units
- meaningful advantage on 39.2% of episodes

HINGE:

- old vocabulary adequacy: 95.6%
- HINGE specialist: 97.2%
- mean marginal gain: -0.0020
- meaningful advantage: 15.2%

Conclusion:

> A real distinct mechanism does not automatically deserve a distinct primitive.

The relevant question is:

    does naming this structure create enough marginal system value
    to justify its maintenance/search/complexity cost?

This explains why HINGE did not promote in V217.

---

# V219 — Finite evidence makes primitive promotion uncertain

Complexity cost:

    0.01 marginal-gain units

Empirical population means from V218b:

- STEP: +0.0276 → should promote
- HINGE: -0.0020 → should reject

Mean-only decisions become more reliable with sample count.

Conservative confidence-gated promotion:

- suppresses false HINGE promotion strongly
- but delays useful STEP promotion substantially

Example at n=120:

- STEP confidence-gated promotion: 63.38%
- HINGE false promotion: 0.1%

Conclusion:

> Primitive birth has a speed-vs-false-abstraction tradeoff.

---

# V220 — Provisional operator evidence stopping

Every five observations:

- promote if lower confidence boundary > complexity cost
- reject if upper boundary < cost
- otherwise remain provisional

Heuristic sequential rule:

- STEP correct decision: 84.8%
- HINGE correct: 72.16%
- mean samples:
  - STEP: 19.82
  - HINGE: 20.96
- balanced correctness: 78.48%

This used much less evidence than fixed 80–120 sample decisions, but lower accuracy.

Conclusion:

> Candidate abstractions can remain provisional and spend evidence until a promotion/rejection decision becomes sufficiently clear.

Caveat:

The confidence rule was heuristic, not anytime-valid statistics.

---

# V221 — Challenger operators compose

Residual contains two mechanisms.

Families:

- TREND
- BUMP
- HARMONIC
- FRACTIONAL
- STEP

Best single operator:

- adequacy: 21.5%

Greedy two-operator composition:

- adequacy: 60.5%
- improves error on 89.13% of episodes
- exact hidden-pair recovery: only 28.75%

Conclusion:

> Exact causal identity is not required for useful compositional repair.

But greedy structural credit assignment is imperfect.

---

# V221b — Search more composition orders

Greedy:

- adequacy: 57.83%
- pair recovery: 28.67%
- ~9 family fits

All ordered pairs:

- adequacy: 68.33%
- recovery: 36.5%
- ~25 family fits

Conclusion:

> Better structural credit assignment improves composition, but search cost rises combinatorially.

---

# V222 — Meta-learn composition search

20 ordered operator pairs.

Meta-context:

- normalized residual vector

Meta-target:

- best ordered pair found by past exhaustive search

Held-out:

Top-1:
- adequacy: 57.86%
- 20x search reduction

Top-3:
- 66.43%
- 6.67x reduction

Top-5:
- 66.67%
- 4x reduction

Exhaustive:
- 68.33%

Conclusion:

> Structural composition search can be amortized into a learned beam.

---

# V223 — Learned structural priors need an escape hatch

Leave one mechanism pair entirely out of meta-training.

Average across leave-one-pair-out folds:

- learned top-3 zero-shot adequacy: 57.33%
- top-3 + exhaustive fallback: 65.92%
- exhaustive: 70.50%
- fallback policy tests ~6.20 pairs on average
- full exhaustive tests 20

Conclusion:

> Search priors should be trusted conditionally, with fallback when unfamiliar composition remains inadequately explained.

---

# V224 — Fallback failures become meta-training data

Initial meta-training excludes BUMP+FRACTIONAL.

Stream repeatedly presents that composition.

Only exhaustive fallback episodes are saved.

Block 1:

- fallback: 45%
- top-3 best recovery: 23.33%

Block 8:

- fallback: 6.67%
- top-3 best recovery: 71.67%

Total fallback-derived examples:

- 59

Conclusion:

> An escape hatch can bootstrap its own future replacement:
> expensive failures become training data that compress future structural search.

---

# V225 — Meta-search itself forgets

After adapting to BUMP+FRACTIONAL:

OLD_ONLY:

- new-pair adequacy: 38.33%
- old-pair mean: 67.50%

NEW_ONLY:

- new: 72.50%
- old: 39.44%

SMALL_REPLAY (59 new + 60 old):

- new: 70.83%
- old: 60.56%

FULL_HISTORY:

- new: 70.83%
- old: 66.94%

Conclusion:

> Continual-learning problems recur at the meta-search level.

The machinery that decides how to search also needs memory/consolidation.

---

# V226 / V226b — Integrated hierarchical structural search

Mixed stream:

- single mechanisms
- two-mechanism compositions

Exhaustive:

    all 5 singles
    +
    all 20 ordered pairs
    =
    25 candidate structures

Adaptive:

1. learned top-1 single
2. if inadequate → learned top-3 pairs
3. if inadequate → all 20 pairs

V226:

- adaptive adequacy: 72.86%
- exhaustive: 78.33%
- adaptive candidates: 2.64
- exhaustive: 25
- ~9.46x search reduction
- full-pair fallback only 2.14%

V226b threshold sweep:

At ~11.36 candidates:

- adaptive adequacy: 78.10%
- exhaustive: 78.33%

Conclusion:

> Hierarchical search can approach exhaustive structural quality while spending far less search on easy cases.

---

# V227 — Real handwritten-digits control

Real perception:

- sklearn handwritten digits
- even vs odd task

Stable channel:

- 4 PCA image dimensions

Synthetic shortcut failure mechanisms:

- DIRECT1
- XOR2
- REDUNDANT4

Repair operators:

- neutralize top 1 aux feature
- top 2
- all 10

All three repairs are optimal on different held-out episodes.

Meta-context:

- IID auxiliary permutation drops

Results:

- exact best repair recovery: 70.83%
- mean net-utility regret: 0.00492
- achieved ~99.23% of exhaustive-best net utility

By hidden failure:

- DIRECT1 exact repair: 62.5%
- XOR2: 50%
- REDUNDANT4: 100%

Conclusion:

> On real perception data, exact repair identity was less important than choosing a near-optimal intervention.

Caveat:

Shortcut variables/failures were synthetic.

---

# V228 — Remove named STEP/HINGE challenger operators

This begins the neutral-grammar phase.

No named STEP/HINGE structure supplied.

Primitive language:

Terminals:
- x
- numeric constants

Operations:
- +
- -
- *
- tanh

Randomly generated/deduplicated expression library:

- 7000 expressions
- max depth 4

Each expression receives only affine output calibration.

Hidden residual worlds:

- STEP
- HINGE

Held-out results:

STEP:

- old vocabulary: 50.77%
- neutral expressions: 92.31%
- named specialist oracle audit: 89.62%

HINGE:

- old vocabulary: 79.62%
- neutral expressions: 98.46%
- specialist: 99.62%

Conclusion:

> Useful missing challenger shapes can be synthesized from lower-level computational primitives without a named STEP/HINGE operator.

Important caveat:

The primitive language, constants, max depth, expression generator, and affine calibration are supplied.

---

# V229 — Promote recurring synthesized expressions

Discovery uses full 7000-expression search on separate STEP/HINGE episodes.

Promote expressions by recurrence frequency.

Held-out:

Full 7000:

- overall adequacy: 92.08%

Promoted 32:

- 92.08%
- STEP: 87.92%
- HINGE: 96.25%
- search reduction:
  7000 / 32
  ≈ 218.75x

Promoted 64:

- 93.13%

Random 32:

- 79.17%

Conclusion:

> Expensive primitive discovery can be amortized into a tiny recurring macro vocabulary.

---

# V230 — Novel WINDOW exposes the flat-grammar ceiling

New unseen mechanism:

    WINDOW = A * 1[c1 < x < c2]

32 old macros:

- adequacy: 2.5%

Full depth-4 7000-expression neutral grammar:

- 17.86%

Old macros + fallback:

- 16.43%

Fallback triggered:

- 94.64%

Promoting fallback WINDOW winners only reduced fallback partially:

- ~63–72% depending new macro count

Conclusion:

> The system correctly detects novelty, but reopening the same shallow grammar is not enough when the missing concept requires greater compositional depth.

This is an important negative result.

---

# V231 — Compose old macros to create WINDOW

Use only 32 promoted STEP/HINGE-derived macros.

Search all unordered pairs:

    y = b + a1*M_i + a2*M_j

Pair candidates:

- 496

Results on unseen WINDOW:

- one macro: 2.38%
- flat depth-4 7000 expressions: 20.48%
- two old macros: 69.76%

Search:

- 496 pair structures vs 7000 flat expressions
- ~14.11x fewer candidate structures

Conclusion:

> Hierarchical abstraction can create a function class that the sampled flat grammar missed.

---

# V232 — Meta-learn macro-pair search

496 possible pairs.

Learn top-K pair beam from previous WINDOW exhaustive searches.

Held-out:

- all 496: 69.52%
- top-1: 56.67%
- top-5: 66.90%
- top-10: 68.81%
- top-20: 70.24%

Top-10:

- ~49.6x search reduction
- nearly exhaustive adequacy

Conclusion:

> Search over learned abstractions can itself become an abstraction.

---

# V233 — Synthesize the macro-composition law

Remove the supplied linear-combination rule.

Composition language terminals:

- A
- B
- constants

Operations:

- +
- -
- *
- tanh

Neutral composition-rule library:

- 420 random/deduplicated expressions
- max depth 3

Search:

- learned top-10 macro pairs
- × composition rules

Results:

- top-10 + supplied linear: 61%
- top-10 + synthesized laws: 72%
- all 496 + supplied linear: 62%

Examples of recurrent winning laws include nonlinear interactions such as:

    tanh((A+A)*B)
    tanh(A*A + A*B)
    tanh((B-A)*A)

Conclusion:

> Even the rule for composing abstractions can be synthesized from a lower-level operation language.

Cost:

- 4200 pair-rule combinations per episode

---

# V234 — Promote recurrent composition laws

Use V233 recurrence to create a compact reusable algebra.

All 420 laws:

- adequacy: 73.89%
- 4200 pair-rule combinations

Promoted 16 laws:

- 72.78%
- 160 combinations
- 26.25x rule-search reduction

Promoted 64:

- 75.56%

Random laws are consistently worse.

Conclusion:

> A compact algebra of recurrent composition laws can amortize composition-rule discovery.

Note:

The first V234 implementation timed out because of redundant nested prediction/evaluation.
It produced no scientific result.
The valid cached/batched rerun is the one reported here.

---

# V235 — Transfer the compact algebra to an unseen composition type

New unseen target:

    TENT = normalized relu(x-left) * relu(right-x)

No TENT-specific meta-training.

Results:

- single old macro: 10.0%
- flat 7000 expressions: 46.94%
- all 496 linear macro pairs: 95.83%
- transferred compact WINDOW algebra:
  - 32 recurrent pair structures
  - 16 recurrent laws
  - 94.17%

Conclusion:

> The learned macro algebra transferred beyond the exact WINDOW family to another boundary-composition mechanism.

Caveat:

This remains the same 1-D domain with related boundary structure.

---

# V236 — Complexity / MDL pressure

Select expression by:

    training MSE
      +
    lambda * node count

Unpenalized:

- adequacy: 94.09%
- mean expression nodes: 16.22

lambda = 5e-5:

- adequacy: 93.41%
- mean nodes: 10.06
- median nodes: 10

lambda = 1e-4:

- adequacy: 91.82%
- mean nodes: 8.57

Conclusion:

> Description-length pressure can remove a large amount of expression bloat with little performance loss.

---

# V237 — Recursive abstraction initially fails

Level 1:

- composed WINDOW structures promoted as macros

New higher-order target:

    DOUBLE_WINDOW = window_1 + window_2

Results:

- flat depth-4: 6.11%
- one promoted WINDOW macro: 6.39%
- pair of promoted WINDOW macros: 16.39%

Conclusion:

> Merely stacking fixed promoted functions is not sufficient for deeper reusable abstraction.

This triggered V237b/V238.

---

# V237b — Diversity helps only modestly

Compare Level-1 macro promotion rules for DOUBLE_WINDOW.

32 macros:

- recurrence: 17.31%
- diversity-preserving: 16.15%
- random: 14.62%

64 macros:

- recurrence: 23.85%
- diversity: 25.38%

Conclusion:

> Coverage/diversity matters, but fixed-function macro libraries still hit a strong ceiling.

The next suspected bottleneck:

    promoted macros contain baked-in constants
    rather than reusable parameter slots

---

# V238 — Parameter lifting

Take recurrent synthesized expression skeletons.

For each:

- every numeric constant becomes a tunable parameter
- output remains affinely calibrated

Repeated STEP experiences select the most reusable skeleton.

Selected skeleton:

    tanh(((x*x)*(x*-3.0)) + (x + (3.0+3.0)))

with its numeric constants lifted into parameters.

Held-out STEP:

- 32 frozen macros: 86.67%
- full 7000 search: 86.67%
- one parameterized discovered skeleton: 85.83%
- named STEP oracle audit: 84.17%

Mean optimizer evaluations:

- ~46.19

Conclusion:

> One reusable parameterized template can replace dozens of fixed macros / thousands of flat candidates on the same structural family.

Trade:

    combinatorial library search
        →
    parameter fitting

---

# V239 — Repeated instances of one parameterized template

Greedy residual fitting.

WINDOW:

1 instance:
- 0%

2:
- 22.22%

3:
- 47.78%

4:
- 56.67%

Flat 7000 reference:
- 17.78%

DOUBLE_WINDOW:

1–4 instances:
- ~5.6% → 7.8%

Conclusion:

> Re-instantiating one template increases functional complexity for WINDOW, but greedy fitting and one-template expressivity fail on DOUBLE_WINDOW.

---

# V239b — Joint structural credit assignment

Greedy instances are jointly re-optimized.

WINDOW, 2 instances:

- greedy: 12.5%
- joint: 56.25%

WINDOW, 4:

- greedy: 50%
- joint: 39.58%

DOUBLE_WINDOW, 4:

- greedy: 4.17%
- joint: 12.5%

Conclusion:

> Joint credit assignment can strongly rescue small compositions, but larger nonlinear compositions remain optimization/template-limited.

---

# V240 — Small dictionary of parameterized skeleton types

Select three reusable skeleton types from recurrent neutral expressions.

At every greedy stage:

- fit all three parameterized templates
- choose the best current residual reduction

Results:

WINDOW, 3 instances:

- one template: 42.86%
- 3-template dictionary: 47.62%

WINDOW, 4:

- 47.62%
- 50.0%

DOUBLE_WINDOW, 4:

- 2.38%
- 21.43%

Conclusion:

> Template-type diversity materially improves harder composition, but fitting cost becomes very large.

DOUBLE_WINDOW mean optimizer evaluations:

- ~534.8

---

# V241 — Learn which parameterized template to instantiate

Meta-context:

- normalized current residual vector

Meta-target:

- which of the three templates gives lowest training residual after fitting

Held-out four-stage repair:

WINDOW:

- learned selector adequacy: 47.37%
- fit-all-three: 57.89%
- optimizer evaluations:
  - learned: 168.0
  - all3: 513.9
- ~3.06x optimization reduction

DOUBLE_WINDOW:

- learned: 10.53%
- all3: 18.42%
- ~3.07x cost reduction

Conclusion:

> Parameterized-template search can be amortized into a meta-policy, but selector errors compound through recursive stages.

---

# V242 — Confidence-gated template fallback

If template selector confidence is low:

- fit all three instead of one

Results were mixed.

WINDOW:

- threshold .45:
  - 50% adequacy
  - ~369.9 optimizer evaluations

Higher thresholds:

- more fallback/cost
- no quality improvement in this small test

DOUBLE_WINDOW:

- .45:
  - 29.17%
- .60+:
  - 33.33%
  - ~560 evaluations

Conclusion:

> Fallback helps harder residuals, but classifier confidence is not sufficiently calibrated to be a clean value signal.

---

# V243 — Adaptive structural depth

The first implementation timed out due redundant nonlinear refitting.
It produced no scientific result.

Valid rerun:

- caches each four-stage repair trajectory once
- evaluates stopping rules afterward

Only 8 episodes per family, so treat as directional.

Zero additional growth price:

Mean instances:

- STEP: 1.0
- HINGE: 1.0
- WINDOW: 1.875
- TENT: 2.125
- DOUBLE_WINDOW: 3.625

Aggregate:

Adaptive:

- mean adequacy: 70.0%
- mean optimizer evaluations: 192.75
- mean instances: 1.925

Always 4:

- adequacy: 82.5%
- evaluations: 445.45

Conclusion:

> Residual-driven growth allocates more structural depth to harder mechanisms and cuts compute, but the current stop rule sacrifices too much quality on compositional tasks.

---

# V244 — Matched-size random function control

Critical control for V228.

Compare equal library sizes:

Neutral grammar:

- 7000 expressions from x/constants/+/-/*/tanh

Control:

- 7000 random smooth Fourier-mixture functions
- each = 3 random sinusoidal components + weak linear term

Identical:

- affine calibration
- training-MSE selection
- held-out evaluation

STEP:

- neutral grammar: 89.17%
- random Fourier: 57.08%

HINGE:

- neutral: 97.92%
- random Fourier: 95.0%

Overall:

- neutral: 93.54%
- random: 76.04%

Conclusion:

> V228 is not explained purely by "search enough random smooth functions."

The structured neutral primitive language has a substantial advantage, especially on STEP.

Caveat:

Fourier mixtures are one control family; they are globally smooth/oscillatory while tanh/multiplication are naturally suited to boundaries.

---

# Strongest conclusions after V244

## 1. Search policy is itself a learnable object

The system can learn:

- which mutation operator to try
- which structural composition to try
- which macro pair to test
- which parameterized template to instantiate

This repeatedly cuts structural-search cost.

---

## 2. Fallback is not just a safety mechanism

Repeated pattern:

    learned cheap prior
        |
        v
      failure
        |
        v
    expensive fallback
        |
        v
   save fallback result
        |
        v
    improve future prior

V224 demonstrated this especially clearly.

---

## 3. Primitive birth should be utility-driven, not ontology-driven

HINGE is a distinct real mechanism.

But if the old language already models it well enough:

    new primitive may not be worth maintaining

STEP had a larger marginal gap and deserved promotion more often.

---

## 4. A primitive can be synthesized before it is named

V228:

    x/constants/+/-/*/tanh
        |
        v
    random structural compositions
        |
        v
    STEP/HINGE-like useful shapes

No named STEP or HINGE operator was needed.

This is still not primitive-operation invention, but it is substantially closer to the original Transmutor question.

---

## 5. Repeated useful synthesized expressions can become macros

V229:

    7000 discovery candidates
       ↓
      32 macros
       ↓
    same held-out adequacy
       ↓
    ~219x less expression search

---

## 6. Hierarchy can create functions absent from the sampled flat grammar

WINDOW:

- flat depth-4 grammar: ~20%
- pair of old macros: ~70%

This is one of the strongest results in the batch.

---

## 7. Composition law can itself be synthesized

V233 did not receive:

    "add the two macros linearly"

It searched a neutral two-input rule grammar.

The synthesized law library outperformed the supplied-linear baseline.

---

## 8. Composition laws can also be promoted

V234:

- all 420 laws: 73.89%
- promoted 16: 72.78%
- ~26x reduction

So the hierarchy is recursive not only in representations, but in the operations used to combine representations.

---

## 9. Learned macro algebra showed genuine, but limited, transfer

WINDOW-trained compact algebra:

- unseen TENT: 94.17%

This strongly beats the flat 7000-expression grammar on that transfer task.

But both live in the same boundary-composition domain.

---

## 10. Description-length pressure is useful

Capability did not require maximal expression complexity.

A substantial fraction of formula bloat could be removed with little loss.

---

## 11. Fixed macros are not enough — parameter slots matter

V237 exposed the fixed-function ceiling.

V238 showed:

    32 frozen macros / 7000 flat expressions
        ≈
    1 parameterized discovered template

on STEP.

A reusable abstraction is increasingly looking like:

    structure
      +
    parameter slots
      +
    applicability policy
      +
    composition rules
      +
    complexity cost

not merely a frozen function.

---

## 12. Parameter fitting reintroduces a new search problem

V239–V243 show that after compressing discrete structural search into parameterized templates:

    combinatorial search cost decreases
    but
    nonlinear optimization / credit assignment becomes the bottleneck

This is an important correction.

Compression does not make cost disappear.
It moves cost.

---

# Current candidate architecture

```text
EXPERIENCE
    |
    v
CURRENT EXPLANATION
    |
    v
RESIDUAL / FAILURE SIGNAL
    |
    v
META-SENSOR
    |
    +--> preserve rich context when needed
    |
    +--> compress learned context later
    |
    v
STRUCTURAL SEARCH PRIOR
    |
    +--> choose challenger-generator family
    +--> choose macro pair
    +--> choose parameterized template
    +--> choose composition law
    |
    v
CHEAP CANDIDATE SEARCH
    |
    v
ADEQUATE?
  /   \
 yes   no
 |      |
stop   FALLBACK TO LOWER-LEVEL GRAMMAR
        |
        v
   synthesize structure
        |
        v
  refit / validate
        |
        +--> poor marginal value -> reject
        |
        +--> provisional -> gather evidence
        |
        +--> repeated + valuable -> promote
                                    |
                                    v
                             PARAMETER LIFTING
                                    |
                                    v
                          reusable template/macro
                                    |
                                    v
                          learn applicability
                                    |
                                    v
                       learn composition search
                                    |
                                    v
                         promote recurring laws
                                    |
                                    v
                             deeper hierarchy
```

---

# The remaining deepest supplied assumptions

After V244, these are now unusually specific.

## A. Primitive operation vocabulary is still supplied

We still give:

- x
- constants
- +
- -
- *
- tanh

The system composes these, but does not invent arithmetic/nonlinearity semantics from a completely neutral substrate.

---

## B. Numeric parameter optimization is external

Parameter lifting relies on conventional nonlinear least squares.

We do not yet have a self-discovered/local parameter-learning rule for newly invented templates.

---

## C. Composition depth/growth operators are supplied

We allow:

- pair two macros
- instantiate another template
- promote a recurrent expression
- lift constants into parameters

The system does not yet invent all of these meta-operations from first principles.

---

## D. Structural value metrics are still engineered

Examples:

- validation error
- adequacy threshold
- node-count complexity
- synthetic operator cost
- confidence thresholds

The system does not yet derive its own general resource/value accounting law.

---

## E. Most neutral-grammar experiments remain 1-D synthetic environments

V227 gives a real-data control for meta-repair.

But V228–V244's strongest architecture-synthesis results remain low-dimensional and synthetic.

---

# Strongest next experiments

1. **Learn parameter-update rules for lifted templates**
   instead of calling nonlinear least-squares.

2. **Discover parameter slots automatically**
   rather than using the hand-coded rule:
       every numeric constant becomes tunable.

3. **Discover growth/composition operators**
   rather than handing the controller:
       pair, instantiate, promote, lift.

4. **Move neutral grammar synthesis to a real dataset**
   where a discovered template must repair or explain real structure.

5. **Use hardware-level cost**
   instead of candidate-count / optimizer-evaluation proxies.

6. **Open-ended primitive ecology**
   where promoted macros compete for finite memory/maintenance budget
   and can later die if no longer marginally useful.

7. **Learn an anytime-valid promotion criterion**
   for provisional primitives under nonstationary task distributions.

---

# Current scientific status

Still NOT demonstrated:

- AGI
- ASI
- a post-Transformer paradigm
- autonomous invention of computation from no primitive language

Now demonstrated in these toy/controlled settings:

- challenger-generator allocation can be learned
- partial-feedback meta-search can adapt online
- structural search can be resource-aware
- recurrent lower-level constructions can become named operators
- exact mechanism identity is not required for near-optimal action
- structural composition matters
- learned search beams can approximate exhaustive search cheaply
- fallbacks can train future search priors
- meta-search itself suffers continual forgetting
- missing challenger shapes can be synthesized from a lower-level neutral grammar
- recurring synthesized expressions can compress into macros
- hierarchical macro composition can create functions absent from sampled flat grammar
- composition laws can themselves be synthesized and promoted
- a compact learned algebra can transfer to related unseen compositions
- description-length pressure can remove formula bloat
- parameter lifting can convert a frozen synthesized formula into a reusable template
- adaptive depth can allocate more instances to harder tasks
- large-library success is not fully explained by a matched-size random Fourier lottery

The emerging hypothesis is now sharper:

> Intelligence may be a recursive resource-allocation process that continually decides which representations, computations, experiments, parameters, search rules, abstractions, and abstraction-composition laws deserve to exist — while retaining an escape path back to a lower-level language when its current abstractions fail.
