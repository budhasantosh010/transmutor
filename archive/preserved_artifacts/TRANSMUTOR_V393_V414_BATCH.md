# Transmutor Experiments V393–V414

## Purpose

Continue removing scaffolds from the Transmutor hypothesis.

This batch targets:

1. learning generic reductions without a special SUM_ALL recognizer,
2. selecting update programs from raw sequence likelihood,
3. jointly learning update structure + world parameters,
4. learning macro retrieval rather than hand-routing it,
5. discovering variable-length loop laws from final outputs,
6. selecting loop-body families and loop-state size,
7. crystallizing approximate recurrent behavior into exact reusable algorithms,
8. comparing algebraically equivalent implementations under finite precision,
9. resource-aware implementation selection,
10. synthesizing a numerically stable replacement after implementation failure,
11. learning failure diagnostics from raw residuals,
12. attacking diagnostic OOD/generalization,
13. active counterexample selection,
14. deciding capacity-growth vs function-family failure,
15. representing several named laws inside one differential-law language,
16. removing exact-derivative supervision,
17. adapting the derivative/smoothing scale,
18. fitting differential laws directly from raw trajectories,
19. counting the development-search cost of robust discovery.

---

# V393 — Repeated tree edit learns a reduction schema

Previous V378 used a special anti-unifier that explicitly flattened ADD trees and recognized SUM_ALL.

V393 instead received symbolic expressions for K=2,3,4 and searched a generic local tree-edit family:

- replace subtree S by ADD(S,newvar)
- replace subtree S by MUL(S,newvar)
- replace subtree S by DIV(S,newvar)

Exactly one repeated rule mapped both:
K2 -> K3
and
K3 -> K4

Selected:

at denominator path (1):
S <- ADD(S, NEWVAR)

Recursive extrapolation produced the correct expressions at K=5,8,16.

Errors:
- machine precision exact.

Conclusion:
A generic repeated AST transformation can infer a reduction-growth rule without an explicit SUM_ALL detector.

Caveat:
The tree-edit family and symbolic expressions are supplied. Iteration itself is still not invented.

---

# V394 — Raw sequence likelihood selects the update program

Inputs:
- known A
- known E
- raw observation sequences

No belief/posterior supervision.

Candidate grammar:
- old belief B
- PRIOR = B@A
- LIKE = E@onehot(obs)
- MUL
- ADD
- NORMALIZE

1,053 expressions tested.

Best:
MUL(PRIOR,LIKE)

Held-out test NLL:
0.9030451977387021

Bayesian oracle:
0.9030451977387021

Gap:
0

Conclusion:
The correct inference composition can be selected by raw future prediction alone.

Caveat:
A/E and the expression grammar are supplied; candidate belief states are normalized automatically.

---

# V395 — Joint structure + world-parameter learning from raw sequences

Removed known A/E.

Each candidate update structure learned its own:
- transition matrix
- emission matrix

Structures:
- CORRECT
- NO_TRANS
- ADDITIVE
- PRIOR_ONLY
- LIKE_ONLY

Three optimization restarts each.

Held-out NLL:
- CORRECT: 0.9018804
- ADDITIVE: 0.9110371
- LIKE_ONLY: 0.9512016
- NO_TRANS: 1.0350165
- PRIOR_ONLY: 1.0763881

True oracle:
0.9017977

Correct selected from raw likelihood.

Gap to oracle:
~8.27e-5 NLL.

Conclusion:
Inside this small family, raw experience can jointly select the update structure and learn its numerical world parameters.

---

# V396 — Learned macro router using hand features

Tasks measured whether normalization macros were useful.

A decision tree received generic numerical task descriptors rather than a hard-coded probability-target rule.

Router accuracy:
98.15%

However raw generated-state count alone made ALWAYS_BASE look cheap because failure was not penalized.

Therefore raw-state-count interpretation of V396 is incomplete.

---

# V396b — Failure-adjusted macro routing

Utility:
generated semantic states
+ 250 penalty if unsolved.

Router accuracy on utility-optimal action:
98.15%

Mean utility:
- always base: 175.78
- always macros: 127.89
- learned router: 98.76
- oracle router: 96.56

Learned improvement:
- 43.82% vs always base
- 22.78% vs always macros

Conclusion:
Selective primitive retrieval can be learned from past search outcomes and substantially reduce search/failure cost.

Caveat:
Failure penalty and summary features supplied.

---

# V397 — Macro routing directly from raw task examples

Removed hand-engineered task statistics.

Router sees only sets of:
[x, y, target]

DeepSets-style encoder:
per-coordinate representation
-> pool
-> macro/base decision

Held-out accuracy:
88.89%

Failure-adjusted utility:
- base: 175.78
- all macros: 127.89
- raw learned router: 111.11
- oracle: 96.56

Solve rate:
raw router = 77.78%, same as always-macro and oracle in this task mix.

Conclusion:
Retrieval can be learned directly from raw task examples, though there remains a gap to a feature-engineered router.

---

# V398 — Variable-length reduction learned from final output only

Task:
SUM(sequence)

Training lengths:
2,3,4,5

Generic recurrent accumulator:
state_next = ws*state + wi*x + b

Only final sum supervised.

Learned:
- ws = 1
- wi = 1
- bias ~1.37e-8

Crystallized:
state_next = state + x

Unseen lengths:
6,10,32,64,128

Crystallized loop:
machine precision exact.

Conclusion:
A dimension-generic reduction can emerge from end-to-end variable-length behavior without symbolic anti-unification.

Caveat:
A shared recurrent accumulator scaffold is supplied.

---

# V399 — Loop-body family selection

Candidate loop-body families:

AFFINE:
a*s+b*x+c

BILINEAR:
a*s*x+b*s+c*x+d

Tasks:
SUM
PRODUCT

Training lengths:
2..5

Selection:
simplest family with validation MSE <1e-8.

SUM:
- affine exact
- bilinear also can collapse to affine
- selected AFFINE
- crystallized state <- state+x

PRODUCT:
- affine validation MSE ~1.07
- bilinear ~9.18e-15
- selected BILINEAR
- crystallized state <- state*x

Long lengths up to 64 remain near machine precision.

Conclusion:
Variable-length behavior can select different loop-body function families.

---

# V400 — Two-state variable-length mean

Task:
arithmetic mean of variable-length sequence.

1D recurrent state:
best validation MSE ~0.03975.

2D recurrent state + rational readout:
best validation MSE ~4.38e-5.

2D hidden state linearly decodes:
- sequence sum R² = 0.999985
- sequence count R² = 0.997845

But long-length behavior drifted:
2D length64 MSE ~0.19167.

Conclusion:
The system learned essentially the right information (sum + count), but not an exact update implementation.

Important:
This again separates state information from implementation-law precision.

Caveat:
1D and 2D readouts are not equally expressive, so this is not a pure dimensionality theorem.

---

# V401 — Exact mean through learned primitive reuse

Available learned summaries:
- SUM_X
- SUM_ONES (count)
- PRODUCT_X
- LAST_X

Readouts:
ADD, MUL, DIV, SUB.

Search one summary first.

Best one-summary:
LAST_X
MSE ~0.69275
not exact.

Grow to two summaries.

Best:
DIV(SUM_X,SUM_ONES)

Training MSE:
0

Unseen lengths:
8,16,32,64,128,256

All exact.

Conclusion:
Previously learned reductions can crystallize an approximate neural representation into an exact reusable algorithm.

---

# V402 — Variance exposes memory/compute implementation tradeoff

Two exact implementations:

ONE PASS:
E[x²] - E[x]²
- sequence traversals: 1
- peak accumulators: 3
- replay not required

TWO PASS:
mean((x-mean)²)
- traversals: 2
- peak accumulators: 2
- replay/storage required

Two-pass transform search also found:
x*(x-mean)

which is algebraically exact as well.

All algorithms agree at machine precision on ordinary float64 data.

Conclusion:
The same function can have multiple exact implementations with different resource profiles.

---

# V402b — Equivalent algebraic laws diverge under finite precision

Compared:

A) squared residual
mean((x-mean)^2)

B) x times residual
mean(x*(x-mean))

C) naive one-pass
mean(x²)-mean(x)²

Float64 with offset 1e9:
- squared residual relative error: 0
- x*residual: ~61
- naive: ~130

Float64 offset 1e12:
- squared residual: 0
- x*residual: ~5.87e7
- naive: ~1.35e8

Float32 degrades much earlier and eventually input representation itself loses the small variation.

Conclusion:
Symbolic equivalence is insufficient. Numerical/hardware behavior is part of implementation quality.

---

# V403 — Resource-aware selection among equivalent algorithms

Candidates:
- NAIVE_ONE_PASS
- CENTERED_TWO_PASS
- WELFORD_ONE_PASS

Measured numerical error under:
- float32 / float64
- offsets 1, 1e3, 1e6, 1e9

Resource contexts selected different implementations:

STREAMING_FLOAT32:
WELFORD

MEMORY_TIGHT + REPLAY + FLOAT64:
CENTERED_TWO_PASS

CHEAP_MODERATE_FLOAT64:
NAIVE_ONE_PASS

STREAMING_FLOAT64_HIGH_OFFSET:
WELFORD

All three algorithms were selected somewhere.

Conclusion:
There is no universally best equivalent implementation. Resource context and precision can rationally change the chosen algorithm.

Caveat:
Utility weights supplied.

---

# V404 — Synthesized stable online variance replacement

After naive failure, existing primitive language was used to synthesize:

COUNT_UPDATE:
ADD(n,one)

MEAN_UPDATE:
SUB(mean,DIV(SUB(mean,x),n_next))
= mean + (x-mean)/n_next

M2_UPDATE:
ADD(M2,MUL(delta,delta2))

Packaged as:
WELFORD_STEP

Float64 offset 1e9:
- naive median relative error ~128.46
- synthesized stable step ~2.09e-8

Conclusion:
Once appropriate state variables are exposed, the primitive library can construct a stable replacement rather than merely selecting one from a named list.

Caveat:
Prefix count/mean/M2 targets are supplied during component synthesis.

---

# V405 — Raw residual diagnostic router

Classes:
- AFFINE
- BILINEAR
- RATIONAL
- HARMONIC
- NO_GROW

Input:
sets of raw normalized [x,y,residual] samples.

No handcrafted FFT/polynomial features.

ID held-out:
100% accuracy.

This looked suspiciously clean, so an OOD attack followed.

---

# V405b — OOD attack destroys diagnostic confidence

Hard shifts:
- weak/noisy affine
- bilinear hidden under affine clutter
- near-linear rational
- unseen harmonic frequency + AR noise
- high-phi heavy-tailed stochastic noise

Overall accuracy:
25%

Per class:
- AFFINE: 31.36%
- BILINEAR: 11.36%
- RATIONAL: 6.36%
- HARMONIC: 0.45%
- NO_GROW: 75.45%

Yet confidence remained extremely high.
Harmonic true cases:
mean max softmax confidence ~0.9964.

Conclusion:
Closed-set failure diagnosis can be catastrophically and confidently wrong under distribution shift.

---

# V406 — Prototype novelty detector

Embedding-distance abstention:
threshold = 99th percentile ID distance.

ID:
- 99% accepted
- 100% accuracy among accepted

Hard OOD:
- only 2% rejected
- accepted accuracy ~24.86%

Unseen families:
- 35.1% rejected overall
- CUBIC 60%
- STEP 74.1%
- EXPONENTIAL 0%
- LOG 6.36%

Conclusion:
Simple embedding novelty is inadequate.

---

# V406b — Diagnostic ensemble disagreement

7 independently bootstrapped routers.

Abstention threshold:
99th percentile ID ensemble disagreement.

ID:
- 1% abstain
- 100% accepted accuracy

Hard OOD:
- 40.27% abstain
- accepted accuracy still ~21.46%

Unseen:
- 77.5% abstain

Breakdown:
- CUBIC 99.55%
- STEP 100%
- LOG 68.64%
- EXPONENTIAL 41.82%

Conclusion:
Multiple hypotheses expose more novelty than a single model, but correlated shared blind spots remain severe.

---

# V407 — Predictive competition instead of trusting diagnostic labels

Candidate families must improve held-out residual prediction.

Random point split.

Hard OOD:
mean selected heldout MSE ratio to constant baseline:
0.345

However AR-correlated NO_GROW cases were often assigned HARMONIC because random splitting leaked the same realization across train/test.

Therefore V407's random-split diagnostic is unsafe for correlated data.

---

# V407b — Blocked extrapolation correction

Sort by x.
Train first 65%.
Test final 35%.

Hard OOD:
- overall no-grow rate 13.6%
- true NO_GROW no-grow rate 32%
- harmonic no-grow 26%
- bilinear almost always correctly useful
- rational often approximated by affine

This improves leakage control but still does not fully prevent spurious structure selection.

Unseen families:
known families can often provide useful local/extrapolative approximations:
- EXPONENTIAL often approximated by HARMONIC
- LOG often by RATIONAL
- CUBIC/STEP often by AFFINE locally

Conclusion:
A wrong family can still be useful locally, which can hide the need for a genuinely new family.

---

# V408 — Active counterexample selection

True families outside current library:
- EXPONENTIAL
- LOG
- CUBIC

Initial observations:
x in [0.7,2]

Known model library:
- AFFINE
- QUADRATIC
- RATIONAL
- HARMONIC

Next experiment chosen either:
- random point in [0.1,6]
- point of maximum disagreement among surviving known models

Median current-best-model error:

EXPONENTIAL:
- random: 0.451
- active disagreement: 987.7
- median error multiplier ~207x
- active beats random in 92% trials

LOG:
- random: 0.0614
- active: 0.2363
- ~5.05x median multiplier
- active beats random 97.4%

CUBIC:
- active only ~1.66x median improvement
- known models often share blind spots

Conclusion:
Active counterexamples can expose locally-good wrong models much faster, but disagreement fails when all surviving models are wrong in the same way.

---

# V409 — Capacity growth vs family mismatch

Candidate family:
polynomial degree 1..10.

Train:
[0.1,4.5]

Extrapolate:
[4.5,6]

Exactness threshold:
MSE < 1e-10

CUBIC:
smallest exact degree = 3.

EXPONENTIAL:
degree10 MSE ~7.12e-9
not exact under budget.

LOG:
error worsens at high polynomial degree;
degree10 MSE ~1.95.

Conclusion:
Some failures are solved by growing capacity within a family; others remain budget-relative evidence for changing representation/family.

Caveat:
Polynomials can approximate exponential/log on bounded intervals, so this is not a mathematical impossibility proof.

---

# V410 — Generic sparse differential-law language

Instead of named EXP/LOG/CUBIC families, use:

y'' =
c0
+ c1*x
+ c2*y
+ c3*y'
+ c4*(y')²

Exact derivatives supplied.

Sparse fit + integration.

Discovered laws included:

CUBIC:
y'' = -0.44 + 1.86*x

LOG:
y'' = -0.769230...*(y')²

EXPONENTIAL:
threshold regression found an exact but nonminimal collinear representation.

Extrapolation improved enormously vs degree-10 polynomial.

---

# V410b — Minimum-description differential-law selection

Enumerated all 31 non-empty feature subsets.

Choose:
fewest terms among derivative MSE <1e-20.

Recovered minimal forms:

CUBIC:
y'' = -0.44 + 1.86*x

EXPONENTIAL:
y'' = 0.62*y'

LOG:
y'' = -0.7692307692*(y')²

All correct.

Conclusion:
Several named function families collapse into compact laws in one common differential representation.

Caveat:
Feature library and exact derivatives supplied.

---

# V411 — Noisy y-only differential discovery

Removed exact derivative supervision.

Learner sees noisy y(x).

Savitzky-Golay:
window 61, order 4

Then estimate derivatives and select sparse differential law by BIC.

Exact structure recovery:

sigma=.01:
- cubic 92%
- exponential 92%
- log 100%

sigma=.05:
- cubic 0%
- exponential 76%
- log 64%

Median extrapolation error increases rapidly with noise.

Conclusion:
Derivative/state estimation becomes the bottleneck even when the law language is adequate.

---

# V412 — Predictively select derivative/smoothing scale

Candidate Savitzky windows:
21,31,41,61,81,121,161.

For each:
- estimate derivatives on [0.1,3.8]
- discover differential law
- integrate into heldout [3.8,4.5]
- choose window by noisy heldout y prediction

At sigma=.05:

CUBIC:
- fixed61 recovery 0%
- adaptive 50%
- median extrapolation MSE 0.02857 -> 0.00627

EXPONENTIAL:
- recovery 76% -> 77.78%
- extrapolation 0.04719 -> 0.01048

LOG:
- recovery 64% -> 66.67%
- extrapolation 7.77e-5 -> 1.50e-6

Conclusion:
The measurement/derivative scale itself can be selected by downstream predictive value.

Caveat:
Smoother family/window candidates supplied.

---

# V413 — Raw-trajectory differential-law fitting, no numerical derivatives

Observation:
raw noisy y(x) only.

Candidate laws:
all 1- or 2-term subsets of the five-term second-order ODE library.

For each candidate:
- fit structural coefficients
- fit unknown initial y and y'
- integrate the ODE
- minimize raw trajectory error
- choose by BIC

Noise sigma=.05, 5 runs:

CUBIC:
- exact minimal support recovery 100%
- median extrapolation MSE ~0.00144

EXPONENTIAL:
- 100%
- ~0.00440

LOG:
- 80%
- ~0.000190

This is substantially more robust than V411's fixed derivative route in this small sample.

Unexpected clean-data issue:
- exponential/log sometimes selected an unnecessary extra term at sigma=0 because tiny numerical integration-fit differences outweighed description penalties.
- prediction remained essentially exact.

Conclusion:
Raw-trajectory fitting can remove derivative estimation but creates a much more expensive nonlinear structure-search problem and its own model-selection quirks.

---

# V414 — Development-search cost of robust discovery

Compare at sigma=.05:

DERIVATIVE_ROUTE recovery reference:
- cubic 0
- exponential .76
- log .64
mean ~.467

RAW_TRAJECTORY_ROUTE:
- cubic 1
- exponential 1
- log .8
mean ~.933

Development proxy:

Derivative route:
31 linear subset fits.

Raw trajectory, mean residual/integration calls across 15 nonlinear candidate subsets:
- cubic ~873
- exponential ~1,019
- log ~1,388

Ratio to 31 linear fits:
~28x to ~45x in count,
while each raw-trajectory residual call itself integrates ~100 time steps.

Conclusion:
More robust law discovery can require dramatically more development/search compute.

This reinforces an earlier Transmutor theme:
**development cost, runtime cost, representational compactness, and reliability are separate quantities.**

---

# Strongest surviving conclusions from V393–V414

## 1. End-to-end raw prediction can select internal update laws
V394/V395 remove posterior supervision, and V395 also learns world parameters.

## 2. Iterative/reduction behavior can emerge without symbolic anti-unification
V398 learns state <- state+x from final variable-length targets.

But:
iteration/recurrent parameter sharing remains supplied.

## 3. Learned primitives should be retrieved selectively
V396b/V397:
always enabling a growing primitive library wastes search.
A router can learn when macros pay for themselves.

## 4. Approximate learned state can be crystallized into exact algorithms
V400 learned sum/count-like state but drifted.
V401 recombined SUM + COUNT + DIV and became exact through length 256.

## 5. One function can have multiple equivalent implementations
V402:
one-pass vs two-pass variance.

## 6. Algebraic equivalence does not imply implementation equivalence
V402b:
finite precision separates mathematically identical programs by many orders of magnitude.

## 7. Algorithm choice should depend on resources/hardware
V403:
different contexts rationally chose naive one-pass, centered two-pass, or Welford.

## 8. Stable replacement algorithms can be synthesized from an existing primitive library
V404 constructed Welford-style recurrence components after naive numerical failure.

## 9. Failure diagnosis is currently fragile
V405 looked perfect ID and collapsed to 25% OOD while remaining highly confident.

## 10. Diversity helps but does not solve correlated blind spots
V406b rejects many unseen families but still accepts many shifted known-family errors.

## 11. Structural growth should be falsifiable
V407/V407b:
candidate families should earn heldout predictive value rather than being trusted by label/class resemblance.

## 12. Active experiments can reveal locally-good wrong models
V408:
model disagreement made exponential/log counterexamples much stronger than random probing.

## 13. Capacity growth and language/family growth are different responses
V409:
cubic stops at degree 3; exp/log remain outside the chosen polynomial budget.

## 14. A broader law language can collapse several named families into one representation
V410b:
cubic, exponential and log all become sparse second-order differential laws.

## 15. Measurement/state estimation can dominate law-discovery quality
V411:
noise destroys derivative estimates before the symbolic law language itself fails.

## 16. The measurement process should also be adaptive
V412:
selecting smoothing scale by downstream prediction improved high-noise discovery.

## 17. Removing an intermediate estimator can improve robustness but greatly increase development search cost
V413/V414:
raw trajectory fitting roughly doubled mean structural recovery at sigma=.05 in this small comparison, but with orders of magnitude more nonlinear search/integration work.

---

# Current narrowed hierarchy

The emerging architecture is no longer a single "model".

It looks more like:

RAW EXPERIENCE
    |
    v
MEASUREMENT / FEATURE ESTIMATION
    |
    v
PREDICTIVE STATE
    |
    v
WORLD MODEL
    |
    v
UPDATE / DYNAMICAL LAW
    |
    v
ALGORITHM / IMPLEMENTATION EQUIVALENCE CLASS
    |
    v
PRIMITIVE + MACRO LIBRARY
    |
    v
CONTEXTUAL RETRIEVAL
    |
    v
FAILURE DIAGNOSIS
    |
    +--> parameter refinement
    +--> more search
    +--> more state
    +--> different implementation
    +--> more family capacity
    +--> new function/law representation
    +--> new experiment
    |
    v
ACTIVE FALSIFICATION
    |
    v
CRYSTALLIZE / PROMOTE / ARCHIVE / PRUNE
    |
    loop

Resource constraints exist at every layer:

- data
- measurement noise
- state/memory
- runtime arithmetic
- replay/storage
- numerical precision
- search/development compute
- diagnostic reliability
- hypothesis diversity

---

# Biggest remaining gaps after V414

1. **The generic recurrent/iteration scaffold is still supplied.**
   V398 discovers a loop body, not iteration itself.

2. **The differential feature library is supplied.**
   Need feature/law-library growth rather than fixed five-term search.

3. **Raw-trajectory structure search is expensive.**
   Need learned search policies / amortized retrieval.

4. **Diagnostic OOD remains unsolved.**
   Ensemble disagreement and novelty are only partial defenses.

5. **Closed model families still share blind spots.**
   Active disagreement fails if every surviving model makes the same mistake.

6. **New-family invention is still mostly template-based.**

7. **Many experiments remain synthetic.**
   Need harder real or richer simulated dynamical environments.

8. **Numerical stability must be included in program fitness, not checked afterward.**

9. **Need joint optimization of state + law + implementation + resource policy**, rather than solving each separately.

10. **Need persistent lifelong model/library evolution**, not mostly batch-isolated experiments.

---

# Strong next targets

- Learn a reusable search policy that predicts which law subsets/programs are worth testing before expensive integration.
- Jointly select state dimension + differential law from raw trajectories.
- Allow the differential feature library itself to grow from residual patterns.
- Active counterexample generation when all current hypotheses agree.
- Persistent lifelong task stream with macro archive/retrieval and family birth/death.
- Numerical-stability-aware program synthesis as part of fitness.
- Test discovered differential/reduction abstractions on richer physical or control-like systems.
