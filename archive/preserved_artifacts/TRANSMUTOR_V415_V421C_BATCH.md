# Transmutor Experiments V415–V421c

## Purpose

Continue attacking the major remaining scaffolds after V414:

1. amortize expensive raw-trajectory structure search,
2. separate search failure from observational identifiability failure,
3. infer minimal predictive state and understand measurement-noise effects,
4. grow differential feature vocabularies,
5. compose missing features from generic primitives,
6. design active experiments when model hypotheses share blind spots,
7. compare brute-force observation history against compact latent uncertainty state,
8. grow parameterized feature families from residual structure,
9. alternate structural levels and continuous parameter refinement.

This batch includes several important failed attempts/corrections.

---

# V415 — Learned retrieval for expensive raw-trajectory law search

Law library:
y'' = c0 + c1*x + c2*y + c3*y' + c4*(y')²

Candidate supports:
all one- or two-term subsets = 15 structures.

A raw-trajectory retriever was trained on 2,600 synthetic law tasks.

Retriever recall:
- @1: 23.17%
- @2: 38.67%
- @3: 51.50%
- @5: 72.50%

On 45 held-out raw nonlinear-fitting tasks:

Top-3 search:
- true-support accuracy: 28.89%
- agreement with exhaustive search: 46.67%
- candidate-fit reduction vs exhaustive: 80%
- residual/integration-call reduction: ~84.46%

Exhaustive 15-candidate search:
- true generating support selected only 33.33%

Important conclusion:
The main problem was not just retrieval/search speed.
**A single noisy trajectory often did not identify the generating law uniquely enough.**

This is an identifiability/data problem.

---

# V415b first attempt — FAILED due development-search cost

Tried to fit 15 candidate ODE structures jointly across four trajectories with:

- shared law coefficients,
- separate initial states per trajectory.

The computation timed out.

This is consistent with V414:
raw-trajectory structural fitting becomes extremely expensive as candidate/state complexity grows.

No result claim from this attempt.

---

# V415b correction — Multiple independent trajectories improve identifiability

Same law observed through four independent initial conditions.

Raw multi-trajectory retriever recall:

- @1: 36.6%
- @2: 58.2%
- @3: 69.2%
- @5: 88.8%

For a cheaper shared derivative-regression identifiability test:

One trajectory:
- true-support recovery 24.8%

Four trajectories:
- 72.2%

Absolute gain:
+47.4 percentage points.

Conclusion:
**Diverse experiences of the same law can be more valuable than additional search on one trajectory.**

A system should distinguish:

search harder

from

collect a more informative experience.

Caveat:
The corrected support evaluator uses numerical derivatives because joint raw fitting was too expensive.

---

# V416 — FAILED naive predictive-state-order selection

Raw scalar sequences generated from:

1. observable 1D latent system
2. observable 2D oscillator
3. physical 2D system with an extra unobservable state

Candidate prediction memories:
AR(1)..AR(4).

Training BIC unexpectedly selected:

- 1D: modal AR(2)
- 2D oscillator: modal AR(4)
- unobservable-extra 2D: modal AR(2)

This did not match clean latent dynamic order.

No clean state-dimension claim.

---

# V416b — FAILED simple blocked predictive-sufficiency correction

Changed rule to:

choose smallest AR order within 10% of best blocked-validation MSE.

Still over-selected large orders:

- 1D: modal AR(3)
- 2D: modal AR(4)
- unobservable extra: modal AR(3)

Reason:
Additive measurement noise changes the observable stochastic process.
Longer observation history really can improve state estimation.

So the apparent "overgrowth" was partly real predictive utility.

---

# V416c — Clean vs noisy state requirement

Separated observation noise.

## Clean observations

Minimal exact recurrence order:

- observable 1D: AR(1), 220/220
- observable 2D oscillator: AR(2), 220/220
- unobservable extra physical state: AR(1), 220/220

Thus clean experience identifies the **minimal observable predictive dynamics**, not physical hidden dimensions that never affect observations.

## Noisy observations

Best predictive AR order among 1..4 became overwhelmingly AR(4) for all three systems.

Interpretation:

latent dynamics
+
measurement uncertainty

can require additional predictive memory/state.

The right architecture may be:

compact latent state
+
uncertainty/filter state

rather than naive physical-state dimension.

---

# V417 — Residual-driven feature vocabulary growth

Base differential features:

1, x, y, y', (y')²

New systems:

PENDULUM:
y'' = -sin(y) - 0.12y'

VAN DER POL:
y'' = 1.4y' - 1.4y²y' - y

Candidate feature bank included several transforms/interactions.

Correct missing features were selected:

- Pendulum -> sin(y)
- Van der Pol -> y²y'

Heldout errors fell from:

Pendulum:
~0.3765 -> ~1e-30

Van der Pol:
~3.7126 -> ~7e-30

However V417 then added useless extra features because relative improvement from 1e-30 to a smaller numerical number looked enormous.

---

# V417b — Adequacy stop prevents solved systems from overgrowing

Added:

stop structural growth once heldout MSE <= 1e-12.

Result:

Pendulum:
only birth = sin(y)

Van der Pol:
only birth = y²y'

No numerical-noise proxy growth.

Conclusion:
Structural evolution needs both:

improvement pressure

and

an adequacy / "good enough" stop.

Without it, solved systems can keep mutating forever.

---

# V418 — Disagreement and coverage expose different blind spots

Initial observations:
x in [0.7,2]

Candidate experiments:
x in [0.1,6]

True unseen laws:
exponential, log, cubic.

Strategies:
- random
- model disagreement
- coverage/farthest point
- simple hybrid

Median current-model error at chosen experiment:

## Exponential
- random: 0.306
- disagreement: 546.3
- coverage: 27.0

## Log
- random: 0.040
- disagreement: 0.227
- coverage: 0.227

## Cubic
- random: 2.63
- disagreement: 1.20
- coverage: 36.08

For cubic, coverage was ~30x stronger than disagreement by median.

Conclusion:
Model disagreement is powerful only when hypotheses disagree.
If they share the same blind spot, coverage/exploration can be much better.

The simple additive hybrid did not solve the strategy-choice problem.

---

# V418b — Diverse two-probe experiment portfolio

Query budget = 2.

Strategies:
- two random
- two disagreement peaks
- two coverage points
- one disagreement + one coverage

Median strongest counterexample:

## Exponential
- two random: 3.16
- disagreement: 1142
- coverage: 34.1
- diverse portfolio: 1142

## Log
- two random: 0.101
- disagreement: 0.222
- coverage: 0.222
- portfolio: 0.222

## Cubic
- two random: 8.28
- disagreement: 1.15
- coverage: 35.91
- portfolio: 36.19

Across all three families, the diverse portfolio improved median strongest counterexample by at least ~2.20x vs two random probes.

Conclusion:
When the architecture is uncertain about its own blind spot, **diverse experiment policies can hedge diagnostic uncertainty**.

Cost:
two experiments instead of one.

---

# V419 first attempt — INVALID unequal NLL accounting

Compared AR history to one-state latent Kalman model.

But AR(k) skipped its first k observations while Kalman was charged for all observations.

This made BIC/NLL comparison unfair.

No claim from first attempt.

---

# V419 corrected — Long observation history vs compact uncertainty-aware latent state

True world:

z_{t+1} = 0.91 z_t + process noise(q=.035)

y_t = z_t + observation noise(r=.16)

Candidate architectures:

AR(1)..AR(8)

vs

one-state linear-Gaussian latent filter with learned:
a, q, r

All models scored on identical horizon:
t >= 8.

Learned latent parameters:

- a = 0.90195
- q = 0.03662
- r = 0.16280

Heldout NLL:

Best AR:
AR(8) = 0.7027035

One-state latent:
0.7026277

Latent advantage:
~7.58e-5 NLL.

BIC:

Best AR BIC:
AR(6) = 23778.84

Latent BIC:
23725.44

Latent advantage:
~53.40 BIC points.

Conclusion:
The value of long history under measurement noise can sometimes be explained more compactly as:

small latent state
+
explicit uncertainty/filter model.

So state growth and observation-model growth are competing explanations.

Caveat:
The correct linear-Gaussian state-space family is supplied.

---

# V420 — Compose missing law features from generic primitives

Removed named domain features such as:

sin_y

y_squared_times_yprime.

Generic terminals:

x, y, v

Unary primitives:

SIN, COS, TANH, SQUARE, ABS

Binary primitive:

MUL

Automatically generated candidate expressions:
185.

Results:

Pendulum:
birth = SIN(y)

Van der Pol:
birth = MUL(v,SQUARE(y))

Both drove heldout derivative MSE to ~1e-30.

Conclusion:
Domain-specific composite law features can emerge from a lower-level compositional grammar.

Remaining scaffold:
the primitive transforms themselves are supplied.

---

# V421 — Parameterized feature-family growth after compositional insufficiency

New forced Duffing world:

y'' =
-0.20y'
-1.0y
-0.50y³
+0.70 cos(1.70x)

V420 grammar contains COS(x) but cannot express unknown-frequency cos(omega*x).

Stage A compositional search became confused by the unmodeled forcing and promoted proxy features:

- MUL(TANH(y),ABS(v))
- MUL(x,y)
- SIN(y)

Validation MSE remained:
~0.147

Residual spectrum detected:

coarse omega:
1.74117

refined omega:
1.7002367

true:
1.7000000

Adding parameterized:

COS_FREQ(x;omega)=cos(omega*x)

reduced validation MSE to:
~1.14e-4

but did not fully solve the system.

Conclusion:
Composition alone can be insufficient.
Residual structure can trigger a new **parameterized feature family**.

---

# V421b — Structural discoveries change the residual landscape

After COS_FREQ existed, discarded the earlier proxy features and re-ran compositional search.

Now the correct missing Duffing nonlinearity became obvious:

MUL(y,SQUARE(y))
= y³

Validation MSE:

pre-family composition:
~0.147

after frequency-family birth:
~1.14e-4

after y³ re-composition:
~1.07e-6 initially.

But small frequency error still generated numerical residual, leading to additional weak proxy features.

Important conclusion:
**A higher-level structural discovery can invalidate earlier lower-level feature choices.**

Structural search may need to alternate and reconsider earlier commitments.

---

# V421c — Joint continuous refinement after discrete structure discovery

Restricted to meaningful structure:

base features
+
y³
+
cos(omega*x)

Then jointly refined omega while refitting linear coefficients.

Recovered:

omega:
1.699999999906

absolute error:
~9.36e-11

coefficients:

y:
-1.000000000045

y':
-0.200000000351

y³:
-0.500000000006

cos forcing:
0.700000000076

Heldout MSE:
~1.58e-19

Adequacy reached.

Conclusion:
Structural search must coordinate:

discrete feature discovery

with

continuous parameter refinement.

Otherwise small parameter error can masquerade as missing structure and cause proxy growth.

---

# Strongest new conclusions from V415–V421c

## 1. Search failure and information failure are different
V415/V415b:

Trying harder on one trajectory is not equivalent to observing more independent trajectories.

Four diverse trajectories increased structural support recovery from ~24.8% to ~72.2% in the shared regression control.

## 2. The right "state dimension" depends on what must be predicted
V416–V416c:

Clean latent dynamics may be first/second order.

Measurement noise can make longer history useful.

Therefore:

physical state size
!=
minimal predictive state size under uncertainty.

## 3. Uncertainty modeling can be an alternative to state/history growth
V419:

AR history kept improving through order 8.

A one-state noise-aware latent model slightly beat AR8 on heldout NLL and had a substantially better BIC.

## 4. Law vocabularies can grow from residual value
V417:

Missing nonlinear features can be promoted when they eliminate heldout residuals.

## 5. Growth must stop when the problem is solved
V417b:

Relative numerical improvements after ~1e-30 error caused pointless feature births.

An adequacy criterion prevents endless evolution.

## 6. Composite law concepts can be built from generic primitives
V420:

SIN(y)

and

y²*y'

were generated automatically from lower-level transforms and multiplication.

## 7. Composition and family invention are distinct structural levels
V421:

The grammar could build y³ but could not express unknown-frequency forcing.

That required a new parameterized feature family.

## 8. Structural levels interact non-monotonically
V421/V421b:

Before external forcing was modeled, greedy search chose proxy nonlinear features.

After family growth, the correct y³ term became discoverable.

Earlier structural decisions may need to be revoked.

## 9. Discrete structure and continuous parameter refinement must alternate
V421c:

A tiny frequency error generated residuals that looked like missing features.

Joint frequency refinement removed the false pressure for further growth.

## 10. Active experiment policies need diversity
V418/V418b:

Disagreement excels when hypotheses differ.

Coverage excels when hypotheses share blind spots.

A two-probe disagreement+coverage portfolio was robust across several unseen-law types.

---

# Updated architecture hypothesis

The architecture is increasingly looking like a system that manages competing explanations at multiple levels:

RAW EXPERIENCE
    |
    v
IS THERE ENOUGH INFORMATION?
    |
    +--> no: acquire diverse / active experience
    |
    v
MEASUREMENT + UNCERTAINTY MODEL
    |
    v
PREDICTIVE STATE
    |
    +--> history growth?
    +--> latent-state growth?
    +--> observation-noise/filter state?
    |
    v
DYNAMICAL LAW
    |
    v
FEATURE / PRIMITIVE GRAMMAR
    |
    +--> compose known primitives
    +--> instantiate parameterized family
    +--> invent/grow family
    |
    v
DISCRETE STRUCTURE
    <---- alternate ---->
CONTINUOUS PARAMETERS
    |
    v
HELDOUT FALSIFICATION
    |
    +--> solved: STOP GROWTH / CRYSTALLIZE
    |
    +--> unsolved:
            reconsider earlier structure
            collect new experience
            expand vocabulary
            change state
            refine parameters
    |
    loop

---

# Major remaining gaps after V421c

1. Primitive transforms such as SIN, SQUARE, MUL are still supplied.

2. Parameterized family templates such as cos(omega*x) are still supplied once spectral structure is detected.

3. The architecture does not yet autonomously decide whether to:
   - collect more data,
   - search more,
   - add state,
   - add uncertainty,
   - compose features,
   - grow a new family,
   except in separate controlled experiments.

4. Joint nonlinear fitting remains extremely expensive.

5. Learned retrieval is still weak when the task is poorly identifiable.

6. Active experiment selection in high dimensions remains open.

7. Earlier structural commitments need a principled revision mechanism.

8. Need persistent lifelong library evolution rather than isolated batches.

9. Need real-world/noisy multivariate systems.

10. Need hardware/numerical stability integrated into every search objective.

---

# Strong next targets

- Build a meta-controller that chooses between:
  SEARCH MORE / GET MORE DATA / ADD STATE / ADD NOISE MODEL / GROW LANGUAGE.

- Learn the adequacy threshold from estimated noise and downstream utility.

- Learn when to revoke earlier proxy features after a higher-level structural birth.

- Persistent lifelong stream where useful primitives/macros are born, retrieved, archived, and revived.

- Extend the law grammar to multivariate coupled systems.

- Learn experiment portfolios under explicit acquisition cost.

- Joint state-space + feature-law discovery from raw noisy observations.

