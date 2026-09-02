# Transmutor Experiments V422–V428

## Purpose

Continue from V421c by attacking the control/lifecycle layer:

1. learn when structural growth should stop from measured noise,
2. revoke earlier proxy structure after better explanations arrive,
3. build a first meta-controller that chooses different intervention types,
4. attack that meta-controller out of distribution,
5. require proposed interventions to prove value before commitment,
6. extend meta-control to temporal state/noise/data/language choices,
7. preserve failures in benchmark design and correct them,
8. build lifelong structural memory with revision,
9. scale archive retrieval beyond trying every learned structure,
10. allocate verification itself as a resource.

---

# V422 — Noise-calibrated adequacy instead of a fixed tiny error threshold

Problem:
V417b stopped growth with a human-set MSE threshold of 1e-12.

New setup:
- noisy quadratic and cubic regression worlds
- repeated measurements at identical x values estimate observation-noise variance
- structural growth starts from degree 1 and can add x²...x⁸

Compare:

FIXED_TINY:
stop only if validation MSE <= 1e-12

NOISE_CALIBRATED:
stop if validation MSE <= 1.15 × estimated noise variance

Noise estimates were accurate:
estimated/true variance was roughly 0.98–1.01.

Examples:

Quadratic, sigma=.01:
- fixed exact structure: 52.5%
- noise-calibrated: 93.33%
- mean max degree: 3.79 -> 2.26

Quadratic, sigma=.15:
- fixed exact: 49.17%
- calibrated: 83.33%
- mean degree: 3.79 -> 2.57

Cubic, sigma=.05:
- fixed exact: 47.5%
- calibrated: 80%
- mean degree: 4.59 -> 3.38

Conclusion:
**Adequacy should be tied to the measurable uncertainty floor.**
A near-zero target causes unnecessary structural growth when irreducible noise exists.

Caveat:
Repeated identical measurements and the 1.15 multiplier are supplied.

---

# V423 — Structural revision: delete earlier proxy features

Revisited forced Duffing:

y'' = -0.20y' - y - 0.50y³ + 0.70 cos(1.70x)

Lifecycle:
1. greedy compositional growth
2. periodic family discovery/refinement
3. continued composition
4. backward deletion of any learned feature whose removal keeps the model adequate

Historical extra features:
- MUL(y,SQUARE(y))
- MUL(x,TANH(y))
- TANH(x)

Revision removed:
- TANH(x)
- MUL(x,TANH(y))

Final learned feature:
- MUL(y,SQUARE(y)) = y³

Refined frequency:
1.69999999389

True:
1.7

Heldout MSE:
~9.15e-16

Conclusion:
**Structural learning must support deletion/revision, not only monotonic growth.**

---

# V424 — First raw intervention meta-controller

Single scalar regression interface.

Controlled situations:
- already adequate
- search deeper in current polynomial language
- obtain more data
- grow a trigonometric language
- refine a continuous frequency parameter

Actions:
- STOP
- SEARCH_MORE
- GET_MORE_DATA
- GROW_LANGUAGE
- REFINE_PARAMETER

Each action was actually executed.
Oracle action = clean test MSE + small intervention cost.

Controller input:
raw current-model sets:
[x, y, prediction, residual]

No scenario label.

Heldout familiar-family action accuracy:
81.43%.

By controlled source:
- adequate: 67.1%
- search more: 82.9%
- more data: 75.7%
- grow language: 92.9%
- refine parameter: 88.6%

Mean utility:
- always stop: 1.5408
- learned controller: 0.001382
- oracle: 0.000817

Conclusion:
Raw failure behavior contains enough information to route several qualitatively different structural interventions in a controlled setting.

---

# V424b — OOD attack on the intervention controller

New, unseen failure regimes:
- quartic search
- data-limited cubic
- smooth square-wave-like target
- high-frequency parameter refinement
- already-adequate linear law

Accuracy fell:
81.4% familiar -> 63.75% OOD.

Important failure:
ADEQUATE_LINEAR:
- oracle STOP in 79/80
- controller accuracy: 0%

Overall confidence remained high:
mean max softmax ~0.825.

Mean utility:
- learned: 0.0515
- oracle: 0.0192
- ratio ~2.68x oracle

Conclusion:
The meta-controller has the same closed-set brittleness previously observed in failure-diagnostic classifiers.

---

# V424c — Verify proposed interventions before committing

Correction to V424b:

Controller ranks actions.

Fresh verification data is used to test only:
- STOP
- controller top-1
- controller top-2

Then choose the lowest verification MSE + action cost.

OOD results:

Raw controller:
- action accuracy 64.57%
- mean clean utility 0.10808

Verify top2 + STOP:
- action accuracy 91.71%
- mean utility 0.02218

Verify all five:
- action accuracy 93.71%
- mean utility 0.02242

Oracle:
- utility 0.01991

Adequate linear:
- raw accuracy 1.43%
- top2+STOP verification 94.29%

Top2 verification improved clean utility by ~79.48% vs the raw controller.

Conclusion:
**The meta-controller should propose actions, not be trusted as an unquestioned oracle.**
Structural interventions should prove predictive value on fresh evidence.

---

# V425 — Temporal meta-controller, INITIAL VERSION INVALID FOR GROW_LANGUAGE CLAIM

Temporal situations:
- adequate AR1
- more history/state
- noise-filter state
- more data
- nonlinear law

Actions:
- STOP
- ADD_STATE
- ADD_NOISE_MODEL
- GET_MORE_DATA
- GROW_LANGUAGE

Overall action accuracy:
81.78%.

However the nonlinear recurrence used for GROW_LANGUAGE often settled into a regime where linear AR1 was already enough.

Oracle GROW_LANGUAGE count:
0.

Therefore V425 cannot support a full five-action temporal-controller claim.

This benchmark flaw was preserved and corrected.

---

# V425b — Correct temporal meta-control benchmark

Replaced weak nonlinear task with noisy logistic dynamics:

y_{t+1} = r y_t (1-y_t) + noise

r in [3.55,3.90]

The polynomial autoregressive language can represent this nonlinear update.

Now all five interventions appeared as oracle-best choices.

Overall action accuracy:
82.4%.

Per controlled source:
- adequate: 92%
- add state: 98%
- add noise model: 64%
- get more data: 58%
- grow language: 100%

Mean utility:
- learned: 0.06537
- oracle: 0.06365

Hardest ambiguity:
ADD_NOISE_MODEL vs GET_MORE_DATA.

Conclusion:
Raw temporal prediction failures can route state, noise/filter, data, and language interventions in a controlled benchmark.

---

# V425c — Verify temporal interventions on fresh sequences

Fresh verification:
2 new sequences.

Compare:

Raw controller:
- action accuracy 81.09%
- mean utility 0.06910

Verify top2 + STOP:
- 88.0%
- mean utility 0.06895

Verify all:
- 87.64%
- mean utility 0.06937

Oracle:
- 100%
- 0.06683

By source:

ADD_NOISE_MODEL:
- raw 60%
- verified top2 80%

GET_MORE_DATA:
- raw 50.91%
- verified top2 72.73%

But total utility improved only ~0.21%, because many competing temporal actions had similar predictive value.

Conclusion:
Verification can strongly improve structural-action identification while producing only a small utility gain when interventions are near-equivalent.

---

# V426 — Lifelong structural archive and reuse

Persistent task stream:
1. Pendulum
2. Van der Pol
3. Duffing
4. linear oscillator
5. Pendulum again
6. Van der Pol again
7. Duffing again

Generic candidate pool:
185 compositional features.

First discoveries:

Pendulum:
SIN(y)

Van der Pol:
MUL(v,SQUARE(y))

Duffing:
ultimately includes MUL(y,SQUARE(y))

First-time structural tests:
~185–186.

Returning tasks:
3 retrieval tests.

Rediscovery reductions:
~98.38%.

Linear task:
archive contains 3 learned features but active learned features = 0.

Conclusion:
**Persistent memory does not require every learned structure to remain active.**
Archived abstractions can drastically reduce future discovery cost.

Issue:
First Duffing task temporarily activated archived proxies.

---

# V426b — Lifelong archive + revision

Lifecycle changed to:

retrieve
-> discover
-> solve
-> backward revise
-> promote only surviving new structure

Duffing before revision:
- SIN(y)
- y³

Revision removed:
SIN(y)

Final Duffing active:
y³ only.

Final archive:
- SIN(y)
- y²v
- y³

Returning task cost:
4 total structural evaluations including revision.

First vs repeat reductions:
- Pendulum ~97.85%
- VDP ~97.86%
- Duffing ~97.89%

Conclusion:
A lifelong system can keep durable abstractions while preventing temporary explanatory proxies from polluting the active structure.

---

# V427 — Learned content retrieval from a 185-feature archive: PARTIAL FAILURE

Archive:
all 185 compositional grammar features.

A generic neural pair scorer was trained on synthetic tasks:

input:
[residual, candidate feature value]

Pair-classification accuracy:
99.56%.

Physical tests:
84 Pendulum / VDP / Duffing tasks.

Top-5 exact candidate fitting:
- only 5 exact fits instead of 185
- overall adequate solve 67.86%

But severe family failure:

Pendulum top-5 adequate:
3.57%

VDP top-5:
100%

Duffing top-5:
100%

Reason:
Raw candidate-feature similarity ignores the portion of the candidate already explainable by the current base model.

Conclusion:
High synthetic retrieval accuracy did not transfer reliably to real residual context.

---

# V427b — Correct retrieval representation by residualizing the candidate

For current base design X:

target residual:
r_y = target - projection_X(target)

candidate residual:
r_f = feature - projection_X(feature)

Retrieval score:
corr(r_y, r_f)^2

This asks whether a candidate explains information not already in the current representation.

Physical evaluation:
120 tasks.

Results:

Top-1 candidate:
- adequate solve 100%
- matches exhaustive best 100%

Top-3:
- 100%
- exact candidate-fit reduction vs 185 = 98.38%

Pendulum:
V427 top-5 = 3.57%
V427b top-1 = 100%

Conclusion:
For additive linear feature growth, **retrieval must be conditioned on the current representation**, not only on raw content similarity.

This is a particularly clean result.

Caveat:
The residualized single-feature rule does not solve nonlinear joint-synergy retrieval.

---

# V428 — Verification itself should be resource-allocated

Always verifying meta-actions consumes new experience.

Policy:
verify top2+STOP only if controller confidence < threshold.

Temporal V425c episodes:

Raw:
- action accuracy 81.09%
- verification rate 0%

Threshold .70:
- verify only 29.82% of episodes
- action accuracy 86.18%
- mean utility 0.06803

Threshold .80:
- verify 43.64%
- accuracy 88.0%
- utility 0.06829

Always verify:
- verification 100%
- accuracy 88.0%
- utility 0.06895

Interestingly, intermediate verification rates produced slightly better utility than always verifying.

Conclusion:
**Verification is itself a computational/data action that should be selectively allocated.**

Caveat:
Softmax confidence is not a robust uncertainty measure under strong distribution shift.

---

# Strongest conclusions from V422–V428

## 1. "Solved" should be defined relative to uncertainty

V422:
Noise-calibrated adequacy strongly reduced overgrowth and improved true structural recovery.

Intelligence should not chase zero error when the world itself is noisy.

## 2. Structural evolution must be reversible

V423 / V426b:
Earlier proxy features can become unnecessary after a stronger explanation emerges.

Growth-only evolution accumulates conceptual garbage.

## 3. Different failures really require different actions

V424 / V425b:
Controlled raw behavior contains signals distinguishing:
- stop,
- search more,
- get more data,
- add predictive state,
- add filter/noise state,
- grow language,
- refine parameters.

## 4. Meta-control is not trustworthy without falsification

V424b:
OOD action routing collapsed.

V424c:
Fresh evidence checking top2+STOP repaired most of the damage.

This suggests:

proposal
-> cheap test
-> commitment

rather than:

classifier
-> irreversible action.

## 5. More information and more computation are different interventions

V415b plus V425:
Some ambiguity comes from insufficient observations, not insufficient search.

## 6. Persistent structural memory can amortize discovery

V426b:
~98% reduction in structural evaluations on repeated dynamical families.

## 7. Long-term memory should be archival and sparse-active

A linear task used none of the archived features even though all remained available for later revival.

## 8. Retrieval must be contextual

V427 raw feature retrieval failed badly on Pendulum.

V427b residualized features against the current representation and recovered the exhaustive-best archive feature at rank 1 on all 120 tested additive physical tasks.

A concept is useful not because it resembles the error in isolation, but because it explains error **left over after what is already known**.

## 9. Control layers themselves consume resources

V428:
Verifying only ~30–44% of uncertain episodes captured much of the accuracy gain of always verifying.

Therefore the architecture must decide not only:
"what should I do?"

but also:
"how much should I spend deciding what to do?"

---

# Updated architecture hypothesis

RAW EXPERIENCE
    |
    v
ESTIMATE NOISE / UNCERTAINTY FLOOR
    |
    v
CURRENT REPRESENTATION + PREDICTIVE STATE
    |
    v
FAILURE / RESIDUAL
    |
    v
META-CONTROLLER
    |
    +--> STOP
    +--> SEARCH MORE
    +--> GET MORE / DIVERSE DATA
    +--> ADD STATE
    +--> ADD FILTER / NOISE MODEL
    +--> GROW LANGUAGE
    +--> REFINE PARAMETERS
    |
    v
PROPOSE TOP INTERVENTIONS
    |
    v
IS VERIFICATION WORTH THE COST?
    |
    +--> yes: test on fresh evidence
    +--> no: commit
    |
    v
EXECUTE
    |
    v
DID A BETTER EXPLANATION MAKE OLD STRUCTURE REDUNDANT?
    |
    +--> yes: delete/revise old structure
    |
    v
ADEQUATE RELATIVE TO NOISE?
    |
    +--> yes: stop growth / crystallize
    +--> no: continue
    |
    v
ARCHIVE SURVIVING ABSTRACTIONS
    |
    v
CONTEXTUAL RETRIEVAL ON FUTURE TASKS
    |
    v
REUSE / REVIVE
    |
    loop

---

# Major remaining gaps

1. Static and temporal intervention controllers are still separate.

2. The action set is human-supplied.

3. Intervention costs are manually specified.

4. Verification uses fresh data that may be expensive/unavailable.

5. Confidence is not robustly calibrated OOD.

6. Archive retrieval V427b exploits linear additive structure.

7. Retrieval is still O(archive size) for cheap screening; true sublinear indexing is open.

8. The system does not yet invent a new intervention type when none of the existing actions works.

9. State/law/noise/data decisions are not jointly optimized in one persistent world.

10. Most experiments remain controlled synthetic systems.

---

# Strong next targets

- unify static + temporal intervention routing,
- let the controller abstain when none of its intervention types is adequate,
- learn intervention cost/value from lifecycle outcomes,
- discover a new intervention action rather than selecting a fixed list,
- test joint-synergy archive retrieval where no single feature is useful alone,
- build sublinear archive indexing,
- persistent changing environment where controller, archive, state, and law co-evolve,
- move to harder noisy multivariate dynamical systems.
