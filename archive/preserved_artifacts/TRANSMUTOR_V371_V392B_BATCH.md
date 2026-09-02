# Transmutor Experiments V371–V392b

## Purpose

Push beyond learned state representations and world-model parameters into:

1. discovery of inference update programs,
2. recursive macro / primitive promotion,
3. primitive routing to avoid search bloat,
4. bootstrapping higher-level vector operations from scalar arithmetic,
5. discovering missing primitive operations,
6. selecting primitive function families,
7. growing new function families from structured failure residuals,
8. distinguishing deterministic missing structure from stochastic colored noise,
9. changing-world model-family growth and reuse.

This batch contains several deliberate failures and corrected follow-ups.

---

## V371 — Discover probabilistic update composition from lower-level vector ops

Inputs:
- belief vector b
- transition matrix A
- precomputed likelihood vector l

Available instructions:
- MATMUL_A(vector)
- MUL(vector,vector)
- ADD(vector,vector)
- SUM(vector)
- DIV(vector,scalar)

No normalize/Bayes primitive.

Breadth-first synthesis discovered exactly:

1. v2 = MATMUL_A(v0)
2. v3 = MUL(v1,v2)
3. s0 = SUM(v3)
4. v4 = DIV(v3,s0)

Search:
- 7,001 BFS states expanded
- 34,306 deduplicated states seen

Unseen K=3 test:
- max error 0
- MSE 0

Conclusion:
The factorized probabilistic update composition can be synthesized from lower-level linear algebra/arithmetic primitives.

Caveat:
Primitive vocabulary still supplied.

---

## V372 — Cross-state-count algorithmic transfer

The V371 program was discovered only at K=3.

No retraining/re-search.

Tested:
K = 2,4,5,8,16

Stress cases:
- highly peaked beliefs
- near-deterministic transition matrices
- likelihood values spanning 1e-4 to 1

All tests:
- exact machine-precision zero error.

Conclusion:
The discovered composition is shape-generic rather than a K=3 lookup trick.

---

## V373 — FAILED raw-observation program search

Harder inputs:
- b
- A
- emission matrix E
- raw one-hot observation o

Search needed to construct:
- prior = b @ A
- likelihood = E @ o
- normalized product

A target-MSE beam search failed.

Best intermediate normalized-view error became good, but useful intermediate concepts were discarded because they did not immediately resemble the final posterior.

Conclusion:
**Search heuristic failure can destroy correct compositional paths even when the primitive language is sufficient.**

No valid V373 success claim.

---

## V373b — Landmark-guided correction

Same task/primitives as V373.

Stage 1:
- exhaustive semantic search to depth 3
- identify intermediate vectors whose **proportions** match the target under hypothetical unit-sum rescaling

Exactly one proportional landmark emerged.

Stage 2:
- continue only from landmark states using actual executable primitives

Discovered:

1. v2 = VM(v0,A)
2. v3 = MV(E,v1)
3. v4 = MUL(v2,v3)
4. s0 = SUM(v4)
5. v5 = DIV(v4,s0)

Unseen test:
- 2,500 examples
- max error 0
- MSE 0

Caveat:
The unit-sum landmark is informed search knowledge, though normalization is not executable unless SUM+DIV are synthesized.

---

## V374 — Raw-symbol filter transfers across K

V373b program discovered only at K=3.

Tested K:
2,4,5,8,16

Including:
- peaked beliefs
- near-deterministic transitions/emissions
- unlikely observed symbols

All:
- exact zero error.

---

## V375 — Automatic macro promotion

Successful program traces repeatedly contained:

SUM(v)
DIV(v, sum(v))

Register-number abstraction found:

S0=SUM(V0) ; V1=DIV(V0,S0)

Occurrences:
4 / 4 successful normalization programs.

Promoted macro:

UNIT_SUM(v)

Description length:
- before: 14 instructions
- after: 10

Reduction:
- 28.57%

Macro tested on unseen 13D positive vectors:
- exact.

Conclusion:
Successful program traces can compress themselves into reusable primitives.

Caveat:
Syntactic contiguous macro mining; only four programs.

---

## V376 — Recursive macro growth and search acceleration

After V375 compression, repeated pattern:

MUL(x,y)
UNIT_SUM(result)

occurred in 3 programs.

Promoted:

NORM_MUL(x,y)

Fresh K=4 raw inference search discovered:

1. VM(b,A)
2. MV(E,o)
3. NORM_MUL(prior,like)

Instruction depth:
- pre-macro reference: 5
- macro: 3
- 40% depth reduction

Search states:
- V373b reference approx: 7,738
- macro search expanded: 316
- ~95.92% reduction

Unseen test:
- exact.

Important:
Macro library changes the search problem itself.

---

## V377 — Macro bloat and routing

Tasks:
- RELATED: normalized probabilistic product
- UNRELATED: additive transform

Library policies:
1. BASE
2. ALL_MACROS
3. CONTEXT_GATED macros

On unrelated task:

BASE:
- 4,853 semantic states

ALL_MACROS:
- 7,340
- +51.25% search bloat

CONTEXT_GATED:
- 4,853
- 0% bloat

Related task:
- BASE unsolved within depth 3
- macro library solved at depth 3

Conclusion:
A growing primitive library requires **selective routing**; globally enabling everything creates combinatorial tax.

Caveat:
Context gate was hand-designed using probability-target invariants.

---

## V378 — Bootstrap normalization from scalar arithmetic

Removed:
- SUM primitive
- vector DIV primitive

Available only:
- scalar ADD
- scalar MUL
- scalar DIV

Scalar symbolic synthesis found:

K=2:
x0/(x0+x1)

K=3:
x0/((x0+x1)+x2)

Anti-unification recognized:
- numerator x_i
- denominator additive fold over every component

Inferred generic schema:

NORMALIZE_i(x) = x_i / SUM_ALL_j x_j

Unseen K:
4,8,16

Exact.

Conclusion:
UNIT_SUM can be bootstrapped one level down from scalar arithmetic.

Caveat:
Reduction-pattern anti-unification supplied.

---

## V379 — Bootstrap row-vector × matrix aggregation

Low-level atoms:
scalar products b_i * A_pq plus distractors.

K=2 synthesis:
b0*A00 + b1*A10

K=3:
b0*A00 + b1*A10 + b2*A20

Anti-unified schema:

OUT_j = SUM_i b_i * A_i,j

Unseen K=4,8,16:
- exact.

Conclusion:
ROWVEC_MATMUL can be bootstrapped from scalar product-sum patterns.

Caveat:
Search biased toward additive combinations of products; index-pattern anti-unifier supplied.

---

## V380 — Bootstrap emission aggregation and reconstruct full filter

Synthesized:

K=2:
E00*o0 + E01*o1

K=3:
E00*o0 + E01*o1 + E02*o2

Inferred:

LIKE_j = SUM_k E_j,k * o_k

Then combined lower-level schemas:

- V378 normalization
- V379 transition aggregation
- V380 emission aggregation
- scalar multiplication

Full inference tested K=4,8,16:
- machine-precision exact.

Conclusion:
Every high-level arithmetic piece of the filter was reconstructed from scalar arithmetic plus reusable reduction/index schemas.

Caveat:
Separate synthesis subtasks and anti-unification supplied.

---

## V381 — Hierarchical inference language

Starting conceptual substrate:
- scalar ADD
- scalar MUL
- scalar DIV

Bootstrapped macros:
- NORMALIZE
- ROWVEC_MATMUL
- MATVEC
- POINTWISE_MUL

Macro-level synthesis rediscovered full filter:

1. ROWVEC_MATMUL
2. MATVEC
3. POINTWISE_MUL
4. NORMALIZE

Unseen K=7:
- exact.

Expanded scalar operation counts:
- K=2: 17
- K=3: 38
- K=4: 67
- K=8: 263
- K=16: 1,039
- K=32: 4,127
- K=64: 16,447

Hierarchical algorithm description:
- constant 4 macro instructions

Important:
This compresses **description/search**, not runtime arithmetic; runtime still scales roughly O(K²).

---

## V382 — Proof that primitive vocabulary is insufficient in continuous inference

Existing scalar language:
ADD, MUL, DIV

All primitive inputs strictly positive.

Closure:
- positive + positive > 0
- positive × positive > 0
- positive / positive > 0

Therefore every finite expression in old language stays positive.

Kalman-style innovation:
observation - predicted_mean

was negative on ~51.26% of valid examples.

Therefore:
- exact solution in old language is impossible
- this is representational failure, not search failure

Candidate SUB solves exactly.

---

## V383 — Select smallest useful language extension

Base:
ADD, MUL, DIV

Candidate extensions:
NEG, SUB, ABS, EXP, LOG, MIN, MAX

Task family:
- obs - pred
- pred - obs
- abs(obs - pred)

Enumerative expression search + description cost.

Only two 2-primitive extensions solved all tasks:

Best:
{SUB, ABS}
score 30

Second:
{NEG, ABS}
score 33

Selected:
SUB + ABS

Conclusion:
Language extension can be selected by reusable explanatory compression rather than by one task only.

Caveat:
Candidate new operations supplied.

---

## V384 — Second inference family from expanded language

Scalar language:
ADD, MUL, DIV, SUB, ABS

Synthesized Kalman-style components:

PREDICT_MEAN:
MUL(a,m)

PREDICT_VAR:
ADD(MUL(MUL(a,a),v),q)

GAIN:
DIV(vp,ADD(r,vp))

INNOVATION:
SUB(y,mp)

CORRECT_MEAN:
ADD(MUL(e,k),mp)

REDUCE_VAR:
SUB(vp,MUL(k,vp))

Combined full update.

Unseen stress:
30,000 examples spanning roughly six orders of magnitude in variances/noises.

Errors:
- mean update max error 0
- variance update max error ~7.1e-15

Conclusion:
The expanded primitive language transfers to a second inference family.

Caveat:
Subtask intermediate targets supplied.

---

## V385 — Cross-family macro discovery

Two independently synthesized expressions:

Binary normalization:
x/(x+y)

Kalman gain:
vp/(vp+r)

Canonical expression-tree pattern:

DIV(X, ADD(X,Y))

Promoted:

RATIO(X,Y)=X/(X+Y)

New transfer tasks:
- resource share
- two-model posterior share
- fractional allocation

All exact.

Conclusion:
A macro can be shared across discrete probabilistic normalization and continuous estimation.

Caveat:
Expression anti-unifier supplied; new tasks algebraically same family.

---

## V386 — Invent subtraction from generic affine family

No named SUB candidate.

Generic primitive family:

f(x,y)=w1*x+w2*y+b

Fit residual examples.

Learned:
- w1 = 1.000000...
- w2 = -0.999999...
- b ~ 0

Crystallized:
(1,-1,0)

Interpretable law:
x-y

Unseen test over ~12 orders of magnitude:
- exact

Reverse residual by argument swap:
- exact

Conclusion:
A missing operation can be fitted inside a generic function family and crystallized into a simple symbolic law.

Caveat:
Affine family and integer snapping supplied.

---

## V387 — Invent ratio from generic rational family

No named RATIO template.

Generic family:

(a*x+b*y+c)/(d*x+e*y+f)

Homogeneous nullspace fitting from examples of x/(x+y).

Normalized learned coefficients:
~[1,0,0,1,1,0]

Crystallized exactly:
x/(x+y)

Unseen ~12-order-magnitude stress:
- exact.

Conclusion:
The specific ratio operation can crystallize from a more generic rational family.

Caveat:
Degree-(1,1) rational family supplied.

---

## V388 — Select the primitive function family

Candidate generic families:

AFFINE (3 coefficients)
BILINEAR (4)
RATIONAL_LINEAR (6)

Tasks:

SUB:
- AFFINE exact
- larger families also can represent it
- minimum-description selector chose AFFINE

MUL:
- AFFINE fails
- BILINEAR exact
- RATIONAL_LINEAR fails
- selected BILINEAR

RATIO:
- AFFINE/BILINEAR MSE ~0.0157
- RATIONAL_LINEAR exact
- selected RATIONAL_LINEAR

Conclusion:
Primitive structural family itself can be selected from evidence + description cost.

Caveat:
Family set supplied.

---

## V389 — Structured failure residual grows a new family

Existing families:
AFFINE, BILINEAR, RATIONAL_LINEAR

New target:
1.3*sin(2.2*x+0.4)+0.2

Old best MSE:
~0.844

Residual FFT detected:
omega = 2.2000000000000233

Added generic harmonic family:

a*sin(w*x)+b*cos(w*x)+c

Fit:
- amplitude 1.3000
- phase 0.4000
- bias 0.2000

Training MSE:
~5.9e-25

Unseen MSE:
~6.0e-24

Conclusion:
Structured residuals can suggest what new function family is missing.

Caveat:
FFT analyzer and harmonic template supplied.

---

## V390 — Noise + white-noise null control

Periodic signal amplitude:
1.3

Noise sigma:
0.2,0.5,1,2,4

Global peak/median spectral threshold:
40

Periodic detection:
- 100% through sigma 2
- 99.44% at sigma 4

Pure white-noise false positives:
- 0% all settings

Initially looked strong.

---

## V391 — CRITICAL FAILURE: colored noise fools detector

Same global spectral rule.

AR(1) null:

phi=0:
0% false growth

phi=.5:
10.75%

phi=.8:
100%

phi=.95:
100%

phi=.99:
100%

Conclusion:
Global spectral concentration confuses broad correlated stochastic structure with deterministic periodic law.

V390 diagnostic is not universal.

---

## V391b — Corrected local line-prominence diagnostic

Replace global peak/global median with:

FFT-bin power /
median neighboring-bin power

Window:
8 bins

Threshold:
35

Periodic signal:
- sigma 1: 100%
- sigma 2: 100%
- sigma 4: 97.33%

AR false growth:
- phi 0: 0.33%
- .5: 1.33%
- .8: 1.0%
- .95: 0%
- .99: 0.33%

Worst:
~1.33%

Conclusion:
A narrow spectral line can be separated much better from broad colored stochastic spectra.

Caveat:
Still tailored specifically to deterministic narrowband structure.

---

## V392 — FAILED changing raw dynamics lifecycle

Raw transition windows:

W1 affine
W2 bilinear
W3 periodic
W4 affine again

Initial family library:
AFFINE/BILINEAR/RATIONAL_LINEAR

W3 residual correctly triggered HARMONIC family growth.

However coarse FFT estimated:
omega ~1.7245
true =1.7

Long-horizon extrapolation was bad:
harmonic W3 test MSE ~1.20

System incorrectly selected AFFINE for W3.

Important distinction:
**correct structural family + inaccurate continuous parameter can still fail badly.**

V392's intended lifecycle claim is invalid.

---

## V392b — Correct structure + parameter refinement

After structure discovery:
- use coarse FFT only as proposal
- refine omega continuously by local MSE search

W3:
coarse omega = 1.724517
refined omega = 1.700002

Harmonic W3 test MSE:
~1.35e-8

Selected family lifecycle:

W1: AFFINE
W2: BILINEAR
W3: HARMONIC
W4: AFFINE

Expected:
exact match.

New harmonic family grew only in W3.

When affine dynamics returned, old AFFINE family was reused despite HARMONIC remaining in library.

Conclusion:
Structure discovery and parameter estimation must be verified separately.
A persistent model-family library can grow and later reuse older structures.

---

# Strongest surviving conclusions from V371–V392b

## 1. Inference laws can be synthesized compositionally
V371/V373b:
The full stochastic update emerges from lower-level linear algebra + arithmetic operations.

## 2. Search strategy is part of intelligence
V373:
Final-output similarity destroyed useful intermediates.

V373b:
Landmark preservation recovered them.

## 3. Successful programs can change the future search language
V375/V376:
UNIT_SUM and then NORM_MUL were promoted from repeated programs.

This reduced future solution depth and search dramatically.

## 4. Primitive libraries need routing
V377:
Unused learned macros caused ~51% search bloat on an unrelated task.

## 5. Higher-level primitives can be bootstrapped from lower-level arithmetic
V378–V381:
Normalization, matrix-vector aggregation, emission aggregation, and full filter hierarchy were reconstructed from scalar arithmetic plus supplied anti-unification/lifting mechanisms.

## 6. Primitive language can be provably insufficient
V382:
Positive closure of ADD/MUL/DIV cannot express signed innovation.

This separates vocabulary failure from search failure.

## 7. Operations can crystallize from generic families
V386/V387:
SUB and RATIO emerged as sparse simple coefficient patterns inside affine/rational families rather than named candidate choices.

## 8. The primitive function family can itself be selected
V388:
Affine vs bilinear vs rational structure selected according to task behavior and description cost.

## 9. Missing families can be proposed from structured residuals
V389:
Periodic residual structure caused harmonic family growth.

## 10. Diagnostic tests themselves require falsification
V391 destroyed V390's global spectral detector on colored noise.
V391b repaired it with local line prominence.

## 11. Structure and parameter precision are separate
V392 discovered the right harmonic family but estimated frequency poorly and failed.
V392b refined the parameter and succeeded.

## 12. A model-family library can grow and reuse old structures
V392b:
AFFINE -> BILINEAR -> HARMONIC -> AFFINE

The newest family does not necessarily replace older ones.

---

# Current narrowed architecture hypothesis

The emerging computational object now has several nested adaptive layers:

RAW EXPERIENCE
    ↓
PREDICTIVE STATE
    ↓
CURRENT WORLD MODEL
    ↓
PROGRAM / UPDATE LAW
    ↓
PRIMITIVE LIBRARY
    ↓
FUNCTION-FAMILY LIBRARY
    ↓
FAILURE DIAGNOSTICS
    ↓
LANGUAGE GROWTH / MACRO PROMOTION / ROUTING / PRUNING
    ↺

At each layer the system must distinguish at least:

1. parameter failure
2. optimization/search failure
3. state-capacity failure
4. update-law failure
5. primitive-vocabulary failure
6. function-family failure
7. stochastic/noise structure that should NOT trigger deterministic growth

And these failures demand different responses.

---

# Important remaining gaps

1. **Anti-unification / loop invention is still supplied.**
   The system does not yet autonomously invent generic indexed reductions from a truly neutral substrate.

2. **Many decomposition targets are supplied.**
   V378–V384 often learn subcomponents from privileged component targets rather than raw end-to-end experience.

3. **Residual analyzers are hand-designed families of diagnostics.**
   There is no universal mechanism for inventing a new diagnostic when failure structure is unfamiliar.

4. **Function-family search space is tiny.**
   Affine/bilinear/rational/harmonic families are only a controlled sandbox.

5. **Scale is microscopic.**
   The combinatorial search costs will explode without stronger learned search policies.

6. **Macro usefulness/routing is still partly hand-designed.**
   Need learned lifecycle utility and context retrieval.

7. **No open-ended primitive creation outside parameterized templates yet.**

8. **No hardware-level compute accounting for these symbolic searches.**

---

# Strong next targets

- Learn anti-unification and indexed reduction schemas rather than supplying them.
- End-to-end program/language discovery from raw sequence likelihood, without intermediate targets.
- Learn a residual-diagnostic router over many failure types.
- Archive/retrieve macros rather than keeping all active.
- Learn macro utility from downstream search savings.
- Jointly evolve state representation + update law + primitive family.
- Move from finite hand-listed function families to compositional grammar growth.
- Test whether discovered macros transfer across genuinely different datasets/tasks rather than algebraically equivalent toy domains.
