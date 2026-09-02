# Transmutor Experiments V434–V440

## Purpose

After the V429–V433b four-target milestone, this batch pushed underneath the supplied meta-scaffolds.

Primary questions:

1. Can the named `JOINT_ADD` intervention be removed?
2. Can the hand-written coverage policy be removed?
3. Can experiment/falsification strategies be composed over time?
4. Can the system distinguish search failure from representation-language failure?
5. Can the intervention-program language recognize that its own composition depth is inadequate and grow?
6. Can those mechanisms work in a persistent lifelong stream rather than isolated tests?

All failed gates, benchmark confounds, numerical failures, and retries are preserved below.

---

# V434 — Remove supplied JOINT_ADD

Previous intervention synthesis used a named high-level operation:

`JOINT_ADD(A,B)`

V434 removed that named operator.

Lower-level primitives remained:

- construct polynomial/trigonometric basis
- concatenate matrices
- fit linear coefficients
- predict

Candidate low-level programs included:

- FIT_SINGLE(POLY)
- FIT_SINGLE(TRIG)
- FIT_LINEAR(CONCAT(BASIS(POLY),BASIS(TRIG)), target)
- FIT_LINEAR(CONCAT(BASIS(TRIG),BASIS(TRIG)), target)

Birth results:

MIXED_POLY_SIN:
- selected low-level POLY+TRIG basis concatenation
- normalized error ~1.186× noise

TWO_FREQUENCY:
- selected TRIG+TRIG basis concatenation
- normalized error ~0.997× noise

Heldout:
- 160 episodes
- adequate solve: 100%
- aggregate median improvement over best single intervention: ~39.26×

PASS.

Important caveat:
CONCAT, FIT_LINEAR, component basis constructors, and spectral proposals remain supplied.
This removes a named domain-level meta-operation, not the deeper algebraic primitives.

---

# V435 — Remove hand-written coverage policy

Previously the intervention pipeline explicitly requested structured domain coverage.

V435 instead supplied lower-level acquisition scoring primitives:

- DIST = distance to existing observations
- DISAG = model disagreement

Small score-expression grammar:

- DIST
- DISAG
- ADD(DIST,DISAG)
- MUL(DIST,DISAG)
- MAX(DIST,DISAG)
- SQUARE(DIST)
- SQUARE(DISAG)

The learner selected the acquisition expression by downstream solve quality.

With 110 additional samples:

Promoted:
`ADD(DIST,DISAG)`

Heldout:
- 140 episodes
- solve: 100%

PASS under its initial gate.

However this result was recognized as too easy because nearly every policy solved the task.

---

# V435b — Scarce acquisition budget audit: FAIL

Acquisition budget reduced:

110 -> 24

RANDOM added as baseline.

Result:

- RANDOM solve: 100%
- learned DIST solve: 100%
- gain over RANDOM: 0

The "policy matters" gate required +15 percentage points.

FAIL.

Interpretation:

The starting data already contained enough global information.
The acquisition benchmark itself was too easy.

---

# V435c — Narrow-support information bottleneck: FAIL, but meaningful

Corrected benchmark:

Initial data only:
x in [-1,1]

Final prediction:
x in [-3,3]

Acquisition budget:
18 points.

Results:

RANDOM:
- solve 31.5%

DIST:
- solve 72.0%

Absolute gain:
+40.5 percentage points.

By family:

MIXED_POLY_SIN:
- RANDOM 31%
- DIST 84%

TWO_FREQUENCY:
- RANDOM 32%
- DIST 60%

Gate required:
- >=80% solve
- >=15pp gain

The gain gate passed.
Overall solve gate failed.

Therefore V435c = FAIL.

Important result:
Acquisition strategy genuinely matters once initial experience is narrow.

---

# V436 — Compose multi-stage falsification strategies: FAIL

Generated experiment programs from:

- ACQUIRE_DIST(n)
- ACQUIRE_DISAG(n)
- REFIT_HYPOTHESES

Candidate schedules included:

- DIST(18)
- DISAG(18)
- DIST(15) -> REFIT -> DISAG(3)
- DIST(12) -> REFIT -> DISAG(6)
- etc.

Promoted:

`DIST(15) -> REFIT -> DISAG(3)`

Heldout:
- solve 68.18%
- RANDOM18 27.73%
- gain +40.45pp

Gate required 80%.

FAIL.

Conclusion:
Multi-stage experiment composition helped substantially, but a fixed 18-query budget remained insufficient.

---

# V436b — Adaptive evidence budget: FAIL narrowly

Program:

1. DIST(12)
2. PROBE_DIST(6)
3. evaluate prediction on those fresh probes before training on them
4. if probe MSE > noise floor:
   - REFIT
   - DISAG(12)
5. otherwise stop

Maximum acquisition:
30

Average:
23.8

Result:

- adaptive solve 77.08%
- resource-matched RANDOM 42.08%
- +35pp gain
- average acquisition <=28 gate passed

But solve gate required 80%.

FAIL.

---

# V436c — Learn the post-falsification continuation: PASS

Rather than hard-code DISAG after a failed probe, candidate continuation operations were tested:

- DIST
- DISAG
- ADD(DIST,DISAG)
- MUL(DIST,DISAG)

Promoted continuation:

`ADD(DIST,DISAG)`

Full program:

`DIST(12) -> PROBE_DIST(6) -> IF FAIL -> REFIT -> ADD(DIST,DISAG)(12)`

Heldout:
- solve 81.54%
- matched RANDOM 47.69%
- gain +33.85pp
- average acquisitions 24.28

PASS.

Interpretation:

The system can compose a conditional experiment program from lower-level information-gathering operations and use falsification evidence to allocate more experiment budget.

Remaining scaffolds:
DIST, DISAG, REFIT, IF/conditional structure, noise estimate.

---

# V437 — First intervention-language depth growth test: FAIL

Current language maximum:
2 components.

New worlds intended to require:
3 components.

Depth-3 generated programs:

- POLY+TRIG+TRIG
- TRIG+TRIG+TRIG

Problem:
degree-5 polynomial often acted as a proxy for a missing periodic component.

Old depth<=2 inadequacy:
only 43.08%

New depth3 solve:
98.46%

Median improvement:
only ~1.20×

The test did not cleanly force language growth.

FAIL.

---

# V437b — Hardened non-proxyable depth test: FAIL

New worlds used:

- stronger high-frequency components
- well-separated frequencies
- lower observation noise

Now old depth<=2 language inadequacy:

100%

Good.

But depth3 solve:
27.86%

The representation depth was potentially right, but frequency search was not accurate enough.

FAIL.

---

# First V437c joint optimizer attempt — no result

A Nelder-Mead joint frequency optimizer was added.

Execution timed out during the heldout audit.

No scientific result.

---

# V437c deterministic refinement — FAIL

Replaced expensive nonlinear optimization with local coordinate-grid refinement.

Results:

- old language inadequate: 100%
- depth3 solve: 77.27%
- median improvement: ~700.94×

FAIL on 90% solve gate.

Important:
The huge improvement suggested the grown language was largely right, but search precision still caused outliers.

---

# V437d — Bounded variable projection: FAIL narrowly

Optimized only nonlinear frequencies while exactly solving linear coefficients at every evaluation.

Results:

- old language inadequate: 100%
- depth3 solve: 87.5%
- median improvement: ~1006×

POLY_TWO_FREQ_HARD:
100% solved

THREE_FREQ_HARD:
75% solved

FAIL.

---

# V437e — Multi-start refinement for three-frequency branch: FAIL narrowly

Three-frequency search:
- multiple coarse spectral triples
- bounded variable-projection refinement from several starts

Results:

- old language inadequate: 100%
- grown depth3 solve: 89.0625%
- median improvement: ~974.83×

Gate remained 90%.

FAIL.

No rounding up.

---

# V437f — Multi-start refinement for both depth3 branches: PASS

Applied multi-start variable projection to both:

- POLY+TRIG+TRIG
- TRIG+TRIG+TRIG

Fresh heldout:
68 episodes.

Results:

Old depth<=2 language inadequate:
100%

Depth3 solve:
91.176%

Median improvement:
~1008.63×

By family:

POLY_TWO_FREQ_HARD:
- solve 97.06%
- median improvement ~918.49×

THREE_FREQ_HARD:
- solve 85.29%
- median improvement ~1262.50×

PASS under fixed aggregate gate.

Interpretation:

Once the benchmark truly forced missing representational depth and search was made sufficiently robust, increasing low-level program composition depth became a meaningful response.

Important caveat:
Recursive CONCAT and the ability to increase depth are still supplied.
The system does not invent recursion itself.

---

# V438 — First persistent lifelong integration: FAIL

Persistent language depth starts at 1.

Lifecycle:

1. evaluate current language
2. if adequate -> stop
3. if inadequate -> grow one depth level
4. retain expanded language forever
5. repeated tasks should not trigger further growth

12-task stream.

Language depth evolved:

1 -> 2 -> 3

No unnecessary growth.
No growth on repeat tasks.

But task T8 (THREE_FREQ_HARD) failed even at depth3.

All tasks solved = false.

FAIL.

This exposed a new issue:

The representation language can be adequate while the search procedure still fails.

---

# V438b — Explicit search escalation vs language growth: FAIL

New control distinction:

If inadequate and depth < 3:
- grow language.

If inadequate and depth == 3:
- escalate search inside the existing language.

T8 still failed.

FAIL.

Important:
"More of the same search" was insufficient for this outlier.

---

# V439 — Search strategy diversity: PASS

Fixed representation:

TRIG + TRIG + TRIG

Hard three-frequency tasks:
90.

Search strategies:

MULTISTART:
- solve 84.44%

DEFLATION:
- solve 95.56%

HYBRID:
- solve 98.89%

PORTFOLIO:
- solve 98.89%

Hybrid search combined:

- global spectral candidates
- iterative residual deflation candidates
- joint variable-projection refinement

PASS.

Key conclusion:

Search strategy is itself a structural resource.
Different search methods fail on different instances even inside the same correct representation language.

---

# V438c — Feed ordinary hybrid search into lifelong stream: FAIL

The T8 outlier still failed.

This was surprising because V439's hybrid solved 98.89% over a large fresh sample.

Therefore T8 was isolated directly.

---

# V438d — Diagnose exact T8 outlier

Exact task:
THREE_FREQ_HARD seed 43808.

Normalized test error:

CHEAP:
~16.32× noise

MULTISTART:
~16.32×

DEFLATION:
~4.03×

HYBRID:
~22.39×

STRONG_HYBRID:
~1.036×

Only STRONG_HYBRID was adequate.

This showed:

- representation depth3 was sufficient,
- ordinary search strategies could all fail on one hard instance,
- a broader candidate portfolio solved it.

Diagnostic only:
This reused a known failing seed and was not an independent generalization result.

---

# V438e — Lifelong rerun with strong hybrid: PASS on diagnosed stream

No task changes.
No adequacy changes.
No depth changes.

Only search-escalation mechanism changed according to V438d diagnosis.

All 12 tasks solved.

Language depth:

1 -> 2 -> 3

Repeat growth:
0

T8:
- search escalation = 1
- selected TRIG+TRIG+TRIG
- normalized error ~1.036
- solved

PASS on the original stream.

But because the mechanism was diagnosed using T8 itself, a fresh audit was required.

---

# V440 — Fresh independent lifelong audit: PASS

No algorithm changes after V438e.

Six fresh lifelong streams.

Each stream:
8 tasks.

Total:
48 fresh tasks.

Results:

Overall task solve:
100%

Perfect-stream rate:
100%

Language-growth steps on repeat tasks:
0

Total search escalations:
1

Per family solve:

- SIMPLE_POLY: 100%
- SIMPLE_TRIG: 100%
- TWO_FREQUENCY: 100%
- POLY_TWO_FREQ_HARD: 100%
- THREE_FREQ_HARD: 100%

PASS.

This is the strongest result in the batch.

---

# Strongest conclusions from V434–V440

## 1. A named high-level intervention can sometimes be removed

V434 replaced supplied JOINT_ADD with lower-level:

- basis construction
- concatenation
- linear fitting

The useful intervention emerged as a low-level program.

This does not remove the lower algebraic primitives.

---

## 2. Information acquisition must be tested under a real information bottleneck

V435 looked perfect but was too easy.

V435b exposed that random acquisition was equally good.

V435c narrowed the initial observation domain and finally made acquisition strategy matter:

DIST:
72%

RANDOM:
31.5%

The benchmark quality mattered more than the first positive result.

---

## 3. Experiment policy can itself be a program

V436c produced a conditional information-gathering program:

DIST exploration
-> fresh falsification probes
-> if still wrong, refit
-> combine distance + disagreement acquisition

This passed while using an adaptive ~24.3 samples on average.

---

## 4. Representation failure and search failure are different

This became one of the clearest findings of the batch.

V437 hardened the representation benchmark until old depth2 language genuinely failed.

But even a sufficient depth3 language often failed because its search was inaccurate.

The progression:

depth3 solve:
27.9%
-> 77.3%
-> 87.5%
-> 89.1%
-> 91.2%

without changing the depth3 language itself.

Search quality was responsible for much of the apparent "representation failure."

---

## 5. Search-strategy diversity can be more important than simply spending more search

V439:

MULTISTART:
84.4%

DEFLATION:
95.6%

HYBRID:
98.9%

Different search algorithms expose different parts of the same hypothesis language.

Therefore a system may need a portfolio of search procedures, not only a larger search budget.

---

## 6. Program-language capacity can grow only when current capacity is inadequate

In the lifelong streams:

depth1:
simple worlds

depth2:
two-component worlds

depth3:
hard three-component worlds

Once depth3 existed, later tasks reused it.
Repeat tasks did not cause further depth growth.

---

## 7. A persistent system can separate language growth from search escalation

Fresh V440 audit:

48/48 tasks solved.

The system's control loop behaved approximately as:

CURRENT LANGUAGE
    |
    v
try normal search
    |
    +--> adequate:
    |      solve
    |
    +--> inadequate:
           |
           +--> language depth can grow:
           |      grow representation language
           |
           +--> language already deep enough:
                  escalate / diversify search

This is more precise than:

"failure -> make the model bigger."

---

# Updated architecture hypothesis after V440

EXPERIENCE
    |
    v
NOISE / ADEQUACY ESTIMATE
    |
    v
CURRENT MODEL + CURRENT PROGRAM LANGUAGE
    |
    v
NORMAL SEARCH
    |
    +--> ADEQUATE
    |       |
    |       v
    |      STOP / USE / ARCHIVE
    |
    +--> INADEQUATE
            |
            v
      WHAT KIND OF FAILURE?
            |
      +-----+--------------------+
      |                          |
SEARCH MAY BE WEAK         LANGUAGE MAY BE WEAK
      |                          |
      v                          v
DIVERSIFY / ESCALATE       GROW COMPOSITION DEPTH
SEARCH STRATEGY                  |
      |                          |
      +------------+-------------+
                   |
                   v
             TEST AGAIN
                   |
                   v
        ACQUIRE NEW EVIDENCE
        ONLY IF IT EARNS VALUE
                   |
                   v
             PERSIST / REUSE
                   |
                   loop

---

# What remains supplied

Major scaffolds still remain:

1. CONCAT itself.
2. FIT_LINEAR.
3. Polynomial and trigonometric basis families.
4. Spectral-analysis primitives.
5. DIST and DISAG acquisition primitives.
6. IF/conditional program skeletons.
7. Noise-estimation/adequacy rule.
8. The ability to increase program depth.
9. Search-strategy candidates.
10. The maximum language depth.
11. Synthetic task generators.

Therefore these experiments still do not demonstrate:

- autonomous invention of arbitrary computational languages,
- a universal self-modifying machine,
- AGI,
- a Transformer replacement,
- or a new fundamental computational primitive.

---

# Next frontier

The strongest remaining question is now lower-level:

Can the system discover useful computational operators such as:

- concatenation,
- conditional control,
- repeated application / recursion,
- residual formation,
- search diversification,

from an even more neutral substrate?

Strong next experiments:

1. Remove explicit CONCAT and allow only message/state transformations.
2. Make recursion/depth growth emerge from repeated program reuse rather than a supplied depth parameter.
3. Replace named DIST/DISAG with learned acquisition-value functions from raw candidate consequences.
4. Let search strategies themselves be stored, retrieved, composed, and revised like other learned programs.
5. Move from scalar one-dimensional regression to multivariate coupled dynamical systems.
6. Test persistent memory/search/language evolution under nonstationary objectives.
7. Begin removing explicit polynomial/trigonometric component families entirely.
