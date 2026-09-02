# Transmutor Research Addendum — V192 through V206

This phase attacked a narrow question created by V191/V191b:

> How should a self-developing system falsify a promoted abstraction without either accepting a bad macro too easily or destroying a useful abstraction through indiscriminate stress testing?

The phase then pushed deeper into:
- active query design
- joint refitting after experiments
- abstraction-level priors
- value-of-information stopping
- query-location construction
- adaptive measurement precision
- repeat-vs-relocate decisions
- detecting that the lower-level grammar itself is missing a structure
- temporary shadow grammars
- champion-challenger scientific rivalry
- evidence-budget scaling

---

# V192 — Pairwise maximum-disagreement falsification fails

Promoted macro family:

    {x^2, x^4, x^8}

Lower-level alternatives:

    {x^1, x^3, x^5, x^6, x^7}

One active query was chosen by:

    squared prediction disagreement
    ------------------------------
             query cost

Result:

- active macro-status accuracy: 54.54%
- random query: 55.15%
- active query cost was also slightly larger

Conclusion:

> Maximizing disagreement between one incumbent and one rival is not enough when uncertainty is distributed across many plausible structures.

V192 is a negative control.

---

# V192b — Decision-relevant group information helps, but creates overconfidence

Maintain posterior over all candidate powers.

Active query targets uncertainty in:

    MACRO FAMILY
        vs
    LOWER-LEVEL FAMILY

Three queries.

Results:

- active status accuracy: 57.58%
- random: 55.50%
- posterior family entropy:
  - prior: ~0.959 bits
  - random: ~0.257
  - active: ~0.106

Important mismatch:

    certainty improved much more than correctness

Conclusion:

> A scientific controller can become more certain faster than it becomes more correct.

The fixed fitted parameters were suspected.

---

# V192c — Refit every explanation after every experiment

After every active observation:

    refit every candidate structure
    re-optimize its frequency and coefficients
    then design the next experiment

Results:

- active macro-status: 66.0%
- random: 60.0%
- active exact-power recovery: 69.54%
- random: 64.77%

V192b active fixed-parameter reference:

    57.58%

Conclusion:

> New evidence must update not only hypothesis weights but also the parameters inside every competing hypothesis.

This mirrors V173b's integration lesson.

---

# V192d — Prior mass belongs at the abstraction level

Previous model placed equal prior on each power.

That accidentally gave:

    5/8 prior mass
    to non-macro structures

because the lower-level family contained more hypotheses.

V192d instead set:

    P(MACRO FAMILY)=0.5
    P(NON-MACRO FAMILY)=0.5

and distributed mass uniformly inside each family.

Results:

- active macro-status accuracy: 76.77%
- random: 64.15%
- active exact-power recovery: 71.23%
- random: 64.15%

By true power, active status accuracy:

- x²: 97.69%
- x³: 53.08%
- x⁴: 75.38%
- x⁵: 82.31%
- x⁸: 75.38%

Conclusion:

> "More hypotheses in a family" must not automatically mean "more prior belief in that family."

Structural priors need to operate at the abstraction level.

---

# V193 — Value-aware stopping for falsification

Instead of always asking a fixed number of questions:

Current loss:

    probability of making wrong
    keep-macro vs reopen decision

For every candidate experiment:

    expected reduction in Bayes decision error
        -
    experiment price × physical query cost

Run only when net value > 0.

Results:

Price .01:
- accuracy: 88.57%
- mean queries: 4.73

Price .03:
- accuracy: 88.86%
- mean queries: 4.52

Price .07:
- accuracy: 88.29%
- mean queries: 4.06

Reference V192d:
- fixed 3 active: 76.77%
- fixed 3 random: 64.15%

Conclusion:

> Falsification itself can use a value-of-information stopping rule.

---

# V193b — Adaptive stopping really saves experiments

Fixed active query-count frontier:

1 query:
- 54.0%

2:
- 64.86%

3:
- 78.29%

4:
- 84.29%

5:
- 88.29%

Adaptive price=.03:

- 88.86%
- 4.52 queries average

Compared with fixed 5:

- ~9.54% fewer queries
- slightly higher measured accuracy

Conclusion:

> The value-aware stop rule was not merely an accuracy trick; it avoided some unnecessary fifth experiments.

---

# V194 — Construct query locations instead of enumerating them

Full experiment search:

    36 candidate query locations
    × 3 experiments
    = 108 action evaluations

Coarse-to-fine constructor:

1. evaluate 5 coarse locations
2. refine around best coarse region
3. choose best constructed location

Results:

Full grid:
- macro-status: 80.33%
- exact power: 75.0%
- 108 candidate-action evaluations

Coarse-to-fine:
- macro-status: 82.33%
- exact power: 74.67%
- 22.81 action evaluations

Random:
- macro-status: 68.0%
- exact power: 64.0%
- 3 action evaluations

Experiment-design search reduction:

    108 / 22.81
    ≈ 4.73x

Conclusion:

> Useful falsification actions can be constructed hierarchically rather than fully enumerated.

---

# V195 — Jointly choose experiment location and precision

One experimental round.

Controller can choose:

    query location x
    and
    repetitions R ∈ {1,2,4,8}

Cost includes:
- launch overhead
- location cost
- replicate cost

Results:

Best x, fixed R=1:
- status accuracy: 55.11%
- cost: 2.156

Best x, fixed R=4:
- status: 56.67%
- cost: 2.808

Joint x + precision:
- status: 58.67%
- cost: 2.342
- mean R: 1.849

Chosen R counts over 450 episodes:
- R=1: 202
- R=2: 183
- R=4: 64
- R=8: 1

Conclusion:

> Measurement precision should itself be allocated adaptively.

---

# V196 — Online repeat vs relocate

Each new observation can either:

    REPEAT existing experiment
    or
    MOVE to a new location

Actions are priced differently.

Results:

MOVE ONLY:
- status: 86.67%
- power: 82.67%
- cost: 6.919
- ~5.01 new locations

REPEAT ONE LOCATION:
- status: 55.56%
- power: 50.22%
- cost: 1.748

ADAPTIVE MOVE/REPEAT:
- status: 85.33%
- power: 83.11%
- cost: 5.806
- ~3.84 new locations
- ~1.73 repeats

Conclusion:

> Repetition alone is cheap but epistemically narrow.
> Relocation is powerful but expensive.
> Adaptive mixture preserves nearly all status accuracy at lower total cost and slightly improves exact-power recovery.

---

# V197 — Generic residual criticism fails to detect missing grammar

Initial grammar:

    integer powers 1..8

Hidden out-of-grammar worlds:

    2.5
    3.5
    6.5

After active observations, a separate critic measured normalized residual MSE.

Threshold calibrated at 95th percentile on known integer worlds.

Results:

- false expansion: 4.0%
- unknown detection: 3.33%
- model-class status accuracy: 61.25%
- unknown exact-power recovery after half-step expansion: 1.67%

Conclusion:

> "The residual is large" is a very weak detector of missing model language when the current grammar can approximate the unknown structure locally.

V197 is a strong negative control.

---

# V198 — Temporary local structural mutants

Best current integer explanation gets temporary challengers:

    p - 0.5
    p + 0.5

These challengers are NOT permanently added.

Three active probes are chosen where incumbent and best mutant disagree.

Trigger grammar expansion when mutant posterior ≥ .80.

Results:

- false expansion: 0.44%
- unknown detection: 11.48%
- unknown exact recovery: 11.48%

By power:
- 2.5: 24.44%
- 3.5: 10.0%
- 6.5: 0%

Conclusion:

> Temporary structural mutations help more than generic residual criticism, but the local challenger family is still weak.

---

# V198b — Threshold tuning cannot solve weak challenger evidence

Sweep mutant trigger threshold.

Best balanced accuracy occurred near threshold .30:

- false expansion: 55.11%
- unknown detection: 84.81%
- balanced accuracy: 64.85%

At conservative .80:

- false expansion: 0.44%
- detection: 11.48%

Conclusion:

> The known/unknown mutant-evidence distributions overlap substantially.

This is not merely a bad threshold.

---

# V199 — Broader temporary local mutation neighborhood

Temporary mutations:

    ±0.25
    ±0.50
    ±0.75

Five active probes.

Threshold calibrated on separate known worlds.

Results:

- false expansion: 11.08%
- unknown detection: 29.23%
- grammar-status accuracy: 66.54%
- unknown exact recovery: 22.05%

By hidden power detection:
- 2.5: 30.77%
- 3.5: 32.31%
- 6.5: 24.62%

Conclusion:

> A broader temporary language helps, but incurs more false expansion and computation.

---

# V200 — Oracle identifiability audit

Question:

> Are half-integer worlds simply too close to integer-power models to distinguish?

Oracle procedure:

1. observe noiseless true function over whole allowed domain
2. find strongest integer-power approximation
3. compare its residual scale to observation noise

Median best-integer approximation MSE / noise variance:

- p=2.5: 0.1716
- p=3.5: 0.1435
- p=6.5: 0.1505

So wrong integer models are globally quite close.

But maximum one-query KL at the best discriminating input:

- 2.5: 0.482 nats
- 3.5: 0.554
- 6.5: 1.269

Heuristic ln(20)-evidence query counts at oracle-best point:

- 2.5: 6.21
- 3.5: 5.40
- 6.5: 2.36

Conclusion:

> The worlds are globally similar but contain isolated high-information experiments.

This points to experiment selection/challenger quality, not total absence of signal.

---

# V201 — Oracle headroom

For two simple equal-variance Gaussian hypotheses:

    accuracy
      =
    Phi( sqrt(n * KL_max / 2) )

Using V200's oracle-best discriminating input:

Five-query expected discrimination accuracy:

- p=2.5: 84.67%
- p=3.5: 85.81%
- p=6.5: 92.68%

Mean:

    87.72%

V199 actual unknown detection with five active probes:

    29.23%

Conclusion:

> There is enormous headroom. The dominant difficulty is finding/maintaining the right structural challenger and therefore the right experiment.

---

# V202 — Broad sandboxed shadow grammar fails through dilution

Permanent grammar:

    integers

Temporary shadow grammar:

    21 fractional powers

Prior:

    0.5 integer family
    0.5 fractional family

Five active group-disagreement probes.

Results:

- false expansion: 2.91%
- unknown detection: 16.36%
- unknown exact recovery: 10.30%

This was WORSE than V199.

Conclusion:

> A broad temporary grammar can dilute its strongest challenger.

More alternatives are not automatically more useful.

The first implementation timed out and was discarded; these are the valid vectorized rerun results.

---

# V203 — Beam width of temporary shadow grammar

Keep only strongest B structures in each family.

B=1:
- false expansion: 3.33%
- unknown detection: 42.86%
- grammar status: 76.49%
- unknown recovery: 30.16%

B=3:
- false expansion: 4.76%
- detection: 25.40%

B=5:
- false expansion: 4.76%
- detection: 26.19%

B=10:
- false expansion: 3.81%
- detection: 21.43%

Conclusion:

> Temporary hypothesis breadth has a focus-vs-coverage tradeoff.

The best tested policy is a CHAMPION CHALLENGER:

    one strongest incumbent
        vs
    one strongest shadow rival

---

# V203b — Champion-challenger result replicates

Larger sample:

- 320 calibration worlds
- 120 held-out episodes per hidden power

Results:

- false expansion: 5.17%
- unknown detection: 41.11%
- grammar-status accuracy: 74.69%
- known power recovery: 83.50%
- unknown power recovery: 30.56%

Unknown detection:
- 2.5: 49.17%
- 3.5: 39.17%
- 6.5: 35.0%

Conclusion:

> B=1 result is not a small-sample accident.

---

# V204 — Challenger identity must be replaceable

FIXED policy:
- choose first best integer
- choose first best fractional challenger
- never allow identities to change

Results:
- false expansion: 7.38%
- unknown detection: 8.21%
- grammar-status accuracy: 60.96%

DYNAMIC policy:
- after every observation:
  - refit
  - reselect strongest integer
  - reselect strongest fractional challenger

Results:
- false expansion: 6.15%
- unknown detection: 42.05%
- grammar-status accuracy: 74.42%
- known power recovery: 84.92%

Conclusion:

> Scientific rivals must be provisional.

New evidence can make a different alternative the best falsifier of the incumbent.

---

# V205 — Value-aware stopping for dynamic grammar rivalry

Dynamic champion challenger plus entropy-value stopping.

Another probe is purchased iff:

    expected incumbent/challenger entropy reduction
        >
    probe price × physical cost

Price .005:
- false expansion: 3.33%
- unknown detection: 50.0%
- grammar-status accuracy: 79.17%
- mean probes: 6.08

Price .02:
- false expansion: 3.75%
- detection: 31.94%
- status: 72.14%
- probes: 4.04

Price .05:
- false expansion: 5.42%
- detection: 18.75%
- status: 66.15%
- probes: 2.24

Conclusion:

> Grammar criticism has its own evidence-price frontier.

---

# V206 — Evidence-budget scaling

Fixed active champion-challenger probe count.

1 probe:
- false expansion: 0.95%
- unknown detection: 3.97%
- status: 63.39%

3:
- false expansion: 3.33%
- detection: 15.08%
- status: 66.07%

5:
- false expansion: 4.29%
- detection: 44.44%
- status: 76.49%

7:
- false expansion: 2.38%
- detection: 51.59%
- status: 80.36%

10:
- false expansion: 5.24%
- detection: 62.70%
- status: 82.74%

12:
- false expansion: 3.81%
- detection: 56.35%
- status: 81.25%

The 12-probe drop is not treated as a law; it may reflect sampling/policy variance.

Strongest tested point:

    10 active probes
    ~62.7% unknown detection
    ~5.2% false expansion

Still below V201's optimistic oracle benchmark.

---

# What V192–V206 narrow down

## 1. Falsification must target a decision

Generic stress testing is inferior to asking:

    "What experiment best distinguishes
     the structural decision I actually need to make?"

---

## 2. New evidence must refit the models, not only update their weights

Frozen internal parameters produced overconfidence.

---

## 3. Prior mass must be normalized across abstraction families

Otherwise a family gains belief merely by containing more enumerated hypotheses.

---

## 4. Scientific evidence has a price

The controller can stop experiments when expected decision-value gain falls below physical/economic cost.

---

## 5. Experiment design itself can be constructed hierarchically

Coarse-to-fine search retained full-grid quality with ~4.7x fewer action evaluations.

---

## 6. Precision is part of the action

The system can decide:
- where to measure
- how many repeats to buy
- whether to repeat or relocate

---

## 7. Missing-language detection is much harder than ordinary model selection

Residual error alone almost completely failed.

---

## 8. Temporary hypotheses should be sandboxed

A system may need alternatives that are:

    allowed to exist
    allowed to compete
    allowed to guide experiments

without immediately becoming permanent primitives.

---

## 9. Temporary breadth must be controlled

Too broad:
- evidence dilution
- higher compute

Too narrow/fixed:
- misses the right challenger

Best tested pattern:

    dynamic champion challenger

---

## 10. Rivals must be replaceable

The strongest challenger is not a permanent identity.

A useful scientific loop is:

    incumbent
       vs
    strongest current rival
          |
        experiment
          |
        refit all
          |
    strongest rival may change
          |
         repeat

---

## 11. There is still a major gap to oracle experiment design

Five oracle-positioned observations could theoretically discriminate these simple fractional-vs-integer worlds at ~87.7% average accuracy.

Actual five-probe dynamic criticism is ~40–44% detection.

At ten probes:

    ~62.7%

So the dominant unsolved issue is not only "how many experiments?"

It is:

> How does the system generate the right structural rival early enough that its experiment planner can discover the truly discriminating observations?

---

# Updated architecture fragment

```text
PERMANENT COMPUTATIONAL LANGUAGE
             |
             v
        incumbent model
             |
             v
         uncertainty
             |
             v
   TEMPORARY SHADOW SANDBOX
             |
      generate rivals
             |
             v
      select CHAMPION rival
             |
             v
   DESIGN DISCRIMINATING EXPERIMENT
             |
       expected decision value
       -----------------------
          physical cost
             |
             v
          observe
             |
             v
      REFIT ALL STRUCTURES
             |
             v
   incumbent / challenger may change
             |
      +------+------+
      |             |
 evidence weak   challenger wins
      |             |
      v             v
 keep testing   promote / expand
      |
 value too low?
      |
     STOP
```

---

# Next hard target after V206

The biggest remaining supplied component is the challenger generator.

Current experiments still give the system:

- integer-power semantics
- fractional-power shadow semantics
- allowed query coordinate
- observation mechanism
- cost model

The next clean question is:

> Can a challenger generator itself learn which *kind* of structural mutation has historically produced high-value falsifiers, and allocate shadow-search effort toward those mutation operators instead of being handed "try fractional powers"?

That would move meta-learning one level above V204's dynamic challenger replacement.
