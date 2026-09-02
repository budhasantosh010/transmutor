# Transmutor Research Addendum — V146 through V161

This phase attacked the next wall after V145:

> Can the system choose or construct the experiment that tells it how it should change itself?

The experiments moved from passive model selection toward active scientific behavior:
- intervention
- falsification
- experiment ordering
- synergy discovery
- noisy sequential evidence
- active type discovery
- restructuring diagnosis
- compound experiment construction
- experiment cost optimization
- information-theoretic limits
- self-calibration of experiment noise
- sparse multi-cause group testing
- continuous-parameter active design
- model-misspecification failure
- exploration tradeoffs

---

# V146 — Active falsification separates shortcut from cause

World:

- 4 observed binary variables
- one variable directly causes Y with 8% noise
- another is a downstream shortcut copying Y with only 2% noise

Pure observational prediction always preferred the shortcut.

Results:

- observational cause identification: 0%
- active intervention cause identification: 100%
- mean interventions: 2.28
- 95th percentile: 3

Intervention:

    do(X_j = random)

A direct cause changes Y's distribution under intervention.
A shortcut/noise variable does not.

Conclusion:

> Prediction quality alone can prefer a noncausal shortcut. Active interventions can answer a structural question that passive predictive fit cannot.

Caveat:
candidate interventions and one-direct-cause hypothesis family were supplied.

---

# V147 — Active experiments break the synergy trap

Hidden rule over 32 binary ±1 variables:

    y = product_{i in S} x_i

with:

    |S| = 2..8

Under the uniform passive distribution, every individual variable has exactly zero population correlation with y whenever |S|>=2.

Thus greedy single-variable relevance sees no useful variable.

Active experiment:

1. baseline all +1
2. flip variable j alone
3. output flips iff j belongs to S

Results over 500 trials:

- exact hidden subset recovery: 100%
- designed queries: 33

Passive exhaustive interaction candidates vs 33 queries:

|S|=2:
- 496 candidates
- 15.0x more

|S|=4:
- 35,960
- 1,089.7x

|S|=6:
- 906,192
- 27,460.4x

|S|=8:
- 10,518,300
- 318,736.4x

Conclusion:

> Synergistic information that is invisible to passive marginal screening can sometimes become individually identifiable under designed interventions.

---

# V148 — Exact cost-aware falsification ordering

One of N hypotheses is true.

Testing hypothesis j:

- perfectly answers whether j is true
- costs c_j
- prior probability p_j

The final remaining hypothesis can be inferred without testing.

Pairwise exchange gives the exact ordering rule:

    test i before j
    iff
    p_i / c_i > p_j / c_j

So sort by decreasing:

    probability / experiment cost

Across 5,000 random 18-hypothesis problems:

mean expected cost:

- optimal p/c ordering: 5.878
- probability-only: 7.756
- cheapest-first: 7.815
- random: 14.111

Ratios relative to optimal:

- probability-only: 1.32x
- cheapest-first: 1.33x
- random: 2.40x

Conclusion:

> The best experiment is not simply the most probable hypothesis or the cheapest test. Information must be valued relative to action cost.

---

# V149 — Noisy active synergy discovery

Same parity/product membership test as V147, but each oracle output flips independently with probability:

    p = 0.10

A paired baseline/flip vote has error:

    q = 2p(1-p) = 0.18

Fixed repeated voting:

R=3:
- exact 64-variable subset recovery: 0.33%
- 384 oracle evaluations

R=5:
- 5.00%
- 640 evals

R=7:
- 21.33%
- 896

R=9:
- 47.33%
- 1152

R=11:
- 64.00%
- 1408

Sequential Bayesian evidence:

- exact recovery: 96.33%
- mean oracle evaluations: 845.3

Conclusion:

> Experimentation itself needs adaptive compute allocation. Fixed repetition wastes measurements on easy variables and underspends on ambiguous ones.

---

# V150 — Active discovery of unknown type classes

30 atoms belonged to an unknown number of latent types:

    K = 2..6

Diagnostic composition:

    succeeds with probability .9 if same type
    succeeds with probability .1 otherwise

Learner was not told K.

Active strategy:

- process atoms sequentially
- compare new atom against representatives of discovered classes
- accumulate Bayesian evidence sequentially
- create a new class if none match

Passive comparator spent the same query budget on random pair tests and clustered compatibility fingerprints.

Results:

Active:
- exact K recovery: 97%
- pairwise partition accuracy: 99.74%

Passive:
- exact K: 66%
- pairwise partition: 91.34%

Mean composition attempts:

    269.34

Conclusion:

> Active composition tests can infer latent compatibility/type structure much more reliably than randomly accumulating the same amount of interaction data.

Caveat:
a diagnostic same-type composition operator was supplied.

---

# V151 — Diagnose which self-modification is needed

Three hidden failure classes:

PARAM:
- useful restructuring = parameterize by context

SPLIT:
- useful restructuring = split incompatible regimes

STATE:
- useful restructuring = expand history/state

Candidate diagnostic experiments:

CONTEXT:
- intervene on context

HISTORY:
- hold current state fixed and vary previous state

REPLICATE:
- repeat same visible condition and measure hidden multimodality

Likelihoods were calibrated from prior synthetic experience.

Bayesian policy:

    choose test maximizing
    expected posterior entropy reduction / test cost

Stop when posterior restructuring class >= .95.

Results over 1,500 episodes:

Active:
- correct restructuring diagnosis: 99.93%
- mean diagnostic cost: 3.331
- mean tests: 1.665

Random diagnostic order:
- accuracy: 99.87%
- mean cost: 7.817

Cost reduction:

    57.39%

Conclusion:

> The system can actively choose which evidence to generate before deciding how to modify itself.

Caveat:
the diagnostic experiment families were still supplied.

---

# V152 — Construct compound experiments by hypothesis bisection

128 candidate causal variables.
Exactly one is causal.

An experiment is now a binary action vector assigning 0/1 to every candidate.

Construct intervention so that approximately half surviving hypotheses receive 1 and half receive 0.

Observed Y reveals which half contains the causal variable.

Results:

- compound experiments: exactly 7
- single-variable tests: 65.69 average
- experiment-count reduction: 9.38x

Scaling:

single testing:

    O(N)

balanced hypothesis splitting:

    O(log N)

Conclusion:

> The system need not choose among human-named experiments. It can construct an intervention from the current hypothesis set itself.

Caveat:
compound manipulation had no extra physical cost in V152.

---

# V153 — Experiment count is not experiment cost

Cost model:

    total
      =
    experiment_count * h
      +
    manipulated_variable_count * gamma

For N=128:

Single tests:
- expected experiments: 64.49
- expected manipulated variables: 64.49

Compound bisection:
- experiments: 7
- manipulated variables: 127

Break-even:

    h/gamma ≈ 1.087

If experiment overhead is cheap relative to manipulating variables:

    single-variable probes win

If experiment overhead is expensive:

    compound tests win

Conclusion:

> Information-efficient and resource-efficient experiments are different objectives.

---

# V154 — Dynamic programming constructs the optimal experiment size

Instead of choosing between singles and 50/50 bisection, allow any subset size m.

For n equally likely remaining hypotheses:

Experiment cost:

    h + gamma*m

Bellman equation:

    E(1)=0

    E(n)=min_m [
        h + gamma*m
        + (m/n) E(m)
        + ((n-m)/n) E(n-m)
    ]

At N=128 the optimal first subset changed continuously with physical cost:

h/gamma=0.05:
- m=2
- 1.6% of hypotheses

0.2:
- m=5

0.5:
- m=7

1:
- m=10

2:
- m=13

5:
- m=18

20:
- m=32
- 25%

Conclusion:

> Experiment structure itself is a resource-allocation decision. The system can synthesize broader or narrower interventions from the physical cost model.

---

# V155 — Exact noiseless information limit

A binary-outcome experiment contains at most:

    1 bit

Distinguishing N equiprobable hypotheses therefore requires at least:

    log2(N)

binary outcomes.

For powers of two, balanced bisection attains:

    ceil(log2 N)

exactly.

For N=128:

    lower bound = 7
    balanced bisection = 7

Thus V152's experiment count is information-theoretically optimal in its idealized binary noiseless setting.

---

# V156 — Noisy probabilistic bisection

Binary outcomes flip with:

    p = 0.10

Binary symmetric channel capacity:

    C = 1 - H2(p)
      ≈ 0.531 bits/observation

Simple information lower bound for N=128:

    7 / C
      ≈
    13.18 observations

Adaptive probabilistic bisection:

- mean observations: 15.01
- p95: 28
- identification accuracy: 99.67%

Naive five-repeat bisection:

    35 observations

Conclusion:

> In this supplied noisy binary experiment family, Bayesian experiment construction operated close to the simple information limit.

---

# V157 — Separate self-calibration of unknown experiment noise

True experiment noise varied by episode:

    p ~ Uniform(.02,.25)

Compare:

ORACLE:
- knows p

FIXED:
- assumes p=.10

CALIBRATED:
- repeat one diagnostic 17 times
- infer noise from minority outcome fraction
- then plan with p_hat

Results:

Oracle:
- accuracy 99.4%
- mean observations 20.53

Fixed:
- accuracy 96.4%
- 19.25 observations

Separate calibration:
- accuracy 99.0%
- total observations 39.24
- noise MAE ~0.0525

Conclusion:

> Self-calibration restores reliability but a dedicated calibration phase can be too expensive.

---

# V157b — Joint inference of answer and experiment reliability

Belief state:

    P(cause=j, noise=p_k)

No separate calibration.

At each experiment:

1. marginalize cause posterior
2. construct ~50% posterior-mass intervention
3. observe outcome
4. jointly update cause and noise

Results:

- joint identification accuracy: 99.4%
- oracle accuracy: 99.6%
- joint mean observations: 21.76
- oracle: 20.25
- noise posterior mean MAE: ~0.0441

The separate-calibration V157 needed:

    39.24 observations

Conclusion:

> The same evidence stream can simultaneously teach the system what is true and how reliable its own measurements are.

---

# V158 — Multiple hidden causes via adaptive group splitting

N:

    1024

Unknown relevant set size:

    k = 1,2,4,8,16,32

Group experiment returns positive iff the tested subset contains at least one relevant variable.

Recursive strategy:

- negative group -> discard entire group
- positive group -> split
- positive singleton -> relevant variable found

Exact recovery:

    100% at every tested k

Mean tests:

k=1:
- 21.0
- 48.8x fewer than 1024 individual tests

k=2:
- 37.1
- 27.6x

k=4:
- 65.6
- 15.6x

k=8:
- 115.1
- 8.9x

k=16:
- 197.7
- 5.2x

k=32:
- 333.8
- 3.1x

Conclusion:

> Constructed experiments can scale with sparse hidden structure instead of raw environmental dimension.

Caveat:
the group-test observation is a noiseless OR.

---

# V159 — Continuous-parameter active experiment design

Hidden mechanism:

    y = sin(theta*x) + noise

theta:

    0.5..4

Learner maintains posterior over theta.

Active probe:

    choose x maximizing
    posterior variance of sin(theta*x)

That asks where currently plausible mechanisms disagree most.

Eight observations per episode.

Results:

Random probes:
- mean parameter MAE: 0.04012
- mean posterior std: 0.04886

Active probes:
- parameter MAE: 0.02004
- posterior std: 0.02423

Parameter error reduction:

    ~2.00x

Conclusion:

> Active scientific experiment design generalizes beyond finite discrete hypothesis elimination.

Caveat:
the sinusoidal model family was supplied.

---

# V160 — Active disagreement fails under model misspecification

Learner assumes:

    y = sin(theta*x)

True world:

    y = sin(theta*x)
        + 0.35 * local_bump(x)
        + noise

The hidden discrepancy was located near a region frequently selected by the V159 active policy.

Results:

Parameter recovery:

Active:
- theta MAE: 0.07393

Random:
- 0.06383

Active became:

    15.8% worse

However predictive MSE:

Active:
- 0.01061

Random:
- 0.01382

So the biased active parameter partly compensated for the missing model component.

Conclusion:

> Under misspecification, the parameter that predicts best can move away from the true mechanism parameter. Active disagreement can amplify this effect by repeatedly probing where the model family is structurally wrong.

V160 is intentionally adversarial and should count as a failure/control, not a positive result.

---

# V161 — Exploration/exploitation under misspecification

epsilon-greedy experiment policy:

with probability 1-epsilon:
- disagreement-maximizing probe

with probability epsilon:
- random probe

Results:

Best epsilon for recovering the true theta:

    epsilon = 1.0
    pure random exploration

theta MAE:

    ~0.06113

Best epsilon for predictive MSE:

    epsilon = 0.10

predictive MSE:

    ~0.01043

Thus:

> The exploration policy that best identifies the true mechanism can differ from the policy that best predicts under a misspecified model.

No single epsilon optimized both objectives.

---

# What V146–V161 narrow down

## Exact / theorem-level

### 1. Cost-aware single-hypothesis test ordering

    decreasing p_j/c_j

is optimal under the stated perfect-test model.

### 2. Binary noiseless experiment lower bound

    >= log2(N)

outcomes for N equiprobable hypotheses.

Balanced bisection attains the bound for powers of two.

### 3. Experiment-construction Bellman equation

For uniform hypotheses and subset-test cost:

    h + gamma*m

optimal policy satisfies:

    E(n)=min_m[
      h+gamma*m
      +(m/n)E(m)
      +((n-m)/n)E(n-m)
    ]

---

# Strong empirical findings

### 1. Active interventions can separate causal structure from stronger predictive shortcuts.

### 2. Designed experiments can reveal synergistic variables that passive marginal relevance cannot see.

### 3. Sequential evidence allocation is dramatically more efficient under noisy interventions than fixed repetition.

### 4. Active composition tests improve latent type discovery.

### 5. A controller can choose which diagnostic experiment to run before selecting how to restructure itself.

### 6. Compound experiment construction can change identification scaling from linear to logarithmic in ideal settings.

### 7. The experiment should be synthesized from both information value and physical action cost.

### 8. The system can jointly learn task structure and the reliability of its own experimental channel.

### 9. Sparse multi-cause structure can be located with far fewer group experiments than individual tests.

### 10. Active design also helps continuous-parameter mechanism estimation.

### 11. Active learning is not automatically epistemically safe under model misspecification.

### 12. Prediction-optimal and mechanism-identification-optimal experiment policies can diverge.

---

# Updated architecture hypothesis

```text
CURRENT SYSTEM
    |
    v
residual uncertainty / competing explanations
    |
    v
HYPOTHESIS / MODEL SET
    |
    +---------------------------+
    |                           |
    v                           v
what evidence differs?       what actions are possible?
    |                           |
    +-------------+-------------+
                  |
                  v
          EXPERIMENT SYNTHESIS
                  |
       information value
             /
          action cost
             /
       measurement reliability
                  |
                  v
          perform experiment
                  |
                  v
       Bayesian / evidence update
          /               \
         /                 \
   task/world belief    experiment-noise belief
         |                 |
         +--------+--------+
                  |
                  v
       enough evidence to restructure?
                  |
        +---------+----------+
        |         |          |
        v         v          v
  parameterize   split    expand state
        |         |          |
        +---------+----------+
                  |
                  v
          validate / falsify
                  |
                  v
        library / meta-library
                  |
                  v
              repeat
```

---

# New hard frontier

The largest remaining limitation has moved again.

The experiments can now:

- choose interventions
- construct compound interventions
- allocate measurement effort
- optimize experiment breadth from cost
- infer measurement reliability
- diagnose restructuring needs

But the following are still supplied:

1. the hypothesis/model language
2. the allowed intervention/action language
3. the variables that can be manipulated
4. the observation channel form
5. the diagnostics available to detect misspecification
6. the semantics of what counts as a valid intervention

The next clean research target is:

> Can the system discover or expand its own experiment/action language when its current set of possible interventions cannot distinguish the remaining explanations?

That is the experimental analogue of the earlier primitive-language problem.

There is also a second major problem exposed by V160/V161:

> How should a system detect that all of its current hypotheses are wrong, rather than merely choosing the least-wrong one more confidently?

That model-misspecification detector should probably become a first-class component of Transmutor.
