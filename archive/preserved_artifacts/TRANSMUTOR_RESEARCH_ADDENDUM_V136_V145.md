# Transmutor Research Addendum — V136 through V145

This phase tested whether the previously separate mechanisms could begin to operate under a common self-reorganization principle rather than the experimenter manually deciding which restructuring operation to run.

The most important new themes were:

- one common objective can still be gamed by structurally inappropriate moves
- falsification / invariance tests are needed alongside predictive fit
- unnecessary restructuring must clear an evidence/noise threshold
- representation invariance has an exact error-vs-resource tradeoff
- useful information can be synergistic and invisible one variable at a time
- the restructuring-operation library itself can grow and become parameterized
- IID predictive superiority is not sufficient evidence of stable structure

---

# V136 — Shared self-restructuring objective: failed control

Episode families:

- NONE
- PARAMETERIZE
- SPLIT
- EXPAND_STATE

Candidate moves available to every episode:

- KEEP
- PARAMETERIZE
- SPLIT
- EXPAND_STATE

Common objective:

    J
      =
    error(move) / error(KEEP)
      +
    lambda * complexity(move)

with:

    lambda = 0.03

Results across 64 episodes:

- overall intended-move selection: 75%
- NONE: 100%
- SPLIT: 100%
- STATE: 100%
- PARAM: 0%

Every parameterized-function episode selected EXPAND_STATE.

Why?

The static functions were sampled on a fixed ordered coordinate grid. An autoregressive state model exploited adjacency on that grid and achieved slightly lower error at lower assigned complexity than the intended parameterized abstraction.

This was a genuine shortcut / objective-gaming failure.

Lesson:

> A common resource objective does not guarantee structurally meaningful self-modification. Candidate restructuring moves can exploit accidental regularities outside their intended semantic domain.

---

# V136b — Falsification by representation-preserving perturbation

Static observations were presented in random coordinate orders.

Moves such as KEEP / PARAMETERIZE / SPLIT were evaluated using coordinate identity and therefore survived presentation permutation.

The fake state model relied on presentation adjacency, so its shortcut disappeared.

Results:

- 64 episodes
- KEEP/NONE: 16/16
- PARAMETERIZE: 16/16
- SPLIT: 16/16
- EXPAND_STATE: 16/16

Overall:

    100%

Mean errors illustrate the separation.

PARAM episodes:

- KEEP: 1.4031
- PARAMETERIZE: 0.00705
- SPLIT: 0.7328
- fake EXPAND_STATE: 3.1812

STATE episodes:

- KEEP: 0.3573
- PARAMETERIZE: 0.3591
- SPLIT: 0.3565
- EXPAND_STATE: ~6.15e-15

Conclusion:

> Structural changes should be tested under perturbations that preserve the intended problem while destroying accidental shortcuts.

This is stronger than ordinary held-out validation.

---

# V137 — Invariance selection from task evidence

Candidate representations:

- IDENTITY
- ABS
- SIGN

Task families:

SIGNED_VALUE:

    y = x

MAGNITUDE:

    y = |x|

SIGN_ONLY:

    y = sign(x)

A simple affine downstream readout selected the representation with the best held-out prediction.

Across 300 noisy trials:

- SIGNED_VALUE -> IDENTITY: 100%
- MAGNITUDE -> ABS: 100%
- SIGN_ONLY -> SIGN: 100%

Overall:

    100%

Conclusion:

> An invariance should be adopted only when the distinctions it merges are irrelevant to the task evidence.

Candidate invariances were still supplied.

---

# V138 — Infer the number of latent compatibility/type classes

V135 had been given:

    K = 2 latent classes

V138 removed that count.

Hidden compatibility systems contained:

    K = 2,3,4,5

Candidate K:

    1..7

Procedure:

1. observe composition success/failure
2. construct atom compatibility fingerprints
3. cluster for each candidate K
4. learn operator signatures between clusters
5. evaluate validation cross entropy
6. select the smallest K within a small tolerance of best validation score

Clean results across 100 episodes:

- K=2: 100%
- K=3: 100%
- K=4: 100%
- K=5: 100%

Held-out composition validity:

    100%

Thus the type-system granularity did not need to be explicitly supplied in this clean finite setting.

---

# V139 — Noisy / missing type evidence

Stress conditions:

- 10% composition outcomes flipped
- only 38% of all compositions used for training
- hidden K from 2 to 6

Results across 60 episodes:

Exact K recovery:

    96.67%

Prediction of the underlying true compatibility relation:

    98.09%

By K:

K=2:
- K recovery: 100%
- truth validity accuracy: 97.91%

K=3:
- K recovery: 91.67%
- mean chosen K: 3.083
- truth validity: 97.76%

K=4:
- K recovery: 91.67%
- mean chosen K: 4.083
- truth validity: 97.77%

K=5:
- K recovery: 100%
- truth validity: 98.55%

K=6:
- K recovery: 100%
- truth validity: 98.45%

Important distinction:

> Exact structural count can be slightly wrong while behavioral compatibility prediction remains highly accurate.

---

# V140 — Split vs parameterize vs keep: over-restructuring failure

Three cross-domain regimes:

FIXED:

    same coefficient every domain

CONTINUOUS_CONTEXT:

    coefficient varies continuously and is explained by context

DISCRETE_UNEXPLAINED:

    coefficient belongs to two modes not explained by supplied context

Candidate moves:

- KEEP
- PARAMETERIZE
- SPLIT

Results across 300 episodes:

- CONTINUOUS_CONTEXT -> PARAMETERIZE: 100%
- DISCRETE_UNEXPLAINED -> SPLIT: 100%
- FIXED -> KEEP: only 28%

Overall:

    76%

Why FIXED failed:

All three representations were already at the noise floor.

Mean MSE:

- KEEP: 0.001616
- PARAMETERIZE: 0.001590
- SPLIT: 0.001609

Tiny random advantages triggered unnecessary structural change.

Lesson:

> Error minimization alone encourages needless reorganization at the noise floor.

---

# V140b — Confidence / effect-size gate

A restructuring move was accepted only if paired validation improvement was:

    > 2 standard errors

and:

    > 5% of current KEEP error

Otherwise:

    KEEP

Results:

- FIXED: 100%
- CONTINUOUS_CONTEXT: 100%
- DISCRETE_UNEXPLAINED: 100%

Overall:

    100%

This is a heuristic rather than a theorem, but it establishes an important architecture rule:

> Self-modification needs an evidence threshold. Do not reorganize merely because a larger model wins by numerical noise.

---

# V141 — Exact invariance/resource threshold

Full representation preserves:

    magnitude + sign

Invariant representation preserves:

    magnitude only

Target:

    y
      =
    |x| + epsilon * sign(x) + noise

Representation costs:

- invariant dimension: 1
- full dimension: 2

Objective:

    J = MSE + lambda * dimension

with:

    lambda = 0.04

If sign is discarded, balanced signs produce irreducible additional squared error:

    epsilon^2

Keeping sign costs one additional unit:

    lambda

Therefore the exact clean decision is:

    preserve sign

iff:

    epsilon^2 > lambda

Equivalent threshold:

    epsilon > sqrt(lambda)

For lambda=0.04:

    epsilon* = 0.20

Experiment:

- below 0.20: invariant representation selected 100%
- at 0.20: full representation selected 65.5% because finite noise puts the system exactly at the boundary
- >=0.225: full representation selected 100%

The first tested majority switch was:

    epsilon = 0.20

exactly matching theory.

This gives a precise interpretation of learned invariance:

> Forget a distinction only when the predictive value of preserving it is smaller than its representation cost.

---

# V142 — Synergistic information defeats greedy expansion

Variables:

    z1,z2,z3,z4 ∈ {-1,+1}

Target:

    Y = z1*z2

Base constant predictor MSE:

    1

Add z1 alone:

    MSE = 1

Add z2 alone:

    MSE = 1

Every single variable gives exactly zero improvement.

But allow pair interaction:

    z1*z2

Then:

    MSE ≈ 3.7e-32

essentially exact.

All other pairs remained at MSE 1.

Therefore:

> A greedy representation-expansion rule that only considers individually useful information can provably miss jointly useful variables.

The architecture must sometimes escalate from:

    single-variable proposals

into:

    interaction / grouped proposals

when residual uncertainty remains high.

---

# V143 — Grow the restructuring-operation library itself

Initial restructuring library:

    KEEP_ONE_STATE

Hidden tasks required a two-coordinate delay state.

Controller rule:

If every existing restructuring move keeps validation error above:

    0.08

propose a generic meta-move:

    ADD_HISTORY_LAG

Promote it only if it reduces validation error by more than 80%.

First episode:

- one-state model remained bad
- ADD_HISTORY_LAG was generated
- validation passed
- move promoted

Results across 24 episodes:

Mean one-state error:

    0.2936

Mean promoted delay-state error:

    1.73e-14

Promoted move used:

    24 / 24 episodes

This is the first test where the persistent library being expanded is not the task-program library, but the library of architectural restructuring actions.

Caveat:

The meta-generator for ADD_HISTORY_LAG was supplied.

---

# V144 — Shortcut falsification under distribution shift

Stable feature:

    C

Target y followed C with 10% noise.

Shortcut S was copied from y with only 2% error in the ordinary IID environment.

IID validation accuracy:

- C: 90.12%
- S: 98.02%

An IID-only selector chose S:

    500 / 500 trials

Then the environment shifted so S became anti-correlated with y.

Shifted accuracy of the IID-selected representation:

    1.994%

A second validation environment deliberately randomized the shortcut relation while preserving the C relationship.

Intervention validation:

- C: 89.87%
- S: 50.09%

A robust selector using worst-case performance across IID + falsification environments chose C:

    500 / 500

Shifted accuracy:

    90.03%

Conclusion:

> IID predictive superiority is insufficient evidence that a learned representation captures stable structure. Candidate representations should be tested under perturbations chosen to break plausible shortcuts.

The shortcut intervention was supplied by the experiment.

---

# V145 — Meta-abstraction over restructuring operations: failed control

Hidden processes required state/history orders:

    2,3,4

Candidate restructuring operations:

    SET_HISTORY(k)

Pure validation-error minimization produced:

    SET_HISTORY(4)
    SET_HISTORY(4)
    SET_HISTORY(4)

for true orders 2,3,4.

Why?

Once the correct order had been reached, additional lags slightly reduced finite-noise prediction error.

Therefore no useful pattern such as:

    SET_HISTORY(2), SET_HISTORY(3), SET_HISTORY(4)

was available to parameterize.

Again:

> Pure best-error selection destroys minimal structural information by over-expanding.

---

# V145b — Parameterize the smallest sufficient self-modification

Observation noise:

    sigma = 0.004

Predictive sufficiency threshold:

    10 * sigma^2
      =
    0.00016

Successful restructuring was defined as:

> the smallest history size whose held-out rollout error falls below the sufficiency threshold.

Training meta-tasks now produced:

true q=2:

    SET_HISTORY(2)

true q=3:

    SET_HISTORY(3)

true q=4:

    SET_HISTORY(4)

These closed successful moves anti-unified into:

    SET_HISTORY(k)

Held-out unseen process:

    true order = 5

Closed old library only had k=2,3,4.

Best closed move:

    k=4

MSE:

    0.0071813

Parameterized restructuring operator searched k and selected:

    k=5

MSE:

    4.088e-6

Improvement:

    ~1756.6x

Conclusion:

> Repeated successful architectural modifications can themselves become parameterized abstractions, provided "success" is defined by minimal predictive sufficiency rather than unrestricted error minimization.

---

# Strongest architectural update after V145

Several previously separate lessons now form one loop.

```text
CURRENT REPRESENTATION / ARCHITECTURE
              |
              v
       solve / predict / act
              |
              v
       residual uncertainty
              |
      +-------+--------+
      |                |
 uncertainty high   near noise floor
      |                |
      v                v
propose expansions   KEEP unless
      |             evidence clears
      |             confidence gate
      v
falsify shortcuts / test invariances
      |
      v
remove structurally invalid moves
      |
      v
search useful information
      |
      +---------------------------+
      |             |             |
   context        state        subtype
      |             |             |
      v             v             v
parameterize   expand state      split
      |             |             |
      +-------------+-------------+
                    |
                    v
            successful change?
                    |
                    v
         store restructuring move
                    |
                    v
      repeated related modifications?
                    |
                    v
          parameterize meta-move
                    |
                    v
            restructuring library
                    |
                    ↺
```

---

# New exact / near-exact lessons

## 1. Invariance threshold

Under the V141 model:

    preserve distinction iff predictive error avoided > representation cost

Specifically:

    epsilon^2 > lambda

## 2. Greedy information expansion is not sufficient

V142 gives an exact XOR/parity-style counterexample where every individual variable has zero value but a pair has complete value.

## 3. Structural sufficiency matters more than absolute error minimization

V140 and V145 independently show that unrestricted error minimization encourages unnecessary state/library growth at the noise floor.

## 4. Falsification must accompany fit

V136 and V144 show two different shortcuts that ordinary validation rewards but structure-preserving / intervention validation destroys.

## 5. Self-modification can itself become an abstraction domain

V143 stores a successful architectural move.

V145b parameterizes a family of architectural moves.

This pushes recursive abstraction one level above task programs.

---

# Remaining frontier after V145

1. Generate falsification/intervention tests autonomously rather than receiving them from the experimenter.

2. Generate new restructuring operators rather than selecting from supplied meta-generators.

3. Search synergistic information groups without combinatorial explosion.

4. Learn confidence/noise thresholds from experience rather than supplying them.

5. Learn the resource valuation lambda dynamically from real hardware/task budgets.

6. Jointly optimize task primitives and restructuring primitives.

7. Avoid catastrophic meta-library growth: routing, typing, merging, splitting, and pruning must also apply to restructuring operations.

8. Test whether the same architecture can discover:

    task abstraction
    -> representation argument
    -> state variable
    -> type constraint
    -> falsification test
    -> restructuring operator
    -> parameterized restructuring operator

without the researcher selecting the next algorithmic stage.

The clearest next target is therefore:

> an active scientific controller that does not merely choose a representation, but chooses which experiment to perform next in order to decide how its own representation and architecture should change.
