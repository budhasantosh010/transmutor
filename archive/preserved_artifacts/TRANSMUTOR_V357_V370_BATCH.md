# Transmutor Experiments V357–V370

## Purpose
Continue from V347–V356 and attack the hardest surviving problems:

1. compress predictive UPDATE DYNAMICS, not merely hidden information,
2. learn compact stochastic world models from raw observations,
3. select latent-state count from evidence,
4. grow/shrink world-model complexity under nonstationarity,
5. learn active sensing from learned world models,
6. distinguish uncertainty about the world state from uncertainty about the model,
7. prevent false certainty from killing curiosity.

---

## V357 — Simple compressed dynamics are insufficient
3D GRU teacher in the three-hidden-state stochastic world.
Teacher hidden states projected to 2D PCA coordinates.

A simple 2D tanh-affine transition was trained on projected teacher transitions.

Results:
- 3D teacher NLL: 0.90290
- 2D projected teacher NLL: 0.90278
- 2D tanh-affine autonomous NLL: 0.91769
- V355 naive 2D recurrent distillation: 0.92465
- 2D tanh-affine one-step state MSE: 0.05125
- mean autonomous state drift MSE: 0.06489

Conclusion:
The correct information dimension is not enough. A too-simple update law accumulates drift.

Status:
A first plotting attempt failed; the clean rerun above is the valid V357 result.

---

## V358 — Correct state dimension still needs nonlinear update law
Use the exact 2D Bayesian belief coordinate directly in the 3-state HMM.

Compare:
- tanh-affine update
- small nonlinear MLP update
- exact Bayesian filter

Results:
One-step belief-update MSE:
- tanh-affine: 8.97e-3
- small MLP: 3.21e-6

Autonomous length-70 belief MSE:
- tanh-affine: 1.376e-2
- small MLP: 7.67e-6

Observation NLL:
- tanh-affine: 0.923519
- MLP: 0.903166
- exact Bayesian: 0.903165

MLP gap to oracle:
- 8.3e-7 NLL

Conclusion:
Predictive STATE DIMENSION and UPDATE-LAW COMPLEXITY are separate quantities.

---

## V359 — Continuous latent world needs estimate + uncertainty
Linear-Gaussian hidden process with randomly missing observations.

Exact Kalman predictive state:
- posterior mean
- posterior variance

Results:
Future-distribution PCA:
- PC1: 50.18%
- PC2: 49.82%
- PC1+PC2: 100%

Next hidden-state probabilistic prediction:
- mean + uncertainty NLL: 0.36455
- mean only / global variance NLL: 0.40117
- improvement: 0.03661

Posterior variance was strongly calibrated to empirical squared prediction error.

Conclusion:
In continuous stochastic worlds, uncertainty itself is future-relevant state.

---

## V360 — Learn estimate + uncertainty from raw observations
GRU sees only:
- observed value or 0
- observation mask

Predicts next observed value as a Gaussian.

Results:

Hidden 1:
- NLL 1.03605
- posterior mean R² 0.805
- posterior variance R² 0.027

Hidden 2:
- NLL 1.00097
- posterior mean R² 0.980
- posterior variance R² 0.762

Hidden 3:
- NLL 1.00104
- posterior mean R² 0.987
- posterior variance R² 0.706

Conclusion:
1D learns the estimate but almost none of uncertainty.
2D captures both and essentially saturates predictive performance.
3D adds no held-out gain.

---

## V361 — Structured dynamics compress the update law
Three-state stochastic world.

Compare belief-update models:
- generic MLP: 435 trainable parameters
- factorized transition/emission/normalization law: 18 logits

Factorized law structure:
1. prior = belief @ transition
2. weighted = prior × emission likelihood
3. normalize

Numerical transition/emission matrices are learned.

Results:
- parameter compression: 24.17×
- generic MLP one-step belief MSE: 1.04e-5
- factorized one-step MSE: 1.22e-15
- generic MLP autonomous length-70 MSE: 3.44e-5
- factorized autonomous MSE: 3.06e-15
- learned factorized observation NLL: 0.90175048
- true model oracle NLL: 0.90175050

Learned transition/emission matrices recovered the true matrices essentially exactly.

Conclusion:
The right compositional primitives can make predictive dynamics dramatically smaller and more stable than a generic approximator.

Caveat:
The Bayesian factorization was supplied; only its numerical parameters were learned.

---

## V362 — Learn hidden stochastic dynamics from raw sequences only
Removed privileged belief-transition targets.

Learner sees only raw observed symbol sequences.
Model class:
- 3-state HMM
- transition + emission probabilities
Learning:
- Baum-Welch / EM

Five random starts.

Results:
Oracle held-out NLL: 0.903572

Three runs recovered the true hidden model almost perfectly:
- run 1 NLL 0.903616
- run 3 NLL 0.903617
- run 5 NLL 0.903615
- matrix MSE ~3–4e-5
- belief MSE ~1.1–1.4e-4

Two runs fell into worse local optima:
- run 2 NLL 0.91774
- run 4 NLL 0.96828

Conclusion:
Compact hidden dynamics are learnable from raw sequences with no hidden-state/belief supervision, but search/local-optimum reliability remains a major issue.

Status:
A first differentiable-filter implementation timed out and produced no valid result. The EM rerun is the valid V362 result.

---

## V363 — Discover latent state count from raw data
True hidden-state count: 3.
Candidate HMMs: K=2,3,4,5.
Three EM restarts each.
Select by held-out prediction and BIC-like fit + complexity.

Held-out NLL:
- K2: 0.96634
- K3: 0.90241
- K4: 0.90268
- K5: 0.90257

Both best held-out NLL and BIC-like score selected:
- K = 3

Conclusion:
Predictive fit improves sharply until the needed hidden complexity, then saturates; description cost can stop unnecessary growth.

---

## V364 — Initial nonstationary grow/shrink attempt
True windows:
[2,2,3,3,2] latent states

Using only 2 EM restarts:
selected:
[2,2,4,4,3]

Result:
- detected growth and later shrink directionally
- exact window accuracy only 40%

Failure interpretation:
Bad search masqueraded as excess structural need.

V364 is superseded by V364b for the corrected state-count claim.

---

## V364b — Search diversity before structural change
Same nonstationary windows.
Candidate K={2,3,4}.
Five EM restarts per K.

Selected:
[2,2,3,3,2]

Result:
- 5/5 windows correct
- growth detected
- shrink detected

Conclusion:
Before changing architecture, spend enough independent search effort to ensure optimization failure is not being mistaken for representational failure.

Cost:
More development/search compute.

---

## V365 — Learned world model drives active sensing
Hidden binary state.
Sensors:
- cheap accuracy .65, cost 1
- expensive accuracy .95, cost 8

Exploration:
- random sensor actions
- raw action + observed bit only
- hidden state never shown

Action-conditional HMM EM recovered:
- transition matrix
- cheap sensor reliability
- expensive sensor reliability

Matrix MSE after label alignment:
- 5.29e-6

Then learned model drove expected-information + energy sensing.

At lambda=.08:

Learned active:
- logloss 0.37141
- energy 2.00166
- expensive fraction 14.309%

Oracle active:
- logloss 0.37239
- energy 2.00192
- expensive fraction 14.313%

Conclusion:
A world model learned from raw exploration can drive active sensing almost identically to a model with true parameters.

---

## V366 — Single-run exploration-budget sweep
Tested random expensive-sensor exploration fractions:
2%, 5%, 10%, 25%, 50%.

The curve was highly non-monotonic:
- some tiny-budget runs succeeded
- 5% fell into a disastrous local optimum

Therefore:
The single-run claim about "minimum exploration needed" is invalid.

V366 is superseded by V366b for exploration-budget reliability.

---

## V366b — Repeated exploration-budget control
Expensive exploration fractions:
1%,2%,5%,10%,25%

Five independent datasets per fraction.
Four EM restarts each.

Good-model recovery rate (matrix MSE < .005):
- 1%: 20%
- 2%: 20%
- 5%: 40%
- 10%: 60%
- 25%: 80%

No tested fraction reached 100%.

Conclusion:
More costly experience improves recovery, but model search quality remains intertwined with data quantity.
Random timing is inefficient.

---

## V367 — Active calibration of unknown expensive sensor
Transition and cheap sensor known.
Expensive sensor accuracy unknown among:
[.55,.65,.75,.85,.95]
True=.95.

Hidden state never revealed.

Compare:
- random expensive calibration times
- active times chosen by information gain about SENSOR-MODEL hypotheses

Mean sensor-accuracy error:

Budget 2:
- active .1800
- random .1902

Budget 4:
- active .1593
- random .1821

Budget 6:
- active .1400
- random .1730

Budget 10:
- active .1108
- random .1551

Budget 16:
- active .0861
- random .1279

Posterior mass on true sensor model was also consistently higher under active calibration.

Conclusion:
Expensive exploration should be allocated where it is informative about the MODEL, not merely sampled randomly.

---

## V368 — State uncertainty and model uncertainty want opposite experiments
Known expensive sensor accuracy .95:
- information gain about CURRENT HIDDEN STATE peaks at belief b=.5

Unknown sensor accuracy:
- information gain about SENSOR MODEL peaks near b≈0 or b≈1

Results:
- state-info peak: b=.5
- sensor-model-info peaks: b=.001 and .999
- correlation of the two information-value curves: -0.99893

At b=.5:
- state IG: .7136 bits
- model IG: 0

At b=.99:
- state IG: .0371 bits
- model IG: .0792 bits

Conclusion:
There is no single scalar "uncertainty."
A system must track uncertainty about:
1. current world state
2. its own model
and possibly many other levels.

Different uncertainties demand different experiments.

---

## V369 — False consensus shuts down curiosity
Unknown expensive-sensor reliability.
Active calibration happens only when model hypotheses disagree enough.

Start conditions:

DIVERSE:
uniform sensor-model hypotheses.

COLLAPSED WRONG:
100% belief in accuracy=.75 while truth=.95.

REJUVENATE:
start collapsed wrong; inject 8% uniform model mass every 25 steps.

Across 2,000 runs:

Diverse:
- mean queries 15.31
- accuracy error .0881

Collapsed wrong:
- mean queries 0
- never-query rate 100%
- accuracy error .2000
- true-model mass 0

Rejuvenation:
- mean queries 15.65
- accuracy error .1321
- first query ~160 steps

Conclusion:
No disagreement => no information gain => no curiosity.
Irreversible hypothesis death can make a wrong system permanently certain.

Rejuvenation helps but is late and incomplete.

---

## V370 — Permanent diversity floor
Initial model belief is completely wrong:
P(.75 accuracy)=1, truth=.95.

At each step:
w <- (1-epsilon)w + epsilon*uniform

Results:

epsilon 0:
- no queries ever
- error .200

epsilon .001:
- still no queries
- error .200

epsilon .005:
- queries ~16
- first query step ~104.6
- error .1581

epsilon .01:
- queries 16
- first query ~58.7
- error .1881

epsilon .02:
- first query ~33.5
- error .1992

epsilon .05:
- first query ~16.8
- error ~.2000

Best tested epsilon:
- .005

Conclusion:
A small diversity reserve can revive curiosity, but too much permanent model mass on bad alternatives prevents concentration.
Diversity preservation itself has an optimal resource level.

Status:
A non-vectorized first V370 attempt timed out; the vectorized rerun above is the valid result.

---

# Strongest surviving conclusions from V357–V370

## 1. Predictive information complexity != dynamics complexity
V358:
2D state is sufficient, but tanh-affine dynamics are not.
A richer nonlinear update is required.

## 2. Structured primitives can radically compress dynamics
V361:
18-parameter factorized update beat/stabilized a 435-parameter generic MLP and recovered exact dynamics.

## 3. Compact world models can be learned from raw sequences
V362:
No belief/hidden-state supervision is required in principle.
But local optima remain severe.

## 4. Latent complexity can be selected from predictive evidence
V363:
Fit improves until K=3 and saturates; complexity cost stops growth.

## 5. World-model complexity should be reversible
V364b:
Under changing worlds, evidence selected 2→2→3→3→2.
Growth and death/pruning are both necessary.

## 6. Search reliability must precede structural conclusions
V364 vs V364b:
Too little search falsely implied more hidden states.

## 7. Learned models can drive active resource allocation
V365:
Raw-experience model produced almost the same sensing frontier as oracle parameters.

## 8. Model learning and state estimation are different information problems
V368:
State-information and model-information values were almost perfectly anticorrelated.

## 9. Curiosity requires maintained alternatives
V369:
Wrong single-model certainty produced zero questions forever.

## 10. Diversity must be preserved but not maximized
V370:
No diversity => irrecoverable certainty.
Too much diversity => inability to concentrate.
A small reserve helped.

---

# Current narrowed architecture hypothesis

The emerging object is increasingly not "a model" but a hierarchy of predictive hypotheses under finite resources:

EXPERIENCE
    ↓
PREDICTIVE STATE
    ├─ estimate of world
    └─ uncertainty about world
    ↓
WORLD-MODEL HYPOTHESES
    ├─ transition laws
    ├─ observation laws
    ├─ state dimension / latent count
    └─ uncertainty about those laws
    ↓
DISAGREEMENT
    ↓
choose which information is worth buying
    ├─ learn current state
    ├─ learn model
    ├─ test architecture
    └─ test primitive
    ↓
UPDATE / GROW / SPLIT / PRUNE / REVIVE
    ↓
compress into reusable structured laws
    ↓
repeat

A key new distinction:

**STATE uncertainty and MODEL uncertainty are not the same thing.**

And another:

**Predictive information dimension, update-law complexity, search difficulty, development cost, and runtime cost are different quantities.**

---

# Strong next experiments

1. Discover the factorized Bayesian-style update composition rather than supplying it.
2. Online birth/death of hidden states inside one persistent nonstationary world model.
3. Joint uncertainty over state count + transition law + sensor reliability.
4. Active experiment selection across levels:
   state vs model vs architecture vs primitive.
5. Hypothesis birth outside a fixed candidate list.
6. Continuous model uncertainty rather than finite sensor-accuracy candidates.
7. Learn diversity/rejuvenation rate itself from lifecycle performance.
8. Integrate primitive-library growth with stochastic predictive-state world models.
