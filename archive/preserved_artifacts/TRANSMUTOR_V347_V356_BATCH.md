# Transmutor Experiments V347–V356

## Purpose
Move beyond exact finite deterministic worlds into stochastic, partially observed environments.

Core question:
**Does "state = distinctions about the past that matter for relevant futures" survive when predictive state is continuous, noisy, uncertain, and learned rather than enumerated?**

---

## V347 — Continuous stochastic predictive state
Two-hidden-state noisy HMM.

Result:
- Future-prediction manifold PC1 variance explained: 1.000
- Exact Bayesian belief predicts all tested future observation probabilities affinely, numerical error ~0.
- Last-observation-only next prediction MSE: 0.01297
- Posterior-belief MSE: ~3.4e-32

Conclusion:
A stochastic predictive state need not be a discrete label. It can be a continuous belief coordinate.

Caveat:
Exact Bayesian belief and world parameters were supplied.

---

## V348 — Learn the belief coordinate from observations
1D GRU trained only for next-observation prediction.

Result:
- Memoryless BCE: 0.64599
- Learned 1D recurrent BCE: 0.62169
- Bayesian oracle BCE: 0.61896
- Learned scalar vs exact belief correlation: -0.9815
- Linear R² belief decoding: 0.9633

Conclusion:
A single learned recurrent coordinate can rediscover most of the sufficient belief state from raw observation streams.

Caveat:
GRU architecture and backpropagation supplied.

---

## V349 — Discrete approximation of continuous belief
Quantize exact belief into K states.

Result:
- 2 states: BCE 0.63186
- 4 states: 0.62284
- 6 states: 0.62131
- 8 states: 0.62081
- Oracle: 0.62054
- Smallest K within 0.001 BCE of oracle: 6

Conclusion:
Finite states approximate continuous belief with diminishing returns; forcing continuity into discrete states has a measurable representation cost.

Caveat:
Quantization was applied directly to exact Bayesian belief.

---

## V350 — Active sensing under finite energy
Cheap sensor: accuracy .65, cost 1.
Expensive sensor: accuracy .95, cost 8.
Policy chooses sensor from expected information gain minus energy cost.

Result examples:
- Cheap only: log loss 0.5469, energy 1
- Expensive only: 0.1093, energy 8
- Active λ=.08: log loss 0.3686, energy 2.00, expensive sensor 14.3%

Conclusion:
Predictive uncertainty can allocate expensive sensing/computation selectively, generating an information-energy frontier.

Caveat:
Sensor models and transition model known exactly; policy formula supplied.

---

## V351 — Predictive dimension rises with world dimension
Three-hidden-state HMM.

Result:
Future prediction manifold:
- PC1: 69.87%
- PC2: 30.13%
- PC1+PC2: ~100%

Learned GRU:
- 1D NLL: 0.96289, belief R² 0.383
- 2D NLL: 0.90737, belief R² 0.941
- 3D NLL: 0.90743, belief R² 0.944

Conclusion:
The world requires two predictive coordinates. 1→2 dimensions matters; 2→3 adds essentially nothing to held-out prediction.

Caveat:
Hidden dimension manually swept.

---

## V352 — Automatic predictive-state growth
Rule:
Start at dimension 1. Train h+1 candidate. Grow if validation NLL improves > .01.

Six runs/world.

Result:
Two-hidden-state world:
- final dimensions [1,2,1,1,1,2]
- correct stop at 1D: 66.7%

Three-hidden-state world:
- [2,3,2,2,2,2]
- correct stop at 2D: 83.3%

Conclusion:
Same structural rule roughly adapts internal dimension to world complexity.

Failure:
Optimization noise causes overgrowth.

---

## V353 — Replicated structural evidence
Three independent training replicas per hidden size; compare median NLL.

Result:
Two-state world:
- 4/4 stopped correctly at 1D

Three-state world:
- only 1/4 stopped at 2D
- 3/4 overgrew to 3D

Important failure:
Replication removed false growth in the easy world but worsened overgrowth in the harder world.

Interpretation:
Minimal predictive dimension != easiest dimension to optimize.

---

## V354/V354b — Learning speed vs minimality
Train 2D vs 3D longer in the three-state world.

Result:
At epoch 6:
- 3D ahead by 0.02132 NLL

At epoch 30:
- 3D ahead by only 0.00221

3D advantage shrank ~89.6%.
2D entered the <=.003 catch-up band at epoch 24.

Conclusion:
Overcomplete state can be substantially easier to learn early even when it contains more dimensions than theoretically needed.

Correction:
V354b corrected an overly permissive catch-up checkpoint in the original V354 interpretation; raw V354 NLLs remain valid.

---

## V355 — Naive overgrow then distill
Train 3D teacher then distill into 2D recurrent student.

Result:
- 3D teacher mean NLL: 0.90740
- direct 2D: 0.91083
- distilled 2D: 0.92465

Failure:
Naive recurrent distillation was worse overall, although one seed improved.

Conclusion:
Compression of recurrent dynamics is not automatically easy.

---

## V356 — Static information compression vs dynamics compression
Train 3D teacher, PCA its hidden states, retrain only a readout on projected states.

Result:
Teacher hidden variance:
- PC1: 61.09%
- PC1+PC2: 98.71%
- PC3: 1.29%

Prediction:
- original 3D teacher NLL: 0.90286
- 1D projected readout: 0.94969
- 2D projected readout: 0.90324
- gap 2D projection to teacher: 0.00038
- V355 standalone 2D recurrent distill: 0.92465

Conclusion:
The teacher's **information** is almost perfectly compressible to 2D, but compressing the **recurrent transition dynamics** is much harder.

This is a major distinction.

---

# Current narrowed hypothesis

The strongest surviving idea is no longer simply:

"learn a small state."

It is:

**Maintain an internal representation whose distinctions are justified by different relevant futures, while separately optimizing the dynamics that update that representation under finite data, energy, and development compute.**

This creates at least four different quantities:

1. Predictive information dimension
2. Learnable implementation dimension
3. Runtime resource cost
4. Development/search cost

These need not be equal.

A system may rationally:
- temporarily overgrow because an overcomplete representation is easier to learn,
- later discover that much of its state is informationally redundant,
- then face a separate hard problem of compressing the update dynamics.

---

# Strong next targets

1. Learn stochastic predictive states without supplied GRU/backprop architecture.
2. Learn active sensing policy and sensor reliability from experience rather than known models.
3. Compress recurrent dynamics, not merely hidden information.
4. Continuous latent worlds where belief requires mean + uncertainty, not categorical probabilities.
5. Predictive-state growth under nonstationary world changes.
6. Maintain multiple stochastic world models without false consensus.
7. Integrate predictive-state dimension growth + active sensing + primitive/library growth in one continual environment.
