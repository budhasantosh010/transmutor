# V837 Learned-Reference Calibration Report

## 1. Why this calibration was necessary

The V837/V837b/V837c neutral-structural lineage failed, followed by V837d fixed sparse input access, V837g learned scalar state persistence, and V837h low-rank interaction diagnostics. Those results made the neutral tanh-centered cell family the strongest representation-level suspect, but oracle solvability alone did not establish that ordinary learned recurrent models could solve the same benchmark under the same training/data regime. V837j+ therefore calibrates benchmark learnability before further Transmutor cell-law redesign.

The reference models are controls only. They are never eligible for primitive mining, archive admission, or structural-search operators.

## 2. Current representation-failure evidence

Before V837j, the strongest supported statement was that the tested neutral continuous-cell substrate/training regime did not provide reliable generalizable competence across the five required task families. V837j+ separates four possible causes: representation family, optimization budget, sample efficiency, and model capacity.

Historical V836 remains `PASS`; exact V836 reproduction remains `CANNOT_REPRODUCE_MISSING_SOURCE`. Historical V837 through V837h artifacts are immutable. Fresh-audit seeds 90000-90499 remain unused.

## 3. Neutral baseline

V837j first reran the preserved high-capacity neutral reference. The compatibility probe reproduced the preserved blocker-diagnostic validation scores with zero drift, so learned-reference comparisons are not explained by implementation/environment drift.

At the primary matched 1x regime (128 unique development episodes, 128 validation episodes, 192 AdamW steps), the 856-parameter neutral high-capacity model passed 1/5 families.

| Family | Neutral validation median | Capacity result |
| --- | ---: | --- |
| conditional routing | 0.3359 | FAIL |
| delayed recall | 0.8125 | FAIL |
| iterative state | 0.9766 | PASS |
| partial observation | 0.6328 | FAIL |
| variable composition | 0.5547 | FAIL |

## 4. Reference architectures

V837j used task-independent recurrent controls behind the same training/evaluation semantics:

- GRU reference: input projection -> `GRUCell` -> output projection; hidden size 13; 875 parameters.
- Residual recurrent MLP: dense recurrent tanh candidate plus fixed residual coefficient; hidden size 26; 885 parameters.
- Optional vanilla dense tanh RNN: dense hidden-state recurrence; hidden size 26; 885 parameters.

No reference model receives task family ID/name/generator-class information.

## 5. Parameter matching

The neutral high-capacity diagnostic has 856 parameters. Deterministic hidden-size selection chose the closest reference sizes:

| Model | Parameters | Difference from neutral |
| --- | ---: | ---: |
| Neutral high-capacity | 856 | 0% |
| GRU | 875 | +2.22% |
| Residual RNN | 885 | +3.39% |
| Vanilla RNN | 885 | +3.39% |

All primary learned references therefore satisfy the frozen +/-10% parameter-matching target.

## 6. Training/data matching

V837j primary comparison held constant:

- optimizer: AdamW;
- optimizer steps: 192;
- learning rate: 0.005;
- weight decay: 0.0001;
- gradient clipping: 5.0;
- development episodes: seeds 10000-10127 (128 unique episodes);
- validation episodes: seeds 20000-20127 (128 episodes);
- five independently seeded training replicates per model/family;
- same existing task generators and success criteria;
- CPU execution; fresh-audit data forbidden.

V837k changed only optimizer-step budget. V837l changed only unique development episodes while restoring optimizer steps to 192. V837m then used the learnability-calibrated 4x unique-data regime for every cell-law condition.

## 7. Per-family results

### V837j matched-budget calibration

| Model | Params | Families passing | routing | recall | iterative | partial observation | composition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 856 | 1/5 | 0.3359 | 0.8125 | 0.9766 | 0.6328 | 0.5547 |
| GRU | 875 | 2/5 | 0.5234 | 0.8594 | 0.9922 | 0.8281 | 0.7578 |
| Residual RNN | 885 | 2/5 | 0.3516 | 0.8672 | 0.9844 | 0.7422 | 0.6016 |
| Vanilla RNN | 885 | 1/5 | 0.4375 | 0.8125 | 0.9844 | 0.7500 | 0.6094 |

V837j therefore did not establish benchmark learnability at the historical 1x regime.

### V837k optimizer-step escalation

Optimizer steps alone were increased from 1x=192 to 2x=384 and 4x=768 with the same 128 unique development episodes.

| Step budget | Neutral | GRU | Residual RNN |
| --- | ---: | ---: | ---: |
| 1x | 1/5 | 2/5 | 2/5 |
| 2x | 1/5 | 2/5 | 2/5 |
| 4x | 1/5 | 2/5 | 2/5 |

The reference failure therefore was not rescued by simply applying up to four times as many optimizer steps to the same data.

### V837l unique-data escalation

V837l restored optimizer steps to 192 and changed only the number of unique development episodes: 128 -> 256 -> 512. The validation set stayed exactly fixed.

| Unique-data budget | Neutral | GRU | Residual RNN |
| --- | ---: | ---: | ---: |
| 1x (128) | 1/5 | 2/5 | 2/5 |
| 2x (256) | 1/5 | 3/5 | 2/5 |
| 4x (512) | 2/5 | **5/5** | 3/5 |

At 4x unique data the GRU validation medians were:

| Family | GRU validation median | Capacity result |
| --- | ---: | --- |
| conditional routing | 0.9375 | PASS |
| delayed recall | 0.9844 | PASS |
| iterative state | 1.0000 | PASS |
| partial observation | 0.8906 | PASS |
| variable composition | 0.8828 | PASS |

The 856-parameter neutral high-capacity model under the identical 4x data regime passed only 2/5 (recall and iterative state). This establishes benchmark learnability under a calibrated data regime and strengthens the evidence that the neutral representation remains deficient relative to a small conventional gated recurrent model.

## 8. Learning curves

V837j learning curves show that most failed 1x conditions fit the development data very strongly while validation performance stayed materially lower. V837k demonstrates that prolonging optimization on those same episodes does not resolve the gap. V837l shows that introducing additional unique development episodes does resolve all five families for the GRU.

The combined pattern is therefore inconsistent with a simple "not enough gradient steps" explanation. It is most directly consistent with sample-efficiency/generalization limitation at 1x data, with architecture-dependent generalization efficiency.

## 9. Training-vs-validation behavior

At 1x and expanded-step V837k conditions, many models achieved development success near 0.98-1.00 while failing held-out validation criteria. Those rows are appropriately treated as generalization failures rather than optimization underfitting.

V837l changes this conclusion: with 4x unique data and the original 192-step optimizer budget, the GRU passes all five validation families. The benchmark/training pipeline is therefore learnable by a small conventional recurrent model once the data support is sufficient.

## 10. Compute accounting

Actual V837j/k/l/m work, avoiding double-counting reused 1x conditions:

| Component | Model fits | Optimizer steps | Examples processed | Environment interactions | Forward calls | Worker wall sec | CPU sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V837j primary | 100 | 19,200 | 2,457,600 | 175,860 | 23,000 | 2,008.10 | 1,335.89 |
| V837k added 2x/4x | 150 | 86,400 | 11,059,200 | 263,790 | 94,125 | 9,307.81 | 6,895.77 |
| V837l added 2x/4x data | 150 | 28,800 | 11,059,200 | 532,290 | 32,250 | 2,925.19 | 2,462.19 |
| V837m | 100 | 19,200 | 9,830,400 | 444,900 | 21,500 | 6,377.81 | 5,267.88 |
| **Total** | **500** | **153,600** | **34,406,400** | **1,416,840** | **170,875** | **20,618.91** | **15,961.72** |

GPU usage: 0.

## 11. Diagnosis

The calibration resolves the original ambiguity in two layers:

1. **Sample-efficiency limitation is real.** Neither GRU nor residual RNN reaches 4/5 at the original 1x data regime, and 4x optimizer steps on the same data does not help. Increasing only unique development episodes to 4x makes the small GRU pass 5/5. This is classified `SAMPLE_EFFICIENCY_FAILURE` for the original regime.
2. **A representation gap remains under a learnable regime.** Under the exact 4x data regime where the 875-parameter GRU passes 5/5, the 856-parameter neutral high-capacity model passes 2/5. This strengthens, but does not universalize, the representation-family diagnosis.

The strongest justified claim is therefore not that the benchmark was learnable under the original V837 data budget. It was not demonstrated to be. The stronger calibrated claim is that the benchmark becomes fully learnable for a small GRU by changing only unique development data, while the tested neutral cell family still remains far below the 4/5 capacity prerequisite under that same learnable regime.

## 12. Whether V837k/l were needed

Both were required by the frozen decision tree.

- V837k was needed because V837j GRU/residual references were only 2/5 at matched budget. It ruled out optimizer-step budget through 4x as the sole explanation.
- V837l was then required to isolate sample efficiency. It resolved learnability at 4x unique development data, so a separate reference-capacity escalation was **not run**. Capacity escalation would have been unnecessary confounding after an 875-parameter GRU already reached 5/5.

## 13. Whether V837m was justified

Yes. V837l provided the required calibrated evidence: a conventional learned recurrent reference reached 5/5 under a modest, explicitly characterized data escalation. V837m therefore tested the next single Transmutor cell-law property under that shared learnable 4x data regime.

V837m compared:

| Condition | Parameters | Families passing |
| --- | ---: | ---: |
| Historical direct tanh | 856 | 2/5 |
| Scalar persistence | 866 | 2/5 |
| Stable general linear transport | 1,016 | 2/5 |
| Parameter-matched additive control | 1,016 | 1/5 |

The transport matrix was constrained to spectral norm approximately 0.95. Linear transport did not restore >=4/5 competence and did not dominate the historical/scalar controls. It is therefore classified `LINEAR_STATE_TRANSPORT_INSUFFICIENT`. Full structural search remains blocked.

V837m also exposed a useful numerical property: despite the transport matrix itself being norm-bounded, accumulated state norms were much larger than historical controls, with nonzero exploding-state diagnostic frequency. The result therefore gives no basis for promoting general linear transport as the missing mechanism.

## 14. What remains unresolved

The calibrated GRU-vs-neutral gap is now clear, but it does not identify which GRU property is necessary. General linear transport alone failed. The next isolated property should be a **generic adaptive state-update coefficient** conditioned on ordinary state/message/input, rather than importing a full GRU or adding several gates at once.

The next experiment should ask whether adaptive multiplicative carry/update control closes the calibrated 5/5-vs-2/5 gap. It should remain a cheap fixed-topology capacity diagnostic first. Structural search, motif mining, primitive promotion, and fresh-audit use remain prohibited until a neutral Transmutor substrate reaches the high-capacity prerequisite and then passes the original full structural-search competence gates.

## Final scientific state

- Benchmark learnability at original 1x data: **not established**.
- More optimizer steps through 4x: **does not resolve reference failure**.
- Benchmark learnability at 4x unique data: **established by 875-parameter GRU, 5/5**.
- Sample-efficiency confound: **supported**.
- Neutral representation gap under calibrated learnable data: **supported** (neutral 2/5 vs GRU 5/5).
- General linear state transport: **insufficient** (2/5).
- Full structural search allowed: **no**.
- Primitive mining allowed: **no**.
- Fresh-audit episodes consumed: **0**.
- Primitives promoted: **0**.
