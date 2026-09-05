# V837t — Dynamic Control Granularity Localization

V837t interrogates the already-successful explicit GRU before transferring any new controller into the neutral Transmutor substrate.

The only primary variable is dynamic gate **output granularity**. The original vector-producing gate networks remain intact and trainable in every condition. Scalarized conditions compute the normal post-sigmoid vector gate, take its hidden-dimension mean, and broadcast that mean back across all hidden dimensions.

Run order is guarded:

1. `python run_dynamic_granularity.py --phase anchors`
2. only if T0/T1/T3 remain >=4/5, `python run_dynamic_granularity.py --phase scalarized`
3. `python analyze_results.py`
4. consume `diagnostics/decision_state.json`; do not choose V837u manually.

Fresh-audit, structural-search, primitive-mining, sample-efficiency and V838 remain locked.
