# V837n — Successful-GRU mechanism localization

V837n is a diagnostic control experiment. It does not add a Transmutor mechanism.

The experiment reimplements the successful V837l GRU explicitly and then ablates update/carry and reset/candidate-conditioning mechanisms under the same calibrated 4x unique-development-data regime where the learned GRU reference reached 5/5 families.

Fresh-audit seeds remain unused. Primitive mining and structural search remain blocked.

Run in scientific order:

```text
python experiments/v837_primitive_invention/v837n/run_mechanism_ablation.py --phase full
python experiments/v837_primitive_invention/v837n/run_mechanism_ablation.py --phase ablations
python experiments/v837_primitive_invention/v837n/analyze_results.py
```

The ablation phase refuses to run unless the explicit full-GRU positive control first reproduces at least 4/5 families and passes the frozen compatibility check against V837l.
