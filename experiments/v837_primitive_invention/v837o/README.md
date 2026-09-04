# V837o — Shared-Property Factorial Localization

V837o asks whether the successful 4×-data GRU requires dynamic adaptive state control or whether learned static recurrent pathways are sufficient.

It reuses the frozen explicit GRU tensors/equations from V837n and crosses update/reset pathway states across dynamic, static-vector, static-scalar, and off controls. The required G0–G9 matrix is frozen in `config.json` and `frozen_factorial_gate.json`.

This is a successful-reference diagnostic only. It does not create Transmutor primitives, reopen structural search, or consume fresh-audit seeds.

Run the positive control first:

```text
python experiments/v837_primitive_invention/v837o/run_factorial_localization.py --phase full
```

Only if it passes, run the factorial batch and analysis.
