# V837n Research Specification

## Question

Which property of the successful V837l GRU reference is causally necessary for >=4/5 competence under the calibrated 4x unique-development-data regime?

## Scientific boundary

V837n interrogates the learned reference only. It does not transfer GRU machinery into the neutral Transmutor substrate. V837o/p/q/r remain blocked until V837n narrows the mechanism.

## Positive control

The explicit GRU preserves the exact PyTorch `GRUCell` convention used by V837j/l:

```text
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
h_next = (1-z) * n + z * h
```

It also preserves the V837j/l input projection and readout. The full explicit model must contain exactly 875 parameters and reproduce >=4/5 families at 4x unique data before ablations may be interpreted.

## Conditions

- `full_gru`: no ablation.
- `static_update_vector`: learned time-independent update coefficient per hidden dimension; dynamic update tensors remain registered but inactive.
- `static_update_scalar`: one learned time-independent update scalar; dynamic update tensors remain registered but inactive.
- `no_update`: force update coefficient to zero, so state is overwritten by the candidate; update tensors remain registered but inactive.
- `no_reset`: force candidate-conditioning coefficient to one; reset tensors remain registered but inactive.
- `static_reset_vector`: learned time-independent candidate-conditioning vector; dynamic reset tensors remain registered but inactive.
- `no_update_no_reset`: force update=0 and reset=1, yielding the dense tanh recurrence inside the same outer GRU infrastructure.

## Matched regime

Every condition uses the exact V837l 4x regime:

- 512 unique development episodes, seeds 10000–10511.
- 128 validation episodes, seeds 20000–20127.
- AdamW, 192 optimizer steps, lr 0.005, weight decay 0.0001.
- gradient clipping 5.0.
- hidden size 13.
- five paired initialization replicates per family using the existing `v837j-primary-init` namespace.
- exact existing five task generators and capacity criterion.

No task-family label enters the model.

## Positive-control compatibility

A baseline implementation failure is declared if the full explicit GRU fails >=4/5 or differs from the preserved V837l 4x GRU validation median by >0.10 on at least two families.

## Mechanism evidence

A mechanism is strongly implicated if disabling it reduces aggregate competence to <=2/5, or if the ablation produces a paired validation loss of at least 0.10 with a 95% bootstrap interval entirely below zero on at least two families. A decline to 3/5 is contribution evidence but not sole necessity.

These rules are frozen before V837n results.

## Locks

- fresh audit 90000–90499: unused.
- primitive mining: blocked.
- structural search: blocked.
- V838: not started.
- V837/V837b/c/d/g/h/j/k/l/m result artifacts: immutable.
