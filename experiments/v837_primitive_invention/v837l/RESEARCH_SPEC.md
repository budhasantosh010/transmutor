# V837l Research Spec

## Question

Does increasing only the number of unique development episodes allow the matched learned recurrent references to reach the unchanged V837 high-capacity competence criterion?

## Frozen comparison

- 1x data: reuse V837j, 128 unique development episodes.
- 2x data: 256 unique development episodes.
- 4x data: 512 unique development episodes, only if 2x remains below 4/5.
- Optimizer steps remain 192 in every condition.
- Validation remains the same 128 episodes and seed IDs.
- GRU and residual-RNN hidden sizes remain frozen from V837j.
- Neutral high-capacity reference remains included as a matched control.
- Initialization seed namespace remains `v837j-primary-init`.

Fresh-audit seeds 90000–90499 are forbidden. Primitive mining remains blocked.
