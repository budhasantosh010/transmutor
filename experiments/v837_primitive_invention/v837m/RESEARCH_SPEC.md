# V837m Research Spec

## Calibrated premise

V837j showed matched learned references did not solve >=4/5 at the original 128-episode regime. V837k showed 2x/4x optimizer steps did not rescue them. V837l then showed the small GRU reaches 5/5 at 4x unique development episodes with optimizer steps fixed at 192. The benchmark is therefore learnable, but the original regime has a sample-efficiency limitation.

## V837m question

Does allowing a stable learned linear transform of the previous cell state to bypass the tanh candidate bottleneck recover neutral-cell competence under the same 4x calibrated data regime?

## Conditions

1. Historical direct tanh update.
2. V837g scalar persistence (`learned_leaky`).
3. Stable general linear state transport: `s_next = A s + candidate`, with `A = 0.95 B / ||B||_2`.
4. Exactly parameter-matched additive control: the same stable matrix contributes inside the tanh preactivation instead of as a direct state carry path.

The transport and matched-additive conditions both have 1,016 parameters. Historical has 856; scalar persistence has 866. All conditions use the same 10-cell/55-edge topology, initialization seed pairing, tasks, 512 development episodes, 128 validation episodes, AdamW hyperparameters, and 192 optimizer steps.

## Gate

The original high-capacity criterion remains unchanged: a family is capable only when median development success >=0.90 and median validation success >=0.85, and the representation screen requires >=4/5 families. A transport mechanism claim additionally requires the transport condition to outperform the exactly parameter-matched additive control rather than merely benefiting from parameter count.

Fresh-audit seeds are forbidden. Primitive mining remains blocked.
