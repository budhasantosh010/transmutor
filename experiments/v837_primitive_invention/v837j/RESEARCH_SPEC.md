# V837j Research Spec

## Single diagnostic question

Separate neutral representation failure from training/data/optimization/capacity ambiguity by comparing the frozen high-capacity neutral graph against parameter-matched conventional recurrent controls.

## Controls

- B0: current 10-cell/55-edge neutral high-capacity graph.
- B1: small GRU reference with deterministic parameter-count matching.
- B2: simple residual recurrent MLP with fixed residual coefficient and no gates.
- B3: optional cheap dense vanilla tanh RNN control.

All receive only observation tensors and lengths?never family IDs/names. Primary runs use the same 128 development episodes, same 128 validation episodes, AdamW, 192 optimizer steps, learning rate 0.005, weight decay 1e-4, gradient clipping 5.0, and five independent initialization replicates.

The historical compatibility probe uses the exact preserved blocker seeds/restarts and must stay within the frozen >0.10-on-two-families drift guard before primary interpretation.

## Gate

The capacity criterion is imported from `common/gates.py`: median development >=0.90 and median validation >=0.85 per family; >=4/5 families establishes conventional learned benchmark learnability.

Fresh audit and primitive mining remain locked.
