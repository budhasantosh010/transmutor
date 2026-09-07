# V837ab Research Specification

Frozen parent: V837aa. Frozen successful reference: V837t `T2_scalarized_update_no_reset`.

Primary causal comparison: AB0 factorized T2 versus AB1 fully folded direct T2, with exact step-zero effective-function matching. Secondary localization conditions isolate candidate-only factorization (AB2), update-only factorization (AB3), frozen shared preconditioning (AB4), and ordinary direct GRU initialization (AB5).

All conditions use 512 development + 128 validation episodes per family, five families, five replicates, 192 AdamW steps, the V837t seed ranges and initialization namespace, and zero fresh-audit episodes.
