# V837v Research Specification

## Question

Does a distributed neutral substrate require fewer independent temporal control decisions so that local computations evolve coherently?

## Single variable

Only the mapping from existing local scalar controller outputs to controlled cells changes. Controller information remains local to the fixed earliest source cell in each deterministic contiguous domain.

## Conditions

- V0: 10 single-cell domains (exact V837u U2 anchor)
- V1: five 2-cell domains
- V2: two 5-cell domains
- V3: one 10-cell domain using cell 0's local gate

## Frozen regime

512 development episodes/family, 128 validation episodes/family, five families, five replicates, 192 AdamW steps, same paired seeds and unique-data accounting as V837u.

## Gate

Representation adequacy requires at least 4/5 families under the existing frozen V837 capacity criterion. If multiple conditions pass, prefer the passing condition with the most independent domains.

No gate pooling, global state visibility, vector modulation, global recurrent coupling, extra data, structural search, primitive mining, fresh audit, or V838 is allowed.
