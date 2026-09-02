# V836 Failure Tree

## Status

Historical V836 is preserved as **PASS**, not FAIL. The stronger recovery program requires reproduction before any root-cause ablation or repair can be scientifically interpreted. Exact reproduction is currently blocked because the preserved inventory references `transmutor_v828_v836_experiments.zip` (23,579 bytes; SHA-256 `dcb0bfc2e4927ee89511ae7f87701ba465f6a7193a838cdb1b2dc0e1c122f953`) but that source archive is absent from the migrated repository and migration package.

Therefore the diagnostic tree stops at the reproduction gate rather than inventing a failure.

```text
historical V836 = PASS
        |
        v
recover executable + generator + seeds + exact gate
        |
        +-- missing source archive --> CANNOT_REPRODUCE_MISSING_SOURCE
                                      |
                                      v
                           root-cause ablations BLOCKED
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
                 v                    v                    v
          implementation         benchmark/oracle    search-vs-repr
          checks not run         checks not run      checks not run
                 |                    |                    |
                 +--------------------+--------------------+
                                      |
                                      v
                           V836b/c repair NOT CREATED
```

## Required diagnostics once the historical source is recovered

The order is frozen by the recovery specification:

1. implementation correctness: indexing, leakage, metrics, normalization, seeds, masking/state reset, ranking, off-by-one errors, and resource accounting;
2. benchmark validity: oracle/generating solution, noise floor, statistical power, and whether the original gate is achievable;
3. search versus representation: oracle representation + normal search, normal representation + strong search, oracle representation + strong search;
4. optimization precision if continuous parameters exist;
5. credit assignment if V836's library admission decision used delayed or resource-adjusted outcomes;
6. development, same-family fresh seeds, shifted family, and new-composition generalization.

## Current classification

This is **not** a scientific V836 failure classification. It is a preservation/reproduction blocker:

`CANNOT_REPRODUCE_MISSING_SOURCE`

No `IMPLEMENTATION_FAILURE`, `SEARCH_FAILURE`, `REPRESENTATION_FAILURE`, or other scientific failure class is assigned because doing so would exceed the preserved evidence.

## Repair status

No V836b/V836c variant has been run. Historical V836 remains unchanged. A repair variant must not be created unless exact/statistical reproduction is first established and a controlled ablation isolates a genuine cause.
