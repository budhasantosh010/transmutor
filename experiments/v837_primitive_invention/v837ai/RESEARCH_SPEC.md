# V837ai frozen research spec

Question: what is the minimum tested V837l-compatible unique-development multiplier at which exact AF1D retains >=4/5 representation adequacy?

- Architecture: exact imported `CandidateInputFactorizationY3`, condition `AF1D_deshared_candidate_input_factorization`.
- Data: nested development seeds 10000..10127, 10000..10255, 10000..10511; validation 20000..20127.
- Training: AdamW, 192 steps, lr 0.005, wd 0.0001, clip 5.0, five replicates.
- AI4: reuse accepted V837af rows; no new 4x fits.
- Gate: family dev >=0.90 and val >=0.85; representation >=4/5 families.
- Claim discipline: directly tests unique-data threshold only; parameter/compute comparisons remain descriptive.
- Historical-protocol caveat: increasing unique development data also increases examples contributing to each of the fixed 192 full-development-set optimization steps.
