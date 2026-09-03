# Live Repository Verification Index

This file is the compact entry point for independent review of the live private repository `budhasantosh010/transmutor`. It indexes the research state at `4d065927ad0cc2ccf92808cd2e984c2e6c0e7ca7`. The commit containing this verification-only package has a later SHA by construction; use `git rev-parse HEAD` for the container commit.

## Historical boundary

- V836 historical status: **PASS**.
- Exact V836 reproduction status: **CANNOT_REPRODUCE_MISSING_SOURCE**.
- V837 is an independent post-V836 lineage, not a V836 repair.
- V837 representation recovery is closed under the three-variant stop rule; primitive mining remains blocked.

## Primary evidence paths

| Evidence | Repository path |
| --- | --- |
| V837 original results | `experiments/v837_primitive_invention/v837/results.json` |
| V837b results | `experiments/v837_primitive_invention/v837b/results.json` |
| V837c results | `experiments/v837_primitive_invention/v837c/results.json` |
| V837d sparse-input recovery | `experiments/v837_primitive_invention/v837d/results.json` |
| V837g state-update recovery | `experiments/v837_primitive_invention/v837g/results.json` |
| V837h interaction-basis recovery | `experiments/v837_primitive_invention/v837h/results.json` |
| Frozen V837 gates | `experiments/v837_primitive_invention/frozen_gates.json` |
| V837 lineage status | `experiments/v837_primitive_invention/lineage_status.json` |
| Original V837 blocker analysis | `experiments/v837_primitive_invention/BLOCKER_ANALYSIS.md` |
| Representation blocker analysis | `experiments/v837_primitive_invention/REPRESENTATION_BLOCKER_ANALYSIS.md` |
| Original V837 resources | `experiments/v837_primitive_invention/final_resource_accounting.json` |
| Representation-recovery resources | `experiments/v837_primitive_invention/representation_recovery_resource_accounting.json` |
| Active validator | `scripts/validate_active_research.py` |
| Fast live-repo verifier | `scripts/verify_live_repo.py` |
| Reproduction dispatcher | `scripts/reproduce_v837_recovery.py` |
| Tests | `tests/` |
| Machine-readable verification manifest | `verification/live_repo_manifest.json` |
| Active-file SHA-256 list | `verification/active_research_sha256.txt` |
| Model snapshot metadata | `verification/model_snapshot_manifest.json` |

## Historical SHA-256 anchors

- V836 result: `0ed63ee1e1c5903c1c90b58942aaf968b747df19d4c4a51c1d73a6b36f91527d`
- V837 result: `5fed69cc990be5c6f64a5229f59ff7f27af0c1fc26398bdfbe80ee46255eef14`
- V837b result: `f131110969e7700ec0cd9a82825e8554a51a9c05bb308d54625452db54e35cb0`
- V837c result: `994195fdd0e32e12ec44521ea782c1fc3561b8f596fd4a70e9d59f335fe7d009`
- Frozen V837 gate: `a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`

## Locked scientific state

- Fresh-audit episodes consumed: **0**.
- Primitives promoted: **0**.
- Current scientific frontier: **generic tanh-centered cell-update family**.

## Fast verification

Run:

```text
python scripts/verify_live_repo.py
python scripts/validate_active_research.py
python scripts/validate_registry.py
python -m unittest discover -s tests
```

The fast verifier is read-only and does not rerun expensive experiments. Reproduction entrypoints are dry-run by default; pass `--execute` only when an explicit rerun is intended.
