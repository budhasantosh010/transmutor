# V836 Reproduction Failure

**WHAT failed?** Exact historical reproduction could not start because the historical executable/source bundle is missing from the migrated repository.

**WHERE did it fail?** At the baseline-reproduction source-discovery gate, before any scientific metric was recomputed.

**WHEN did it fail?** During the V836 recovery pass on branch `research/v836-recovery`, after preserved V833–V836 result/log evidence had been hashed and inspected.

**WHY is it believed to have failed?** `registry/artifact_inventory.json` records `transmutor_v828_v836_experiments.zip`, size 23,579 bytes, SHA-256 `dcb0bfc2e4927ee89511ae7f87701ba465f6a7193a838cdb1b2dc0e1c122f953`, but that archive is absent from the repository and from the provided migration ZIP. The remaining V836 JSON/log do not contain the executable, task generator, seed policy, exact pass-gate calculation, or complete per-task library-cost matrix needed for faithful rerun.

**HOW was the failure reproduced?** `python experiments/v836_recovery/baseline/reproduce_v836.py` checks the preserved V836 result hash and expected source-archive hash/path. It returns `CANNOT_REPRODUCE_MISSING_SOURCE` and exits non-zero by design when the source is unavailable.

**WHAT was tried?** The active repository, the migration ZIP contents, the preserved-artifact tree, and the relevant local Transmutor workspace were searched for the named source archive. Only the inventory reference, historical Markdown log, and V836 result JSON were found.

**WHAT changed in each retry?** Nothing scientific changed. Searches were widened only enough to verify that the named archive was not present in the supplied preservation package/workspace.

**WHAT evidence rules out alternative explanations?** The inventory supplies an exact filename, size, and SHA-256, proving a source archive existed at preservation time. Its absence is independently observable. The current result JSON still hashes to `0ed63ee1e1c5903c1c90b58942aaf968b747df19d4c4a51c1d73a6b36f91527d`, so the blocker is not corruption of the preserved V836 result.

**WHAT remains unknown?** The exact historical executable logic, task generator, seeds, original Boolean gate calculation, and full resource/search accounting remain unknown until the exact source archive (or equivalent original files with matching provenance) is recovered.

**WHAT is the one-line current fix/hypothesis?** Recover the exact `transmutor_v828_v836_experiments.zip` matching SHA-256 `dcb0bfc2e4927ee89511ae7f87701ba465f6a7193a838cdb1b2dc0e1c122f953`; do not substitute reconstructed code and call it historical reproduction.
