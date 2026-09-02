# V836 baseline reproduction

The migrated evidence preserves V836's narrative log and final result JSON, but not the historical executable source archive referenced by the registry.

Historical source reference:

- path: `transmutor_v828_v836_experiments.zip`
- expected size: 23,579 bytes
- expected SHA-256: `dcb0bfc2e4927ee89511ae7f87701ba465f6a7193a838cdb1b2dc0e1c122f953`

The source archive is absent from both the repository and `transmutor_github_migration_full.zip`. `reproduce_v836.py` therefore refuses to invent an implementation or replacement numbers. It verifies the preserved result hash and emits the required reproduction classification.

If the exact historical archive is later restored, pass its path using `--source-archive`. The next step is to inspect it, record executable source paths and SHA-256 values, copy the historical logic into this active directory with explicit attribution, and only then rerun the original algorithm/generator/seeds/gate.
