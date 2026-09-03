# Transmutor

Transmutor is an experimental research program exploring computational systems that can adapt not only parameters, but also structure, memory, relationships, reusable primitives, search processes, and higher-level abstractions under finite resources.

## Repository status

This repository is being migrated from a long-running experimental research conversation. The migration preserves failures and provenance rather than presenting reconstructed code as original code.

### Provenance labels

- `RAW_SOURCE_PRESERVED` — original source file is present.
- `RESULT_ARTIFACT_PRESERVED` — result JSON/CSV/figure exists but raw source may not.
- `DOCUMENTED_IN_LOG` — experiment is documented in a preserved research log.
- `MISSING_DIRECT_ARTIFACT` — no direct artifact is currently available; do not silently reconstruct it as original.

## Core research loop

`variation -> protected evaluation -> selection -> reuse -> compression -> higher-level abstraction -> repeat`

## Layout

- `docs/` — research overview, rules, claims, and migration notes.
- `registry/` — machine-readable experiment registry.
- `experiments/` — normalized experiment directories as migration proceeds.
- `archive/preserved_artifacts/` — original preserved outputs/logs copied without pretending they are normalized source.
- `scripts/` — migration and validation tooling.

The repository is private during cleanup and evidence reconstruction.

## Research verification navigation

- [Post-V836 frontier audit](docs/POST_V836_FRONTIER_AUDIT.md)
- [V837 primitive-invention report](docs/V837_PRIMITIVE_INVENTION_REPORT.md)
- [V837 representation-recovery report](docs/V837_REPRESENTATION_RECOVERY_REPORT.md)
- [Live repository verification index](docs/LIVE_REPO_VERIFICATION.md)
