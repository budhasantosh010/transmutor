# V837b — Full-parameter refinement

Parent: V837.

Single change: after the unchanged V837 structural search selects evolved and matched-random graphs, refine all continuous parameters for 48 AdamW steps. Topology, task generators, seed partitions, structural penalties, and pass gates remain frozen.

Result: **FAIL (`SEARCH_FAILURE`)**. Development fitting rises to approximately 100% across families, but held-out generalization remains inadequate and the evolved-minus-random gap collapses to +1.47 percentage points. This rules out simple readout under-training as the sole V837 blocker.

See `results.json`, `FAILURE.md`, and `plots/`.
