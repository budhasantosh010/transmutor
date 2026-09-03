# V837c — Wider structural search

Parent: V837b.

Single change: double structural-search offspring breadth from 4 to 8 and return to the original readout-only candidate adaptation. Population, generation cap, substrate, tasks, seeds, penalties, and gates are unchanged.

Result: **FAIL (`SEARCH_FAILURE`)**. Full family passes remain 0/5. The overall evolved-minus-matched-random validation gap improves to +20.97 percentage points, showing that structural search matters, but the family-competence prerequisite still fails.

This is the third scientifically distinct failure at the same prerequisite layer and triggers the mandated stop/reassess rule. See `results.json`, `FAILURE.md`, and `plots/`.
