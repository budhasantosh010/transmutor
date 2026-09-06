# V837y Frozen Research Specification

Question: **Can the neutral substrate reach representation adequacy when the previously useful joint global scalar temporal controller and the previously useful rank-4 cross-cell candidate transformation are present simultaneously?**

The factorial is fixed before execution:

- Y0: controller OFF, candidate coupling OFF.
- Y1: exact V837x X2 joint global scalar carry, coupling OFF.
- Y2: controller OFF, exact V837r R3 rank-4 cross-block candidate coupling.
- Y3: exact Y1 controller + exact Y2 candidate coupling.
- Y3C: exact Y1 controller + exact V837r C3 rank-4 parameter-matched local candidate branch.

Both Y3 mechanisms read the same previous-state snapshot. Rank-4 coupling is inside candidate preactivation; scalar carry is applied after candidate formation. Historical mixed same-step/recurrent graph message scheduling and per-cell input transforms are unchanged.

All conditions use 512 development and 128 validation seed-defined episodes per family across five families, the same 3,200 unique family/seed episodes reused across conditions and five replicates, AdamW for 192 optimizer steps, lr 0.005, wd 0.0001, clip 5.0. No fresh audit, structural search, primitive mining, V838, rank sweep, vector gates, dense coupling, extra state, extra cells, extra data, or extra optimization is authorized.
