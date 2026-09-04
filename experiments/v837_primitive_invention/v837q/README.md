# V837q — Shared-State Organization Diagnostic

V837q tests one variable only: **recurrent-state ownership / fragmentation**.

The historical high-capacity neutral graph contains 10 cells with 4 private recurrent dimensions each (40 total). V837q preserves the same 10-cell graph, historical direct tanh cell law, 4D message interface, broadcast input access, 40-wide readout, optimizer, 4x unique-data calibration regime, seeds, and V837 capacity criterion while progressively reducing the number of recurrent state owners:

- Q0: 10 private groups × 4D = 40D
- Q1: 5 shared groups × 8D = 40D
- Q2: 2 shared groups × 20D = 40D
- Q3: 1 shared group × 40D = 40D

Shared conditions reuse the historical trainable cell/edge/readout parameters. Cells read a fixed task-independent 4D projection of their assigned group state and write candidate contributions back through the fixed projection transpose. No projection is trainable. Recurrent group state is snapshotted at each timestep and committed simultaneously after all pathway contributions are computed; existing non-recurrent message ordering remains represented through sequential candidate outputs.

V837q does **not** enable V837p dynamic modulation, attention, routing, memory modules, structural search, motif mining, fresh-audit seeds, or V838.

The diagnostic also includes a 40D dense vanilla RNN control and the calibrated 13D GRU positive control. These are references only and cannot enter the Transmutor primitive archive.

Run the baseline first:

```text
python experiments/v837_primitive_invention/v837q/run_state_organization_diagnostic.py --phase baseline
```

Only if `diagnostics/baseline_compatibility.json` reports `compatible=true` may the primary shared-state/reference batch run:

```text
python experiments/v837_primitive_invention/v837q/run_state_organization_diagnostic.py --phase primary
python experiments/v837_primitive_invention/v837q/analyze_results.py
```

Conditional no-message/projection-sensitivity controls are guarded by the completed Q3 result and are not part of the default run.
