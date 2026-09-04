# V837r — Global Recurrent Coupling Localization

V837r isolates one architectural variable after V837q closed the state-ownership hypothesis: **direct cross-dimensional recurrent coupling**.

The recurrent state remains the historical 10 cells × 4 dimensions = 40 total dimensions. Historical cell transforms, 55-edge message graph, broadcast input, direct tanh update, 40-wide readout, 4× unique-data regime, optimizer, seeds, and V837 capacity criterion remain fixed.

Primary global conditions add only a recurrent term computed from the previous-timestep concatenated 40D state. The primary global matrix is cross-block-only: every 4×4 same-cell diagonal block is masked to zero, so added recurrence can only carry information between different cells.

Low-rank conditions use configured ranks 1, 2, 4, and 8. The dense condition uses a 40×40 matrix with the ten 4×4 diagonal blocks masked. Every coupling scale has an exactly parameter-matched local-only factorized control that participates in computation but cannot access another cell's state.

Execution is staged for compute discipline:

```text
python experiments/v837_primitive_invention/v837r/run_coupling_diagnostic.py --phase baseline
python experiments/v837_primitive_invention/v837r/run_coupling_diagnostic.py --phase screen
python experiments/v837_primitive_invention/v837r/analyze_results.py --screen-only
```

Only if `diagnostics/screen_decision.json` authorizes localization may rank1/rank8 run:

```text
python experiments/v837_primitive_invention/v837r/run_coupling_diagnostic.py --phase localization
python experiments/v837_primitive_invention/v837r/analyze_results.py
```

If the screen closes the hypothesis, final analysis is run immediately without rank1/rank8. V837s is never auto-run; its permission is machine-readable in `diagnostics/decision_state.json` and requires the frozen interaction guard.

V837r does not enable V837p dynamic modulation, V837q shared state, attention, routing, external memory, structural search, primitive mining, fresh-audit seeds, or V838.
