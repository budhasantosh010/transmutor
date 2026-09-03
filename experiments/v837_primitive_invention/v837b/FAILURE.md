# V837b FAILURE

WHAT failed? The unchanged V837 competence gate still failed after full-parameter refinement; failing families: ['delayed_recall', 'conditional_routing', 'iterative_state', 'variable_composition', 'partial_observation'].

WHERE? On the exact V837-selected topologies and matched-random controls.

WHEN? After structural search, before any motif mining.

WHY suspected? If refinement materially improves absolute performance but not the family-level reliability gate, structural search/representation rather than readout optimization remains limiting.

HOW reproduced? Same V837 graphs, same paired seeds, same gate, equal 48-step all-parameter AdamW refinement for evolved and random controls.

WHAT evidence supports diagnosis? `results.json` records before-lineage resource inheritance and every refined paired result.

WHAT alternatives were ruled out? Benchmark validity, first-observation leakage, and simple readout-only underfitting.

WHAT single change next? V837c strengthens structural search breadth only while returning to the original readout-only candidate adaptation.
