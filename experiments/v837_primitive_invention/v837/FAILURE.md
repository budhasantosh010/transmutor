# V837 FAILURE

WHAT failed? Neutral-substrate competence gate; failing families: ['delayed_recall', 'conditional_routing', 'iterative_state', 'variable_composition', 'partial_observation'].

WHERE? Independent bounded structural searches under the common generic cell substrate.

WHEN? Before motif mining; no downstream primitive claim is attempted.

WHY suspected? The initial fixed readout-only parameter-adaptation scope may underfit useful recurrent dynamics even when topology/parameter seeds are searched.

HOW reproduced? Thirty independent searches per family with frozen development/validation partitions and matched random controls.

WHAT evidence? `results.json` contains every run, search cost, matched-random result, and family-level gate.

WHAT alternatives ruled out? Oracle benchmark validity and first-observation leakage are checked before search.

WHAT single change next? V837b changes parameter adaptation only: full-cell AdamW under the same task, search, seed, structure, and pass gates.
