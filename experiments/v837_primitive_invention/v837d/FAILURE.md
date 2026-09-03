# V837d FAILURE

WHAT: fixed sparse raw-input access did not restore >=4/5 high-capacity competence.

WHERE: failing aggregate families: ['delayed_recall', 'conditional_routing', 'variable_composition', 'partial_observation'].

WHY: Sparse access increased generic message dependence but did not restore competence, so more input sparsity is not justified.

CONTROLS: historical broadcast, degree-preserving shuffled sparse masks, and same-mask no-message training/evaluation were all run with paired task seeds and matched parameter budget.

DIAGNOSTICS: selected density=0.5; message-dependency median=0.4620145910450697; mean pairwise-state-correlation median=0.1583026025392675; saturation median=0.02904040404040404; raw-ablation median summary=0.010432047955691814; message-ablation median summary=0.013992452109232546.

CLASSIFICATION: ['INPUT_ACCESS_FAILURE', 'MESSAGE_MEDIATION_FAILURE']; outcome=MESSAGE_MEDIATION_INDUCED_BUT_REPRESENTATION_STILL_INSUFFICIENT.

NEXT: Skip further sparsity and test the generic state update law as the next single representation variable.

Fresh-audit seeds consumed: NO. Primitives promoted: 0.
