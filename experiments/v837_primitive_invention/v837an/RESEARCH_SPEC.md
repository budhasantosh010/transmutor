# TRANSMUTOR — V837an

# Causal / Routing / Distributed Primitive Redefinition

## Program identity

- Repository: `budhasantosh010/transmutor`
- Required START SHA: `f556c92a895b141f73f214e5eda3ac29b1927ea9`
- Branch: `research/v837-primitive-redefinition-causal-routing-distributed`
- Version: `V837an`
- Machine predecessor instruction: `V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED`
- Fresh audit: locked / false
- Primitive archive: locked / false
- V838: locked / false

## Scientific question

**What is the lowest-dimensional interventionally faithful causal computation implemented across independently trained AF1D organisms, and how is that computation routed through their distributed state and communication systems?**

The tested possibilities are:

1. `CAUSAL_MACROVARIABLE`: a low-dimensional distributed variable implements the task's true computational state.
2. `CAUSAL_ROUTING_MECHANISM`: the reusable object is the communication mechanism that transports/manipulates that variable.
3. `DISTRIBUTED_CAUSAL_COALITION`: the computation exists only through synergistic joint activity of multiple cells/channels.
4. `NO_SHARED_INTERNAL_CAUSAL_ABSTRACTION`: independent organisms implement the same behavior using genuinely different internal causal decompositions.

All four are legitimate outcomes.

## Fundamental V837an science lock

V837an **never requires a cross-organism bijection or invertible state map**.

The model is:

```text
private microstate s_A -> shared causal macrostate z
private microstate s_B -> shared causal macrostate z
```

The primitive candidate is `(z, F, causal interface)`, not the complete 40D private state.

Any prerequisite equivalent to `recipient_state <-> donor_state must be bijective` is invalid in V837an.

## Historical interpretation refinement

V837am's machine diagnosis remains exactly:

`DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE`

The historical files are not rewritten. V837an appends three interpretation refinements:

- `REF-AN-001`: 105 / 126 AM-A configurations were invalid because their state-bearing maps failed the dynamic-state invertibility / conditioning gate.
- `REF-AN-002`: all seven AM-C configurations produced `row_count = 0`; all 38 causal pairs were invalid for every configuration.
- `REF-AN-003`: the historical diagnosis is preserved, but it must not be inflated into a claim that temporal causal operators or all shared internal causal abstractions were directly disproven.

## Architecture lock

No change is permitted to the AF1D substrate or trained source organisms:

- 10 cells
- 4D state per cell
- 40D aggregate recurrent state
- rank-4 cross-block global coupling
- one global scalar controller
- de-shared 6→6 input projections
- existing learned message topology
- trained source parameters
- readout
- V837aj structural-search history
- V837ak classes / reconstructed checkpoints
- V837al pairing history
- V837am results

Expected training work in V837an:

```text
new organism fits             0
optimizer steps               0
adapter gradient steps        0
processed training examples   0
```

## Source population

Load the exact 50 V837ak reconstructed organisms:

- 40 competent organisms: primary science
- 10 incompetent organisms: controls / diagnostics only

Family counts are derived from the committed source population, never hard-coded for decisions.

A strong family claim requires at least five adequately powered competent organisms and at least 60% organism-level support. Both V837aj structural-discovery engines must be represented where the source population permits.

## Frozen data partitions

```text
AN_FIT               10000–10127   128 episodes / family
AN_SELECT            10128–10255   128 episodes / family
AN_ROUTING_FIT       10256–10319    64 episodes / family
AN_ROUTING_SELECT    10320–10383    64 episodes / family
AN_META_CONFIRM      10384–10447    64 episodes / family
AN_FINAL_DEV         10448–10511    64 episodes / family
AN_FINAL_VALIDATION  20000–20127   128 episodes / family
FRESH_AUDIT          90000–90499   untouched
```

Validation is inaccessible until `raw/frozen_family_abstractions.json` is written and hash-verified. After validation starts there is no new carrier, dimension, phase, routing subset, coalition, winner, or second-best retry.

## Ground-truth semantic probes

Primary decision probes are exactly:

```text
conditional_routing   ROUTING_CONTROL_STATE       z = control c

delayed_recall        RECALL_MAINTAIN             z = remembered value

iterative_state       ITERATIVE_RUNNING_STATE     z = running state s_t
                                                s_(t+1)=0.65*s_t+0.35*x_t

partial_observation   PARTIAL_LATENT_Z             candidate abstraction = physical hidden z_t
                                                (not automatically a proven network belief state)

variable_composition  COMPOSITION_RUNNING_STATE   z = running value v_t
                                                v_(t+1)=tanh(g_t*v_t+d_t)
```

Secondary probes are diagnostic only and cannot replace the primary family winner.

## Oracle reality gate

Experiment-local instrumentation reproduces the historical task generators exactly for all five families and every development seed 10000–10511.

Required equivalence:

- observation shape exact
- float32 observations array-equal or max absolute error ≤ `1e-7`
- target absolute error ≤ `1e-12`
- sequence length exact

Failure type: `ORACLE_INSTRUMENTATION_MISMATCH`.

## Paired counterfactuals

Every counterfactual originates from an existing historical seed. No new task seed is introduced.

```text
conditional_routing   CONTROL_FLIP                c' = -c

delayed_recall        MEMORY_VALUE_FLIP           value' = -value

iterative_state       MID_INPUT_FLIP              flip deterministic middle x_j

partial_observation   INITIAL_LATENT_SIGN_FLIP     z0' = -z0
                      RHO_MIRROR                   diagnostic only

variable_composition  INITIAL_VALUE_SIGN_FLIP      v0' = -v0
```

All nuisance variables, random innovations/noise, sequence lengths, and nonintervened coordinates are held fixed exactly as specified.

Pair eligibility requires:

- base episode solved
- counterfactual episode solved
- same length
- finite predictions
- semantic target displacement at least the historical family success tolerance

Minimum eligible pairs per organism:

```text
AN_FIT              48
AN_SELECT           32
AN_META_CONFIRM     24
AN_FINAL_DEV        24
FINAL_VALIDATION    32
```

Below threshold is `ORGANISM_COUNTERFACTUAL_POWER_INSUFFICIENT`, not a scientific failure.

## Instrumented AF1D runtime

The experiment-local runtime records:

```text
STATE40
OUTPUT40
MESSAGE40
GLOBAL40
LOCAL40
INPUT40
GATE1
COUPLING_FACTOR4  (diagnostic only)
```

No-intervention behavior must match the historical runtime within `1e-6` for prediction, state, output, message, global term, and gate across all source organisms.

Intervention hooks:

- post-update state patch
- output patch before later same-step recipients consume it
- incoming message patch before `Wm`
- global recurrent term patch before candidate preactivation
- scalar gate patch before state interpolation

## AN-A — causal macrovariable discovery

Primary causal carriers:

```text
STATE40    k = 1,2,4,8
OUTPUT40   k = 1,2,4,8
MESSAGE40  k = 1,2,4,8
GLOBAL40   k = 1,2,4,8
GATE1      k = 1
```

Diagnostic carrier:

```text
COUPLING_FACTOR4  k = 1,2,4
```

For each family / organism / carrier on AN_FIT:

```text
Delta x = x_counterfactual - x_base
Delta X = U Sigma V^T
Q_k = first k right-singular directions
```

`Q_k` is explicitly many-to-one and need not be invertible.

Held-out source-swap patch:

```text
x_patch = x_base + Q_k Q_k^T (x_cf - x_base)
```

Reverse `cf -> base` direction is required too.

Semantic compiler:

```text
a_i = Q_k^T Delta x_i
Delta x_hat = Q_k C Delta z
ridge lambda = 1e-6
intercept = 0
```

Zero semantic delta must produce zero intervention.

Matched controls:

- 32 deterministic Haar-random subspaces
- shuffled-semantic SVD
- same-norm noise
- orthogonal-residual intervention

Primary per-organism source-swap / compiler gate:

```text
eligible SELECT pairs >= 32
median recovery >= 0.60
direction agreement >= 0.75
counterfactual task success >= 0.70
candidate - median random recovery >= 0.20
paired one-sided p <= 0.01
median OOD ratio <= 2.0
```

Family winner selection is complexity-first:

1. smallest `k`
2. lower carrier deployment cost
3. lower carrier dimension
4. stable config ID

Validation score never selects the winner.

If a distributed STATE40 family winner exists, all 10 individual 4D cell states are tested at k=1,2,4 as a localization control. This diagnostic cannot replace the primary winner.

## Phase diagnostics

Delayed recall uses the selected family carrier/config at WRITE, MID_DELAY, and PRE_QUERY.

Possible qualifiers:

- `PHASE_STABLE_CAUSAL_REPRESENTATION`
- `PHASE_CONDITIONAL_CAUSAL_REPRESENTATION`

Conditional routing additionally evaluates `ROUTING_SELECTED_VALUE` after payload B. This is secondary evidence only.

## AN-B — causal routing

The communication universe is exactly:

- every actual message edge
- 10 global-source-cell contributions
- global scalar gate

Message-edge contributions and global-source contributions must reconstruct the historical natural terms within `1e-6` before routing science is accepted.

Task windows are frozen by family. There is no free window search.

Channels are ranked using AN_ROUTING_FIT only. Selection uses AN_ROUTING_SELECT only.

Message prefix sizes: `1,2,4,8,16,32` capped by edge count.

Global-source prefixes: `1,2,4,8,10`.

Combined prefixes: `1,2,4,8,16,32`.

Each candidate uses 64 same-cardinality random routing controls.

Routing gate:

```text
median recovery >= 0.60
counterfactual task success >= 0.70
direction agreement >= 0.75
random-subset recovery margin >= 0.20
paired p <= 0.01
```

Compact qualifier: smallest passing set ≤8 channels or ≤25% of available channels. Otherwise the routing mechanism is distributed.

Predeclared bus comparison:

```text
MESSAGE_ONLY
GLOBAL_ONLY
MESSAGE + GLOBAL
GATE_ONLY
MESSAGE + GATE
GLOBAL + GATE
MESSAGE + GLOBAL + GATE
```

`RANK4_GLOBAL_COMMUNICATION_BUS_SUPPORTED` requires intervention evidence across at least three families; configured rank alone is never evidence.

## AN-C — distributed synergy

Coalition science is triggered when AN-A lacks a strong compact explanation or AN-B requires broad/absent routing support.

Ten cells imply exactly `2^10 - 1 = 1023` nonempty subsets. Every triggered organism scans all 1023 exactly.

A coalition patches only selected cells' natural paired counterfactual 4D state blocks. All unselected cells remain base-state coordinates. No cross-organism state map is used.

Minimum sufficient coalition gate:

```text
median recovery >= 0.60
counterfactual success >= 0.70
direction agreement >= 0.75
paired p <= 0.01
median OOD ratio <= 2.0
```

Strong synergy additionally requires:

```text
best member recovery < 0.30
SynergyGain >= 0.30
coalition cardinality <= 4
```

If the first sufficient coalition requires at least seven cells, qualify as organism-scale distributed computation. If only all ten cells pass, qualify as whole-organism causal state required.

All seven nonempty MESSAGE / GLOBAL / GATE combinations are also examined for cross-channel synergy.

## META, FINAL_DEV, freeze, validation

Each adequately powered family contributes at most one frozen AN-A candidate, one AN-B candidate, and one AN-C candidate to META_CONFIRM.

META_CONFIRM uses 10384–10447 with no refit. Passing branches are compared by causal-interface complexity:

1. intervention degrees of freedom
2. physical component count
3. intervention MAC footprint
4. lower OOD ratio where needed
5. stable config ID

FINAL_DEV uses 10448–10511, no refit. A failed family winner is not replaced by a second-best candidate.

Only after FINAL_DEV is complete:

- write `raw/frozen_family_abstractions.json`
- record its semantic hash in `diagnostics/final_freeze.json`
- unlock final validation

FINAL_VALIDATION uses untouched seeds 20000–20127 with no basis fitting, reranking, new coalition search, routing search, or alternative winner.

## Machine outcomes

Possible primary diagnoses include:

- `CAUSAL_LATENT_PRIMITIVE_ESTABLISHED`
- `GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN`
- `CAUSAL_MACROVARIABLE_ESTABLISHED` + `SEMANTIC_COMPILER_NOT_ESTABLISHED`
- `CAUSAL_ROUTING_PRIMITIVE_ESTABLISHED`
- `DISTRIBUTED_SYNERGISTIC_PRIMITIVE_ESTABLISHED`
- `PHASE_CONDITIONAL_CAUSAL_PRIMITIVES`
- `COMPUTATION_DISTRIBUTED_AT_ORGANISM_SCALE`
- `NO_SHARED_INTERNAL_CAUSAL_PRIMITIVE_AT_TESTED_GRANULARITY`

`LOW_DIMENSIONAL_GLOBAL_CAUSAL_BUS_SUPPORTED` is a qualifier unless the family causal gates are independently met.

## Claim discipline

- decodable feature → `decoded semantic variable`
- intervention success → `causal carrier`
- cross-organism intervention faithfulness → `causal macrovariable abstraction`
- high-level Delta z alone generating held-out interventions → `semantic compiler`
- held-out validation + cross-organism support + matched controls + OOD gate → `causal latent primitive`

Decodability alone is never a pass gate.

## Archive and next-version lock

V837an is definition/validation, not archival reuse.

Even on success:

```text
primitive_archive_allowed_next = false
primitives_promoted = 0
fresh_audit_consumed = false
v838_started = false
```

Successful causal-latent evidence can authorize V837ao canonicalization, but not PrimitiveArchive promotion inside V837an.

## Permanent failure memory

Every failed scientific configuration must remain machine-searchable in `raw/failure_ledger.json` and `diagnostics/failure_ledger.json`, with append-only human synthesis in `docs/V837_FAILURE_LEDGER.md` and `FAILURE_ANALYSIS.md`.

Engineering failures are typed separately from scientific failures. Underpowered families are not counted as scientific negatives.

## Resource discipline

- no retraining
- trace/cache artifacts are reconstructable and gitignored when large
- no giant tensor dumps are committed
- track forward-call classes, SVDs, ridge solves, permutation tests, CPU/wall/GPU time, cache hits/misses, and unique historical task seeds consumed

## Final scientific target

The strongest desired evidence is not a recurring group of similar cells. It is:

```text
different microcircuits
different topology
different private coordinates
        ->
same causal computation
same semantic state
same counterfactual law
```

The experiment is optimized to resolve uncertainty, not to force that result.
