# V837 Primitive-Invention Frontier Report

## 1. Starting frontier after V836

The complete V450–V836 audit identified the deepest remaining human-supplied scaffold as the **primitive/operator vocabulary**. Earlier Transmutor work already demonstrated neutral-cell differentiation, local structural invention, macro discovery/reuse, recursive callable structures, search portfolios, and library-level admission. What remained unproven was the stronger chain:

`neutral cells -> recurring useful motif -> causal proof -> callable primitive -> later retrieval -> cheaper future discovery -> fresh transfer`.

V836 remains historical `PASS`. Its exact reproduction remains `CANNOT_REPRODUCE_MISSING_SOURCE` because the inventory-referenced historical source ZIP is absent. The user subsequently authorized V837 as a **new independent post-V836 lineage**, not a V836 repair. No V836b/V836c was created and no historical artifact was altered.

## 2. Exact human scaffold targeted for removal

The intended V837 milestone removes named high-level computational concepts from the learner's primitive vocabulary. The model is not supplied operators named `MEMORY`, `ROUTER`, `COUNTER`, `ATTENTION`, `STACK`, `SEARCH`, `CONCAT`, `ADD`, `SUB`, `MUL`, `NAND`, or `XOR`.

The allowed initial structural edits are low-level graph mechanics only:

- `ADD_CELL`
- `REMOVE_CELL`
- `ADD_EDGE`
- `REMOVE_EDGE`
- `PERTURB_EDGE_WEIGHT`
- `PERTURB_CELL_PARAMETERS`
- `ADD_RECURRENT_EDGE`
- `REMOVE_RECURRENT_EDGE`
- `DUPLICATE_SUBGRAPH`

The lineage stopped before it could make a primitive-invention claim because the prerequisite neutral substrate did not meet the competence gate.

## 3. Neutral substrate

Every task uses the same generic recurrent continuous cell. State and message dimensions are both 4. The generic state update is implemented as a `tanh` transformation of previous state, aggregated messages, raw observation input, and bias, followed by a generic outgoing-message projection. There are no LSTM/GRU gates, attention keys/queries, memory-write gates, routing probabilities, or task/domain labels.

A graph begins with two generic cells and minimal sparse connectivity. Development caps are 16 cells and 64 edges. Graph identity is deterministic from canonicalized cells, parameter seeds, edge direction/weight, and recurrence flags.

One caveat exposed by the blocker analysis is that every cell currently receives the raw observation directly and the terminal readout sees the concatenated state of all cells. Those low-level interface choices remain human-supplied and are candidates for the next diagnostic lineage.

## 4. Task generators

Five families share one six-dimensional numeric observation schema and a common stateful interface:

1. delayed recall;
2. conditional routing;
3. iterative state;
4. variable composition;
5. partial observation.

Family identity exists only in evaluation metadata, never in the model-facing observation tensor or retrieval query.

Frozen seed partitions:

- development: `10000..10999`
- validation: `20000..20499`
- ablation: `30000..30499`
- negative controls: `40000..40499`
- fresh audit: `90000..90499`

The partitions are asserted disjoint in code and tests.

## 5. Benchmark validity

Before structural search, an intentionally capable oracle was evaluated on each family. All five oracle success rates were 100%, exceeding the frozen >=98% validity gate.

A simple first-observation family classifier achieved approximately **19.05%** accuracy with five classes, below the frozen <=35% leakage gate and near the 20% chance level. Thus the initial failure cannot be attributed to a trivially family-revealing first observation.

## 6. Frozen gates

The gate file was frozen before the first experimental result. SHA-256:

`a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`

For V837 competence, a family must have >=90% of independent runs solve the development criterion and >=85% solve held-out validation. At least four of five families must pass. Median final graph size must stay below 75% of the 16-cell cap, and evolved matched-size graphs must beat matched random graphs by at least 20 percentage points overall.

No threshold was changed after observing results.

## 7. Search procedure

The initial search is a bounded CPU evolutionary search with population 16, maximum 40 generations, four offspring per generation, fixed structural penalties, and equal candidate-training budgets. Candidate fitness is validation loss plus small cell/edge costs. Matched-random graph controls preserve graph-size/recurrent-fraction characteristics and receive the same parameter-training budget.

V837c changes only structural-search breadth: offspring per generation increases from 4 to 8. The population size, generation cap, penalties, tasks, seed partitions, substrate dimensions, and scientific gate remain unchanged.

## 8. Every failed attempt

### V837 — initial neutral-substrate competence

**Single change:** establish the first post-V836 neutral-substrate competence experiment using readout-only AdamW adaptation while internal cell parameter seeds/topology are searched by low-level mutations.

**Result:** `FAIL` / `SEARCH_FAILURE`.

- full family passes: **0/5**;
- overall evolved minus matched-random validation gap: **+17.13 points**;
- median final graph: **4 cells**, so the size-restraint gate passed.

Per-family mean validation success:

| Family | Evolved | Matched random | Gap |
|---|---:|---:|---:|
| delayed recall | 0.2617 | 0.1283 | +13.33 pp |
| conditional routing | 0.4417 | 0.3317 | +11.00 pp |
| iterative state | 0.9333 | 0.6817 | +25.17 pp |
| variable composition | 0.5133 | 0.3533 | +16.00 pp |
| partial observation | 0.8883 | 0.6867 | +20.17 pp |

The iterative and partial-observation families showed strong promise, but the frozen family-level run gates were not met.

### V837b — full continuous-parameter refinement

**Single change:** after the unchanged V837 structural search selects an evolved graph and matched-random control, train all continuous parameters for 48 AdamW steps. Topology and the scientific gate remain frozen.

**Result:** `FAIL` / operationally `SEARCH_FAILURE`; diagnostically this rules out simple readout underfitting.

- full family passes: **0/5**;
- all five families reach essentially 100% development-run gate success;
- held-out performance remains poor on most families;
- overall evolved-minus-random gap collapses to **+1.47 points**.

This is a strong generalization warning: simply increasing trainable parameter adaptation overfits the small development sample and does not reveal a hidden structural advantage.

### V837c — doubled structural-search breadth

**Single change:** double offspring breadth from 4 to 8 and return to the original readout-only candidate adaptation.

**Result:** `FAIL` / `SEARCH_FAILURE`.

- full family passes: **0/5**;
- overall evolved-minus-random gap improves to **+20.97 points**, passing that particular control threshold;
- median final graph: **3.5 cells**.

Per-family mean validation success:

| Family | Evolved | Matched random | Gap |
|---|---:|---:|---:|
| delayed recall | 0.3883 | 0.1500 | +23.83 pp |
| conditional routing | 0.4600 | 0.3567 | +10.33 pp |
| iterative state | 0.9350 | 0.6267 | +30.83 pp |
| variable composition | 0.5583 | 0.3683 | +19.00 pp |
| partial observation | 0.9100 | 0.7017 | +20.83 pp |

More structural search helps, but not enough to make the common substrate reliable across four families. This is the third scientifically distinct failure at the same prerequisite layer, so the specified stop condition is triggered.

## 9. Blocker diagnostics

### High-capacity task-independent neutral reference

To separate search failure from representation/optimization capacity, a generic task-independent 10-cell / 55-edge graph inside the same neutral substrate was trained strongly with all continuous parameters. Three restarts were run for each family.

Best validation success:

- conditional routing: 0.4063;
- delayed recall: 0.7109;
- iterative state: 0.9922;
- partial observation: 0.7578;
- variable composition: 0.5859.

Only **1/5** families demonstrated capacity at the 0.85 validation criterion. This rules out structural-search breadth as the sole explanation.

### Increased-data reference diagnostic

The same high-capacity generic reference was then given more than twice the fitting data, while topology, substrate and strong parameter training remained fixed.

Best validation success:

- conditional routing: 0.43;
- delayed recall: 0.81;
- iterative state: 1.00;
- partial observation: 0.85;
- variable composition: 0.78.

Only **2/5** families demonstrate capacity. More data improves three families materially but does not rescue four-family competence. The original small training sample is therefore not the sole blocker.

The strongest narrowed classification is **`REPRESENTATION_FAILURE`** under the tested substrate/training regime, with search failure as the observed main-line symptom.

## 10. Differentiation evidence

No formal V837 differentiation claim is made. Differentiation was specified to run only after the neutral-substrate competence prerequisite passes. Although evolved structures beat matched random structures substantially in several families, using unsuccessful organisms to claim meaningful task-specific differentiation would violate the phase ordering.

Status: **NOT RUN — prerequisite failure**.

## 11. Motif recurrence evidence

Motif extraction/canonicalization infrastructure is implemented and determinism-tested, but scientific motif mining was not run because no V837 competence variant passed.

Candidate motifs promoted: **none**.

Status: **NOT RUN — prerequisite failure**.

## 12. Causal ablation evidence

Causal replacement infrastructure supports disabling/removing motifs and size/topology-matched randomized replacement, but no scientific causal motif experiment was run because recurrence was never validly established after a competence pass.

Status: **NOT RUN — prerequisite failure**.

## 13. Primitive compression

The callable-primitive representation and equivalence machinery are implemented. The unit test verifies deterministic expanded/callable numerical equivalence within the frozen tolerance for an extracted subgraph representation.

Scientific primitives promoted: **none**.

Status: **NOT RUN — prerequisite failure**.

## 14. Primitive archive and retrieval

The primitive archive implementation supports add, inactive removal, list, task-label-free embedding retrieval, usage/success/failure histories, serialization/deserialization, and simulated pruning. Retrieval code does not accept or inspect task-family labels.

No scientific archive was populated because no primitive passed recurrence + causal validation + compression gates.

## 15. Primitive reuse and search-cost savings

The required `NO_ARCHIVE`, `RANDOM_ARCHIVE`, and `VALIDATED_ARCHIVE` scientific comparison was not run. Doing so without a causally validated primitive would turn the control itself into the experiment and violate the staged protocol.

Therefore there is **no claim of primitive reuse or future-search savings** in this lineage.

## 16. Random-control comparison

Matched-random graph controls were run for every V837/V837b/V837c competence search. Random-macro/archive controls are implemented as a required future phase but were not scientifically invoked because no primitive archive existed.

## 17. Fresh audit

Fresh-audit seeds `90000..90499` were never consumed during this unsuccessful lineage. `audit/run_fresh_audit.py` explicitly refuses to run while `lineage_status.json` reports Outcome B. The audit result records zero episodes consumed.

Status: **NOT RUN — prerequisite failure**.

This is intentional preservation of test-data purity, not missing work.

## 18. Resource accounting

Actual recorded research work through the stop-condition diagnostics consumed approximately:

- structural/reference candidate evaluations: **50,688**;
- optimizer steps: **1,235,952**;
- environment steps/interactions: **12,061,933**;
- summed worker/model-fit wall time: **9,391.52 seconds** (~2.61 worker-hours; parallel wall-clock is lower);
- V837 candidate evaluations: 18,343;
- V837c candidate evaluations: 32,315;
- V837b added refinement optimizer steps: 14,400;
- blocker diagnostics: 30 high-capacity reference model fits.

The experiments stayed on CPU. No GPU resource was used.

## 19. Scientific caveats

1. The failure applies to the tested neutral `tanh` cell substrate and training/search regime, not to every possible neutral substrate.
2. The generic cells still receive direct raw-observation input and a globally concatenated terminal readout; these are human-supplied interface assumptions.
3. Gradient learning remains human-supplied continuous-parameter adaptation.
4. The task generators, loss, success thresholds, resource penalties, mutation vocabulary, and hard graph caps are still human-defined.
5. The high-capacity diagnostics intentionally bypass structural-search difficulty; they diagnose feasibility but are not V837 pass attempts.
6. V837b demonstrates substantial overfitting under full-parameter refinement, so development success alone is especially unreliable in this setting.

## 20. Strongest justified claim

**Under the tested synthetic environments, low-level neutral continuous-cell structural search produces clear held-out advantages over matched random structures on some task pressures, but the tested substrate/training regime does not reliably generalize across the required task-family set. After three controlled competence variants and two capacity diagnostics, the primitive-invention milestone cannot validly proceed to motif recurrence, causal promotion, reuse, or fresh audit.**

No claim is made about new fundamental computing primitives, AGI, human intelligence, Transformer replacement, universal program induction, or self-aware self-improvement.

## 21. What remains human-supplied

The deepest originally targeted scaffold—high-level named primitive vocabulary—remains unresolved because the program stopped one layer earlier. The newly isolated prerequisite scaffold is the **low-level representation/interface itself**: cell update form, observation injection, terminal readout, state/message dimensionality, parameter optimizer, graph-edit vocabulary, and task loss/gate.

The most immediate scientifically actionable question is which *single* low-level representation change permits reliable generalization across at least four families before primitive mining resumes.

## 22. Next deepest frontier

Do **not** proceed automatically to V838 experiment invention. The next experiment must come from the isolated V837 blocker.

Recommended next hypothesis:

> Restrict or learn generic observation-to-cell access/message mediation (rather than broadcasting the full raw observation to every cell) while keeping the cell nonlinearity, graph mutation vocabulary, tasks, seeds, budgets, and competence gate fixed; test whether this single representation change improves task-independent held-out competence.

A competing diagnostic candidate is replacing the concatenated terminal readout with a generic graph-level pooled/message readout, again as a single-variable experiment. Only one should be changed per variant.

## 23. Outcome

**Outcome B — milestone failed honestly with a narrowed blocker.**

The V837 lineage is scientifically closed under the requested stop rule. Primitive-invention phases beyond competence remain intentionally unexecuted until a future blocker-focused lineage restores the prerequisite.
