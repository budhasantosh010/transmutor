# V837 Primitive-Invention Blocker Analysis

## Stop condition

The V837 lineage reached the required stop condition after **three scientifically distinct variants failed at the same prerequisite layer**. The downstream motif -> causal validation -> primitive compression -> retrieval/reuse -> fresh-audit chain was therefore not executed.

The historical archive is not part of this active lineage and remains immutable. V836 remains historical `PASS`; its exact reproduction remains `CANNOT_REPRODUCE_MISSING_SOURCE`. V837 was explicitly authorized later as a new independent lineage, not as a V836 repair.

## What failed

The prerequisite was neutral-substrate competence: the same generic continuous-cell substrate, with no task-family label or named high-level computational operator, had to solve at least four of five task families under the frozen V837 gate.

All three variants achieved **0/5 full family passes**.

| Variant | Single scientific change | Families passing | Overall evolved - matched-random validation gap | Verdict |
|---|---|---:|---:|---|
| V837 | Initial bounded low-level structural search; readout-only AdamW candidate adaptation | 0/5 | +17.13 points | FAIL |
| V837b | Full continuous-parameter AdamW refinement after the unchanged V837 structure search | 0/5 | +1.47 points | FAIL |
| V837c | Double structural-search offspring breadth from 4 to 8; otherwise return to V837 adaptation | 0/5 | +20.97 points | FAIL |

V837c crosses the frozen **overall** +20-point evolved-vs-random control threshold, which is evidence that structural search can matter, but it still does not meet the core family-competence requirement. It therefore remains a failure.

## Where the failures occurred

The strongest family-level pattern is stable across V837 and V837c:

- `iterative_state`: strong validation performance and a large evolved-vs-random advantage;
- `partial_observation`: strong validation performance and a meaningful evolved-vs-random advantage;
- `delayed_recall`: improved with broader search but still below the run-level competence gate;
- `variable_composition`: improved with broader search but remained below the gate;
- `conditional_routing`: remained weak.

V837b showed a different pathology: full-parameter refinement made development success nearly perfect across families but caused poor held-out generalization and nearly eliminated the evolved-vs-random structural advantage. This rules out the simple explanation that V837 failed only because the readout was under-trained.

## How the failure was reproduced

Each main variant used the frozen seed partitions and the same frozen gate hash:

`a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867`

The main experiments each retained 30 independent runs per family, paired development/validation seeds, matched random graph controls, complete run-level metrics, mutation counts, and resource accounting. V837c used a larger structural-search breadth but did not alter the scientific pass gate.

## Diagnostic 1: high-capacity substrate reference

After the three failures, a diagnostic intentionally removed structural-search difficulty without adding task semantics. It used a generic task-independent **10-cell / 55-edge** neutral graph within the same substrate, trained all continuous parameters strongly, and tested three independent restarts per family.

Capacity at the frozen 0.85 validation criterion was demonstrated on only **1/5** families:

- conditional routing: best validation 0.40625;
- delayed recall: 0.71094;
- iterative state: 0.99219 — PASS;
- partial observation: 0.75781;
- variable composition: 0.58594.

This rules out structural-search breadth as the sole blocker: even when topology search is bypassed by a large task-agnostic graph, the tested substrate/training regime does not reliably generalize across the family set.

## Diagnostic 2: >2x fitting-data increase

A second diagnostic kept the same high-capacity generic graph/training regime but increased fitting data to 300 development episodes per family. It still failed to establish four-family competence. Capacity was demonstrated on **2/5** families:

- conditional routing: best validation 0.43;
- delayed recall: 0.81;
- iterative state: 1.00 — PASS;
- partial observation: 0.85 — PASS;
- variable composition: 0.78.

The extra data materially improved delayed recall, partial observation, and variable composition, but did not close the required family set. Therefore the original small candidate-training set is not the sole cause.

## Failure classification

**Primary narrowed blocker: `REPRESENTATION_FAILURE`.**

This label is used conservatively: under the tested generic `tanh` recurrent-cell substrate plus the tested optimization regime, task-independent high-capacity reference graphs still do not reliably generalize on the required four of five families. The evidence does **not** prove that all neutral continuous-cell substrates are insufficient.

Secondary evidence:

- `SEARCH_FAILURE` accurately describes V837/V837c at the operational level: bounded search did not find enough competent graphs;
- V837b rules out simple readout underfitting and exposes severe overfit/generalization collapse when all parameters are refined on the original small development set;
- the high-capacity diagnostics show that simply searching more topology is insufficient;
- increasing development data more than 2x improves some families but does not rescue the milestone.

## Alternatives ruled out or narrowed

- **Benchmark invalidity:** ruled out at the tested level; explicit task oracles solve all five families at 100% on the validity sample, above the frozen >=98% gate.
- **Trivial task-family leakage:** ruled out at the tested level; a first-observation classifier scores ~19.05%, below the frozen <=35% leakage ceiling and near the 20% five-class chance rate.
- **Only insufficient structural-search breadth:** ruled out as sole cause by V837c and the high-capacity reference diagnostic.
- **Only insufficient readout adaptation:** ruled out by V837b.
- **Only insufficient development data:** ruled out as sole cause by the >2x-data diagnostic.
- **Graph-size cap alone:** not supported; main-line median final graphs remain far below the 16-cell hard cap, and the diagnostic explicitly uses a larger 10-cell/55-edge graph.

## What remains unknown

The current evidence cannot cleanly separate these lower-level possibilities without starting a new research lineage:

1. the specific `tanh` cell update may be a poor inductive bias for routing/composition/long-delay generalization;
2. the model exposes every cell directly to the raw observation, which may encourage shortcut fitting rather than useful message-mediated specialization;
3. the optimizer/training curriculum may need explicit regularization or longer-horizon distributional coverage;
4. one fixed state/message dimension of four may be too restrictive for some families even though graph size is not saturated;
5. the fixed model readout over concatenated terminal states may itself be a representation bottleneck.

Changing several of these simultaneously would violate the one-variable scientific discipline.

## Why motif/primitive experiments did not run

V837B-style motif mining is meaningful only after successful organisms exist reliably across enough task pressures. Mining motifs from a lineage that fails the prerequisite would select structures from an inadequate search/representation regime and could produce misleading recurrence or ablation claims.

Accordingly, no primitive was promoted, no archive was used for scientific retrieval, no random-macro reuse comparison was run, no primitive-composition claim was attempted, and fresh-audit seeds `90000..90499` remained untouched by experiment development.

## One-line current hypothesis/fix

**Test a single lower-level substrate change that improves task-independent held-out competence—starting with observation access/message mediation or the terminal readout—before reopening motif invention; do not increase search again first.**
