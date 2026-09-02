# Transmutor Research Log — V0 through V52

## Scientific status

These are prototype / falsification experiments, mostly on small synthetic tasks plus the sklearn handwritten-digits dataset.

They do **not** prove AGI, ASI, or a successor to Transformers.

The purpose is to turn the original hypothesis into measurable subproblems and keep both successes and failures.

---

# Core hypothesis being tested

Can useful computation increasingly organize itself under finite resources by learning:

- what to compute
- what not to compute
- what to remember
- what to forget
- when to simulate / search
- when to grow
- when to prune
- how to repair damage
- how to protect old knowledge
- how to compress repeated experience
- how its own learning rule should work

rather than humans permanently specifying one fixed monolithic computation?

---

# Major results through V21

Earlier results include:

- V0: topology growth/pruning solved four Boolean tasks.
- V1: lifetime weight learning caused interference.
- V2: recurrent state alone did not preserve delayed memory.
- V2b: retention/write gating gave 100% delayed recall.
- V3: blind structural growth 2 -> 8 cells still failed XNOR.
- V3b: directed structural repair solved XOR/XNOR with only 2 cells.
- V4: energy pressure reduced active-cell proxy 7.92 -> 1.28 at 100%.
- V5: generic selective memory collapsed to "write nothing."
- V5b: addressed memory reached 100% with ~1.2 writes / 20 items.
- V6: random feedback solved XOR in 30/30 seeds.
- V7: search found a useful learning-rule coefficient mixture.
- V8: primitive search found exact compact Boolean programs.
- V9: reusable modules accelerated related tasks.
- V10: repeated symbolic computation compiled into cheaper skills.
- V11: input-dependent routing used more compute on harder cases.
- V12: learned world model enabled planning around a wall.
- V13: selective planning kept 100% success at much lower planning cost.
- V14: uncertainty sampling modestly improved data efficiency.
- V15: identical tiny seeds developed different body sizes by environment.
- V16: near-cloning useful cells failed.
- V16b: diverse offspring succeeded 12/12.
- V17: resource competition pruned 12 -> 3 cells at 100%.
- V18b: protected progressive modules removed forgetting under conflicting tasks.
- V21: recurrent-rule search discovered `state*x` for parity and `state+x` for majority, generalizing to sequence length 64.

---

# V22 — Structural recurrence emerges

Initial population had **zero recurrent edges**.

Delayed-memory task required remembering a bit through 14 timesteps.

Result:

- first >=99% training accuracy: generation 4
- best train accuracy: 100%
- held-out accuracy: 100%
- final best genome: only **1 recurrent edge**

Lesson:
when recurrence is structurally available and actually useful, selection can retain a very small recurrent pathway.

Caveat:
the mutation system was explicitly allowed to add recurrence.

---

# V23 — Content addressing emerges

A generic bilinear query-key compatibility matrix was learned.

Result:

- uniform-read baseline: 63.60%
- learned retrieval: 100%
- attention mass on relevant item: 0.9989
- mean mass per distractor: 0.0001

The learned compatibility matrix became strongly diagonal.

Caveat:
softmax routing and bilinear query-key interaction were supplied.

---

# V24 / V24b — Search initially failed

On the first maze benchmark:

- greedy success: 77.14%
- evolved search: 77.14%
- evolution stayed at depth 0, beam 1

Even large structural jumps did not help.

This was a useful failure: the supplied lookahead implementation did not create a fitness advantage in that environment.

---

# V24c — Search evolves once the environment really rewards it

The maze family was redesigned with deliberate greedy traps.

Result:

- greedy held-out success: 3.75%
- best lookahead setting: depth 7, beam 8
- evolved setting: depth 7, beam 8
- evolved held-out success: 77.50%

Lesson:
the earlier failure was partly a benchmark / incentive failure.
A computational mechanism does not emerge merely because it exists; the environment must make it useful enough to cross its discovery cost.

---

# V26 — Multiple memory timescales

Prediction stream contained slow and fast latent dynamics.

One timescale:

- alpha ~0.341
- held-out MSE 0.10624

Two timescales:

- fast alpha ~0.0436
- slow alpha ~0.9156
- held-out MSE 0.10088
- ~5.04% relative reduction

Lesson:
different environmental timescales can make multiple retention speeds useful.

---

# V27 — Offline replay / consolidation

Sequential conflicting tasks.

Result:

- awake-only final mean accuracy: 61.25%
- offline replay: 78.50%
- mean forgetting: 34.75 pp -> 5.50 pp

Lesson:
offline rehearsal can substantially consolidate knowledge.

---

# V28 — Structural self-repair on toy tasks

Three-cell system damaged to one cell.

Result:

- intact: 100%
- damaged: 60.94%
- one-cell weight-only repair: 80.47%
- structural regrowth: 96.88%

Lesson:
some damage cannot be recovered by changing weights alone; structural capacity may need to regrow.

---

# V29 — Capacity scaling

Random six-bit mapping:

- 1 cell: 70.31%
- 2: 76.56%
- 4: 96.88%
- 8: 100%
- 16: 100%
- 32: 100%

Lesson:
capacity helps until the task is solved; beyond that, extra cells gave zero gain.

---

# V31 — Equal compute accounting exposes structural-search tax

4-bit parity.

Mean compute proxy:

- fixed 4: 74,880, success 40%
- fixed 8: 9,216, success 100%
- adaptive parallel candidate search: 353,152, success 100%

Important lesson:
a small final adaptive architecture can be extremely expensive to discover.

---

# V31b / V31c — Reducing structural-search tax

Adaptive compute:

- V31 parallel candidates: ~353k
- V31b single-growth: ~115k
- V31c early plateau-triggered growth: ~28.5k

V31c:

- 100% success
- mean final size ~4.9 cells

Still more training compute than fixed 8-cell network.

---

# V31d — Lifecycle break-even

Using measured V31c values:

- adaptive extra development cost: 19,328 compute units
- inference body: ~4.9 cells vs 8 fixed
- break-even: ~6,235 future queries

At 100,000 queries, adaptive lifecycle proxy was ~35.9% lower.

Caveat:
hidden-cell count is only a toy inference-cost proxy.

---

# V5c — Relevance discovered without explicit equality signal

Memory gate saw raw one-hot query and raw item key.
No explicit `query == key` signal.

Result:

- overwrite baseline: 53.90%
- learned-address memory: 100%
- effective writes: 1.006 / 20
- relevant write gate: 0.9968
- distractor gate: 0.0005

The compatibility matrix learned strong positive diagonals and negative off-diagonals.

Caveat:
bilinear compatibility scaffold was supplied.

---

# V18c — Protected modules can later be consolidated

V18b progressive system used 15 hidden units.

Offline consolidation into one shared network:

- 4 units: 97.50%
- 6 units: 99.06%
- 8 units: 100%

So the protected 15-unit structure could later compress to 8 units at 100%.

---

# V7b — Learning-rule structure discovery

Search expression language saw:

- feedback error `e`
- hidden state `h`
- constants
- add/subtract/multiply/negate/tanh

It discovered:

`- ( e * (h*h - 1) )`

which simplifies to:

`e * (1 - h^2)`

This is exactly the tanh derivative-shaped hidden learning signal.

Held-out:

- discovered rule: 97.40%
- raw feedback `e`: 95.83%
- hand-coded exact tanh derivative: 97.40%

Important:
the search independently reconstructed a known useful derivative structure from lower-level pieces.

---

# V33 / V33b — Development moves to real images

Dataset: sklearn handwritten digits.

Fixed 64-cell model:

- test ~96.71%
- training compute proxy ~2.26M

Developmental frontier:

- ~12.8 cells: 93.96%, ~0.194M compute
- ~32 cells: 96.31%, ~1.08M compute
- ~49.6 cells: 96.40%, ~2.23M compute

The ~32-cell point kept almost all accuracy at roughly half the training proxy.

---

# V34 — Real selective compute

Cheap classifier + deep expert.

Result:

- cheap only: 95.33%
- always deep: 97.56%
- selective: 96.67%
- deep expert used on only 4.44% of images
- inference proxy reduction: ~82%

Lesson:
"when not to think deeply" can save large compute.

---

# V35 — Real structural self-repair

32-cell digit model damaged to 8 cells.

Result:

- intact: 96.78%
- damaged: 82.61%
- fine-tune 8 survivors: 94.44%
- structural repair: 96.61%

Structural regrowth nearly restored intact accuracy.

---

# V36 / V37 — Real continual learning and memory budget

Digits arrived as five sequential class pairs.

No replay:

- final all-10-class accuracy ~48.8–51.9%

Replay:

- 15 examples/class: ~85–87%

V37 memory-budget curve:

- 0 stored: 51.93%
- 20: 60.96%
- 50: 72.59%
- 100: 80.22%
- 150: 84.89%
- 300: 93.56%

Lesson:
continual learning exposes a direct memory/retention tradeoff.

---

# V36b / V38 — Naive "smart" memory selection failed

KMeans representative examples:

- 5 random/class: 73.61%
- 5 prototypes/class: 72.89%

Low-margin "fragile" memories:

- random 5/class: 71.78%
- fragile 5/class: 66.58%
- easy 5/class: 70.44%

Lesson:
"representative," "hard," and "fragile" do not automatically mean "best for preventing forgetting."

---

# V39 — Protect parameters instead of storing raw examples

EWC-like parameter importance.

Result:

- no protection: 49.06%
- parameter protection: 75.61%
- replay 15/class: 83.61%

So significant retention is possible without raw replay, but replay remained stronger.

---

# V40 / V41 — Real structural death

Train 64 cells, then prune.

V40:

- 64: 97.39%
- 32: 96.89%
- 16: 96.83%
- 12: 96.50%
- 8: 94.39%

Smallest pruned model within 1 percentage point of original: **12 cells**.

V41 directly compared 12 survivors:

- guided survival: 96.94%
- random survival: 96.36%

So the survival signal helped, but only modestly.

---

# V42 — Hybrid replay + parameter protection

Final real-digit continual accuracy:

- none: 51.89%
- replay 5/class: 74.28%
- parameter protection: 76.33%
- replay 5/class + protection: 78.56%
- replay 15/class: 87.11%

The mechanisms complement each other, but large replay remained best.

---

# V43 / V43b / V43c / V44 — Compressed generative long-term memory

Instead of raw old images, store class statistics and generate pseudo-memories.

## V43

Per class:
- 64 means
- 64 standard deviations

Approx storage:
- Gaussian stats: 1,280 floats total
- raw 15/class: 9,600 floats

Result:
- no replay: 50.17%
- Gaussian pseudo-replay: 88.78%
- raw 15/class: 85.11%

## V43b — equal storage across multiple splits

Both Gaussian memory and raw 2/class used ~1,280 floats.

Across four splits:

- none: 51.28%
- raw 2/class: 62.28%
- Gaussian memory: 87.33%
- raw 15/class: 82.00%

## V43c — control replay quantity

When replay quantity is equal:

- raw 2: 61.56%
- Gaussian 2: 61.78%
- raw 15: 83.48%
- Gaussian 15: 80.37%
- Gaussian 30: 87.78%

Interpretation:
the Gaussian memory's advantage is not magic information compression alone.
It uses a compact stored distribution to generate **more rehearsal experiences**, trading offline compute for storage.

## V44 — how much of the distribution must be stored?

All pseudo modes generated 30 examples/class.

- class mean only, 640 floats: 82.81%
- class mean + one scalar spread/class, 650 floats: 88.15%
- diagonal Gaussian, 1,280 floats: 88.67%
- raw 15/class, 9,600 floats: 84.15%

A surprisingly compact class distribution was enough on this dataset.

Caveat:
generated vectors need not look like valid digit images; they only need to support the classifier's boundaries in standardized feature space.

---

# V45 / V46 — Sparse computational organs on real data

8 experts, each 64 -> 8 -> 10.

V45:

- top-1: 91.04%
- top-2: 96.59%
- top-4: 96.81%
- all 8: 96.15%

Assuming real sparse dispatch:
- top-2 expert compute proxy ~67.7% lower than all 8.

V46:
after training top-2, per-image routing chose 1 or 2 experts.

Result:

- fixed top-2: 95.89%
- dynamic 1-or-2: 95.61%
- average experts: 1.151
- 84.89% of images used only one expert
- ~29.6% further ops-proxy reduction vs fixed top-2

---

# V47 — Integrating compressed memory and sparse compute

Continual digit learning:

- dense 32 + compressed replay: 87.11%
- sparse top-2 + compressed replay: 83.63%
- sparse top-2 without replay: 60.89%

Sparse proxy:
~28.4% lower inference cost than dense 32.

Lesson:
useful mechanisms do not combine for free; sparsity introduced an accuracy cost in the integrated continual system.

---

# V48 — Random feedback on real images

One hidden layer, 64 -> 64 -> 10.

- exact backprop: 94.56%
- fixed random feedback: 94.06%

The hidden layer never used exact downstream transpose weights.

---

# V49 — Two hidden layers with direct random feedback

64 -> 64 -> 64 -> 10.

- exact backprop: 95.48%
- direct random feedback: 92.96%

Random feedback still learned, but the gap widened with depth.

---

# V50 — Local self-supervised representation

32 hidden features.

- random frozen representation: 92.80%
- local reconstruction-trained representation: 95.96%
- end-to-end supervised: 97.96%

The locally trained hidden representation never received class labels.

---

# V51 — Hebbian representation with no backprop

Sanger / generalized Hebbian-Oja rule.

- random frozen features: 90.94%
- Hebbian local features: 93.72%
- V50 local autoencoder reference: 95.96%
- end-to-end reference: 97.96%

The hidden representation used no labels and no backprop.

---

# V52 — No backprop anywhere

Hebbian/Sanger features + one centroid per class.

- random features + centroids: 81.51%
- Hebbian features + centroids: 86.22%

No gradient descent was used in either the representation or readout.

Labels were used only to form 10 class centroids.

Readout memory:
10 × 32 = 320 floats.

---

# Strongest current lessons

## 1. Intelligence is not just "more compute"

Several experiments show clear saturation:
- V3: more cells did not solve the missing computation.
- V29: 8 cells solved the benchmark; 16/32 added nothing.
- V33b: larger developmental bodies produced diminishing returns.

## 2. Structural credit assignment matters more than raw growth

Blind growth failed.
Directed repair and useful survival signals were better.

## 3. The cost of discovering structure must be counted

Adaptive architectures can be small at inference yet expensive to discover.
Lifecycle amortization matters.

## 4. Memory appears to benefit from generative compression

On the digit stream, compact class distributions could regenerate rehearsal experience more efficiently than storing a few literal examples.

## 5. "Important memory" is not obvious

Representative, fragile, and hard examples all failed to beat random memory in tested settings.

## 6. Exact backprop symmetry is not fundamental on these tasks

Random feedback nearly matched exact backprop with one hidden layer and remained viable with two.

## 7. Fully local unsupervised learning can build useful representations

Hebbian/Sanger features reached >93% with a trained head and >86% with a centroid readout.

## 8. Sparse routing can preserve most accuracy

Real-image compute can be allocated selectively across experts and even vary per input.

## 9. Search only emerges when the fitness landscape rewards it

V24/V24b failed.
V24c succeeded once greedy traps made lookahead decisively useful.

## 10. Integration remains hard

Sparse compute + generative memory worked together, but lost accuracy versus dense + generative memory.

---

# Biggest unresolved problems after V52

1. **Discover the primitive vocabulary itself**
   - current searches still operate inside human-supplied operation languages.

2. **Discover search / attention / recurrence from a more neutral substrate**
   - current experiments expose structural options but still define what those options mean.

3. **Make structural growth/death/reproduction decisions locally**
   - many current experiments use an external evaluator.

4. **Learn memory-selection policy**
   - naive hand rules failed.

5. **Generate useful replay from learned compact world models**
   - Gaussian replay is crude and dataset-specific.

6. **Scale local/random-feedback learning deeper**
   - the gap widens with depth.

7. **Integrate memory, sparse compute, repair, continual learning, and local learning without accuracy collapse**

8. **Run stronger real-world benchmarks with equal hardware-level accounting**
   - current compute proxies ignore vectorization, memory traffic, sparse-dispatch overhead, and device utilization.

9. **Test orders-of-magnitude larger populations**
   - current systems are tiny.

10. **Test genuinely open-ended environments**
    - where new skills, states, and objectives arrive without a fixed task list.

---

# Current architectural picture

```text
                         ADAPTIVE SYSTEM
                              |
        +---------------------+----------------------+
        |                     |                      |
      MEMORY               COMPUTE                STRUCTURE
        |                     |                      |
  compress history       route selectively        grow
  generate replay        vary depth               prune
  retain slowly          simulate/search          repair
  update quickly         compile skills           reproduce
        |                     |                      |
        +---------------------+----------------------+
                              |
                         LEARNING RULES
                              |
                    global / random / local
                              |
                              v
                           WORLD
                              |
                          experience
                              |
                              +----> repeat
```

All of this remains constrained by finite:

- energy
- memory
- data
- time
- compute
- communication bandwidth

A working hypothesis emerging from the experiments is:

> Intelligence may be less about maximizing raw computation and more about continuously deciding what computation, memory, structure, and learning should exist under finite resources.

That hypothesis is now much more precise than the original idea, but it is still a research hypothesis, not a demonstrated post-Transformer paradigm.
