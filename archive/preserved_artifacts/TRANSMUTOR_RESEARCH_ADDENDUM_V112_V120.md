# Transmutor Research Addendum — V112 through V120

This phase pushed directly on grammar growth, reusable abstraction, routing, local mechanism discovery, and state expansion.

The central question was:

> Can the system build new reusable computation from previous computation, and can it avoid the costs created by an ever-growing grammar?

---

# V112 — Naive repeated-subtree macro invention

Base grammar:

    x, 1, +, -, *

Hidden recurring composite used only to generate tasks:

    H(x) = x*x + x

Training tasks were synthesized using the base grammar.

Naive learner:

1. solve training programs
2. count repeated syntactic subtrees
3. promote the most frequent/large subtree

It invented:

    M = x*x + x + 1

rather than H itself.

Held-out search effects:

- 2H: 0.24x speedup — worse
- H^2+1: 1.89x
- H^3: 4.03x
- H^3+H: 0.77x — worse

Lesson:

> Syntactic recurrence alone is not a reliable abstraction criterion.

Algebraically equivalent programs can expose different subtrees, and a frequent chunk is not necessarily the abstraction that best compresses future computation.

---

# V113 — Corpus compression learns the correct macro

Instead of subtree frequency, candidate macros were evaluated by total description length:

    macro definition cost
    +
    minimum program lengths for the whole solved-task corpus

Candidate macros were base-grammar expressions of size <=5.

Training corpus base description length:

    43

Best learned macro:

    M = x*x + x

which exactly matched the hidden recurring H.

Corpus description length with macro:

    20

Compression:

    53.49%

Held-out results:

3H:
- expression size 9 -> 5
- search 215 -> 73
- 2.95x speedup

H^2+1:
- 13 -> 5
- 1769 -> 59
- 29.98x

H^3:
- 17 -> 5
- 11442 -> 75
- 152.56x

H^3+H:
- base search did not solve within the smaller bounded representation used previously, while macro grammar solved compactly

This established:

> Within a fixed meta-grammar, corpus-level compression can discover a reusable abstraction that is more useful than naive syntactic frequency.

Caveat:
the objective and candidate macro grammar are supplied.

---

# V114 — Grammar pollution

The V113 macro was excellent on related tasks but expensive on unrelated ones because it adds another branch to the enumerator.

Related tasks:

- 3H: macro/base cost 0.340
- H^2+1: 0.033
- H^3: 0.007
- H^3+H: 0.008

Mean related-task macro/base ratio:

    0.010

~100x cheaper.

Unrelated polynomial tasks:

- x^3: 1.533x
- x^4+x: 3.370x
- x^5+x^2: 5.451x
- x^6+x+1: 14.237x

Mean unrelated macro/base ratio:

    13.334

Thus:

> A useful abstraction can badly pollute the active grammar for tasks that do not reuse it.

For the particular measured future-task mixture, keeping the macro globally active became cheaper only when roughly:

    55%

or more future tasks were related.

This motivates selective routing rather than a permanently flat growing grammar.

---

# V115 — Functional primitive learned from recurring residuals

V108/V109 empirical primitives were fixed vectors over one coordinate grid.

V115 changed the representation.

Each task used different random coordinates:

    y_t(x)
      =
    task-specific quadratic background
      +
    amplitude_t * hidden recurring function h(x)
      +
    noise

The hidden h was an unnamed smooth composite.

Learner:

1. fit quadratic background per task
2. normalize residual
3. pool residual samples across varying x coordinates
4. train a generic MLP phi_theta(x)

On new tasks at new coordinates:

4 recurring training tasks:
- base MSE 0.544
- functional primitive MSE 0.00347
- 156.7x improvement

8 tasks:
- 165.2x

16:
- 428.4x

32:
- base 0.528
- primitive 0.000933
- 565.8x

Thus the primitive became an actual input->output function, not a lookup vector.

But extrapolation outside the coordinate range remained poor:

- ~2.95
- ~2.08
- ~1.80
- ~2.00 MSE

Lesson:

> A generic learned functional primitive can generalize across new coordinates and tasks inside its experience distribution while still failing to behave like a generative law outside that distribution.

---

# V116 — Learned grammar routing

The system now had two grammars:

- base
- base + M

Past solved tasks were labelled only by:

    which grammar actually cost fewer enumeration steps?

A Random Forest router saw normalized target examples and finite-difference descriptors.

Held-out:

- routing accuracy: 95.05%
- always base mean cost: 7337
- always macro: 1893
- learned route: 1817.6
- oracle route: 1749.4

Learned routing:

- saved 75.2% vs always base
- saved 4.0% vs always macro
- cost only 1.039x oracle

Thus:

> Grammar growth is much more useful when abstractions are selectively activated rather than globally active.

Caveat:
only two grammars and a fixed-grid task representation were tested.

---

# V117 — Discover a local generative mechanism from trajectories

Rather than memorizing whole functions, the learner observed trajectories only over:

    t in [0,2]

and attempted to infer a local law.

Generic supplied local grammar:

    1, y, y^2, y^3

Hidden system A:

    dy/dt = 1.30 y

Discovered:

    dy/dt ≈ 1.300 y

Rollout to t=4:
- median final relative error: 0.016%

Hidden system B:

    dy/dt = 1.45 y - 0.72 y^2

Discovered:

    dy/dt ≈ 1.449 y - 0.7195 y^2

Rollout:
- median final relative error: 0.003%

This is more rule-like than V115 because the discovered object is a local generative mechanism that can be recursively applied beyond the observed horizon.

Caveat:
derivatives, polynomial local grammar, and sparse regression are supplied.

---

# V118 — Detect insufficient state and expand it

Hidden damped oscillator:

    dx/dt = v
    dv/dt = -omega^2 x - gamma v

Initially learner observed position x only and tried:

    dx/dt = f(x)

This failed.

Scalar derivative MSE:

    2.3948

Conditional velocity variance within narrow x bins divided by total velocity variance:

    1.006

Meaning knowing x removed essentially none of the ambiguity in dx/dt.

Same position could have very different velocities.

Expansion candidate:

    v := dx/dt

With state (x,v), sparse regression over:

    1, x, v, x^2, xv, v^2

discovered:

    dx/dt = 1.000 v

and:

    dv/dt = -4.6165 x - 0.1801 v

Hidden coefficients were approximately:

    -4.6225 x - 0.18 v

Rollout trained conceptually through t=4 and evaluated through t=8:

- mean state MSE: ~3.4e-5
- median final-state relative error: 0.879%

This demonstrates:

> Residual ambiguity can reveal that the current state description is non-Markov-sufficient, and adding a derived state variable can restore a compact local law.

Caveat:
the candidate augmentation v=dx/dt was supplied.

---

# V119 — Failed automatic memory-order selection

Hidden discrete systems had order 1, 2 and 3.

Candidate history orders:

    1..4

Selection used one-step validation error plus complexity penalty.

Result:

- true order 1: selected order 1 100%
- true order 2: selected order 1 100%
- true order 3: selected order 1 100%

This benchmark failed as a test of state-order discovery.

Why:

The higher-order systems were stable/decaying enough that a lower-order approximation predicted their long-term behavior cheaply, and the complexity penalty rewarded the simpler model.

Important lesson:

> "True hidden order" is not necessarily the same as "minimum useful predictive state order" under a specific data distribution and objective.

Therefore V119 should not be counted as positive evidence.

---

# V119b — Harder persistent dynamics and rollout-based state selection

The benchmark was rebuilt with persistent oscillatory dynamics and multi-step rollout validation.

True order 1:
- selected correctly 100%

True order 2:
- selected correctly 91.11%
- mean selected order 2.09

True order 3:
- selected exactly 36.11%
- mean selected order 3.63
- often over-selected order 4

Thus persistent dynamics forced higher state complexity to become visible, but exact complexity selection remained difficult under noise.

Notably, overparameterized order 4 often had lower rollout MSE than the true hidden order.

Lesson:

> State complexity should be judged by predictive/resource utility, not blindly by recovering the generator's nominal hidden order.

This remains unresolved beyond the tested autoregressive family.

---

# V120 — Recursive abstraction / hierarchical grammar growth

Starting point:

    M1 = x*x + x

which was learned in V113.

New task family repeatedly used:

    J = M1*M1 + M1

After routing into the M1-related subgrammar:

    M1, 1, +, -, *

the learner evaluated compact second-level macro candidates using corpus description length.

Stage-2 corpus base description length:

    43

Learned:

    M2 = M1 + M1*M1

which semantically exactly matched J.

Description length:

    20

Compression:

    53.49%

Held-out:

3J:
- size 9 -> 5
- search 215 -> 73
- 2.95x

J^2+1:
- size 13 -> 5
- search 1769 -> 59
- 29.98x

J^3:
- size 17 -> 5
- search 11442 -> 75
- 152.56x

This demonstrates recursive library learning:

```text
raw arithmetic
    |
    v
learn M1
    |
    v
route M1-related tasks
    |
    v
use M1 as an atom
    |
    v
learn M2 from compositions of M1
    |
    v
future programs become much shorter/searchable
```

Caveat:
all macros remain compositions inside the same arithmetic meta-language.

---

# What V112–V120 narrow down

## Strong empirical findings

### 1. Corpus compression is a better abstraction objective than raw syntactic recurrence.

V112 failed; V113 recovered the intended reusable composite.

### 2. Grammar growth has a real cost.

A macro can make unrelated search dramatically worse.

### 3. Routing is therefore a core companion to abstraction.

V116 approached oracle grammar selection closely.

### 4. A reusable primitive can exist at several levels.

```text
fixed empirical vector
    <
functional approximator
    <
local generative mechanism
    <
compact symbolic/programmatic abstraction
```

The higher levels generally support stronger out-of-context reuse.

### 5. State representation itself can be insufficient.

V118 gives a concrete residual-ambiguity diagnostic and successful state augmentation.

### 6. Exact latent complexity recovery is not always the right target.

V119 and V119b show that predictive utility, noise and complexity price can favor lower or higher orders than the nominal generator.

### 7. Abstractions can recursively compose into higher-level abstractions.

V120 created M2 using previously learned M1.

---

# Updated architecture hypothesis

```text
experience / tasks
        |
        v
active routed subgrammar
        |
        v
solve / predict / act
        |
        v
residuals + solved programs
        |
        +------------------------------+
        |                              |
 recurring unexplained pattern    repeated solved subprogram
        |                              |
        v                              v
learn functional primitive       corpus compression
        |                              |
        v                              v
test transfer / mechanism        promote macro
        |                              |
        +---------------+--------------+
                        |
                        v
                 primitive library
                        |
                        v
                learned routing
                        |
                        v
              activate only useful
                 primitives/modules
                        |
                        v
          detect residual ambiguity
                        |
                        v
              expand state/history
                        |
                        v
              learn local mechanism
                        |
                        v
                 reuse recursively
                        |
                        v
                 higher abstractions
```

---

# Still unresolved

1. Inventing semantics outside the supplied meta-language.

2. Inventing arbitrary latent state variables rather than selecting supplied candidates such as derivatives or lag histories.

3. Scaling library learning and routing without combinatorial search exploding.

4. Knowing when to prune, merge or retain old primitives over long nonstationary task streams.

5. Distinguishing useful complexity from overfitting under realistic noise and limited evidence.

6. Showing gains on real language, perception, robotics or code tasks.

7. Demonstrating an advantage over established program-synthesis, library-learning, model-selection, representation-learning and neural-routing methods.

The frontier is therefore no longer simply "invent a primitive."

It is:

> Build a computational system whose representation, state, reusable library, and active grammar can all change over time—while controlling the combinatorial and statistical cost of that self-expansion.
