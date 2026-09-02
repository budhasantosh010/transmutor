# Transmutor Research Addendum — V129 through V135

This phase attacked four assumptions that were still too clean after V128:

1. abstractions matched exactly
2. types were handed to the system
3. library consolidation had an obvious "correct" size
4. latent-state discovery was mostly linear or based on supplied state variables

The new experiments tested noisy approximate parameterization, type induction, nonlinear delay-state reconstruction, library-size utility, and automatic splitting of overloaded abstractions.

---

# V129 — Approximate parameterized abstraction

Each domain had:

    y = a*u^2 + b*u + c + noise

with different unseen:

    a, b, c

A single global closed quadratic was forced across all domains.

Held-out clean MSE:

    3.7331

Instead, use one shared parameterized template:

    F(u;a,b,c)
      =
    a*u^2 + b*u + c

and infer a,b,c from 5 demonstrations in each new domain.

Held-out MSE:

    0.009475

Improvement:

    ~393.99x

Parameter estimation MSE:

    ~0.001143

Conclusion:

> Exact symbolic identity is not necessary for abstraction. A stable shared transformation family with varying parameters can remain highly reusable under noisy approximate recurrence.

Caveat:
the quadratic template family was supplied.

---

# V130 — Exact typed-grammar counting

Atoms:

Number:
- x
- 1

Bool:
- p
- q

Operators:

Number x Number -> Number:
- +
- -
- *

Bool x Bool -> Bool:
- AND
- OR

Number x Number -> Bool:
- >

Untyped enumerator sees:

    4 atoms
    6 binary operators

Typed dynamic-programming recurrence:

    N_L
      =
    sum_i 3*N_i*N_(L-i)

    B_L
      =
    sum_i [
        2*B_i*B_(L-i)
        +
        N_i*N_(L-i)
    ]

Results:

4 leaves:
- untyped / all valid: 64x
- untyped / numeric-target space: 128x

8 leaves:
- 16,384x
- 32,768x

12 leaves:
- 4,194,304x
- 8,388,608x

14 leaves:
- 67,108,864x
- 134,217,728x

Thus type constraints can remove exponentially growing regions of invalid syntax.

This is exact under the specified grammar.

Caveat:
types were supplied in V130.

---

# V131 — Merge near-duplicate primitives

A lifelong library contained:

    300 empirical primitives

generated from 5 hidden drifting families:

- WAVE1
- WAVE2
- EXP
- KINK
- BUMP

Unsupervised KMeans + silhouette selected:

    K = 7

not the hidden family count 5.

Yet:

- cluster purity: 100%
- flat family retrieval accuracy: 100%
- prototype retrieval accuracy: 100%
- flat reconstruction MSE: 0.005305
- prototype reconstruction MSE: 0.005275

Routing comparisons:

    300 -> 7

Reduction:

    ~42.86x

Important failure:

> The system over-split real within-family drift into extra prototypes.

Therefore "recover the true hidden family count" is not automatically the right consolidation objective.

---

# V131b — A naive MDL/BIC-like correction also failed

A validation distortion + centroid complexity objective was tried.

It selected:

    K = 10

the maximum tested count.

Family routing remained perfect, but the complexity score did not recover 5.

This failure matters:

> There is no representation-independent or objective-free "correct library size."

Any compression criterion embeds assumptions about what storage, routing, distortion, and parameter complexity are worth.

---

# V132 — Nonlinear hidden-state reconstruction

Hidden Hénon-like dynamics:

    x_(t+1) = 1 - 1.2*x_t^2 + 0.3*y_t

    y_(t+1) = x_t

Learner observes:

    x only

Since:

    y_t = x_(t-1)

the minimal sufficient delay state is:

    [x_t, x_(t-1)]

but this was not supplied.

Candidate delay states:

    1..6

Each used a degree-2 polynomial predictor and multi-step validation rollout.

Validation MSE:

delay 1:

    0.690387

delay 2:

    5.25e-13

delay 3:

    1.21e-12

delay 4:

    1.77e-12

delay 5:

    1.60e-12

delay 6:

    1.01e-11

Selected:

    delay = 2

Held-out:

15-step MSE:

    ~6.93e-19

85-step chaotic rollout:

chosen delay 2:

    0.0541

delay 1:

    0.8422

delay 6:

    0.0737

Conclusion:

> Observation history can reconstruct missing nonlinear state within an appropriate delay-state family.

Because the system is chaotic, tiny errors eventually amplify over long horizons, so exact long-trajectory alignment is not a realistic invariant metric.

Caveat:
delay embeddings and the polynomial predictor family were supplied.

---

# V133 — Library size is a utility tradeoff

Let:

    E(K)

be held-out reconstruction distortion using K prototypes.

Define:

    J_lambda(K)
      =
    E(K) + lambda*K

where lambda is the cost/value assigned to one additional persistent prototype.

Exact local rule under this objective:

    add prototype K+1

iff

    E(K) - E(K+1) > lambda

Observed optimal K varied with lambda:

lambda = 0:
- K=19

1e-6:
- K=19

3e-6:
- K=11

1e-5:
- K=7

3e-5:
- K=7

1e-4:
- K=7

3e-4:
- K=7

Thus:

> Data gives a fidelity curve, not a unique library size. The final size depends on how the architecture values fidelity versus persistent complexity.

This explains why V131/V131b need not recover the hidden generator count 5.

---

# V134 — Failed overloaded-primitive split

Two hidden regimes:

A:

    y = u^2 + u

B:

    y = u^3 - u

One global primitive was fit across both.

Residual clustering was attempted.

But residual vectors were sign-canonicalized using a preprocessing rule inherited from earlier empirical-primitive experiments.

That preprocessing transformed approximately opposite residual modes into similar orientations and erased the key distinction.

Result:

- chosen K=2
- cluster purity ~51.25%
- global MSE 1.5213
- split MSE 1.4426
- only ~1.05x improvement

V134 is a failure and should not count as evidence for automatic splitting.

---

# V134b — Preserve residual orientation

The sign-canonicalization step was removed.

Residual magnitude was normalized, but direction/sign was preserved.

Now:

- chosen residual modes: 2
- training purity: 100%
- global overloaded MSE: 1.52144
- split-child MSE: 1.785e-5
- improvement: ~85,219x
- routing from 3 demonstrations: 100%

Conclusion:

> Multimodal residuals can be used to split an over-generalized abstraction, but only if the representation preserves the information that distinguishes the modes.

This produced a broader lesson:

> Representation preprocessing itself can determine whether the architecture is capable of discovering structure.

A supposedly harmless invariance can destroy causal/task-relevant information.

---

# V135 — Learn type-like classes from composition outcomes

V130 supplied types manually.

V135 removed that hint.

12 atoms had hidden types:

- 6 Number
- 6 Bool

Operators had hidden compatibility:

Number,Number:
- +
- *
- >

Bool,Bool:
- AND
- OR

Learner saw only 72% of attempted compositions:

    (operator, atom_i, atom_j)
        ->
    success / failure

It built compatibility fingerprints for each atom and clustered them into two latent classes.

Results:

- latent class purity: 100%
- held-out composition validity accuracy: 100%

Baselines:

assume every composition is valid:

    26.73%

majority validity baseline:

    73.27%

True valid fraction of raw compositions:

    25%

Learned compatibility system accepted:

    25%

Thus in this clean setup, the system recovered exactly the latent compatibility partition needed to reproduce the type-valid search space.

Conclusion:

> Type-like structure can emerge from repeated composition success/failure rather than always being hard-coded.

Caveats:

- number of latent classes K=2 was supplied
- type structure was simple
- compatibility was noiseless
- no polymorphism, subtyping, context-dependent types, or higher-order types

---

# Updated architecture implications

The system now appears to need not only primitive invention and routing, but also a representation-governance layer.

```text
EXPERIENCE
    |
    v
ACTIVE REPRESENTATION
    |
    +-----------------------------+
    |                             |
    v                             v
composition outcomes          prediction residuals
    |                             |
    v                             v
infer compatibility          recurrence / multimodality
classes                      / ambiguity
    |                             |
    v                             v
learned type-like           merge / split /
constraints                 parameterize
    |                             |
    +-------------+---------------+
                  |
                  v
             LIBRARY
                  |
          fidelity / cost curve
                  |
                  v
        choose persistent size
                  |
                  v
              ROUTER
                  |
                  v
        ACTIVE SUBGRAMMAR
                  |
                  v
             STATE MODEL
                  |
      incomplete observation?
                  |
                  v
       expand delay/latent state
                  |
                  v
         solve / predict / act
```

---

# Strongest new lessons from V129–V135

## 1. Approximate recurrence can still support reusable abstraction.

Exact tree equality is not necessary.

## 2. Types can be enormously powerful search constraints.

The benefit compounds exponentially with expression depth in the tested raw syntax model.

## 3. Type-like classes can themselves be inferred from composition behavior.

At least in a simple clean compatibility world.

## 4. There is no universally correct library size.

Library size is a resource-allocation decision.

## 5. Over-generalized primitives should sometimes split.

Residual multimodality is one possible signal.

## 6. Learned invariances are dangerous.

Sign invariance helped earlier primitive clustering but destroyed the distinction needed in V134.

## 7. Nonlinear state can sometimes be reconstructed from history.

The Hénon test selected exactly the two-coordinate delay state implied by the hidden dynamics.

---

# Remaining frontier

1. Learn the number and hierarchy of type classes rather than supplying K.

2. Soft/contextual/polymorphic types.

3. Approximate structural anti-unification without supplying the template family.

4. Decide merge/split/prune/parameterize jointly rather than with separate algorithms.

5. Learn representation invariances instead of manually choosing normalization rules.

6. Learn nonlinear latent coordinates more compact than delay embeddings.

7. Integrate all operations into one online objective:

    predict / solve quality
    +
    computation cost
    +
    library cost
    +
    state cost
    +
    communication/description cost

8. Test whether one architecture can autonomously move through:

    raw primitive
      ->
    empirical chunk
      ->
    parameterized function
      ->
    routed typed abstraction
      ->
    split/merge hierarchy
      ->
    generative mechanism

without a human choosing which transformation to run at each stage.

That end-to-end self-reorganization problem is now the cleanest remaining target.
