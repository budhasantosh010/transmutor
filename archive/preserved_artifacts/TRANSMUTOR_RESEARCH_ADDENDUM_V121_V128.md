# Transmutor Research Addendum — V121 through V128

This phase focused on whether a self-growing computational language can remain usable over time.

The central problems were:

- distribution shift
- pruning obsolete primitives
- combinatorial grammar pollution
- scaling of hierarchical abstraction
- latent-state dimension
- parameterized abstraction
- unsafe over-generalization
- discovering which context variables should become primitive arguments

---

# V121 — First lifelong-library distribution-shift test

Two already-discovered macro candidates:

- H(x) = x^2 + x
- K(x) = x^3 - x

Four grammars:

- BASE
- BASE+H
- BASE+K
- BASE+H+K

Task stream phases:

- H-heavy
- K-heavy
- mostly unrelated
- mixed

Results:

BASE:
- total search-cost proxy: ~2.742M

KEEP-ALL:
- ~157.6k
- 94.25% saving vs base

ADAPTIVE:
- ~86.2k
- 96.86% saving

ORACLE:
- ~27.8k
- 98.98% saving

However, this first pruning test was not sufficiently adversarial.

Many supposedly unrelated tasks accidentally became cheaper under the macro grammars.

Therefore V121 is useful as a lifelong routing result but should NOT be counted as a clean pruning stress test.

---

# V121b — Adversarial macro-pollution shift

The unrelated task bank was rebuilt empirically.

Tasks were selected only if:

- BASE could solve them
- H grammar was substantially slower
- K grammar was substantially slower
- H+K was at least 2x slower than BASE

Top observed H+K / BASE pollution ratios were roughly:

- 27.9x
- 27.8x
- 27.6x
- 27.5x
- ...

Stream phases:

- H-heavy
- K-heavy
- pollution-heavy
- H-return

Results:

BASE:
- total 2.238M

KEEP-ALL:
- 1.561M
- 30.22% saving vs base

ADAPTIVE PRUNE/REACTIVATE:
- 630.5k
- 71.82% saving

ORACLE:
- 73.9k
- 96.70% saving

During the pollution-heavy phase:

BASE mean:
- 621.5

KEEP-ALL:
- 5549.3

ADAPTIVE:
- 1079.4

ORACLE:
- 214.7

Adaptive activation fractions during pollution:

- H active only ~8%
- K active only ~12%

When H-heavy tasks returned:

- H reactivated ~98.1% of the phase

Conclusion:

> Lifelong primitive libraries need deactivation/pruning/reactivation. Permanent global activation can become catastrophically expensive under distribution shift.

The adaptive heuristic still lagged the oracle and temporarily paid transition cost.

---

# V122 — Exact grammar-pollution law

For an ordered full binary expression tree with:

- L leaves
- A available atom/primitives
- O binary operators

the exact raw syntactic candidate count is:

    N(L,A,O)
      =
    Catalan(L-1) * A^L * O^(L-1)

If m new macros are globally active:

    N(L,A+m,O) / N(L,A,O)
      =
    ((A+m)/A)^L

For the toy base grammar:

    A=2
    O=3

Examples:

4-leaf expression:

- +1 macro: ~5.06x
- +4: 81x
- +16: 6.56e3x
- +64: 1.19e6x

8 leaves:

- +1: 25.6x
- +4: 6.56e3x
- +16: 4.30e7x
- +64: 1.41e12x

12 leaves:

- +1: 130x
- +4: 5.31e5x
- +16: 2.82e11x
- +64: 1.67e18x

This is an exact syntactic counting result under the stated grammar model.

It explains why a permanently flat self-growing grammar cannot scale without routing, typing, priors, pruning, hierarchy, or other strong search constraints.

---

# V123 — Recursive hierarchy changes scaling

Recursive abstraction family:

    M0 = x

    M_{d+1}
      =
    M_d*M_d + M_d

Raw expanded macro size:

    s_d = 2*3^d - 1

Future target:

    T_d = M_d^3

Raw target size:

    R_d = 6*3^d - 1

If each level becomes a routed macro:

Each definition costs:

    5

Total hierarchy definitions + final target:

    H_d = 5d + 5

Once the library is amortized:

    final target size = 5

Examples:

depth 1:
- raw target 17
- hierarchy+target 10

depth 3:
- raw 161
- hierarchy 20

depth 5:
- raw 1457
- hierarchy 30

depth 8:
- raw 39,365
- hierarchy 45

depth 10:
- raw 354,293
- hierarchy 55

At depth 10:

    raw / hierarchy description ratio
      ≈
    6441.7x

The raw syntax count grows vastly faster.

This is an exact synthetic scaling result, conditional on discovering/routing the correct abstraction at each depth.

---

# V124 — Infer latent state dimension from one observable

Hidden finite-dimensional linear dynamical systems had latent dimension:

    d = 1..5

The learner saw only a scalar output y_t.

For generic observable linear systems, the trajectory Hankel matrix has rank d.

## Noiseless

Across 80 systems per dimension:

- d=1: 100% rank recovery
- d=2: 100%
- d=3: 100%
- d=4: 100%
- d=5: 100%

## Noisy

Observation noise std:

    0.003

A singular-value ratio threshold was selected on validation systems.

Validation exact dimension:

    88.0%

Held-out:

d=1:
- exact dimension 88.46%

d=2:
- 96.92%

d=3:
- 86.15%

d=4:
- 93.85%

d=5:
- 78.46%

Important complication:

An always-order-8 predictor frequently had LOWER rollout MSE than a predictor using the true latent dimension.

Therefore:

> Recovering the generator's true hidden dimension is not equivalent to choosing the best predictive state dimension under noise.

This reinforces the V119/V119b distinction.

Caveat:
systems are linear and generically observable; the inferred state is a delay embedding.

---

# V125 — Parameterized abstraction across variables

Previous macros were closed:

    Mx = x*x+x

Training solved programs:

    x*x+x
    y*y+y
    z*z+z

Normalize the varying terminal to a placeholder:

    ($u*$u)+$u

All three programs became identical.

The learner therefore formed:

    F(u) = u*u + u

Closed definitions:

    3 * size 5
      =
    15

One parameterized definition:

    size 5

Held-out variable:

    w

which had never appeared in the training abstractions.

Representation sizes:

F(w):
- base 5
- parameterized 2

F(w)^2:
- 11 -> 5

F(w)^3:
- 17 -> 8

F(w)^3 + F(w):
- 23 -> 11

For F(w)^3, the raw-vs-routed syntactic candidate ratio in the counting model was about:

    9,884,160x

Conclusion:

> Parameterized abstractions can transfer structure across entities/domains in a way closed macros cannot.

Caveat:
the anti-unification mechanism only handled exact corresponding variable substitution.

---

# V126 — One parameterized operator can replace an entire closed macro ladder

Learned:

    F(u)=u*u+u

Recursive target:

    R_d(x)
      =
    F(F(...F(x)...))

Raw expansion:

    2*3^d - 1

Closed hierarchy:

    M1=F(x)
    M2=F(M1)
    ...

Library + final atom:

    5d + 1

One parameterized operator:

Define F once:

    cost 5

Nested target:

    d+1

Total:

    d+6

Examples:

d=3:
- raw 53
- closed 16
- parameterized 9

d=5:
- raw 485
- closed 26
- parameterized 11

d=10:
- raw 118,097
- closed 51
- parameterized 16

d=15:
- raw 28,697,813
- closed 76
- parameterized 21

At d=15:

    raw / parameterized
      ≈
    1,366,562x

Thus parameterization can be much more powerful than learning a separate closed macro at every depth when the same transformation truly recurs.

---

# V127 — Parameterization can be unsafe if context is missing

Hidden family:

    y = u^2 + c*u

where:

    c ∈ {-1,+1}

## A. Force one context-free abstraction

Fit:

    y = u^2 + a*u

Global learned:

    a ≈ -0.0325

Held-out MSE:

    4.697

It averaged incompatible regimes.

## B. Keep two learned variants

Per-domain inferred coefficients clustered into:

    -1
    +1

One demonstration identified the correct variant:

    100%

Held-out MSE:

    0

## C. Expose context c

Fit one parameterized rule:

    y = u^2 + b*c*u

Learned:

    b = 1

Held-out MSE:

    0

Conclusion:

> An abstraction can only safely merge variants if the information controlling their variation is represented as an argument/context. Otherwise the system should split the abstraction or accept error.

---

# V128 — Discover which context variable should become the argument

The system was no longer told which context controlled the abstraction.

Each domain contained:

    12 candidate binary context bits

Hidden behavior:

    y = u^2 + a(context)*u

Two regimes.

## SINGLE

    a = c_j

for one hidden context bit j.

## INTERACTION

    a = c_j * c_k

for one hidden pair.

The system:

1. estimated per-domain coefficient a from behavior
2. searched degree-1 context features
3. searched pair interactions
4. used validation evidence plus a small complexity price
5. selected the controlling context representation

240 trials total.

Results:

SINGLE:
- correct context argument: 100%
- correct context degree: 100%
- held-out behavior MSE: ~4.87e-6

INTERACTION:
- correct pair argument: 100%
- correctly recognized degree-2 context need: 100%
- held-out MSE: ~6.00e-6

This connects several earlier mechanisms:

```text
abstraction varies across domains
        |
        v
infer latent variation parameter
        |
        v
search available context variables
        |
        v
simple context insufficient?
        |
        v
expand context interactions
        |
        v
promote discovered context as primitive argument
```

Caveat:
the context space was restricted to supplied binary bits and pairwise products, with a single true controlling feature/interaction.

---

# What V121–V128 establish

## Exact / mathematical

### 1. Flat syntax-pollution law

    N(L,A,O)=Catalan(L-1) A^L O^(L-1)

Global addition of m macros multiplies same-size syntax by:

    ((A+m)/A)^L

### 2. Synthetic hierarchy scaling

For the tested recursive family:

raw representation grows exponentially:

    O(3^d)

closed hierarchical library grows linearly:

    O(d)

and an already-learned parameterized transformation requires only:

    O(d)

target description with a much smaller constant.

---

# Strong empirical results

### 1. Pruning/reactivation is necessary under hostile distribution shift.

### 2. Hankel structure can reveal latent linear state dimension from scalar observations, exactly without noise and imperfectly with noise.

### 3. Parameterized abstractions transfer to unseen variables.

### 4. Context-free over-generalization can fail badly.

### 5. The controlling context argument can itself be discovered from many candidate variables and interaction depth.

---

# Major conceptual update

The architecture now needs at least four distinct levels of selective representation:

```text
GLOBAL LIBRARY
    |
    v
ROUTER / CONTEXT
    |
    v
ACTIVE SUBGRAMMAR
    |
    v
PARAMETERIZED PRIMITIVES
    |
    v
STATE / MEMORY REPRESENTATION
```

A flat list of learned skills is not enough.

The system needs to answer:

1. Which abstractions should exist?
2. Which should currently be active?
3. What arguments/context should each abstraction receive?
4. What state variables are needed for the current dynamics?
5. Which abstractions should be retired or reactivated?
6. Which recurring closed abstractions should be generalized into parameterized operators?

---

# Still unresolved

1. Automatic discovery of arbitrary argument structure rather than binary bits/pairs.

2. Typed abstractions:
   - scalar
   - vector
   - sequence
   - graph
   - object
   - action
   - program

3. Safe abstraction across approximate rather than exact structural matches.

4. Continual routing with thousands or millions of learned primitives.

5. Library consolidation:
   - merge duplicates
   - split overloaded primitives
   - rewrite old programs using newer abstractions
   - garbage-collect obsolete abstractions

6. Nonlinear latent-state discovery beyond delay embeddings and supplied derivative/state candidates.

7. End-to-end learning of:
   representation + routing + abstraction + state + objective
   without separately hand-designing each layer.

8. Demonstrating advantages on real tasks.

The sharpened research target is now:

> Build a continually reorganizing computational language where primitives can be invented, parameterized, routed, split, merged, retired, reactivated, and composed—while the state representation itself can also expand when current variables fail to make the world predictable.
