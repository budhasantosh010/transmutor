# Transmutor Research Addendum — V87 through V92

This phase returned to the "universal cell / primitive computation" branch and removed several hand-supplied mechanisms.

The goal was not to claim a new architecture, but to determine exactly what this toy task family minimally requires.

---

# V87 — Universal update law from raw primitives, no task-specific skill macros

Previous V56 experiments supplied learned skill macros:

- P = s*x
- M = s+x
- K = tanh(s)+g*x

V87 removed those macros.

Raw terminals:

- state s
- current input x
- memory gate g
- one-hot task context p,m,k
- tanh(s)
- constant 1

All multiplicative monomials up to degree 3 were generated automatically.

Sparse search recovered exactly five terms:

- +1 * s*m
- +1 * x*m
- +1 * k*tanh(s)
- +1 * s*x*p
- +1 * x*g*k

Local transition error was effectively zero.

Long-sequence tests:

- parity length 64: 100%
- majority length 65: 100%
- memory length 96: 100%

This removed the task-specific macro vocabulary.

Caveat:
primitive multiplication/tanh, task context, and desired local transition targets were still supplied.

---

# V88 — tanh is unnecessary

Memory was rewritten exactly as:

    s_next = s + g*(x-s)

Therefore:

- g=0 -> preserve state
- g=1 -> overwrite with x

No if/else.
No tanh.

Raw terminals:

    s, x, g, p, m, k, 1

Sparse search recovered:

- +1 * s*m
- +1 * s*k
- +1 * x*m
- +1 * s*x*p
- -1 * s*g*k
- +1 * x*g*k

Local MSE was ~8e-30.

Long tests:

- parity length 128: 100%
- majority length 129: 100%
- memory length 192: 100%

Thus for this finite task family, addition/subtraction plus multiplicative variable interaction is representationally sufficient for the scalar recurrent cell.

---

# V89 — affine scalar update cannot implement parity transition

Desired local parity transition:

    y = s*x

for:

    s,x ∈ {-1,+1}

Suppose affine:

    y = a*s + b*x + c

The four exact equations are inconsistent.

Matrix ranks:

    rank(A) = 3
    rank([A|y]) = 4

Therefore no exact affine solution exists.

A direct contradiction:

from (+,+) and (-,-):

    c = 1

from (+,-) and (-,+):

    c = -1

Impossible.

The multiplicative rule:

    y = s*x

is exact.

Scoped theorem:

> For a scalar one-step affine update y=a*s+b*x+c over binary ±1 state/input, exact parity transition is impossible; a nonlinear state-input interaction is necessary.

This does NOT mean literal multiplication is universally necessary; threshold networks, logic gates, or higher-dimensional state can implement parity differently.

---

# V90 — affine scalar update cannot implement selective overwrite memory

Desired gated memory:

    y=s if g=0
    y=x if g=1

Ask whether:

    y = a*s + b*x + c*g + d

can satisfy all:

    s,x ∈ {-1,+1}
    g ∈ {0,1}

It cannot.

Ranks:

    rank(A) = 4
    rank([A|y]) = 5

Best affine MSE:

    0.5

Adding interactions:

    s*g
    x*g

gives the exact rule:

    y = s - s*g + x*g

or:

    y = s + g*(x-s)

with numerical error ~0.

Scoped theorem:

> Within scalar affine updates in variables s,x,g, exact selective overwrite is impossible; multiplicative gate interactions make it exact.

---

# V91 — remove explicit task ID using demonstrations

Hidden task:

- parity
- majority
- gated memory

The system was not given the task ID.

It received random demonstrations:

    sequence x
    gates g
    final answer y

Candidate algorithms inconsistent with demonstrations were eliminated.

Across 6,000 episodes:

- unique identification: 99.97%
- mean demonstrations: 2.898
- long-sequence accuracy after identification: 99.97%

By task:

Parity:
- identification 100%
- mean demos 2.55
- p95 6
- long accuracy 100%

Majority:
- identification 100%
- mean demos 2.67
- p95 6
- long accuracy 100%

Memory:
- identification 99.90%
- mean demos 3.48
- p95 8
- long accuracy 99.90%

This removes the explicit task-ID wire for this supplied finite hypothesis family.

Caveat:
the three candidate algorithms themselves are known in advance.

---

# V92 — active identification hits the binary information lower bound

There are 3 possible hidden tasks.

Each diagnostic query returns a binary answer ±1.

Therefore any binary decision tree has worst-case query depth at least:

    ceil(log2 3) = 2

The system searched over all length-3 binary diagnostic queries.

It found an adaptive decision tree with worst-case depth exactly:

    2

Simulation:

- parity identified in 2 queries
- majority identified in 2 queries
- memory identified in 1 query

Thus the information-theoretic lower bound is achieved exactly for this finite family.

First diagnostic query found:

    x = [-1,-1,-1]
    g = [0,0,0]

Responses:

- parity -> -1
- majority -> -1
- memory -> +1

So memory is isolated immediately; a second query distinguishes parity from majority.

Scoped exact result:

> For the supplied three-algorithm family and allowed binary length-3 diagnostic queries, adaptive task identification requires at least 2 queries in the worst case and a 2-query strategy exists.

---

# What this branch actually establishes

## Exact / scoped

1. An affine scalar local update cannot exactly realize parity's y=s*x transition.

2. An affine scalar update in s,x,g cannot exactly realize selective overwrite y=s for g=0 and y=x for g=1.

3. Multiplicative interactions make both exact.

4. For parity + majority + gated memory, a scalar recurrent update built from addition/subtraction and multiplication is sufficient.

5. For the finite known three-task family, explicit task ID is unnecessary if demonstrations are available.

6. If active binary diagnostic queries are allowed, the hidden task can be identified in the information-theoretically optimal worst-case 2 queries.

---

# What remains unresolved

- discovering the primitive operation set itself
- discovering the candidate task/algorithm family itself
- open-ended task inference
- whether scalar state is useful beyond toy algorithmic tasks
- whether these primitives scale to real perception/language/world modeling
- whether any of this offers an advantage over existing neural/recurrent/program-synthesis methods

The strongest remaining novelty question is therefore no longer:

    "Can one cell implement these tasks?"

Yes, under a supplied algebra it can.

The harder question is:

    "Can a system autonomously construct or expand the algebra / hypothesis family it needs when the correct primitive is not already available?"
