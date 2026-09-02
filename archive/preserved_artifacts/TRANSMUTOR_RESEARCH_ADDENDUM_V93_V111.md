# Transmutor Research Addendum — V93 through V111

This phase directly attacked the remaining question from V92:

> Can a system detect that its current primitive language is insufficient, expand it only when necessary, and eventually promote recurring unexplained structure into reusable computation?

The answer became much more precise.

---

# V93 — Residual-triggered primitive expansion

Start with affine features only.

Tasks:

- AFFINE: y=x1+x2
- PARITY: y=x1*x2
- GATED MEMORY: y=s if g=0 else x
- THREE_WAY: y=x1*x2*x3

Progressive interaction degree:

- affine solved at degree 1
- parity impossible at degree 1, exact at degree 2
- gated memory impossible at degree 1, exact at degree 2
- three-way interaction impossible through degree 2, exact at degree 3

The rank test gave an exact representational certificate:

    rank([Phi | y]) > rank(Phi)

means y is outside the current linear feature span.

This is a clean mechanism for distinguishing search failure from representational impossibility in complete noiseless finite data.

---

# V94 — Minimal interaction depth over 800 unseen functions

5 binary variables.
Random sparse multilinear functions with true maximum degree 1..5.

The system was not told the degree.

It started at degree 1 and expanded only when the current span was insufficient.

Results:

- 800 functions tested
- exact minimum-degree recovery: 100%
- degree 1: 160/160
- degree 2: 160/160
- degree 3: 160/160
- degree 4: 160/160
- degree 5: 160/160

This is exact because the complete noiseless truth table is available and the multilinear basis is well behaved.

---

# V95 — Partial/noisy data breaks exact sufficiency detection

6 binary variables, but only:

- 32 train points
- 16 validation points
- 16 test points

Observation noise std = 0.35.

Compare:

- always expand to degree 5
- select degree by validation evidence
- oracle true degree

Results:

- exact true-degree selection: 86.8%
- over-expansion: 7.6%
- under-expansion: 5.6%

Clean held-out MSE:

- validation-selected: 4.0536
- always degree 5: 10.1036
- oracle true degree: 3.8904

Lesson:

> In noisy/partial data, nonzero residual is not an exact certificate of missing primitives. It may be noise.

Evidence-based expansion substantially beat blind expansion, but certainty disappeared.

---

# V96 — Passive finite data cannot certify global primitive sufficiency

Construct two worlds over 5 binary variables:

    f_simple(x) = x1

and

    f_complex(x) = x1 + 4*delta_u(x)

where delta_u is nonzero only at one unobserved vertex.

Observe 31 of 32 domain points.

Results:

- maximum difference over all 31 observed points: exactly 0
- difference at the one hidden point: 4
- simple world minimum multilinear degree: 1
- complex world minimum degree: 5
- degree-1 observed MSE is numerically zero in both worlds

Exact indistinguishability result:

> A passive learner seeing only the shared observations cannot know which world it is in.

Therefore finite passive data alone cannot guarantee global primitive sufficiency without assumptions.

---

# V97 — No-free-lunch for global certification

General finite-domain theorem:

If any input remains unqueried, an unrestricted alternative target can agree with the current model on every queried point and differ only at an unseen point.

Therefore:

> Without structural assumptions restricting the target class, worst-case global certification requires querying every remaining point.

Concrete N=64 demonstration:

- 0 queries -> adversarial alternative possible
- 16 -> possible
- 32 -> possible
- 48 -> possible
- 63 -> still possible
- 64 -> eliminated

Thus active experimentation alone does not solve the problem unless combined with inductive structure.

---

# V98 — Structural priors reduce query cost to hypothesis dimension

8 binary variables -> 256 domain points.

Assume target is multilinear degree <=D.

Hypothesis dimensions:

- D<=1: M=9
- D<=2: M=37
- D<=3: M=93
- D<=4: M=163

Rank-gain active queries exactly identified coefficients using:

- 9
- 37
- 93
- 163

queries respectively.

Exact coefficient errors were numerical noise only.

This converts exhaustive domain coverage into parameter identification.

Exact linear-algebra principle:

> An M-dimensional noiseless linear function class can be uniquely identified from M linearly independent scalar evaluations, while fewer than M cannot uniquely determine arbitrary coefficients in general.

---

# V99 — A low-degree fit is not a sufficiency certificate if higher-degree alternatives remain allowed

Assume degree <=4 on 8 variables.

Full dimension:

    M = 163

After 9 rank-independent queries—enough to identify a degree-1 model—the degree<=4 design matrix still has:

    154-dimensional nullspace

An explicit nullspace perturbation had:

- effect on all 9 observed points: ~1e-15
- maximum effect somewhere in the full domain: ~1.35

Therefore:

> Fitting a degree-1 model perfectly does not certify degree-1 sufficiency when a degree-4 alternative class is still admitted.

To eliminate every arbitrary degree<=4 alternative in the worst case requires 163 independent evaluations.

---

# V100 — Sparsity collapses the evidence requirement

10 binary variables.
Allowed space:

    all monomials degree <=3

Dense hypothesis dimension:

    M = 176

But true functions used only:

    K = 5 active terms

OMP was given K=5.

Exact full-function recovery:

- 12 queries: 0%
- 16: 6.43%
- 20: 21.43%
- 24: 47.86%
- 30: 75.71%
- 40: 94.29%
- 60: 99.29%
- 90: 100%

Thus a strong sparsity/compositionality assumption reduced the needed evidence far below the dense dimension.

---

# V101 — Sparsity can be estimated rather than supplied exactly

True K varied randomly from 3 to 7.

Learner was not told K.

For each task it fit candidate OMP models K=1..12 and selected complexity by validation evidence.

Exact full-function recovery:

- 30 queries: 50.00%
- 40: 84.17%
- 50: 95.83%
- 60: 99.17%
- 80: 100%
- 100: 100%

Exact K selection:

- 30: 43.33%
- 40: 78.33%
- 50: 92.50%
- 60: 96.67%
- 80: 99.17%
- 100: 100%

So exact sparsity need not be supplied, though the sparse monomial hypothesis itself remains a prior.

---

# V102 — Residual-driven parameterized primitive construction

Current language:

    polynomial degree <=5

Hidden tasks:

- polynomial
- unknown sinusoid
- polynomial + unknown sinusoid

If polynomial residual remained, the system searched residual Fourier structure and constructed:

    sin(kx), cos(kx)

at the discovered k.

Results:

POLY:
- expansion rate: 0%
- correct decision: 100%

SINE:
- expansion: 100%
- correct frequency: 86.67%
- mean MSE 0.8816 -> 0.00440

MIX:
- expansion: 100%
- correct frequency: 77.50%
- mean MSE 0.7544 -> 0.00993

Lesson:

> Detecting that a primitive is missing can be easier than identifying the right missing primitive.

The sine family was still supplied.

---

# V103 — Fit quality alone cannot choose between nested primitive languages

Candidate operator families:

- polynomial
- periodic
- exponential
- kink

All richer families included the polynomial base and could set their extra coefficient to zero.

Results:

- periodic family: 100%
- exponential: 100%
- kink: 100%
- pure polynomial: only 3% family identification

Yet polynomial prediction error remained essentially zero.

Reason:

> Equal fit does not imply equal representation quality.

A richer nested language can mimic a simpler one exactly.

---

# V103b — Minimum-description / complexity price resolves nested-model ambiguity

Score:

    validation MSE + tiny complexity penalty

The penalty only breaks equal/nearly-equal fits toward fewer extra primitives.

Results:

- POLY: 100% family + parameter
- PERIODIC: 100%
- EXP: 100%
- KINK: 100%

All test errors were numerical zero.

Lesson:

> Primitive invention needs both explanatory fit and a simplicity / description-length preference.

---

# V104 — Generic growable primitive vs specialized primitive

Instead of named operator families, give one generic expansion:

    hinge_c(x) = max(0, x-c)

The system greedily adds hinges and uses validation to select model size.

Results:

POLY:
- mean hinges: 0
- MSE ~0

PERIODIC:
- mean hinges: 14.93
- mean test MSE: 4.34e-2

EXP:
- mean hinges: 14.10
- mean MSE: 3.34e-6

KINK:
- mean hinges: 4.43
- mean MSE: 6.26e-6

Specialized V103b operators used only 1–2 extra features and had numerical-zero error.

Lesson:

> A generic universal substrate can cover many functions, but a matched primitive can be dramatically more compact and extrapolative.

---

# V105 — Exact amortization law for promoting a repeated composite into a primitive

Let:

- G = generic per-use cost
- S = specialized per-use cost
- I = one-time invention/storage/validation cost
- T = number of uses

Generic total:

    T*G

Promoted primitive total:

    I + T*S

Promotion pays iff:

    I + T*S < T*G

Equivalent:

    T > I / (G-S)

Using V104/V103b feature-count proxies:

Generic vs specialized:

- periodic: 14.93 vs 2
- exponential: 14.10 vs 1
- kink: 4.43 vs 1

For invention cost I=20:

- periodic profitable by 2 uses
- exponential by 2
- kink by 6

This is an exact accounting law once G,S,I are defined.

---

# V108 — Repeated residuals can become a new empirical primitive without named operator families

The learner was not told sine/exp/abs families.

Each task:

    polynomial background + amplitude * hidden recurring structure

Procedure:

1. fit current polynomial language
2. collect unexplained residual vectors
3. SVD residual matrix
4. promote top recurring residual direction as a reusable empirical primitive

Hidden structures used only for evaluation:

- WAVE
- CURVE
- KINK

With only 2 recurring tasks, first residual component explained:

    100%

for each clean fixed-shape family.

New-task error fell from:

WAVE:
- base ~0.80
- learned primitive ~2.5e-31

CURVE:
- base ~0.19
- learned primitive ~2.9e-30

KINK:
- base ~0.011
- learned primitive ~2.1e-31

This is the first experiment in the project where a reusable basis element emerged directly from recurring unexplained experience rather than being selected from a named operator library.

Caveat:
the primitive is an empirical vector on a fixed coordinate grid.

---

# V109 — Unsupervised multiple primitive discovery under noise and parameter drift

Mixed training stream:

- oscillatory family with frequency/phase drift
- exponential-like family with rate drift
- kink family with center drift
- random polynomial backgrounds
- amplitude variation
- observation noise

Learner:

1. fit polynomial base
2. normalize residuals
3. choose K=2..5 by silhouette score
4. KMeans cluster residuals
5. SVD each cluster to create one empirical primitive

Results:

Silhouette selected:

    K = 3

which matched the hidden family count.

Training cluster purity:

    100%

Variance explained by first cluster primitive:

- 99.05%
- 95.61%
- 88.23%

New-task improvements:

WAVE:
- base MSE 0.6426
- primitive MSE 0.05045
- 12.7x better

CURVE:
- 0.1273 -> 7.89e-5
- 1613.5x better

KINK:
- 0.00924 -> 4.48e-4
- 20.6x better

This demonstrates unsupervised expansion of a small empirical primitive vocabulary from recurring residual structure.

---

# V110 — Empirical primitive is not yet a rule

Learn primitive on:

    x in [-1,1]

New tasks live on:

    [-1.5,1.5]

Fit coefficients only inside [-1,1].
Evaluate only outside.

Empirical primitive was interpolated/clamped from the learned vector.

Extrapolation MSE:

WAVE:
- base 0.7366
- empirical 0.7691
- symbolic oracle ~8.8e-31

CURVE:
- base 32.51
- empirical 22.68
- symbolic oracle ~1.8e-29

KINK:
- base 0.2352
- empirical 0.1276
- symbolic oracle ~1.7e-30

Therefore:

> A reusable residual template is not equivalent to an operator-level rule.

It can compress experience in-domain without learning the generative law.

---

# V111 — Consolidate empirical primitive into symbolic rule

Pipeline:

    recurring tasks
        ->
    empirical residual chunk
        ->
    symbolic compression search
        ->
    extrapolating primitive

The symbolic grammar was supplied:

- periodic k
- exponential lambda
- kink center

Learned empirical residuals were compressed into:

WAVE:
- selected PERIODIC
- k = 5

CURVE:
- selected EXP
- lambda = 2.2

KINK:
- selected KINK
- center = 0.23

Extrapolation MSE:

- WAVE: 2.9e-30
- CURVE: 1.8e-29
- KINK: 1.9e-30

So symbolic consolidation restored rule-like extrapolation.

Caveat:
the symbolic operator grammar was still supplied.

---

# What is now genuinely narrowed down

## Exact / theorem-level

### 1. Complete noiseless finite data can certify representational insufficiency by linear-span/rank tests.

### 2. Passive finite observations cannot guarantee global primitive sufficiency over unrestricted targets.

### 3. Without structural assumptions, worst-case global certification requires covering every unseen finite-domain point.

### 4. Under a known M-dimensional noiseless linear class, M independent scalar evaluations suffice and are generically necessary for exact coefficient identification.

### 5. Primitive promotion amortization:

    promote iff T > I/(G-S)

once costs are defined.

---

# Strong empirical findings

### 1. Evidence-based complexity expansion beats blind expansion under noise.

### 2. Sparsity/compositionality can dramatically reduce the evidence needed compared with dense hypothesis dimension.

### 3. Sparsity level itself can be inferred from evidence with enough samples.

### 4. Repeated unexplained residual structure can be discovered unsupervised and promoted into a reusable empirical primitive.

### 5. Empirical residual primitives improve new in-domain tasks even when hidden families are mixed, noisy, and slightly drifting.

### 6. Empirical primitives do not automatically extrapolate.

### 7. Compressing an empirical primitive into a compact symbolic rule restores extrapolation when the correct symbolic family exists in the search grammar.

---

# The architecture hypothesis after V111

```text
experience
   |
   v
current primitive language
   |
   v
fit / act / predict
   |
   v
residual or unexplained structure
   |
   +---- noise / one-off ----> ignore
   |
   v
does it recur across tasks?
   |
   v
cluster / compress recurrence
   |
   v
empirical reusable primitive
   |
   v
does it generalize outside discovery context?
   |                 |
  yes               no
   |                 |
   v                 v
keep primitive   search for shorter rule
                     |
                     v
             symbolic / algorithmic
                consolidation
                     |
                     v
             reusable rule-like primitive
```

This is much closer to the original "self-developing computation" idea than the earlier staged-search experiments.

However, the remaining hard problem is still real:

> The system does not yet invent a symbolic grammar that contains a truly new operator family.

It can:

- detect insufficiency under favorable assumptions
- expand known interaction depth
- infer sparsity
- discover recurring empirical residuals
- cluster them into new reusable features
- compress them into a rule IF the relevant rule family already exists in the grammar

It cannot yet:

- create a genuinely new symbolic operator family outside its supplied meta-language
- prove sufficiency from finite noisy observations without assumptions
- guarantee open-ended primitive invention

That is now the clean frontier.
