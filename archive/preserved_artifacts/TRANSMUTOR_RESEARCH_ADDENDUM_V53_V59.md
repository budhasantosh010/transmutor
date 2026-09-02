# Transmutor Research Addendum — V53 through V59

This addendum extends the V0–V52 research log.

## V53 — One generic state-update substrate

One expression language:

    state_next = F(state, x, gate)

with primitives:
add, subtract, multiply, max, min, negate, tanh

was searched independently for three different sequential problems.

Discovered:

- PARITY: `x * s`
- MAJORITY: `x + s`
- GATED MEMORY: `tanh(s) - (-(g) * x)` = `tanh(s) + g*x`

All three generalized to 100% accuracy at much longer sequence lengths:
- parity to length 64
- majority to 65
- gated memory to 96

The gated-memory rule is interesting:
- when g=0, `tanh(s)` preserves sign
- when g=1, adding ±1 overwrites sign because |tanh(s)| < 1

So the search discovered a sign-preserving/overwriting memory rule without a hand-written if/else gate.

Caveat:
the recurrent state-update loop and primitive vocabulary are still supplied.

---

## V54 — Learned latent generative replay

Compressed continual-learning memory was moved from raw standardized-pixel statistics into a learned unsupervised PCA latent space.

Final continual digit accuracy:

- no replay: 51.83%
- raw isotropic replay, 650 floats: 88.17%
- PCA-8 latent replay, 666 floats: 88.56%
- PCA-16 latent replay, 1,258 floats: 89.17%

Interpretation:
compact generative replay survives representation compression and can operate in a learned latent basis.

Caveat:
PCA was pretrained unsupervised on the full training-input corpus.

---

## V55 — Local prediction is not automatically better

32-dimensional hidden representation, no class labels during representation learning.

- random features: 91.33%
- local exact reconstruction: 95.20%
- denoising/predict-clean objective: 94.89%

Denoising did not beat ordinary reconstruction.

Lesson:
a more "predictive" objective is not automatically a better representation objective.

---

## V56 — One multi-algorithm cell via compiled skills

V53's discovered skills were supplied as reusable macros:

- P = s*x
- M = s+x
- K = tanh(s)+g*x

Evolution searched for one context-conditioned update law.

It fit all three training tasks, but long-sequence results were:

- parity: 100%
- majority: 91.34%
- memory: 100%

The discovered program had 39 nodes.

Lesson:
skill composition can overfit even when the component skills individually generalize.

---

## V56b — Curriculum + simplicity pressure still fails

Multi-length training and stronger complexity penalty were added.

Long tests:

- parity: 100%
- majority: 100%
- memory: 49.9%

So evolutionary search preserved two algorithms and sacrificed the third.

This points to search/credit-assignment limitations rather than substrate incapacity.

---

## V56c — Greedy sparse symbolic search fails despite containing the solution

Candidate feature library included pairwise products of:

P, M, K, p, m, k, 1

The exact solution exists as:

    P*p + M*m + K*k

but Orthogonal Matching Pursuit selected the wrong correlated features.

Long tests collapsed near chance.

Lesson:
even when the representation library literally contains the correct answer, a search heuristic can fail to recover it.

---

## V56d — Exact small combinatorial search finds the clean composition

Every 3-feature subset of the same 35-feature library was tested.

It found exactly:

- `P*p` coefficient 1
- `M*m` coefficient 1
- `K*k` coefficient 1

Local error: effectively zero.

Long-sequence performance:

- parity: 100%
- majority: 100%
- memory: 100%

This cleanly separates:

    solution exists
        !=
    search method can find solution

Search quality is therefore a first-class part of the architecture problem.

---

## V57 — Infer hidden task from demonstrations, no task ID

Each episode used a new random 2D linear separator:

    y = sign(w1*x1 + w2*x2 + b)

The learner did not receive w1, w2, or b.

It saw demonstrations and one query.

A simple learned mean-pooled context encoder achieved:

- query only: 49.80%
- 1 demo: 67.74%
- 3 demos: 64.64%
- 6 demos: 65.69%
- 12 demos: 64.29%

More demonstrations did not help.

Lesson:
the information can exist in experience while the context-processing mechanism fails to exploit it.

---

## V57b — Explicit per-episode inference proves the information is there

Same task family, but an explicit tiny ridge-regression inference algorithm was applied to demonstrations.

Accuracy:

- 1 demo: 57.78%
- 3: 75.08%
- 6: 84.85%
- 12: 89.73%
- 24: 93.01%
- 48: 94.99%

So additional experience was highly informative.
V57's plateau was an inference-architecture failure.

---

## V58 — Learned attention can use more experience

A small Transformer consumed demonstration tokens plus a query token.

Accuracy:

- 1 demo: 66.81%
- 3: 78.29%
- 6: 86.56%
- 12: 91.07%

For 3/6/12 demonstrations it slightly exceeded the explicit ridge baseline.

This is an important step:
task identity no longer needs an explicit task-ID wire.
A learned attention mechanism can infer the current hidden task from examples.

Caveat:
the Transformer architecture itself is supplied.

---

## V59 — Selective experience acquisition

Using the V58 learner:

Policy:
- start with 1 demonstration
- if uncertain, request 3
- if still uncertain, request 6
- otherwise use 12

Validation selected a confidence threshold.

Held-out:

- always 12 demos: 90.26%
- selective policy: 89.84%
- average demos consumed: 7.126 / 12
- experience reduction: 40.62%

Stopping distribution:
- stop after 1: 5.5%
- after 3: 30.0%
- after 6: 26.3%
- use all 12: 38.3%

This extends the finite-resource principle from compute to experience/data acquisition:

    do not gather more information when current evidence is already sufficient.

---

# Strong new lessons from V53–V59

1. One generic state-update language can express arithmetic algorithms and useful memory behavior.

2. Memory can emerge as a dynamical sign-preservation/overwrite rule rather than an explicit if/else gate.

3. Compact long-term replay can live in a learned latent basis.

4. Better-sounding self-supervised objectives do not automatically improve representations.

5. Reusing learned skills is not enough; composing them robustly is itself a search problem.

6. Search algorithm quality can dominate:
   - evolutionary GP failed
   - greedy sparse recovery failed
   - exact small combinatorial search found the perfect program immediately.

7. Task identity can be inferred from examples instead of supplied explicitly.

8. More experience only helps if the architecture can actually integrate it.

9. Attention was far better than simple mean pooling for task inference.

10. Experience itself can be selectively budgeted, just like compute.

---

# Updated architecture hypothesis

```text
                         ADAPTIVE INTELLIGENCE
                                |
        +-----------------------+-----------------------+
        |                       |                       |
      MEMORY                  COMPUTE                STRUCTURE
        |                       |                       |
 compress/generate          route selectively       grow/prune
 preserve/overwrite         vary depth              repair
 replay offline             search                  reproduce
 multiple timescales        stop early              compile skills
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                         CONTEXT / TASK INFERENCE
                                |
                  infer "what problem am I in?"
                         from experience
                                |
                                v
                         LEARNING / SEARCH
                                |
              choose not only parameters, but
             algorithms, compositions, and rules
                                |
                                v
                             WORLD
```

The research increasingly suggests that "search" is not just an outer training procedure.
A future adaptive architecture may need mechanisms for choosing how it searches its own computational possibilities.
