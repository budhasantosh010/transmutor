# Transmutor Research Log — V0 through V21

This directory contains the toy experiments built in the conversation.

## Latest major result: V21
Search was given a recurrent scaffold:

    state_next = F(state, x)

and a primitive vocabulary:
add, subtract, multiply, max, min, negate, tanh.

Search only saw short sequences (length <= 6).

It discovered:

PARITY:
    state_next = state * x

MAJORITY:
    state_next = state + x

Both reached 100% accuracy on unseen sequence lengths up to 64.

This is stronger than the fixed Boolean truth-table experiments because the discovered update rules generalize algorithmically to longer inputs.

Important caveat:
The loop/recurrent scaffold and primitive vocabulary were still supplied by us. The system did not invent recurrence itself.

## Strongest lessons across V0–V21

- Capacity alone is not intelligence.
- State alone is not memory.
- Retention/write control matters.
- Structure needs credit assignment.
- Resource pressure can create sparse computation.
- Memory objectives can collapse into degenerate strategies.
- Exact backward-weight symmetry is unnecessary on simple tests.
- Learning rules can themselves become search targets.
- Reusable modules accelerate new related tasks.
- Repeated computation can be compiled into cheaper skills.
- Learned world models enable planning around local traps.
- Planning itself can be selectively allocated.
- Active experience acquisition can improve data efficiency.
- One seed can develop different amounts of structure in different environments.
- Reproduction needs diversity rather than near-cloning.
- Resource competition can eliminate excess cells.
- Protected modular growth can prevent catastrophic forgetting, but capacity grows.
- Simple recurrent algorithms can be discovered and generalize far beyond training lengths.

## Still unsolved

1. Discover addressing/relevance rather than supplying it.
2. Protect old skills without unbounded module growth.
3. Make growth, death, and reproduction decisions locally.
4. Discover learning-rule structure, not coefficients in a supplied family.
5. Discover the primitive vocabulary itself.
6. Discover recurrence/loops/search/attention rather than giving scaffolds.
7. Move from toy tasks to equal-budget comparisons against strong neural baselines.
8. Scale cell populations by orders of magnitude while keeping adaptation stable.

These results are experiments, not evidence of AGI/ASI or proof of a post-Transformer paradigm.
