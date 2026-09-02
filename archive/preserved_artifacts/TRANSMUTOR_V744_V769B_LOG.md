# Transmutor Experimental Log — V744 to V769B

## V744 — Discover recurrent memory update from sparse final reward
PASS, 3/3.
Recovered algebraic equivalents of:
`m_next = m + g*(v-m)`
Final reward: 100%.

## V745 / V745B — Recursive macro mining
V745: macro promotion succeeded, but the generated application search omitted valid pairings; downstream test failed.
V745B: corrected generic pairing enumeration; exact relation recovered 3/3, mean test accuracy 99.84%.

## V746 — Behavioral macro mining
PASS.
Eight syntactically different discoveries collapsed into one behavioral update macro.
Transferred exactly to vector memories and T=256 sequences.

## V747 / V747B — End-to-end relation + mined memory update
V747 timed out.
V747B PASS, 3/3.
Relation: `x3*x5 + x4*x6`
Mined recurrent update used instead of hand-coded WRITE/HOLD.
Fresh reward: 100%.

## V748 — Zero-shot composed-program generalization
PASS.
Frozen relation + frozen update generalized to:
- T32/K1/D1
- T64/K8/D3
- T256/K20/D8
All 100%.

## V749 — 20% corrupted binary reward
PASS, 3/3.
Correct relation recovered; clean test reward 100%.

## V750 series — Reward corruption boundary
- 35% corruption: initial test succeeded.
- 40% naive search failed.
- V750C robust multi-holdout search restored 3/3, clean reward 100%.
- 45% scratch structural search became unreliable.
- V750F showed parameter calibration can still be perfect when structure is known.

## V751 — Cached primitive library under 45% corruption
PASS, 5/5.
DOT selected from 8 cached alternatives.
Mean clean reward 99.7%.

## V752 series — Library-size multiple-comparison problem
Large noisy libraries can produce false winners.
Controlled nested tests showed selection can degrade as candidates grow.
Important unresolved requirement: retrieval + verification, not unlimited flat macro search.

## V753 / V753B — Successive verification + parameter memory
V753: 2048 -> 64 -> 12 -> 3 -> DOT succeeded on completed seed.
V753B PASS: cached threshold prior stabilized calibration under 45% noise; 5/5 clean test = 100%.

## V754 — Local cell structural credit
PASS, 3/3.
45 pair-product cells receive only:
- own participation bit
- scalar broadcast reward
Correct hidden pair cells became ranks #1/#2.

## V755 — Local self-pruning and compilation
PASS, 3/3.
Largest survival gap automatically kept exactly 2 useful cells.
Compiled accuracy 100%.

## V756 — Scrambled channel layouts
PASS, 3/3.
Local cells found correct pairings despite event/goal coordinates moving.
Compiled accuracy 100%.

## V757 — 20% corrupted broadcast reward
PASS, 3/3.
Exact local structure; 100% compiled accuracy.

## V758 — 35% corrupted broadcast reward
PASS, 3/3.
Exact local structure; 100% compiled accuracy.

## V759 — Growth/death/reproduction/mutation from 12 random cells
PASS, 3/3.
Target cells forced absent initially.
Both useful cells were born by mutation and later dominated.
Compiled accuracy 100%.

## V760 — Reproduction-only exploration
PASS, 3/3.
Random immigrants not necessary, but discovery was ~1.75x slower than with immigrants.

## V761 / V762 — Adaptive exploration
V761 PASS, 3/3.
~1 immigrant/run instead of ~43, but slower discovery.
V762 showed exploration scheduling itself has a cost/speed tradeoff.

## V763 — Strong local stress test
PASS, 3/3.
Combined:
- small population
- target cells absent initially
- growth/death/reproduction/mutation
- local participation credit
- 35% corrupted broadcast reward
All 3 scrambled layouts self-pruned to exact target structure with 100% compiled accuracy.

## V764 / V764B — Evolve operator and wiring
V764 failed at 8k because second target operator appeared too late.
V764B, with longer development, PASS 3/3:
cells evolved both MUL operator and correct input wiring.
Compiled accuracy 100%.

## V765 — Co-learn structure and decision threshold from scalar reward
PASS, 3/3.
No post-hoc threshold fitting.
Mean fresh accuracy ~99.81%, minimum ~99.625%.

## V766 / V766B — Remove global bottom-N death ranking
V766 failed: local death timescale too fast.
V766B PASS 3/3 after matching death lifetime to evidence-accumulation time.
No global bottom-N/top-parent tournament.
Compiled accuracy 100%.

## V767 series — Combine operator evolution with fully asynchronous local structure
V767 failed: mediocre ADD cell stabilized; missing target never appeared.
V767B failed: plateau exploration storm caused destructive churn.
V767C failed: fairer newborn prior still insufficient.
V767D mixed: evidence-based death solved one layout, another never generated one target.
V767E increased structural diversity; solved 2/3 layouts, failed 1/3 due missing target.

Lesson:
selection/retention can work locally, but coverage of a larger operator+wiring mutation space becomes a dominant bottleneck.

## V768 / V768B — Multi-scale mutation
V768 generated both target operators but one was born at step 23,441/24,000 and had no time to mature.
V768B retry with longer development succeeded on the previously hard layout:
exact two MUL cells, 100% compiled accuracy.
Not yet replicated.

## V769 — Make local learning rule itself heritable
FAIL.
Rule gene options included CENTERED, ACTIVE, SIGNED, ANTI, NONE.
Pathological ANTI rules could inflate their own survival state and become difficult to kill.
This is a concrete survival/reward-hacking failure.

## V769B — Protected local fitness audit
FAIL on first run.
Separated:
- mutable action-learning rule
- protected causal audit used for death/reproduction
This prevented direct survival hacking, but several marginally helpful cells all received positive audit credit.
Their compiled combination was poor (~75.9%).

### New unresolved problem
Interaction-aware structural credit:
a cell can be marginally useful alone or in some sampled contexts, yet redundant/harmful when compiled with other survivors.

## Current strongest narrowed picture

A viable local Transmutor-like loop increasingly appears to require:

1. **Local causal credit** from participation and broadcast outcomes.
2. **Evidence-compatible developmental timescales**.
3. **Diversity / mutation coverage**.
4. **Retention based on evidence, not mere age or global ranking**.
5. **Protected evaluation channels** so mutable learning rules cannot hack survival.
6. **Interaction-aware credit** so individually useful cells do not form a collectively bad program.
7. **Compilation / macro mining** after stable useful patterns emerge.
8. **Finite-resource retrieval and verification** as libraries grow.

None of these experiments establish AGI, ASI, or a post-Transformer replacement. They are controlled toy experiments narrowing architectural requirements and failure modes.
