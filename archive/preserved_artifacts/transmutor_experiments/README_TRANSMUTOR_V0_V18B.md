# Transmutor Research Log — V0 through V18b

## Core hypothesis
Can useful computation self-organize from small adaptive primitives under finite resources, with the system learning not only parameters but also what computation, memory, routing, structure, and learning machinery should exist?

## Strongest experimental lessons so far

1. More capacity is not automatically more intelligence.
   - V3: XNOR remained at 75% while blind growth expanded 2 -> 8 cells.
   - V3b: directed structural testing solved it with only 2 cells.

2. State is not automatically memory.
   - V2 failed at chance.
   - V2b learned retention/write control and reached 100%.

3. Finite resource pressure can produce selective computation.
   - V4 reduced an activation proxy from 7.92 to 1.28 equivalent cells at 100%.
   - V11 routed hard inputs to expensive computation and easy inputs away from it.
   - V13 learned when to pay for planning.

4. Memory economics can collapse into degenerate behavior.
   - V5 learned "write nothing".
   - V5b succeeded only after relevance/addressability was made available.

5. Exact symmetric backprop is not necessary on at least simple tasks.
   - V6: random feedback alignment solved XOR in 30/30 seeds.

6. Learning rules can themselves become search targets.
   - V7 found a high-performing rule, but only matched a known hand-designed baseline.

7. Computation can be composed from lower-level primitives.
   - V8 rediscovered exact compact programs for basic Boolean functions.

8. Reusable modules dramatically accelerate related tasks.
   - V9 composite tasks reached 100% in one router step, versus 5–42 from scratch.

9. Repeated computation can be compiled.
   - V10 reduced late-stream abstract execution cost by ~50.5%.

10. Learned world models enable planning beyond immediate local choices.
    - V12 one-step greedy got stuck at a wall.
    - multi-step imagined futures found the correct detour.

11. Experience selection can help, but the current gain is modest.
    - V14 uncertainty-driven sampling beat random sampling at equal label budget.

12. A tiny seed can grow task-dependent structure.
    - V15: same 1-cell seed remained 1 for linear, grew to 2 for XOR/parity3, and ~3–4 for parity4.

13. Reproduction requires diversity.
    - V16 near-cloning: 0/12 success.
    - V16b diverse offspring: 12/12 success.
    - Pure copying trapped the system in the same insufficient computation.

14. Resource competition can structurally eliminate excess cells.
    - V17 pruned 12 -> 3 cells with 100% accuracy retained.

15. Continual learning has a real stability/plasticity tradeoff.
    - V18 was too easy and therefore inconclusive.
    - V18b conflicting tasks:
        fixed shared representation: 66.25% final mean, 24.06 percentage-point forgetting
        progressive protected modules: 98.75% final mean, 0 measured forgetting
    - But protection required growing capacity.

## Current architecture picture

The experiments increasingly suggest a future adaptive system needs interacting mechanisms for:

- local/internal state
- controlled memory retention and writing
- structural credit assignment
- growth, pruning, and reproduction
- diversity preservation
- finite compute/energy allocation
- task/input-dependent routing
- reusable modules
- skill compilation
- learned world models
- selective internal simulation
- active experience acquisition
- lifelong stability without freezing everything forever
- learning-rule adaptation
- eventually primitive/algorithm discovery

## The biggest unsolved problems now

1. Discover relevance/addressing rather than supplying it.
2. Protect old knowledge without unbounded structural growth.
3. Let cells decide growth/death/reproduction locally rather than via an external evaluator.
4. Discover the learning rule itself, not coefficients inside a supplied family.
5. Discover genuinely new primitives instead of composing a supplied primitive vocabulary.
6. Replace hand-defined cheap/expensive experts with self-created computational organs.
7. Scale beyond toy Boolean/grid experiments and compare under equal real compute/data budgets.
8. Test whether the same principles still help when the system has thousands/millions of cells.

## Scientific status
These experiments do NOT demonstrate AGI, ASI, or a successor to Transformers.
They are toy falsification/prototyping experiments.
Their value is that failures and successes are now producing concrete architectural requirements instead of vague speculation.
