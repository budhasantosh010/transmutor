# Transmutor Experimental Log — V801 to V812B

## V801 — Low-rank relationship memory
FAIL.
Truncated SVD of the pairwise interaction matrix did not reliably preserve the weak true relation.
Rank 8 recovered 3/4; rank 12 only 2/4.
Conclusion: useful relation behaves like a sparse exception, not only a dominant low-rank pattern.

## V802 — Low-rank background + sparse exceptions
PASS.
Rank-2 background + 120 positive residual exceptions:
- exact true organ 5/5
- memory reduction ~73.4% vs exact pair table
The true relation was consistently recoverable when sparse residual memory was large enough.

## V803 — Scaling fixed exception memory
FAIL.
With fixed rank-2 + 120 exceptions:
- N=50: 2/2
- N=70: 1/2
- N=100: 0/2
- N=140: 1/2
The fixed exception budget did not scale reliably.

## V804 — Audit density sweep
Diagnostic.
At N=100, p_active=0.02 (about 2 active cells/audit) restored 2/2 shortlist recovery.
At N=140, no tested density restored reliability.
Evidence density matters, but is not the whole scaling solution.

## V805 — Stratified sparse exception memory
Mixed/FAIL overall.
Protected sparse exception slots per operator-pair category helped some runs but remained noisy at N=140/180.
Memory stratification alone does not solve evidence interference.

## V806 — Category-local causal screening
PASS.
Protected separate causal contexts per relation category:
ADD-ADD, ADD-SUB, ADD-MUL, SUB-SUB, SUB-MUL, MUL-MUL.

Results:
- N=100: 3/3 exact
- N=140: 3/3 exact
- N=180: 3/3 exact
Only 120 focal candidates were confirmed:
- 97.6% reduction at N=100
- 98.8% at N=140
- 99.3% at N=180

Key principle:
separate relationship categories can prevent unrelated structures from diluting weak causal evidence.

## V807L — Lightweight continuous integration
FAIL overall, 1/3.
When both target genomes were born, the online organ mechanism solved exactly.
In two runs, one required target genome never appeared.
Structural coverage became the bottleneck.

## V808A — Adaptive structural birth allocator
PASS as isolated search result.
At 90 births / 120 total unique genomes:
- balanced: 45.65% both targets
- 50/50 balance+adaptive: 58.13%
- 25/75 balance+adaptive: 92.85%
Adaptive family allocation can drastically improve coverage in a helpful environment.

## V809 — Sandboxed candidate organs
PASS, 3/3.
Continuous mixed-operator population:
- targets absent initially
- adaptive/diverse births
- category-local relationship screening
- candidate organs do not immediately affect behavior
- protected sandbox proof before certification

Results:
- both targets appeared 3/3
- exact certified organ 3/3
- mean genome space seen ~52.1%
- hundreds of proposals compressed to mean ~12.7 surviving sandbox hypotheses

Key principle:
candidate abstractions should be sandboxed before they are allowed to alter normal behavior.

## V810 — Misleading early evidence
Negative control.
Added a strong but imperfect ADD(8,9) standalone decoy.
Same-seed continuous runs with q_balanced=.25 and .75 both failed because one required MUL target never appeared.

## V810A — Misleading-world birth-policy sweep
At 90 births / 120 unique genomes:
- q=.25: 51.7% both targets
- q=.50: 45.2%
- q=.75: 49.2%
- q=1.0: 50.0%
No fixed exploration mixture solves the misleading case.
Around 40/55 MUL genomes are explored, implying ~50% probability of seeing both arbitrary targets.

## V811 — Family relationship promise
FAIL.
Using max within-family synergy/coverage evidence to steer exploration was unreliable.
ADD/SUB often won from noisy maxima.
Even with both target MUL cells sampled, discovery noise could still make another family look stronger.

Key principle:
do not steer large structural budgets from an unverified maximum.

## V812 — Verified family relationship evidence
High precision, incomplete recall.
Each family:
1. proposes top 5 relations
2. independent focal causal verification
3. isolated organ audit
4. certification only near estimated reward ceiling

Results:
- no ADD or SUB false certifications
- when the true MUL pair was actually proposed, it certified correctly
- but top-5 proposal recall was sometimes too low

Key principle:
independent verification fixes false exploitation, but proposal recall remains separate.

## V812B — Adaptive proposal width
PASS.
With all 55 MUL genomes present:
- true relation discovery rank mean 11
- median 6.5
- max 24
- top 5 recall: 6/12
- top 10: 7/12
- top 20: 8/12
- top 40: 12/12
- every true pair was independently certifiable: 12/12

Key principle:
proposal generation and proof must have different thresholds.
Cheap proposal width should expand when certification has not yet succeeded.

# Strongest narrowed conclusions from V801–V812B

1. Pure low-rank relationship memory loses rare weak causal relations.
2. Low-rank background + sparse exception memory can work at modest scale.
3. Fixed global sparse buffers do not reliably scale.
4. Category-local causal contexts solve much of the evidence-interference problem.
5. Structural coverage and relationship proof are separate bottlenecks.
6. Sandboxing candidate organs before behavioral deployment materially improves stability.
7. Adaptive structural search works when early evidence is aligned, but misleading evidence destroys the advantage.
8. No fixed exploration mixture is universally good.
9. Unverified maxima should not control structural resource allocation.
10. Independent causal verification can prevent false family-level exploitation.
11. Proposal recall should widen adaptively while proof thresholds remain strict.
12. In arbitrary structure spaces without side information, some near-exhaustive search remains unavoidable; structure can only be exploited when the environment supplies exploitable regularities.

These remain controlled synthetic experiments. They do not establish AGI, a Transformer replacement, or a new general computational paradigm.
