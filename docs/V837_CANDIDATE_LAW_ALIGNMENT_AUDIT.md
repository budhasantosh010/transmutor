# V837 Candidate-Law Alignment Audit

## 1. Purpose

V837aa is a diagnosis-only audit of the frozen V837y Y3 parent. It asks whether the ten independently parameterized neutral cells already implement one reusable candidate transformation, one reusable law hidden by exact tanh-compatible signed/permuted coordinates, a small stable vocabulary of reusable cell types, or genuinely different local laws.

No candidate-sharing architecture was trained in V837aa. Representation adequacy therefore remains the parent result: 3/5.

## 2. Why raw equality is insufficient

Each local state is four-dimensional and the candidate activation is elementwise `tanh`. Exact coordinate symmetries therefore include coordinate permutations and sign flips. Two functionally equivalent local laws can look different in raw weights if their local coordinates differ by one of these exact symmetries.

V837aa exhaustively enumerates all `4! * 2^4 = 384` signed-permutation matrices. Every matrix is unique, orthogonal, and contains only `-1`, `0`, and `+1`.

Arbitrary continuous rotations are excluded because they are not exact symmetries of elementwise `tanh`.

## 3. Frozen transformation convention

With canonical coordinates `z = P s`, a cell is transformed as:

- `Ws' = P Ws P^T`
- `Wm' = P Wm`
- `Wx' = P Wx`
- `b'  = P b`
- `Wo' = Wo P^T`

Float32 unit tests verify candidate and output-map equivalence to `<= 1e-6`.

## 4. Parameter views

The audit keeps four views separate:

1. recurrent/message candidate core: `Ws + Wm + b`
2. core + input interface: `Ws + Wm + Wx + b`
3. core + output interface: `Ws + Wm + b + Wo`
4. full local law: `Ws + Wm + Wx + b + Wo`

This prevents specialization in `Wx` or `Wo` from masking evidence for a shared recurrent/message core. In the observed result, however, even the primary recurrent/message core failed the universal common-law thresholds in every fit.

## 5. Why only Y3 was rerun

V837aa audits the exact machine-selected V837y parent `Y3_global_control_rank4_candidate`. Only its 25 family-by-replicate fits were rerun under the frozen 4x regime. Y0, Y1, Y2, and Y3C were not rerun.

The regenerated Y3 parent reproduced the committed V837y rows exactly: 25/25 runs had zero development and validation success drift, all family medians were unchanged, and the parent remained 3/5.

## 6. Initialization null

The same exhaustive 384-way alignment is applied to the untrained initial snapshots. This matched null is essential because exhaustive coordinate search can make unrelated small matrices appear artificially closer.

The trained aligned models did not outperform the initial aligned null. Median aligned synthetic cosine fell from 0.7749 at initialization to 0.6121 after training, while median aligned synthetic NRMSE worsened from 0.3762 to 0.4669. Empirical aligned cosine fell from 0.7943 to 0.6489 and empirical aligned NRMSE worsened from 0.3630 to 0.4317.

Thus the apparent gain from signed-permutation search is not evidence that training converged toward one common law.

## 7. Raw functional findings

Primary recurrent/message core, median across the 25 fits:

| Probe regime | Raw cosine | Raw NRMSE |
| --- | ---: | ---: |
| Synthetic | 0.5070 | 0.5373 |
| Empirical development | 0.4433 | 0.5709 |

The frozen direct-common-basis threshold is cosine >= 0.95 and NRMSE <= 0.20 on both probe regimes in at least 20/25 fits. The observed passing count was 0/25.

## 8. Signed-permutation-aligned functional findings

| Probe regime | Aligned cosine | Aligned NRMSE |
| --- | ---: | ---: |
| Synthetic | 0.6121 | 0.4669 |
| Empirical development | 0.6489 | 0.4317 |

Alignment reduced median NRMSE by only 13.1% on synthetic probes and 24.4% on empirical probes, below the frozen 40% meaningful-gain gate. No fit passed the aligned common-law threshold; no fit materially exceeded the aligned initialization null.

The signed-permutation hypothesis therefore fails rather than merely landing narrowly below threshold.

## 9. Relative basis stability

Relative transforms `R_ij = P_i^T P_j` were compared across all 25 fits. Median exact relative-transform agreement was 0.08 and median entropy was 4.56 bits. Relative bases are therefore unstable. Because the common-law gate itself failed, this is reported descriptively rather than as a common-law subdiagnosis.

## 10. Gradient compatibility

After training, one frozen development-batch backward pass was performed without an optimizer update. Aligned recurrent/message-core gradient direction agreement was weak:

- median of fit-level median cosine: 0.0262
- median fraction negative: 0.4889
- median fraction strongly negative (`< -0.25`): 0.2444
- median minimum pair cosine: -0.6250

The frozen classification is `mixed`, not compatible and not uniformly conflicted. This provides no optimization-based reason to impose hard candidate-core sharing.

## 11. Reusable cell-type clustering

Deterministic aligned-functional clustering tested `k = 2, 3, 4, 5`. No `k` passed the frozen within-type similarity, NRMSE, fit-stability, family-stability, and consensus gates. There is therefore no supported small reusable candidate-law vocabulary in these ten cells.

## 12. Global rank-4 coupling alignment

The trained effective 40x40 cross-block recurrent candidate matrix was transformed by the block-diagonal local signed-permutation basis. Equivariance checks passed to numerical precision (median maximum absolute error approximately `4.44e-16`). This confirms that the local basis audit was composed consistently with the frozen global coupling rather than analyzing cell cores in an incompatible coordinate convention.

## 13. Final diagnosis

`GENUINELY_DIVERSE_CANDIDATE_LAWS`

The ten Y3 cells remain substantially heterogeneous even after exhaustive exact tanh-compatible signed/permutation alignment. The trained aligned similarity is well below the universal common-law gate, worse than the aligned untrained null, relative coordinate relations are unstable, and no stable 2-5 type vocabulary passes.

## 14. Recommended next axis

`NEXT_AXIS_SHARED_INPUT_REPRESENTATION`

Candidate parameter sharing should be downgraded as the next blocker hypothesis. The strongest remaining documented axis is input organization/shared input representation. V837aa does not implement that future experiment.

## 15. Locks

- representation adequacy: still 3/5 parent
- fresh-audit episodes consumed: 0
- primitives promoted: 0
- structural search: blocked
- primitive mining: blocked
- V837ab architecture: not implemented
- V838: not started

V837aa succeeds by reducing uncertainty, not by forcing a positive sharing conclusion.
