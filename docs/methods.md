# Using the inference methods

[`cbt.py`](../cbt.py) is a standalone NumPy module. It takes scores as input; model fitting and experiment data generation live in `simulate.py`. Larger scores must indicate stronger evidence against the null.

## Individual controls

Supply a one-dimensional array `tests` of length `m` and a list `controls` of length `m`. `controls[j]` is a nonempty one-dimensional array of control scores for test `j`; the control counts may differ. Scores must be finite. The e-value and q-finalist constructions require no ties within a competition. Validity relies on the null exchangeability assumptions in the manuscript.

Run this example from the repository root:
```python
import numpy as np
import cbt

rng = np.random.default_rng(42)
tests = rng.normal(size=100)
tests[:20] += 3
controls = [rng.normal(size=30) for _ in tests]

# Ordinary CBT followed by BH.
p = cbt.cbt_pvalues(tests, controls)
rejected_bh = cbt.bh(p, alpha=0.1)

# Construct the e-values and apply e-BH at the same prespecified level.
e = cbt.evalues(tests, controls, alpha=0.1)
rejected_ebh = cbt.ebh(e, alpha=0.1)

# Equal weighting and independent control subsampling.
p_equal = cbt.cbt_pvalues(tests, controls, equal_weights=True)
sampled = cbt.subsample(controls, np.random.default_rng(123))
p_subsampled = cbt.cbt_pvalues(tests, sampled, equal_weights=True)
```

All p-value and e-value functions return one value per test, in the input order. `bh` and `ebh` return Boolean rejection masks; the examples run them at target FDR level 0.1. `evalues` requires an explicit `alpha`; the examples use 0.1 as well. 

| Function | Operation |
| --- | --- |
| `cbt_pvalues(tests, controls)` | Winner-only CBT with weights `1/n_j` and common offset `1/n_min` |
| `cbt_pvalues(..., individual_offset=True)` | CBT with individual offset `1/n_j`, used for Bonferroni in the experiments |
| `cbt_pvalues(..., equal_weights=True)` | Assign every control winner weight `1/n_min` regardless of `n_j` |
| `subsample(controls, rng)` | Independently retain `n_min` controls per test, without replacement |
| `evalues(tests, controls, alpha)` | Individual CBT e-values at the prespecified level |
| `evalues(tests, controls, alpha, return_thresholds=True)` | Return `(evalues, thresholds)` with the hypothesis-specific `T_j^+` values |
| `bh(p, alpha)` | Benjamini–Hochberg rejection mask |
| `ebh(e, alpha)` | e-BH rejection mask |

For the Bonferroni comparison, use:
```python
p_bonferroni = cbt.cbt_pvalues(tests, controls, individual_offset=True)
rejected_bonferroni = p_bonferroni <= 0.2 / len(tests)
```

## Shared controls and q-finalist competitions

`shared_cbt_pvalues(tests, controls, labels)` takes a one-dimensional score array, one block label per test, and a dictionary mapping each tested block label to its nonempty control-score array. It returns winner-only shared CBT p-values using the common `max_g b_g/n_g` offset. Use `random_hypotheses=True` only for the manuscript's random-hypothesis construction, with controls already restricted to the null pool; the fixed-hypothesis simulation (which is the setting of Section 5.3 in the paper) uses the default `False`.

`q_finalist(tests, controls, q)` expects a rectangular `m`-by-`n` control-score array, a length-`m` test-score array, and `1 <= q <= n`. It implements the homogeneous q-finalist CBT comparison. The simulation applies BH to the returned values.

