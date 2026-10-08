"""Competition p-values and individual-control e-values; see docs/methods.md.

Larger scores indicate stronger evidence against the null. Individual-control
methods take one test score and one nonempty control-score array per hypothesis.
P-value and e-value arrays preserve test order; rejection masks do likewise.
Statistical guarantees require the manuscript's exchangeability assumptions.
"""

import numpy as np


def tail(scores, weights, thresholds):
    """Sum weights of scores at or above each threshold, including ties.

    Args:
        scores (array_like): One-dimensional score array, possibly empty.
        weights (array_like): One weight per score.
        thresholds (float or array_like): Cutoffs at which to evaluate the tail.

    Returns:
        numpy.ndarray or numpy.float64: Weighted tail sums with the shape of
        thresholds. An empty score array gives zero.
    """
    order = np.argsort(scores)
    suffix = np.r_[np.cumsum(np.asarray(weights)[order][::-1])[::-1], 0.0]
    return suffix[np.searchsorted(np.asarray(scores)[order], thresholds, side="left")]


def bh(p, alpha):
    """Apply the Benjamini–Hochberg step-up rule to a p-value vector.

    Args:
        p (array_like): One-dimensional p-value array.
        alpha (float): Required testing level in (0, 1).

    Returns:
        numpy.ndarray: Boolean rejection mask in input order, including values
        equal to the selected cutoff. No passing rank means no rejections;
        an empty input gives an empty mask.
    """
    p = np.asarray(p)
    ordered = np.sort(p)
    passed = ordered <= alpha * np.arange(1, len(p) + 1) / len(p)
    return (
        p <= alpha * (np.flatnonzero(passed)[-1] + 1) / len(p)
        if passed.any()
        else np.zeros(len(p), bool)
    )


def ebh(e, alpha):
    """Apply the e-BH step-up rule to a nonnegative e-value vector.

    Args:
        e (array_like): One-dimensional array of nonnegative e-values.
        alpha (float): Required testing level in (0, 1).

    Returns:
        numpy.ndarray: Boolean rejection mask in input order, including values
        equal to the selected cutoff. No passing rank means no rejections;
        an empty input gives an empty mask.
    """
    e = np.asarray(e)
    thresholds = len(e) / (alpha * np.arange(1, len(e) + 1))
    eligible = np.sort(e)[::-1] >= thresholds
    return (
        e >= thresholds[np.flatnonzero(eligible)[-1]]
        if eligible.any()
        else np.zeros(len(e), bool)
    )


def empirical(tests, controls):
    """Return add-one empirical p-values against a single control pool.

    Each p-value is (1 + count(controls >= test)) / (1 + len(controls)), so
    ties count against the test.

    Args:
        tests (array_like): One-dimensional test-score array.
        controls (array_like): One-dimensional array of pooled control scores.

    Returns:
        numpy.ndarray: One p-value per test, in input order. An empty control
        pool yields p-values of one.
    """
    controls = np.sort(controls)
    return (1 + len(controls) - np.searchsorted(controls, tests)) / (1 + len(controls))


def prepare(tests, controls):
    """Validate individual competitions and collect their control summaries.

    Args:
        tests (array_like): Nonempty one-dimensional array of finite test scores.
        controls (sequence of array_like): One nonempty, one-dimensional array
            of finite control scores per test. Control counts may differ.

    Returns:
        tuple: (tests, controls, n, maxima), containing a float test array,
        a list of float control arrays, an integer control-count array, and
        a control-maximum array, all in test order.

    Raises:
        ValueError: If scores cannot be converted to floats, dimensions or
            control counts are invalid, or any score is nonfinite.
    """
    tests = np.asarray(tests, float)
    controls = [np.asarray(row, float) for row in controls]
    if tests.ndim != 1 or not len(tests) or len(controls) != len(tests):
        raise ValueError("Supply one control row per test.")
    if not np.isfinite(tests).all() or any(
        c.ndim != 1 or not len(c) or not np.isfinite(c).all() for c in controls
    ):
        raise ValueError("Scores must be finite and control rows nonempty.")
    n = np.array([len(c) for c in controls])
    maxima = np.array([np.max(c) for c in controls])
    return tests, controls, n, maxima


def cbt_pvalues(tests, controls, *, equal_weights=False, individual_offset=False):
    """Return winner-only CBT p-values for individual competitions.

    A test must strictly exceed its control maximum or receive p-value one.
    By default, control winners have weights 1/n_j and offset 1/n_min.

    Args:
        tests (array_like): Nonempty one-dimensional array of finite test scores.
        controls (sequence of array_like): One nonempty, one-dimensional array
            of finite control scores per test. Control counts may differ.
        equal_weights (bool): Use 1/n_min for both weights and offset, overriding
            individual_offset. Defaults to False.
        individual_offset (bool): Use offsets 1/n_j, as in the Bonferroni
            experiments, unless equal_weights is True. Defaults to False.

    Returns:
        numpy.ndarray: One p-value per test, in input order, capped at one.

    Raises:
        ValueError: If the score inputs fail validation in prepare.
    """
    tests, controls, n, maxima = prepare(tests, controls)
    wins = tests > maxima
    if equal_weights:
        counts = tail(maxima[~wins], np.ones((~wins).sum()), tests)
        return np.where(wins, np.minimum((1 + counts) / (len(tests) * n.min()), 1), 1)
    weights = 1 / n
    offset = 1 / n if individual_offset and not equal_weights else 1 / n.min()
    p = (offset + tail(maxima[~wins], weights[~wins], tests)) / len(tests)
    return np.where(wins, np.minimum(p, 1), 1)


def subsample(controls, rng):
    """Retain the minimum control count independently in each competition.

    Args:
        controls (list of numpy.ndarray): Nonempty list of nonempty,
            one-dimensional control-score arrays.
        rng (numpy.random.Generator): Generator used to sample uniformly
            without replacement; sampling advances its state.

    Returns:
        list of numpy.ndarray: One array per competition, each containing
        n_min scores. Arrays already of that size are returned unchanged,
        without copying or drawing random numbers.
    """
    n_min = min(map(len, controls))
    return [
        row if len(row) == n_min else row[rng.choice(len(row), n_min, replace=False)]
        for row in controls
    ]


def compound_empirical(tests, controls):
    """Return the compound empirical comparator for individual competitions.

    Augment each control row with its own test, evaluate its empirical upper
    tail at every test score (including ties), and average these proportions
    across competitions.

    Args:
        tests (array_like): Nonempty one-dimensional test-score array.
        controls (sequence of array_like): One one-dimensional control-score
            array per test. Control counts may differ.

    Returns:
        numpy.ndarray: One compound empirical p-value per test, in input
        order, capped at one.
    """
    values = np.zeros(len(tests))
    for test, row in zip(tests, controls):
        scores = np.sort(np.r_[test, row])
        values += (len(scores) - np.searchsorted(scores, tests)) / len(scores)
    return np.minimum(values / len(tests), 1)


def evalues(tests, controls, alpha, *, return_thresholds=False):
    """Construct individual-control CBT e-values using T_j^+ in both terms.

    For each test, force its competition to a control win to find T_j^+,
    then evaluate the e-value's numerator indicator and denominator tail
    at that same threshold.
    The final denominator uses the individual offset 1/n_j.

    Args:
        tests (array_like): Nonempty one-dimensional array of finite test scores.
        controls (sequence of array_like): One nonempty, one-dimensional array
            of finite control scores per test. Control counts may differ.
        alpha (float): Required prespecified construction level (tilde alpha)
            in (0, 1). The experiments explicitly use 0.1 for this and e-BH.
        return_thresholds (bool): Return the T_j^+ thresholds alongside the
            e-values. Defaults to False.

    Returns:
        numpy.ndarray or tuple: E-values in test order, or (values, thresholds)
        if return_thresholds is True. A missing threshold is infinity and
        gives e-value zero.

    Raises:
        ValueError: If the score inputs fail validation in prepare, there are
            within-competition ties, or alpha is outside (0, 1).
    """
    tests, controls, n, maxima = prepare(tests, controls)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    if any(len(np.unique(np.r_[t, c])) != len(c) + 1 for t, c in zip(tests, controls)):
        raise ValueError(
            "The e-value construction requires no within-competition ties."
        )
    wins = tests > maxima
    winning_scores = np.maximum(tests, maxima)
    candidates = np.unique(winning_scores)
    control_tail = tail(maxima[~wins], 1 / n[~wins], candidates)
    test_tail = tail(tests[wins], np.ones(wins.sum()), candidates)
    thresholds = np.full(len(tests), np.inf)
    values = np.zeros(len(tests))
    # Force each competition to a control win; both e-value terms use T_j^+.
    for j in range(len(tests)):
        active = winning_scores[j] >= candidates
        forced_controls = control_tail + wins[j] * active / n[j]
        forced_tests = np.maximum(1, test_tail - wins[j] * active)
        accepted = (1 / n.min() + forced_controls) <= alpha * forced_tests
        if accepted.any():
            k = np.flatnonzero(accepted)[0]
            thresholds[j] = candidates[k]
            if wins[j] and tests[j] >= thresholds[j]:
                values[j] = len(tests) / (1 / n[j] + control_tail[k])
    return (values, thresholds) if return_thresholds else values


def q_finalist(tests, controls, q):
    """Return homogeneous q-finalist CBT p-values in test order.

    The top q scores in each augmented competition are finalists. Pool control
    finalists with weight 1/n and use offset 1/n.

    Args:
        tests (array_like): One-dimensional test-score array of length m.
        controls (array_like): Control-score array of shape (m, n), with the
            same positive control count n for each test.
        q (int): Number of finalists per competition, between 1 and n inclusive.

    Returns:
        numpy.ndarray: One p-value per test, in input order, capped at one.
        Tests outside their competition's finalists receive p-value one.

    Raises:
        ValueError: If q is outside [1, n] or there are within-competition ties.
    """
    tests = np.asarray(tests)
    controls = np.asarray(controls)
    m, n = controls.shape
    if not 1 <= q <= n:
        raise ValueError("q must lie in [1, n].")
    scores = np.column_stack([tests, controls])
    ordered = np.sort(scores, axis=1)[:, ::-1]
    if np.any(np.diff(ordered, axis=1) == 0):
        raise ValueError("No within-competition ties are allowed.")
    eligible = controls >= ordered[:, q - 1, None]
    selected = controls[eligible]
    mass = tail(selected, np.full(len(selected), 1 / n), tests)
    return np.where(tests >= ordered[:, q - 1], np.minimum((1 / n + mass) / m, 1), 1)


def shared_cbt_pvalues(tests, controls, labels, *, random_hypotheses=False):
    """Return shared-control CBT p-values with offset max_g b_g/n_g.

    Here b_g counts tests and n_g counts supplied controls in block g.
    By default, each control maximum contributes weight equal to the number
    of tests it beats in its block divided by n_g. Counts always refer to the
    supplied controls; retaining counts from a larger, unfiltered calibration
    pool is implemented separately in applications/analyze_utopia.py.

    Args:
        tests (array_like): Nonempty one-dimensional test-score array.
        controls (mapping): Block labels mapped to nonempty, one-dimensional
            control-score arrays.
        labels (array_like): One block label per test, each present in controls.
        random_hypotheses (bool): Use weight b_g/n_g for each control maximum.
            Requires controls already restricted to nulls. Defaults to False.

    Returns:
        numpy.ndarray: One p-value per test, in input order, capped at one.
        Tests that do not strictly exceed their block's control maximum
        receive p-value one.

    Raises:
        ValueError: If a tested block has an empty control array.
        KeyError: If a test's block label is missing from the control mapping.
    """
    tests, labels = np.asarray(tests), np.asarray(labels)
    groups, counts = np.unique(labels, return_counts=True)
    if any(len(controls[g]) == 0 for g in groups):
        raise ValueError("Each tested block needs controls.")
    maxima = np.array([np.max(controls[g]) for g in groups])
    weights = np.array([b / len(controls[g]) for g, b in zip(groups, counts)])
    if random_hypotheses:
        # Here controls are already restricted to the null control pool.
        sources, source_weights = maxima, weights
    else:
        # Each shared control wins against every lower-scoring test in its block.
        source_weights = np.array(
            [
                np.sum(tests[labels == g] < c) / len(controls[g])
                for g, c in zip(groups, maxima)
            ]
        )
        sources = maxima
    mass = tail(sources, source_weights, tests)
    own_max = maxima[np.searchsorted(groups, labels)]
    return np.where(
        tests > own_max, np.minimum((weights.max() + mass) / len(tests), 1), 1
    )
