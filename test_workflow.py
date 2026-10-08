"""Small mathematical checks and guards for the two reproduction paths."""

import importlib.util
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

import cbt
import simulate


def literal_evalues(tests, controls, alpha):
    n = np.array(list(map(len, controls)))
    maximum = np.array([max(t, max(c)) for t, c in zip(tests, controls)])
    wins = np.array([t > max(c) for t, c in zip(tests, controls)])
    values, thresholds = [], []
    for j in range(len(tests)):
        threshold = np.inf
        for t in sorted(set(maximum)):
            numerator = (
                1 / min(n)
                + (maximum[j] >= t) / n[j]
                + sum(
                    (not wins[k]) * (maximum[k] >= t) / n[k]
                    for k in range(len(tests))
                    if k != j
                )
            )
            denominator = max(
                1, sum(wins[k] * (maximum[k] >= t) for k in range(len(tests)) if k != j)
            )
            if numerator / denominator <= alpha:
                threshold = t
                break
        charge = 1 / n[j] + sum(
            (not wins[k]) * (maximum[k] >= threshold) / n[k] for k in range(len(tests))
        )
        values.append(len(tests) * (wins[j] and tests[j] >= threshold) / charge)
        thresholds.append(threshold)
    return np.array(values), np.array(thresholds)


class InferenceTests(unittest.TestCase):
    def test_evalues_match_formula(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            tests = rng.normal(size=9)
            controls = [rng.normal(size=rng.integers(1, 8)) for _ in tests]
            for alpha in [0.1, 0.3, 0.7]:
                actual = cbt.evalues(tests, controls, alpha, return_thresholds=True)
                expected = literal_evalues(tests, controls, alpha)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b)

    def test_finite_null_orbit(self):
        competitions = [
            np.array([0.2, 2.1]),
            np.array([0.8, 1.4, 3.5]),
            np.array([-0.4, 0.1, 2.8, 4.9]),
        ]
        total = 0
        count = 0
        for choice in itertools.product(*(range(len(c)) for c in competitions)):
            tests = np.array([c[i] for c, i in zip(competitions, choice)])
            controls = [np.delete(c, i) for c, i in zip(competitions, choice)]
            e = cbt.evalues(tests, controls, 0.4)
            total += e.sum()
            count += 1
            self.assertTrue(
                np.all(~cbt.ebh(e, 0.4) | cbt.bh(cbt.cbt_pvalues(tests, controls), 0.4))
            )
        self.assertLessEqual(total / count, 3 + 1e-12)

    def test_subsampling_and_homogeneous_equivalence(self):
        rng = np.random.default_rng(36)
        tests = rng.normal(size=20)
        controls = [rng.normal(size=12) for _ in tests]
        np.testing.assert_allclose(
            cbt.cbt_pvalues(tests, controls),
            cbt.cbt_pvalues(tests, controls, equal_weights=True),
        )
        for a, b in zip(controls, cbt.subsample(controls, rng)):
            np.testing.assert_array_equal(a, b)
        ragged = [np.arange(n) for n in [3, 5, 7]]
        a = cbt.subsample(ragged, np.random.default_rng(1))
        b = cbt.subsample(ragged, np.random.default_rng(1))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(len(set(x)), 3)

    def test_supplementary_does_not_run_main_methods(self):
        original = cbt.cbt_pvalues

        def supplementary_pvalues(*args, **kwargs):
            self.assertTrue(kwargs.get("equal_weights"))
            return original(*args, **kwargs)

        with patch.object(
            cbt,
            "empirical",
            side_effect=AssertionError("Main-text baseline was recomputed"),
        ), patch.object(
            cbt,
            "compound_empirical",
            side_effect=AssertionError("Main-text baseline was recomputed"),
        ), patch.object(
            cbt, "cbt_pvalues", side_effect=supplementary_pvalues
        ):
            rows = simulate.replication(
                ("binary_increasing", 0, [2.0], "supplementary")
            )
        self.assertEqual(
            {r["method"] for r in rows},
            {"Equal-weight CBT", "Subsampled CBT", "CBT e-values"},
        )
        self.assertEqual(len(rows), 5)

    def test_crispr_global_and_within_cell_bh_are_distinct(self):
        sys.path.insert(0, str(Path(__file__).parent / "applications"))
        import pandas as pd
        from support.crispr_metrics import task_curves

        base = pd.DataFrame(
            {
                "block": ["a", "a", "b", "b"],
                "pvalue": [0.04, 0.04, 0.9, 0.9],
                "truth": ["positive"] * 4,
            }
        )
        data = pd.concat(
            [
                base.assign(method="Across-cluster conformal"),
                base.assign(method="Within-cell conformal"),
            ],
            ignore_index=True,
        )
        result = task_curves(data, "unrestricted")
        result = result[result.alpha == 0.05].set_index("method")
        self.assertEqual(result.loc["Dempster (global)", "num_rejections"], 0)
        self.assertEqual(result.loc["Dempster (within-cell)", "num_rejections"], 2)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "Optional application dependencies are not installed",
    )
    def test_matching_helpers_and_training_roundtrip(self):
        sys.path.insert(0, str(Path(__file__).parent / "applications"))
        from support.arc import build_adaptive_cem_compound_inputs
        from support.matching import build_optmatch_shared_compound_inputs
        from support.models import train_regressor, save_checkpoint
        from support.scoring import load_model_from_checkpoint
        import tempfile
        import torch

        rng = np.random.default_rng(7)
        common = dict(
            test_yhat=rng.normal(size=12),
            test_y=np.zeros(12),
            test_clusters=np.zeros(12),
            test_covariates=rng.normal(size=(12, 3)),
            calib_yhat=rng.normal(size=30),
            calib_y=np.zeros(30),
            calib_clusters=np.zeros(30),
            calib_covariates=rng.normal(size=(30, 3)),
            null_number=0.05,
            eligible_clusters=(0,),
            covariate_names=("a", "b", "c"),
            control_pool="null",
        )
        for inputs in [
            build_adaptive_cem_compound_inputs(
                **common,
                target_controls=5,
                min_controls=None,
                max_depth=18,
                guided_ratio=None
            ),
            build_optmatch_shared_compound_inputs(
                **common, max_block_ratio=None, controls_per_block=5
            ),
        ]:
            p = cbt.shared_cbt_pvalues(
                inputs.test_scores,
                inputs.control_scores_by_block,
                inputs.idx_to_block,
                random_hypotheses=True,
            )
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
        x = rng.normal(size=(16, 4)).astype("float32")
        y = rng.uniform(size=(16, 2)).astype("float32")
        model, history = train_regressor(x, y, device="cpu", epochs=1, batch_size=8)
        self.assertTrue(np.isfinite(history["train_loss"]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                model_kind="regressor",
                metadata={
                    "model_params": {
                        "input_dim": 4,
                        "output_dim": 2,
                        "dropout_rate": 0.1,
                    }
                },
            )
            loaded, _ = load_model_from_checkpoint(path)
            model.eval()
            with torch.no_grad():
                np.testing.assert_allclose(
                    model(torch.from_numpy(x)).numpy(),
                    loaded(torch.from_numpy(x)).numpy(),
                )


if __name__ == "__main__":
    unittest.main()
