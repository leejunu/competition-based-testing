"""TLS meta-gene comparisons at scales 1, 2 and 4."""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cbt
from support.batch_effect import utopia_batch_corrected_conformal_pvalues
from support.arc import build_adaptive_cem_compound_inputs
from support.matching import build_optmatch_shared_compound_inputs
from support.features import (
    load_cluster_pca_features,
    build_matching_covariates,
    build_fixed_knn_compound_inputs,
)
from support.crispr_metrics import bh_adjusted


def curves(p, truth, method, scale, labels=None):
    adjusted = bh_adjusted(p)
    if labels is not None:
        for g in np.unique(labels):
            ix = labels == g
            adjusted[ix] = bh_adjusted(p[ix])
    rows = []
    for alpha in np.arange(1, 101) * 0.005:
        rejected = adjusted <= alpha
        true = int(np.sum(rejected & truth))
        false = int(np.sum(rejected & ~truth))
        rows.append(
            dict(
                scale=scale,
                method=method,
                alpha=alpha,
                discoveries=int(rejected.sum()),
                true=true,
                false=false,
                fdp=false / max(1, rejected.sum()),
                hypotheses=len(p),
            )
        )
    return rows


def analyze(cache, data_dir, scale, output):
    with np.load(cache, allow_pickle=False) as f:
        scores = {key: f[key] for key in f.files}
    test_features, calib_features = load_cluster_pca_features(
        data_dir=data_dir, score_cache=scores, scale=scale, batch_size=100000
    )
    test_cov, calib_cov, names, info = build_matching_covariates(
        dim=10,
        score_cache=scores,
        test_cluster=test_features,
        calib_cluster=calib_features,
        eligible_clusters=tuple(range(15)),
        null_number=0.05,
        include_spatial_coords=False,
    )
    common = dict(
        test_yhat=scores["test_yhat"],
        test_y=scores["test_y"],
        test_clusters=scores["test_clusters"],
        test_covariates=test_cov,
        calib_yhat=scores["calib_yhat"],
        calib_y=scores["calib_y"],
        calib_clusters=scores["calib_clusters"],
        calib_covariates=calib_cov,
        null_number=0.05,
        eligible_clusters=tuple(range(15)),
        covariate_names=names,
        control_pool="all",
    )
    rows = []
    # Matching uses histology covariates; CBT uses the original, uncorrected scores.
    builders = {
        "CBT (ARC)": lambda: build_adaptive_cem_compound_inputs(
            **common,
            target_controls=15,
            min_controls=None,
            max_depth=18,
            guided_ratio=None,
        ),
        "CBT (GG)": lambda: build_optmatch_shared_compound_inputs(
            **common,
            max_block_ratio=None,
            controls_per_block=10,
            caliper=None,
            knn_controls=128,
            assignment_method="load_balanced",
            load_balance_penalty=0.0,
            soft_max_block_ratio=5.0,
            soft_max_penalty=0.2,
            soft_max_power=2.0,
        ),
        "CBT (one control)": lambda: build_fixed_knn_compound_inputs(
            **{**common, "control_pool": "null"},
            controls_per_test=1,
            knn_candidates=128,
            query_batch_size=25000,
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    for method, build in builders.items():
        print(f"scale {scale}: {method}", flush=True)
        inputs = build()
        # Offsets use all assigned controls; only null controls enter the tail.
        groups, counts = np.unique(inputs.idx_to_block, return_counts=True)
        maxima = np.array(
            [
                np.max(
                    inputs.control_scores_by_block[g][
                        inputs.control_null_membership[g]
                    ],
                    initial=-np.inf,
                )
                for g in groups
            ]
        )
        weights = np.array(
            [b / len(inputs.control_scores_by_block[g]) for g, b in zip(groups, counts)]
        )
        mass = cbt.tail(maxima, weights, inputs.test_scores)
        p = np.minimum((weights.max() + mass) / len(inputs.test_scores), 1)
        p[inputs.test_scores < maxima[np.searchsorted(groups, inputs.idx_to_block)]] = 1
        rows.extend(curves(p, inputs.test_truth, method, scale))
        np.savez_compressed(
            output / f'scale{scale}_{method.split("(")[1][:-1].replace(" ", "_")}.npz',
            pvalues=p,
            test_indices=inputs.matched_test_indices,
            truth=inputs.test_truth,
            block=inputs.idx_to_block,
        )
    # UTOPIA alone applies its original per-cluster batch correction.
    baseline = utopia_batch_corrected_conformal_pvalues(
        **{
            key: scores[key]
            for key in [
                "test_yhat",
                "calib_yhat",
                "calib_y",
                "test_clusters",
                "calib_clusters",
            ]
        },
        null_number=0.05,
        clusters=range(15),
        correct_batch_effect=True,
        correct_batch_effect_magnitude="medium",
        min_calib_pts=None,
    )
    truth = scores["test_y"] > 0.05
    for method, labels in [
        ("UTOPIA (global)", None),
        ("UTOPIA (within-cluster)", scores["test_clusters"]),
    ]:
        rows.extend(curves(baseline.pvalues, truth, method, scale, labels))
    pd.DataFrame(rows).to_csv(output / f"scale{scale}_curves.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/utopia/demo")
    parser.add_argument("--scores", type=Path, default=ROOT / "results/utopia/scores")
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "results/utopia/analysis")
    args = parser.parse_args()
    for scale in args.scales:
        candidates = sorted(
            args.scores.rglob(f"score_cache_tls_meta_scale{scale}*.npz")
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one score cache for scale {scale}; found {len(candidates)}"
            )
        analyze(candidates[0], args.data_dir, scale, args.output)


if __name__ == "__main__":
    main()
