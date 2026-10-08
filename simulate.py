"""Run the manuscript Gaussian-mixture experiments; all paths are local."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits

import cbt

ROOT = Path(__file__).resolve().parent
ETAS = np.arange(2, 6.01, 0.5)
SETTINGS = {
    "homogeneous_uniform": dict(n=25),
    "homogeneous_weighted": dict(n=25, weighted=True),
    "binary_increasing": dict(mode="binary", direction=1),
    "binary_decreasing": dict(mode="binary", direction=-1),
    "smooth_increasing": dict(mode="smooth", direction=1),
    "smooth_decreasing": dict(mode="smooth", direction=-1),
    "shared_2000": dict(shared=2000, m=1000, fraction=0.1),
    "shared_4000": dict(shared=4000, m=1000, fraction=0.1),
    "q_n50_m500": dict(n=50, q=True),
    "q_n500_m50": dict(n=500, m=50, q=True),
}


def fixed_environment():
    centers = np.random.default_rng(369).uniform(-3, 3, (25, 25))
    u = np.random.default_rng(963).standard_normal(25)
    return centers, (u - u.mean()) / u.std(ddof=0)


def metrics(rejected, truth):
    discoveries = int(rejected.sum())
    false = int(np.sum(rejected & ~truth))
    true = int(np.sum(rejected & truth))
    return dict(
        discoveries=discoveries,
        false=false,
        true=true,
        fdp=false / max(discoveries, 1),
        power=true / truth.sum(),
        fwer=int(false > 0),
    )


def replication(task):
    setting, seed, etas, methods = task
    spec = SETTINGS[setting]
    centers, u = fixed_environment()
    sigma = np.exp(0.25 * u)
    sequences = np.random.SeedSequence(seed).spawn(7)
    train_rng, calibration_rng, test_rng, control_rng, size_rng, _, sub_rng = [
        np.random.default_rng(s) for s in sequences
    ]
    m, fraction = spec.get("m", 500), spec.get("fraction", 0.2)
    train_labels = train_rng.integers(25, size=2000)
    train = centers[train_labels] + sigma[
        train_labels, None
    ] * train_rng.standard_normal((2000, 25))
    truth = np.arange(m) < round(m * fraction)
    if spec.get("shared"):
        # Preserve the original shared-control label stream, including alternative order.
        labels = test_rng.integers(25, size=m)
        labels[truth] = test_rng.choice(np.argsort(u), truth.sum(), replace=True)
    else:
        labels = test_rng.choice(25, m, p=np.full(25, 1 / 25))
        weights = np.exp(u) if spec.get("weighted") else np.ones(25)
        labels[truth] = test_rng.choice(25, truth.sum(), p=weights / weights.sum())
    noise = test_rng.standard_normal((m, 25))
    rows = []

    def record(eta, method, p=None, e=None):
        procedures = (
            ("eBH",)
            if e is not None
            else (
                ("BH",) if spec.get("shared") or spec.get("q") else ("BH", "Bonferroni")
            )
        )
        for procedure in procedures:
            if procedure == "eBH":
                rejected = cbt.ebh(e, 0.1)
            elif procedure == "BH":
                rejected = cbt.bh(p, 0.1)
            else:
                rejected = p <= 0.2 / m * (1 + 8 * np.finfo(float).eps)
            rows.append(
                dict(
                    setting=setting,
                    seed=seed,
                    eta=eta,
                    method=method,
                    procedure=procedure,
                    **metrics(rejected, truth),
                )
            )

    with threadpool_limits(limits=1):
        model = OneClassSVM().fit(train)
        score = lambda x: -model.decision_function(x)
        if spec.get("shared"):
            nc = spec["shared"]
            cal_labels = calibration_rng.integers(25, size=nc)
            calibration = centers[cal_labels] + sigma[
                cal_labels, None
            ] * calibration_rng.standard_normal((nc, 25))
            cal_scores = score(calibration)
            kmeans = KMeans(n_clusters=25, n_init=10, random_state=seed).fit(train)
            counts = np.bincount(kmeans.labels_, minlength=25)
            stable = np.flatnonzero(counts >= 20)
            if not len(stable):
                stable = np.array([counts.argmax()])
            mapping = np.array(
                [
                    stable[
                        np.argmin(
                            np.sum(
                                (kmeans.cluster_centers_[stable] - center) ** 2, axis=1
                            )
                        )
                    ]
                    for center in kmeans.cluster_centers_
                ]
            )
            cal_learned = mapping[kmeans.predict(calibration)]
        else:
            if spec.get("mode") == "binary":
                high = u[labels] >= np.median(u)
                n = np.where(high if spec["direction"] == 1 else ~high, 50, 10)
            elif spec.get("mode") == "smooth":
                n = 10 + size_rng.binomial(
                    40, 1 / (1 + np.exp(-spec["direction"] * u[labels]))
                )
            else:
                n = np.full(m, spec["n"])
            repeated = np.repeat(labels, n)
            controls = centers[repeated] + sigma[
                repeated, None
            ] * control_rng.standard_normal((n.sum(), 25))
            flat_scores = score(controls)
            controls = np.split(flat_scores, np.cumsum(n)[:-1])
            # Retain the same independently sampled indices throughout the eta grid.
            sub_seed = int(sequences[6].generate_state(1, dtype=np.uint64)[0])
            if methods == "supplementary":
                subsampled = cbt.subsample(controls, np.random.default_rng(sub_seed))

        for eta in etas:
            scale = sigma[labels] * np.where(truth, np.sqrt(eta), 1)
            tests_x = centers[labels] + scale[:, None] * noise
            tests = score(tests_x)
            if spec.get("shared"):
                record(eta, "Pooled empirical", cbt.empirical(tests, cal_scores))
                for kind, test_labels, control_labels in (
                    ("Oracle", labels, cal_labels),
                    ("Estimated", mapping[kmeans.predict(tests_x)], cal_learned),
                ):
                    blocks = {
                        g: cal_scores[control_labels == g]
                        for g in np.unique(test_labels)
                    }
                    # A tested block without controls is outside this experiment design.
                    valid = np.array([len(blocks[g]) > 0 for g in test_labels])
                    p = np.ones(m)
                    if valid.all():
                        p = cbt.shared_cbt_pvalues(tests, blocks, test_labels)
                    else:
                        raise RuntimeError(
                            "Every tested shared block needs calibration controls."
                        )
                    record(eta, f"CBT ({kind})", p)
                    empirical = np.ones(m)
                    for g in blocks:
                        ix = test_labels == g
                        empirical[ix] = cbt.empirical(tests[ix], blocks[g])
                    record(eta, f"Block empirical ({kind})", empirical)
            elif spec.get("q"):
                for q in (1, 2, 5, 10, 25):
                    record(eta, f"q={q}", cbt.q_finalist(tests, controls, q))
            else:
                if methods == "main":
                    baseline = {
                        "Pooled empirical": cbt.empirical(tests, flat_scores),
                        "Individual empirical": np.array(
                            [cbt.empirical(t, c) for t, c in zip(tests, controls)]
                        ),
                        "Compound empirical": cbt.compound_empirical(tests, controls),
                    }
                    for method, p in baseline.items():
                        record(eta, method, p)
                    p_bh = cbt.cbt_pvalues(tests, controls)
                    p_bonf = cbt.cbt_pvalues(tests, controls, individual_offset=True)
                    rows.append(
                        dict(
                            setting=setting,
                            seed=seed,
                            eta=eta,
                            method="CBT",
                            procedure="BH",
                            **metrics(cbt.bh(p_bh, alpha=0.1), truth),
                        )
                    )
                    rows.append(
                        dict(
                            setting=setting,
                            seed=seed,
                            eta=eta,
                            method="CBT",
                            procedure="Bonferroni",
                            **metrics(
                                p_bonf <= 0.2 / m * (1 + 8 * np.finfo(float).eps), truth
                            ),
                        )
                    )
                else:
                    record(
                        eta,
                        "Equal-weight CBT",
                        cbt.cbt_pvalues(tests, controls, equal_weights=True),
                    )
                    record(
                        eta,
                        "Subsampled CBT",
                        cbt.cbt_pvalues(tests, subsampled, equal_weights=True),
                    )
                    record(eta, "CBT e-values", e=cbt.evalues(tests, controls, alpha=0.1))
    return rows


def summarize(data):
    group = ["setting", "eta", "method", "procedure"]
    out = data.groupby(group).agg(replicates=("seed", "count"))
    for metric in ("fdp", "power", "fwer", "discoveries"):
        out[metric] = data.groupby(group)[metric].mean()
        out[metric + "_se"] = data.groupby(group)[metric].sem()
    return out.reset_index()


def main(methods):
    supported_settings = [
        name
        for name, spec in SETTINGS.items()
        if methods == "main" or not (spec.get("shared") or spec.get("q"))
    ]
    parser = argparse.ArgumentParser(
        description=(
            "Recompute equal-weight CBT, subsampled CBT, and CBT (e-BH) only."
            if methods == "supplementary"
            else "Recompute original main-text methods and original Appendix H curves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="See docs/simulations.md for the designs, output schema, and plotting commands.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=supported_settings,
        default=supported_settings,
        help="Experiment settings to run; each setting is saved separately.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1000,
        help="Replications per setting (at least 2).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Worker processes; each uses one numerical-library thread.",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0, help="First replication seed."
    )
    parser.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=ETAS.tolist(),
        help="Alternative variance multipliers (each greater than 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/reproduced" / methods,
        help="Directory for raw results, summaries, configuration, and fixed environment.",
    )
    args = parser.parse_args()
    if args.replicates < 2 or args.jobs < 1 or any(e <= 1 for e in args.etas):
        parser.error("Use at least two replicates, positive jobs, and eta > 1.")
    args.methods = methods
    args.output.mkdir(parents=True, exist_ok=True)
    centers, u = fixed_environment()
    np.savez(args.output / "environment.npz", centers=centers, u=u)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update(
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        tau=0.25,
        n_train=2000,
        fdr_alpha=0.1,
        fwer_alpha=0.2,
        offset_standardization="population (ddof=0), preserving original code",
        evalue_definition="T_j^+ in numerator and denominator; final offset 1/n_j",
    )
    config["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "cbt.py", Path(__file__))
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.monotonic()
    # Each completed setting is saved, so a later failure does not erase results.
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for setting in args.settings:
            records = []
            tasks = (
                (setting, seed, args.etas, args.methods)
                for seed in range(args.seed_offset, args.seed_offset + args.replicates)
            )
            for i, result in enumerate(pool.map(replication, tasks), 1):
                records.extend(result)
                if i % 50 == 0 or i == args.replicates:
                    print(
                        f"{setting}: {i}/{args.replicates}; elapsed {time.monotonic()-start:.0f}s",
                        flush=True,
                    )
            data = pd.DataFrame(records)
            data.to_csv(
                args.output / f"{setting}.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )
            summarize(data).to_csv(args.output / f"{setting}_summary.csv", index=False)
    print("Completed all requested settings.", flush=True)
