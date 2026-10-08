"""Manuscript workflow helpers, extracted from scripts/crispr_manuscript_metrics_experiment.py."""

from __future__ import annotations
import numpy as np
import pandas as pd

ALPHAS = np.round(np.arange(1, 41, dtype=float) * 0.005, 3)
METHOD_LABELS = {
    "CEM CBT q=1": "CBT (ARC)",
    "Optmatch CBT q=1": "CBT (GG)",
    "Across-cluster conformal": "Dempster (global)",
    "Within-cell conformal": "Dempster (within-cell)",
}


def bh_adjusted(pvalues: np.ndarray) -> np.ndarray:
    """Ordinary BH adjusted p-values for one global family."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order]
    ranks = np.arange(1, ranked.size + 1, dtype=float)
    adjusted_sorted = np.minimum.accumulate((ranked * ranked.size / ranks)[::-1])[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def task_curves(data: pd.DataFrame, task: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, subset in data.groupby("method", sort=False):
        subset = subset.copy()
        if method == "Within-cell conformal":
            # Form the union of ordinary BH rejection sets across cell lines.
            subset["bh_qvalue"] = subset.groupby("block")["pvalue"].transform(
                lambda values: bh_adjusted(values.to_numpy())
            )
        else:
            subset["bh_qvalue"] = bh_adjusted(subset["pvalue"].to_numpy())
        qvalues = subset["bh_qvalue"].to_numpy()
        known_null = subset["truth"].eq("null").to_numpy()
        known_positive = subset["truth"].eq("positive").to_numpy()
        unknown = ~(known_null | known_positive)
        order = np.argsort(qvalues, kind="mergesort")
        sorted_qvalues = qvalues[order]
        cumulative_null = np.concatenate([[0], np.cumsum(known_null[order])])
        cumulative_positive = np.concatenate([[0], np.cumsum(known_positive[order])])
        cumulative_unknown = np.concatenate([[0], np.cumsum(unknown[order])])
        rejection_counts = np.searchsorted(sorted_qvalues, ALPHAS, side="right")
        for alpha, num_rejections in zip(ALPHAS, rejection_counts):
            num_rejections = int(num_rejections)
            known_false_rejections = int(cumulative_null[num_rejections])
            known_true_rejections = int(cumulative_positive[num_rejections])
            unknown_rejections = int(cumulative_unknown[num_rejections])
            denominator = max(num_rejections, 1)
            rows.append(
                {
                    "task": task,
                    "method": METHOD_LABELS[method],
                    "method_raw": method,
                    "alpha": float(alpha),
                    "num_hypotheses": int(len(subset)),
                    "num_known_nulls": int(known_null.sum()),
                    "num_known_positives": int(known_positive.sum()),
                    "num_unknown": int(unknown.sum()),
                    "num_rejections": num_rejections,
                    "known_false_rejections": known_false_rejections,
                    "known_true_rejections": known_true_rejections,
                    "unknown_rejections": unknown_rejections,
                    "global_null_fdp": (
                        float(num_rejections > 0) if task == "global_null" else np.nan
                    ),
                    "kFDR": (
                        known_false_rejections / denominator
                        if task == "unrestricted"
                        else np.nan
                    ),
                    "known_rejections": (
                        known_true_rejections if task == "unrestricted" else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)
