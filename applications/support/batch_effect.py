"""Manuscript workflow helpers, extracted from code/utopia_compound/batch_effect.py."""

from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Literal
import numpy as np

BatchCorrectionMagnitude = Literal["weak", "medium", "strong"]


@dataclass(frozen=True)
class UtopiaConformalResult:
    """Batch-corrected UTOPIA-style raw conformal p-values."""

    pvalues: np.ndarray
    corrected_test_yhat: np.ndarray
    tolerances: dict[int, float]
    cluster_summaries: tuple[dict[str, object], ...]


def _integer_clusters(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values).reshape(-1)
    finite = np.isfinite(values)
    labels = np.zeros(values.shape, dtype=np.int64)
    labels[finite] = values[finite].astype(np.int64)
    return (labels, finite)


def compute_utopia_batch_tolerance(
    *,
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    test_yhat: np.ndarray,
    null_number: float,
    correct_batch_effect_magnitude: float | BatchCorrectionMagnitude = "medium",
) -> float:
    """Compute the per-cluster tolerance used by UTOPIA inference."""
    calib_yhat = np.asarray(calib_yhat, dtype=float).reshape(-1)
    calib_y = np.asarray(calib_y, dtype=float).reshape(-1)
    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    if calib_yhat.size != calib_y.size:
        raise ValueError("calib_yhat and calib_y must have the same length.")
    if isinstance(correct_batch_effect_magnitude, Real):
        return float(correct_batch_effect_magnitude)
    if calib_yhat.size == 0 or test_yhat.size == 0:
        return 0.0
    quantile = float(np.sum(calib_y <= null_number) / len(calib_y))
    if quantile <= 0.1:
        return 0.0
    calib_sorted = np.sort(calib_yhat)
    test_sorted = np.sort(test_yhat)
    calib_q_threshold = np.quantile(calib_sorted, quantile)
    test_q_threshold = np.quantile(test_sorted, quantile)
    calib_first_q = calib_sorted[calib_sorted <= calib_q_threshold]
    test_first_q = test_sorted[test_sorted <= test_q_threshold]
    if calib_first_q.size == 0 or test_first_q.size == 0:
        return 0.0
    if correct_batch_effect_magnitude == "weak":
        quantiles = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    elif correct_batch_effect_magnitude == "medium":
        quantiles = np.array(
            [0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.995, 0.999]
        )
    elif correct_batch_effect_magnitude == "strong":
        quantiles = np.array(
            [
                0.001,
                0.005,
                0.01,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.99,
                0.995,
                0.999,
            ]
        )
    else:
        raise ValueError(
            "correct_batch_effect_magnitude must be a number or one of 'weak', 'medium', 'strong'."
        )
    calib_quantiles = np.quantile(calib_first_q, quantiles)
    test_quantiles = np.quantile(test_first_q, quantiles)
    quantile_shifts = np.clip(test_quantiles - calib_quantiles, a_min=0.0, a_max=None)
    return float(-np.max(quantile_shifts))


def compute_utopia_cluster_tolerances(
    *,
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    test_yhat: np.ndarray,
    calib_clusters: np.ndarray,
    test_clusters: np.ndarray,
    null_number: float,
    clusters: Iterable[int] | None = None,
    correct_batch_effect: bool = True,
    correct_batch_effect_magnitude: float | BatchCorrectionMagnitude = "medium",
    min_calib_pts: int | None = None,
) -> tuple[dict[int, float], tuple[dict[str, object], ...]]:
    """Compute UTOPIA tolerances for each H&E cluster."""
    calib_yhat = np.asarray(calib_yhat, dtype=float).reshape(-1)
    calib_y = np.asarray(calib_y, dtype=float).reshape(-1)
    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    calib_clusters = np.asarray(calib_clusters).reshape(-1)
    test_clusters = np.asarray(test_clusters).reshape(-1)
    if calib_yhat.size != calib_y.size or calib_yhat.size != calib_clusters.size:
        raise ValueError("calibration arrays must have the same length.")
    if test_yhat.size != test_clusters.size:
        raise ValueError("test_yhat and test_clusters must have the same length.")
    calib_cluster_int, calib_cluster_finite = _integer_clusters(calib_clusters)
    test_cluster_int, test_cluster_finite = _integer_clusters(test_clusters)
    if clusters is None:
        cluster_values = sorted(
            {int(cluster) for cluster in test_cluster_int[test_cluster_finite]}
        )
    else:
        cluster_values = [int(cluster) for cluster in clusters]
    tolerances: dict[int, float] = {}
    summaries: list[dict[str, object]] = []
    for cluster in cluster_values:
        calib_mask = (
            np.isfinite(calib_yhat)
            & np.isfinite(calib_y)
            & calib_cluster_finite
            & (calib_cluster_int == cluster)
        )
        test_mask = (
            np.isfinite(test_yhat) & test_cluster_finite & (test_cluster_int == cluster)
        )
        n_calib = int(calib_mask.sum())
        n_test = int(test_mask.sum())
        null_fraction = (
            float(np.sum(calib_y[calib_mask] <= null_number) / n_calib)
            if n_calib
            else 0.0
        )
        skipped = False
        reason: str | None = None
        if n_test == 0:
            skipped = True
            reason = "no_test_points"
            tolerance = 0.0
        elif n_calib == 0:
            skipped = True
            reason = "no_calibration_points"
            tolerance = 0.0
        elif min_calib_pts is not None and n_calib < int(min_calib_pts):
            skipped = True
            reason = "too_few_calibration_points"
            tolerance = 0.0
        elif not correct_batch_effect:
            tolerance = 0.0
        else:
            tolerance = compute_utopia_batch_tolerance(
                calib_yhat=calib_yhat[calib_mask],
                calib_y=calib_y[calib_mask],
                test_yhat=test_yhat[test_mask],
                null_number=null_number,
                correct_batch_effect_magnitude=correct_batch_effect_magnitude,
            )
        tolerances[cluster] = float(tolerance)
        summaries.append(
            {
                "cluster": int(cluster),
                "n_calib": n_calib,
                "n_test": n_test,
                "null_fraction": null_fraction,
                "tolerance": float(tolerance),
                "skipped": bool(skipped),
                "reason": reason,
            }
        )
    return (tolerances, tuple(summaries))


def apply_utopia_batch_correction(
    *, test_yhat: np.ndarray, test_clusters: np.ndarray, tolerances: dict[int, float]
) -> np.ndarray:
    """Return test predictions shifted by their cluster-specific UTOPIA tolerance."""
    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    test_clusters = np.asarray(test_clusters).reshape(-1)
    if test_yhat.size != test_clusters.size:
        raise ValueError("test_yhat and test_clusters must have the same length.")
    test_cluster_int, test_cluster_finite = _integer_clusters(test_clusters)
    corrected = test_yhat.copy()
    for cluster, tolerance in tolerances.items():
        corrected[test_cluster_finite & (test_cluster_int == int(cluster))] += float(
            tolerance
        )
    return corrected


def utopia_batch_corrected_conformal_pvalues(
    *,
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    test_yhat: np.ndarray,
    calib_clusters: np.ndarray,
    test_clusters: np.ndarray,
    null_number: float,
    clusters: Iterable[int] | None = None,
    correct_batch_effect: bool = True,
    correct_batch_effect_magnitude: float | BatchCorrectionMagnitude = "medium",
    min_calib_pts: int | None = None,
) -> UtopiaConformalResult:
    """Compute raw conformal p-values with UTOPIA's cluster batch correction."""
    calib_yhat = np.asarray(calib_yhat, dtype=float).reshape(-1)
    calib_y = np.asarray(calib_y, dtype=float).reshape(-1)
    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    calib_clusters = np.asarray(calib_clusters).reshape(-1)
    test_clusters = np.asarray(test_clusters).reshape(-1)
    calib_cluster_int, calib_cluster_finite = _integer_clusters(calib_clusters)
    test_cluster_int, test_cluster_finite = _integer_clusters(test_clusters)
    tolerances, summaries = compute_utopia_cluster_tolerances(
        calib_yhat=calib_yhat,
        calib_y=calib_y,
        test_yhat=test_yhat,
        calib_clusters=calib_clusters,
        test_clusters=test_clusters,
        null_number=null_number,
        clusters=clusters,
        correct_batch_effect=correct_batch_effect,
        correct_batch_effect_magnitude=correct_batch_effect_magnitude,
        min_calib_pts=min_calib_pts,
    )
    corrected_test_yhat = apply_utopia_batch_correction(
        test_yhat=test_yhat, test_clusters=test_clusters, tolerances=tolerances
    )
    pvalues = np.ones(test_yhat.size, dtype=float)
    for cluster in tolerances:
        calib_mask = (
            np.isfinite(calib_yhat)
            & np.isfinite(calib_y)
            & calib_cluster_finite
            & (calib_cluster_int == int(cluster))
        )
        test_mask = (
            np.isfinite(corrected_test_yhat)
            & test_cluster_finite
            & (test_cluster_int == int(cluster))
        )
        if not np.any(calib_mask) or not np.any(test_mask):
            continue
        null_scores = np.sort(calib_yhat[calib_mask & (calib_y <= null_number)])
        denom = int(calib_mask.sum()) + 1
        if null_scores.size == 0:
            pvalues[test_mask] = 1.0 / denom
            continue
        greater = null_scores.size - np.searchsorted(
            null_scores, corrected_test_yhat[test_mask], side="right"
        )
        pvalues[test_mask] = (greater.astype(float) + 1.0) / denom
    return UtopiaConformalResult(
        pvalues=pvalues,
        corrected_test_yhat=corrected_test_yhat,
        tolerances=tolerances,
        cluster_summaries=summaries,
    )
