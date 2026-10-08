"""Manuscript workflow helpers, extracted from code/utopia_compound/foundational_matching_sweep.py."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from support.matching import OptmatchSharedCompoundInputs
from support.features import standardize_within_cluster


def ordered_eligible_labels(
    test_clusters: np.ndarray, test_mask: np.ndarray
) -> list[int]:
    ordered: list[int] = []
    for label in test_clusters[test_mask].astype(int):
        if int(label) not in ordered:
            ordered.append(int(label))
    return ordered


def adaptive_leaf_blocks(
    *,
    test_features: np.ndarray,
    control_features: np.ndarray,
    target_controls: int,
    min_controls: int | None,
    max_depth: int,
    guided_ratio: float | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct paper-ARC leaves within one exact cluster."""
    leaves: list[tuple[np.ndarray, np.ndarray]] = []
    test_all = np.arange(test_features.shape[0], dtype=np.int64)
    control_all = np.arange(control_features.shape[0], dtype=np.int64)
    effective_min_controls = 1 if min_controls is None else int(min_controls)

    def child_counts(
        indices_test: np.ndarray, indices_control: np.ndarray, dim: int, split: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_test = indices_test[test_features[indices_test, dim] <= split]
        right_test = indices_test[test_features[indices_test, dim] > split]
        left_control = indices_control[control_features[indices_control, dim] <= split]
        right_control = indices_control[control_features[indices_control, dim] > split]
        return (left_test, right_test, left_control, right_control)

    def valid_child(t_idx: np.ndarray, c_idx: np.ndarray) -> bool:
        return t_idx.size == 0 or c_idx.size >= effective_min_controls

    def split(
        indices_test: np.ndarray, indices_control: np.ndarray, depth: int
    ) -> None:
        if indices_test.size == 0 or indices_control.size == 0:
            return
        ratio = indices_test.size / max(indices_control.size, 1)
        small_enough = indices_control.size <= target_controls
        ratio_ok = guided_ratio is None or ratio <= guided_ratio
        if depth >= max_depth or (small_enough and ratio_ok):
            leaves.append(
                (indices_test.astype(np.int64), indices_control.astype(np.int64))
            )
            return
        pooled = np.vstack(
            [test_features[indices_test], control_features[indices_control]]
        )
        spans = np.ptp(pooled, axis=0)
        candidate_dims = np.argsort(spans)[::-1]
        best: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for dim in candidate_dims[: min(8, candidate_dims.size)]:
            values = pooled[:, dim]
            if not np.isfinite(values).all() or float(np.ptp(values)) <= 1e-12:
                continue
            for quantile in [0.5, 0.4, 0.6, 0.33, 0.67]:
                threshold = float(np.quantile(values, quantile))
                left_test, right_test, left_control, right_control = child_counts(
                    indices_test, indices_control, int(dim), threshold
                )
                if (
                    left_test.size + left_control.size == 0
                    or right_test.size + right_control.size == 0
                ):
                    continue
                if not valid_child(left_test, left_control) or not valid_child(
                    right_test, right_control
                ):
                    continue
                if (
                    left_test.size == indices_test.size
                    and left_control.size == indices_control.size
                ):
                    continue
                if (
                    right_test.size == indices_test.size
                    and right_control.size == indices_control.size
                ):
                    continue
                child_ratios = [
                    (
                        left_test.size / max(left_control.size, 1)
                        if left_test.size
                        else 0.0
                    ),
                    (
                        right_test.size / max(right_control.size, 1)
                        if right_test.size
                        else 0.0
                    ),
                ]
                balance = abs(
                    left_control.size
                    + left_test.size
                    - (right_control.size + right_test.size)
                )
                score = max(child_ratios) + 1e-06 * balance
                candidate = (
                    float(score),
                    left_test,
                    right_test,
                    left_control,
                    right_control,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            leaves.append(
                (indices_test.astype(np.int64), indices_control.astype(np.int64))
            )
            return
        _, left_test, right_test, left_control, right_control = best
        split(left_test, left_control, depth + 1)
        split(right_test, right_control, depth + 1)

    split(test_all, control_all, 0)
    return leaves


def build_adaptive_cem_compound_inputs(
    *,
    test_yhat: np.ndarray,
    test_y: np.ndarray,
    test_clusters: np.ndarray,
    test_covariates: np.ndarray,
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    calib_clusters: np.ndarray,
    calib_covariates: np.ndarray,
    null_number: float,
    eligible_clusters: Sequence[int],
    target_controls: int,
    min_controls: int | None,
    max_depth: int,
    guided_ratio: float | None,
    covariate_names: tuple[str, ...],
    control_pool: str = "null",
) -> OptmatchSharedCompoundInputs:
    """Build shared-control inputs with paper ARC."""
    from scipy.spatial import cKDTree

    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    test_y = np.asarray(test_y, dtype=float).reshape(-1)
    test_clusters = np.asarray(test_clusters).reshape(-1).astype(int)
    test_covariates = np.asarray(test_covariates, dtype=float)
    calib_yhat = np.asarray(calib_yhat, dtype=float).reshape(-1)
    calib_y = np.asarray(calib_y, dtype=float).reshape(-1)
    calib_clusters = np.asarray(calib_clusters).reshape(-1).astype(int)
    calib_covariates = np.asarray(calib_covariates, dtype=float)
    if control_pool not in {"null", "all"}:
        raise ValueError("control_pool must be 'null' or 'all'.")
    effective_min_controls = 1 if min_controls is None else int(min_controls)
    eligible = np.asarray(tuple((int(x) for x in eligible_clusters)), dtype=int)
    test_mask = (
        np.isfinite(test_yhat)
        & np.isfinite(test_clusters)
        & (test_clusters >= 0)
        & np.isin(test_clusters, eligible)
        & np.all(np.isfinite(test_covariates), axis=1)
    )
    calib_mask = (
        np.isfinite(calib_yhat)
        & np.isfinite(calib_y)
        & np.isfinite(calib_clusters)
        & (calib_clusters >= 0)
        & np.isin(calib_clusters, eligible)
        & np.all(np.isfinite(calib_covariates), axis=1)
    )
    if control_pool == "null":
        calib_mask &= calib_y <= float(null_number)
    kept_test_parts: list[np.ndarray] = []
    kept_block_parts: list[np.ndarray] = []
    kept_distance_parts: list[np.ndarray] = []
    control_scores: dict[int, np.ndarray] = {}
    control_membership: dict[int, np.ndarray] = {}
    matched_test_by_block: dict[int, np.ndarray] = {}
    matched_control_by_block: dict[int, np.ndarray] = {}
    matched_distances_by_block: dict[int, np.ndarray] = {}
    block_labels: list[int] = []
    controls_per_block: list[int] = []
    tests_per_block: list[int] = []
    summaries: list[dict[str, object]] = []
    for raw_label in ordered_eligible_labels(test_clusters, test_mask):
        cluster_test_pos = np.flatnonzero(test_mask & (test_clusters == raw_label))
        cluster_control_pos = np.flatnonzero(calib_mask & (calib_clusters == raw_label))
        summary: dict[str, object] = {
            "raw_cluster": int(raw_label),
            "available_test": int(cluster_test_pos.size),
            "available_controls": int(cluster_control_pos.size),
            "matched_test": 0,
            "unmatched_test": int(cluster_test_pos.size),
            "used_controls": 0,
            "unused_controls": int(cluster_control_pos.size),
            "matching_method": "adaptive_cem",
            "target_controls": int(target_controls),
            "min_controls": None if min_controls is None else int(min_controls),
            "effective_min_controls": int(effective_min_controls),
            "max_depth": int(max_depth),
            "guided_ratio": None if guided_ratio is None else float(guided_ratio),
            "control_pool": control_pool,
        }
        if (
            cluster_test_pos.size == 0
            or cluster_control_pos.size < effective_min_controls
        ):
            summaries.append(summary)
            continue
        test_features, control_features = standardize_within_cluster(
            test_covariates[cluster_test_pos], calib_covariates[cluster_control_pos]
        )
        leaves = adaptive_leaf_blocks(
            test_features=test_features,
            control_features=control_features,
            target_controls=target_controls,
            min_controls=effective_min_controls,
            max_depth=max_depth,
            guided_ratio=guided_ratio,
        )
        matched_test_locals: list[np.ndarray] = []
        matched_control_locals: list[np.ndarray] = []
        distances_all: list[np.ndarray] = []
        for test_local, control_local in leaves:
            if test_local.size == 0 or control_local.size == 0:
                continue
            block_id = len(block_labels)
            test_pos = cluster_test_pos[test_local]
            control_pos = cluster_control_pos[control_local]
            nearest_distances, _ = cKDTree(control_features[control_local]).query(
                test_features[test_local], k=1
            )
            nearest_distances = np.asarray(nearest_distances, dtype=float)
            block_labels.append(int(raw_label))
            control_scores[block_id] = calib_yhat[control_pos].astype(float)
            control_membership[block_id] = (
                np.ones(control_pos.size, dtype=bool)
                if control_pool == "null"
                else (calib_y[control_pos] <= float(null_number)).astype(bool)
            )
            matched_test_by_block[block_id] = test_pos.astype(np.int64)
            matched_control_by_block[block_id] = control_pos.astype(np.int64)
            matched_distances_by_block[block_id] = nearest_distances.astype(float)
            kept_test_parts.append(test_pos.astype(np.int64))
            kept_block_parts.append(np.full(test_pos.size, block_id, dtype=np.int64))
            kept_distance_parts.append(nearest_distances.astype(float))
            controls_per_block.append(int(control_pos.size))
            tests_per_block.append(int(test_pos.size))
            matched_test_locals.append(test_local.astype(np.int64))
            matched_control_locals.append(control_local.astype(np.int64))
            distances_all.append(nearest_distances.astype(float))
        if matched_test_locals:
            mt = np.unique(np.concatenate(matched_test_locals))
            mc = np.unique(np.concatenate(matched_control_locals))
            distances = np.concatenate(distances_all)
            block_ratios = np.asarray(
                [
                    t / c
                    for t, c in zip(
                        tests_per_block[-len(leaves) :],
                        controls_per_block[-len(leaves) :],
                    )
                ]
            )
            summary.update(
                {
                    "matched_test": int(mt.size),
                    "unmatched_test": int(cluster_test_pos.size - mt.size),
                    "used_controls": int(mc.size),
                    "unused_controls": int(cluster_control_pos.size - mc.size),
                    "shared_blocks": int(len(matched_test_locals)),
                    "tests_per_block_median": float(
                        np.median([x.size for x in matched_test_locals])
                    ),
                    "tests_per_block_max": int(
                        np.max([x.size for x in matched_test_locals])
                    ),
                    "controls_per_block_median": float(
                        np.median([x.size for x in matched_control_locals])
                    ),
                    "controls_per_block_max": int(
                        np.max([x.size for x in matched_control_locals])
                    ),
                    "block_ratio_max": (
                        float(np.max(block_ratios)) if block_ratios.size else np.nan
                    ),
                    "distance_mean": float(np.mean(distances)),
                    "distance_median": float(np.median(distances)),
                    "distance_max": float(np.max(distances)),
                }
            )
        summaries.append(summary)
    if kept_test_parts:
        matched_test_unsorted = np.concatenate(kept_test_parts).astype(np.int64)
        idx_to_block_unsorted = np.concatenate(kept_block_parts).astype(np.int64)
        distance_unsorted = np.concatenate(kept_distance_parts).astype(float)
        order = np.argsort(matched_test_unsorted, kind="stable")
        matched_test_indices = matched_test_unsorted[order]
        idx_to_block = idx_to_block_unsorted[order]
        test_distances = distance_unsorted[order]
    else:
        matched_test_indices = np.asarray([], dtype=np.int64)
        idx_to_block = np.asarray([], dtype=np.int64)
        test_distances = np.asarray([], dtype=float)
    kept_test_mask = np.zeros(test_yhat.size, dtype=bool)
    kept_test_mask[matched_test_indices] = True
    kept_test_y = test_y[matched_test_indices]
    return OptmatchSharedCompoundInputs(
        test_scores=test_yhat[matched_test_indices].astype(float),
        idx_to_block=idx_to_block,
        control_scores_by_block=control_scores,
        control_null_membership=control_membership,
        block_labels=np.asarray(block_labels, dtype=np.int64),
        test_y=kept_test_y,
        test_truth=kept_test_y > float(null_number),
        kept_test_mask=kept_test_mask,
        matched_test_indices=matched_test_indices,
        matched_test_indices_by_block=matched_test_by_block,
        matched_control_indices_by_block=matched_control_by_block,
        matched_test_distances_by_block=matched_distances_by_block,
        controls_per_block=np.asarray(controls_per_block, dtype=np.int64),
        tests_per_block=np.asarray(tests_per_block, dtype=np.int64),
        match_cluster_summaries=tuple(summaries),
        matching_covariate_names=covariate_names,
        controls_per_match_block=int(target_controls),
        assignment_method="adaptive_cem",
    )
