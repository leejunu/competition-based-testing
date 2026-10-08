"""Manuscript workflow helpers, extracted from code/utopia_compound/refined_matching_suite.py."""

from __future__ import annotations
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
from support.matching import MatchedCompoundInputs
from support.utopia_io import load_mask, parse_utopia_config


def as_covariate_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D covariate array.")
    return matrix


def coordinate_index_image(mask: np.ndarray) -> np.ndarray:
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[np.asarray(mask, dtype=bool)] = np.arange(int(np.sum(mask)), dtype=np.int64)
    return index


def coords_to_mask_positions(
    coords: np.ndarray, index_image: np.ndarray, *, name: str
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    rounded = np.rint(coords).astype(np.int64)
    if not np.allclose(coords, rounded, atol=0.0001):
        max_deviation = float(np.max(np.abs(coords - rounded)))
        raise ValueError(
            f"{name} contains non-integer coordinates; max deviation {max_deviation:g}."
        )
    rows = rounded[:, 0]
    cols = rounded[:, 1]
    in_bounds = (
        (rows >= 0)
        & (rows < index_image.shape[0])
        & (cols >= 0)
        & (cols < index_image.shape[1])
    )
    if not np.all(in_bounds):
        raise ValueError(
            f"{name} contains {int((~in_bounds).sum())} coordinates outside image bounds."
        )
    positions = index_image[rows, cols]
    if np.any(positions < 0):
        raise ValueError(
            f"{name} contains {int(np.sum(positions < 0))} coordinates outside tissue."
        )
    return positions.astype(np.int64)


def stable_component_signs(components: np.ndarray) -> np.ndarray:
    components = np.asarray(components, dtype=np.float32).copy()
    for j in range(components.shape[1]):
        col = components[:, j]
        anchor = int(np.argmax(np.abs(col)))
        if col[anchor] < 0:
            components[:, j] *= -1.0
    return components


def transform_embedding_rows_to_cluster_pca(
    embedding_path: Path, row_indices: np.ndarray, pca: Any, *, batch_size: int
) -> np.ndarray:
    embeddings = np.load(embedding_path, mmap_mode="r")
    row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    components = np.asarray(pca.components_, dtype=np.float32)
    mean = np.asarray(pca.mean_, dtype=np.float32)
    out = np.empty((row_indices.size, components.shape[0]), dtype=np.float32)
    for start in range(0, row_indices.size, int(batch_size)):
        stop = min(start + int(batch_size), row_indices.size)
        x = np.asarray(embeddings[row_indices[start:stop]], dtype=np.float32)
        out[start:stop] = (x - mean) @ components.T
    return out


def load_cluster_pca_features(
    *, data_dir: Path, score_cache: dict[str, Any], scale: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    # Map cached spatial coordinates to the frozen histology PCA features.
    pca_path = data_dir / f"histology_clustering/pca_dim_80_scale_{scale}.pkl"
    with pca_path.open("rb") as handle:
        pca = pickle.load(handle)
    wsi_mask = load_mask(data_dir / f"mask/mask-small_scale_{scale}.png")
    wsi_index = coordinate_index_image(wsi_mask)
    test_positions = coords_to_mask_positions(
        score_cache["test_coords"], wsi_index, name="test_coords"
    )
    test_features = transform_embedding_rows_to_cluster_pca(
        data_dir / f"embeddings/embeddings_scale_{scale}.npy",
        test_positions,
        pca,
        batch_size=batch_size,
    )
    config = parse_utopia_config(data_dir / "utopia_config")
    calib_roi = np.asarray(score_cache["calib_roi"]).reshape(-1).astype(int)
    calib_coords = np.asarray(score_cache["calib_coords"], dtype=float)
    calib_features = np.empty(
        (calib_coords.shape[0], test_features.shape[1]), dtype=np.float32
    )
    for roi in sorted(np.unique(calib_roi)):
        roi = int(roi)
        roi_mask = calib_roi == roi
        roi_origin = np.asarray(config["roi_top_left"][roi], dtype=np.float32) / float(
            scale
        )
        local_coords = calib_coords[roi_mask] - roi_origin[None, :]
        roi_tissue_mask = load_mask(
            data_dir / f"mask/mask-small_roi_{roi}_scale_{scale}.png"
        )
        roi_index = coordinate_index_image(roi_tissue_mask)
        roi_positions = coords_to_mask_positions(
            local_coords, roi_index, name=f"calib_coords_roi{roi}"
        )
        calib_features[roi_mask] = transform_embedding_rows_to_cluster_pca(
            data_dir / f"embeddings/embeddings_roi_{roi}_scale_{scale}.npy",
            roi_positions,
            pca,
            batch_size=batch_size,
        )
    return (test_features, calib_features)


def score_cluster_fit_mask(
    scores: np.ndarray, clusters: np.ndarray, *, eligible_clusters: Sequence[int]
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    clusters = np.asarray(clusters).reshape(-1)
    mask = np.isfinite(scores) & np.isfinite(clusters) & (clusters >= 0)
    mask &= np.isin(
        clusters.astype(int), np.asarray(tuple(eligible_clusters), dtype=int)
    )
    return mask


def fit_matching_pca(
    test_cluster: np.ndarray,
    calib_cluster: np.ndarray,
    *,
    score_cache: dict[str, Any],
    matching_dim: int | None,
    eligible_clusters: Sequence[int],
    null_number: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    if matching_dim is None:
        names = tuple((f"cluster_pc{j + 1}" for j in range(test_cluster.shape[1])))
        return (
            test_cluster.astype(np.float32),
            calib_cluster.astype(np.float32),
            names,
            {
                "cluster_matching_requested_dim": None,
                "cluster_matching_effective_dim": int(test_cluster.shape[1]),
                "cluster_matching_cumulative_explained_variance_ratio": None,
                "cluster_matching_transform": "identity",
            },
        )
    dim = min(int(matching_dim), int(test_cluster.shape[1]))
    test_fit = score_cluster_fit_mask(
        score_cache["test_yhat"],
        score_cache["test_clusters"],
        eligible_clusters=eligible_clusters,
    )
    calib_fit = score_cluster_fit_mask(
        score_cache["calib_yhat"],
        score_cache["calib_clusters"],
        eligible_clusters=eligible_clusters,
    )
    calib_fit &= np.isfinite(score_cache["calib_y"])
    calib_fit &= np.asarray(score_cache["calib_y"], dtype=float).reshape(-1) <= float(
        null_number
    )
    pooled = np.vstack([test_cluster[test_fit], calib_cluster[calib_fit]]).astype(
        np.float32
    )
    pooled = pooled[np.all(np.isfinite(pooled), axis=1)]
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    finite_scaled = (pooled - center) / scale
    covariance = finite_scaled.T @ finite_scaled / max(finite_scaled.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = stable_component_signs(eigenvectors[:, order[:dim]])
    test_projected = (test_cluster - center) / scale @ components
    calib_projected = (calib_cluster - center) / scale @ components
    total_variance = float(eigenvalues.sum())
    cumulative = (
        float(np.sum(eigenvalues[:dim] / total_variance)) if total_variance > 0 else 0.0
    )
    names = tuple((f"match_pc{j + 1}" for j in range(dim)))
    return (
        test_projected.astype(np.float32),
        calib_projected.astype(np.float32),
        names,
        {
            "cluster_matching_requested_dim": int(matching_dim),
            "cluster_matching_effective_dim": int(dim),
            "cluster_matching_cumulative_explained_variance_ratio": cumulative,
            "cluster_matching_transform": "pca_svd",
        },
    )


def build_matching_covariates(
    *,
    dim: int | None,
    score_cache: dict[str, Any],
    test_cluster: np.ndarray,
    calib_cluster: np.ndarray,
    eligible_clusters: Sequence[int],
    null_number: float,
    include_spatial_coords: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    test_match, calib_match, names, info = fit_matching_pca(
        test_cluster,
        calib_cluster,
        score_cache=score_cache,
        matching_dim=dim,
        eligible_clusters=eligible_clusters,
        null_number=null_number,
    )
    if not include_spatial_coords:
        return (
            test_match.astype(np.float32),
            calib_match.astype(np.float32),
            names,
            {
                **info,
                "include_spatial_coords": False,
                "spatial_coord_dim": 0,
                "final_matching_dim": int(test_match.shape[1]),
            },
        )
    test_coords = as_covariate_matrix("test_coords", score_cache["test_coords"])
    calib_coords = as_covariate_matrix("calib_coords", score_cache["calib_coords"])
    return (
        np.hstack([test_match, test_coords]).astype(np.float32),
        np.hstack([calib_match, calib_coords]).astype(np.float32),
        names + ("row", "col"),
        {
            **info,
            "include_spatial_coords": True,
            "spatial_coord_dim": int(test_coords.shape[1]),
            "final_matching_dim": int(test_match.shape[1] + test_coords.shape[1]),
        },
    )


def standardize_within_cluster(
    test_covariates: np.ndarray, control_covariates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([test_covariates, control_covariates]).astype(float)
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return ((test_covariates - center) / scale, (control_covariates - center) / scale)


def build_fixed_knn_compound_inputs(
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
    controls_per_test: int,
    eligible_clusters: Sequence[int],
    control_pool: str = "null",
    knn_candidates: int = 128,
    query_batch_size: int = 25000,
    covariate_names: tuple[str, ...] | None = None,
) -> MatchedCompoundInputs:
    from scipy.spatial import cKDTree

    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    test_y = np.asarray(test_y, dtype=float).reshape(-1)
    test_clusters = np.asarray(test_clusters).reshape(-1)
    test_covariates = np.asarray(test_covariates, dtype=float)
    calib_yhat = np.asarray(calib_yhat, dtype=float).reshape(-1)
    calib_y = np.asarray(calib_y, dtype=float).reshape(-1)
    calib_clusters = np.asarray(calib_clusters).reshape(-1)
    calib_covariates = np.asarray(calib_covariates, dtype=float)
    n = int(controls_per_test)
    eligible = np.asarray(tuple((int(x) for x in eligible_clusters)), dtype=int)
    test_mask = (
        np.isfinite(test_yhat)
        & np.isfinite(test_clusters)
        & (test_clusters >= 0)
        & np.all(np.isfinite(test_covariates), axis=1)
        & np.isin(test_clusters.astype(int), eligible)
    )
    calib_mask = (
        np.isfinite(calib_yhat)
        & np.isfinite(calib_y)
        & np.isfinite(calib_clusters)
        & (calib_clusters >= 0)
        & np.all(np.isfinite(calib_covariates), axis=1)
        & np.isin(calib_clusters.astype(int), eligible)
    )
    if control_pool == "null":
        calib_mask &= calib_y <= float(null_number)
    elif control_pool != "all":
        raise ValueError("control_pool must be 'null' or 'all'.")
    ordered_labels: list[int] = []
    for label in test_clusters[test_mask].astype(int):
        if int(label) not in ordered_labels:
            ordered_labels.append(int(label))
    matched_by_test: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    summaries: list[dict[str, Any]] = []
    for raw_label in ordered_labels:
        cluster_test_pos = np.flatnonzero(
            test_mask & (test_clusters.astype(int) == raw_label)
        )
        cluster_control_pos = np.flatnonzero(
            calib_mask & (calib_clusters.astype(int) == raw_label)
        )
        summary = {
            "raw_cluster": int(raw_label),
            "available_test": int(cluster_test_pos.size),
            "available_controls": int(cluster_control_pos.size),
            "controls_per_test": int(n),
            "matched_test": 0,
            "unmatched_test": int(cluster_test_pos.size),
        }
        if cluster_test_pos.size == 0 or cluster_control_pos.size < n:
            summaries.append(summary)
            continue
        test_features, control_features = standardize_within_cluster(
            test_covariates[cluster_test_pos], calib_covariates[cluster_control_pos]
        )
        k = min(max(int(knn_candidates), n), cluster_control_pos.size)
        tree = cKDTree(control_features)
        batch_size = max(1, int(query_batch_size))

        def query_tree(
            features: np.ndarray, *, query_k: int
        ) -> tuple[np.ndarray, np.ndarray]:
            try:
                distances_batch, locals_batch = tree.query(
                    features, k=query_k, workers=-1
                )
            except TypeError:
                distances_batch, locals_batch = tree.query(features, k=query_k)
            distances_batch = np.asarray(distances_batch, dtype=float)
            locals_batch = np.asarray(locals_batch, dtype=np.int64)
            if query_k == 1:
                distances_batch = distances_batch[:, None]
                locals_batch = locals_batch[:, None]
            return (distances_batch, locals_batch)

        quality = np.empty(cluster_test_pos.size, dtype=np.float32)
        quality_k = min(k, n)
        for start in range(0, cluster_test_pos.size, batch_size):
            stop = min(start + batch_size, cluster_test_pos.size)
            distances_batch, _ = query_tree(
                test_features[start:stop], query_k=quality_k
            )
            quality[start:stop] = np.sqrt(
                np.mean(distances_batch[:, :n] ** 2, axis=1)
            ).astype(np.float32)
        order = np.lexsort((cluster_test_pos, quality))
        available = np.ones(cluster_control_pos.size, dtype=bool)
        matched = 0
        for batch_start in range(0, order.size, batch_size):
            batch_order = order[batch_start : batch_start + batch_size]
            distances, control_locals = query_tree(
                test_features[batch_order], query_k=k
            )
            for row, test_local in enumerate(batch_order):
                candidates = control_locals[row]
                candidate_distances = distances[row]
                usable = available[candidates]
                if int(np.sum(usable)) < n:
                    continue
                selected_candidates = candidates[usable][:n]
                selected_distances = candidate_distances[usable][:n]
                available[selected_candidates] = False
                matched_by_test[int(cluster_test_pos[test_local])] = (
                    int(raw_label),
                    cluster_control_pos[selected_candidates].astype(np.int64),
                    selected_distances.astype(float),
                )
                matched += 1
        summary["matched_test"] = int(matched)
        summary["unmatched_test"] = int(cluster_test_pos.size - matched)
        summaries.append(summary)
    matched_test_indices = np.asarray(sorted(matched_by_test), dtype=np.int64)
    kept_test_mask = np.zeros(test_yhat.size, dtype=bool)
    kept_test_mask[matched_test_indices] = True
    control_scores: dict[int, np.ndarray] = {}
    control_membership: dict[int, np.ndarray] = {}
    matched_controls: dict[int, np.ndarray] = {}
    matched_distances: dict[int, np.ndarray] = {}
    block_labels: list[int] = []
    controls_per_block: list[int] = []
    for block_id, test_pos in enumerate(matched_test_indices):
        raw_label, control_pos, distance = matched_by_test[int(test_pos)]
        control_scores[block_id] = calib_yhat[control_pos].astype(float)
        control_membership[block_id] = (
            np.ones(control_pos.size, dtype=bool)
            if control_pool == "null"
            else calib_y[control_pos] <= null_number
        )
        matched_controls[block_id] = control_pos.astype(np.int64)
        matched_distances[block_id] = distance.astype(float)
        block_labels.append(raw_label)
        controls_per_block.append(int(control_pos.size))
    kept_test_y = test_y[matched_test_indices]
    return MatchedCompoundInputs(
        test_scores=test_yhat[matched_test_indices].astype(float),
        idx_to_block=np.arange(matched_test_indices.size, dtype=np.int64),
        control_scores_by_block=control_scores,
        control_null_membership=control_membership,
        block_labels=np.asarray(block_labels, dtype=np.int64),
        test_y=kept_test_y,
        test_truth=kept_test_y > float(null_number),
        kept_test_mask=kept_test_mask,
        matched_test_indices=matched_test_indices,
        matched_control_indices_by_block=matched_controls,
        matched_distances_by_block=matched_distances,
        controls_per_block=np.asarray(controls_per_block, dtype=np.int64),
        match_cluster_summaries=tuple(summaries),
        matching_covariate_names=covariate_names,
    )
