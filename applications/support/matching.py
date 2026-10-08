"""Manuscript workflow helpers, extracted from code/utopia_compound/matching.py."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np
from support.compound_inputs import CompoundInputs, _finite_cluster_mask

ControlPool = Literal["null", "all"]
OptmatchAssignmentMethod = Literal["nearest", "load_balanced"]


@dataclass(frozen=True)
class MatchedCompoundInputs(CompoundInputs):
    """Compound inputs where each retained hypothesis owns its controls."""

    matched_test_indices: np.ndarray | None = None
    matched_control_indices_by_block: dict[int, np.ndarray] | None = None
    matched_distances_by_block: dict[int, np.ndarray] | None = None
    controls_per_block: np.ndarray | None = None
    match_cluster_summaries: tuple[dict[str, object], ...] | None = None
    matching_covariate_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class OptmatchSharedCompoundInputs(CompoundInputs):
    """Compound inputs produced by paper GG (greedy grouping).

    ``Optmatch...`` is a legacy repository/API name retained for compatibility.
    GG is not R ``optmatch`` or Hansen--Klopfer optimal full matching and makes
    no global optimality claim.
    """

    matched_test_indices: np.ndarray | None = None
    matched_test_indices_by_block: dict[int, np.ndarray] | None = None
    matched_control_indices_by_block: dict[int, np.ndarray] | None = None
    matched_test_distances_by_block: dict[int, np.ndarray] | None = None
    controls_per_block: np.ndarray | None = None
    tests_per_block: np.ndarray | None = None
    match_cluster_summaries: tuple[dict[str, object], ...] | None = None
    matching_covariate_names: tuple[str, ...] | None = None
    max_block_ratio: int | None = None
    controls_per_match_block: int | None = None
    assignment_method: str = "nearest"
    load_balance_penalty: float = 0.0
    soft_max_block_ratio: float | None = None
    soft_max_penalty: float = 0.0
    soft_max_power: float = 2.0


def build_optmatch_shared_compound_inputs(
    *,
    test_yhat: np.ndarray,
    test_clusters: np.ndarray,
    test_covariates: np.ndarray,
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    calib_clusters: np.ndarray,
    calib_covariates: np.ndarray,
    null_number: float,
    max_block_ratio: int | None,
    controls_per_block: int = 5,
    control_pool: ControlPool = "null",
    caliper: float | None = None,
    knn_controls: int = 128,
    assignment_method: OptmatchAssignmentMethod = "nearest",
    load_balance_penalty: float = 0.0,
    soft_max_block_ratio: float | None = None,
    soft_max_penalty: float = 0.0,
    soft_max_power: float = 2.0,
    test_y: np.ndarray | None = None,
    eligible_clusters: tuple[int, ...] | list[int] | None = None,
    covariate_names: Sequence[str] | None = None,
) -> OptmatchSharedCompoundInputs:
    """Build paper-GG (greedy grouping) shared-control strata."""
    test_yhat = np.asarray(test_yhat, dtype=float).reshape(-1)
    test_clusters = np.asarray(test_clusters).reshape(-1)
    calib_yhat = np.asarray(calib_yhat, dtype=float).reshape(-1)
    calib_y = np.asarray(calib_y, dtype=float).reshape(-1)
    calib_clusters = np.asarray(calib_clusters).reshape(-1)
    test_covariates = _as_covariate_matrix(
        "test_covariates", test_covariates, test_yhat.size
    )
    calib_covariates = _as_covariate_matrix(
        "calib_covariates", calib_covariates, calib_yhat.size
    )
    if max_block_ratio is not None and max_block_ratio < 1:
        raise ValueError("max_block_ratio must be at least 1.")
    if controls_per_block < 1:
        raise ValueError("controls_per_block must be at least 1.")
    if control_pool not in {"null", "all"}:
        raise ValueError("control_pool must be 'null' or 'all'.")
    if knn_controls < 1:
        raise ValueError("knn_controls must be at least 1.")
    if assignment_method not in {"nearest", "load_balanced"}:
        raise ValueError("assignment_method must be 'nearest' or 'load_balanced'.")
    if load_balance_penalty < 0:
        raise ValueError("load_balance_penalty must be nonnegative.")
    if soft_max_block_ratio is not None and soft_max_block_ratio < 1:
        raise ValueError("soft_max_block_ratio must be at least 1.")
    if soft_max_penalty < 0:
        raise ValueError("soft_max_penalty must be nonnegative.")
    if soft_max_power <= 0:
        raise ValueError("soft_max_power must be positive.")
    if test_yhat.size != test_clusters.size:
        raise ValueError("test_yhat and test_clusters must have the same length.")
    if calib_yhat.size != calib_y.size or calib_yhat.size != calib_clusters.size:
        raise ValueError("calibration arrays must have the same length.")
    if test_y is not None and np.asarray(test_y).reshape(-1).size != test_yhat.size:
        raise ValueError("test_y must match test_yhat when provided.")
    names = _covariate_names(covariate_names, test_covariates.shape[1])
    test_mask = _finite_cluster_mask(test_yhat, test_clusters) & np.all(
        np.isfinite(test_covariates), axis=1
    )
    calib_mask = (
        _finite_cluster_mask(calib_yhat, calib_clusters)
        & np.isfinite(calib_y)
        & np.all(np.isfinite(calib_covariates), axis=1)
    )
    if eligible_clusters is not None:
        eligible = {int(cluster) for cluster in eligible_clusters}
        test_mask &= np.isin(test_clusters.astype(int), list(eligible))
        calib_mask &= np.isin(calib_clusters.astype(int), list(eligible))
    ordered_labels: list[int] = []
    for label in test_clusters[test_mask].astype(int):
        if label not in ordered_labels:
            ordered_labels.append(int(label))
    test_cluster_int = test_clusters.astype(int)
    calib_cluster_int = calib_clusters.astype(int)
    kept_test_parts: list[np.ndarray] = []
    kept_block_parts: list[np.ndarray] = []
    kept_distance_parts: list[np.ndarray] = []
    control_scores: dict[int, np.ndarray] = {}
    control_membership: dict[int, np.ndarray] = {}
    matched_test_by_block: dict[int, np.ndarray] = {}
    matched_control_by_block: dict[int, np.ndarray] = {}
    matched_distances_by_block: dict[int, np.ndarray] = {}
    block_labels: list[int] = []
    controls_per_shared_block: list[int] = []
    tests_per_shared_block: list[int] = []
    summaries: list[dict[str, object]] = []
    for raw_label in ordered_labels:
        cluster_test_pos = np.flatnonzero(test_mask & (test_cluster_int == raw_label))
        cluster_control_mask = calib_mask & (calib_cluster_int == raw_label)
        if control_pool == "null":
            cluster_control_mask &= calib_y <= null_number
        cluster_control_pos = np.flatnonzero(cluster_control_mask)
        summary: dict[str, object] = {
            "raw_cluster": int(raw_label),
            "available_test": int(cluster_test_pos.size),
            "available_controls": int(cluster_control_pos.size),
            "matched_test": 0,
            "unmatched_test": int(cluster_test_pos.size),
            "used_controls": 0,
            "unused_controls": int(cluster_control_pos.size),
            "control_pool": control_pool,
            "matching_method": "optmatch_shared",
            "max_block_ratio": (
                None if max_block_ratio is None else int(max_block_ratio)
            ),
            "controls_per_block_target": int(controls_per_block),
            "caliper": None if caliper is None else float(caliper),
            "knn_controls": int(knn_controls),
            "assignment_method": assignment_method,
            "load_balance_penalty": float(load_balance_penalty),
            "soft_max_block_ratio": (
                None if soft_max_block_ratio is None else float(soft_max_block_ratio)
            ),
            "soft_max_penalty": float(soft_max_penalty),
            "soft_max_power": float(soft_max_power),
        }
        if cluster_test_pos.size == 0 or cluster_control_pos.size == 0:
            summaries.append(_with_empty_optmatch_balance(summary, names))
            continue
        test_features, control_features = _standardize_covariates(
            test_covariates[cluster_test_pos], calib_covariates[cluster_control_pos]
        )
        local_assignments, local_control_groups = _assign_tests_to_control_groups(
            test_features=test_features,
            control_features=control_features,
            test_original_indices=cluster_test_pos,
            controls_per_block=controls_per_block,
            max_block_ratio=max_block_ratio,
            caliper=caliper,
            knn_controls=knn_controls,
            assignment_method=assignment_method,
            load_balance_penalty=load_balance_penalty,
            soft_max_block_ratio=soft_max_block_ratio,
            soft_max_penalty=soft_max_penalty,
            soft_max_power=soft_max_power,
        )
        cluster_matched_tests: list[np.ndarray] = []
        cluster_matched_controls: list[np.ndarray] = []
        cluster_distances: list[np.ndarray] = []
        block_test_counts: list[int] = []
        block_control_counts: list[int] = []
        for group_local, assigned in enumerate(local_assignments):
            if assigned.shape[0] == 0:
                continue
            block_id = len(block_labels)
            control_local = local_control_groups[group_local]
            control_pos = cluster_control_pos[control_local]
            assigned_test_local = assigned[:, 0].astype(np.int64)
            assigned_distances = assigned[:, 1].astype(float)
            test_pos = cluster_test_pos[assigned_test_local]
            block_labels.append(int(raw_label))
            control_scores[block_id] = calib_yhat[control_pos].astype(float)
            if control_pool == "null":
                control_membership[block_id] = np.ones(control_pos.size, dtype=bool)
            else:
                control_membership[block_id] = (
                    calib_y[control_pos] <= null_number
                ).astype(bool)
            matched_test_by_block[block_id] = test_pos.astype(np.int64)
            matched_control_by_block[block_id] = control_pos.astype(np.int64)
            matched_distances_by_block[block_id] = assigned_distances.astype(float)
            kept_test_parts.append(test_pos.astype(np.int64))
            kept_block_parts.append(np.full(test_pos.size, block_id, dtype=np.int64))
            kept_distance_parts.append(assigned_distances.astype(float))
            cluster_matched_tests.append(assigned_test_local)
            cluster_matched_controls.append(control_local.astype(np.int64))
            cluster_distances.append(assigned_distances.astype(float))
            block_test_counts.append(int(test_pos.size))
            block_control_counts.append(int(control_pos.size))
            tests_per_shared_block.append(int(test_pos.size))
            controls_per_shared_block.append(int(control_pos.size))
        if cluster_matched_tests:
            matched_test_locals = np.unique(
                np.concatenate(cluster_matched_tests).astype(np.int64)
            )
            matched_control_locals = np.unique(
                np.concatenate(cluster_matched_controls).astype(np.int64)
            )
            all_distances = np.concatenate(cluster_distances).astype(float)
            block_ratios = np.asarray(block_test_counts, dtype=float) / np.asarray(
                block_control_counts, dtype=float
            )
            summary.update(
                {
                    "matched_test": int(matched_test_locals.size),
                    "unmatched_test": int(
                        cluster_test_pos.size - matched_test_locals.size
                    ),
                    "used_controls": int(matched_control_locals.size),
                    "unused_controls": int(
                        cluster_control_pos.size - matched_control_locals.size
                    ),
                    "shared_blocks": int(len(block_test_counts)),
                    "tests_per_block_min": int(np.min(block_test_counts)),
                    "tests_per_block_median": float(np.median(block_test_counts)),
                    "tests_per_block_mean": float(np.mean(block_test_counts)),
                    "tests_per_block_max": int(np.max(block_test_counts)),
                    "controls_per_block_min": int(np.min(block_control_counts)),
                    "controls_per_block_median": float(np.median(block_control_counts)),
                    "controls_per_block_mean": float(np.mean(block_control_counts)),
                    "controls_per_block_max": int(np.max(block_control_counts)),
                    "block_ratio_mean": float(np.mean(block_ratios)),
                    "block_ratio_max": float(np.max(block_ratios)),
                    "distance_mean": float(np.mean(all_distances)),
                    "distance_median": float(np.median(all_distances)),
                    "distance_max": float(np.max(all_distances)),
                }
            )
            summary.update(
                _balance_diagnostics(
                    test_covariates[cluster_test_pos],
                    calib_covariates[cluster_control_pos],
                    matched_test_locals,
                    matched_control_locals,
                    names,
                )
            )
        else:
            summary.update(
                {
                    "shared_blocks": 0,
                    "tests_per_block_min": np.nan,
                    "tests_per_block_median": np.nan,
                    "tests_per_block_mean": np.nan,
                    "tests_per_block_max": np.nan,
                    "controls_per_block_min": np.nan,
                    "controls_per_block_median": np.nan,
                    "controls_per_block_mean": np.nan,
                    "controls_per_block_max": np.nan,
                    "block_ratio_mean": np.nan,
                    "block_ratio_max": np.nan,
                }
            )
            summary.update(_with_empty_balance({}, names))
        summaries.append(summary)
    if kept_test_parts:
        matched_test_indices_unsorted = np.concatenate(kept_test_parts).astype(np.int64)
        idx_to_block_unsorted = np.concatenate(kept_block_parts).astype(np.int64)
        test_distance_unsorted = np.concatenate(kept_distance_parts).astype(float)
        order = np.argsort(matched_test_indices_unsorted, kind="stable")
        matched_test_indices = matched_test_indices_unsorted[order]
        idx_to_block = idx_to_block_unsorted[order]
        test_distances = test_distance_unsorted[order]
    else:
        matched_test_indices = np.asarray([], dtype=np.int64)
        idx_to_block = np.asarray([], dtype=np.int64)
        test_distances = np.asarray([], dtype=float)
    kept_test_mask = np.zeros(test_yhat.size, dtype=bool)
    kept_test_mask[matched_test_indices] = True
    kept_test_y: np.ndarray | None = None
    test_truth: np.ndarray | None = None
    if test_y is not None:
        kept_test_y = np.asarray(test_y, dtype=float).reshape(-1)[matched_test_indices]
        test_truth = kept_test_y > null_number
    return OptmatchSharedCompoundInputs(
        test_scores=test_yhat[matched_test_indices].astype(float),
        idx_to_block=idx_to_block,
        control_scores_by_block=control_scores,
        control_null_membership=control_membership,
        block_labels=np.asarray(block_labels, dtype=np.int64),
        test_y=kept_test_y,
        test_truth=test_truth,
        kept_test_mask=kept_test_mask,
        matched_test_indices=matched_test_indices,
        matched_test_indices_by_block=matched_test_by_block,
        matched_control_indices_by_block=matched_control_by_block,
        matched_test_distances_by_block=matched_distances_by_block,
        controls_per_block=np.asarray(controls_per_shared_block, dtype=np.int64),
        tests_per_block=np.asarray(tests_per_shared_block, dtype=np.int64),
        match_cluster_summaries=tuple(summaries),
        matching_covariate_names=names,
        max_block_ratio=None if max_block_ratio is None else int(max_block_ratio),
        controls_per_match_block=int(controls_per_block),
        assignment_method=assignment_method,
        load_balance_penalty=float(load_balance_penalty),
        soft_max_block_ratio=(
            None if soft_max_block_ratio is None else float(soft_max_block_ratio)
        ),
        soft_max_penalty=float(soft_max_penalty),
        soft_max_power=float(soft_max_power),
    )


def _as_covariate_matrix(
    name: str, values: np.ndarray, expected_length: int
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if array.shape[0] != expected_length:
        raise ValueError(f"{name} must have {expected_length} rows.")
    return array


def _covariate_names(names: Sequence[str] | None, width: int) -> tuple[str, ...]:
    if names is None:
        return tuple((f"x{index}" for index in range(width)))
    out = tuple((str(name) for name in names))
    if len(out) != width:
        raise ValueError("covariate_names must match the number of covariate columns.")
    forbidden = {
        "outcome",
        "response",
        "truth",
        "label",
        "target",
        "y",
        "test_y",
        "calib_y",
        "score",
        "scores",
        "yhat",
        "pred",
        "prediction",
        "test_score",
        "calib_score",
    }
    forbidden_used = sorted({name.lower() for name in out} & forbidden)
    if forbidden_used:
        raise ValueError(
            f"matching covariates must be pre-outcome covariates; forbidden covariate names: {forbidden_used}"
        )
    return out


def _assign_tests_to_control_groups(
    *,
    test_features: np.ndarray,
    control_features: np.ndarray,
    test_original_indices: np.ndarray,
    controls_per_block: int,
    max_block_ratio: int | None,
    caliper: float | None,
    knn_controls: int,
    assignment_method: OptmatchAssignmentMethod,
    load_balance_penalty: float,
    soft_max_block_ratio: float | None,
    soft_max_penalty: float,
    soft_max_power: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("optmatch shared-control matching requires scipy.") from exc
    control_groups = _kd_control_groups(
        control_features, max_group_size=controls_per_block
    )
    group_by_control = np.empty(control_features.shape[0], dtype=np.int64)
    for group_id, group in enumerate(control_groups):
        group_by_control[group] = int(group_id)
    capacities = (
        None
        if max_block_ratio is None
        else np.asarray(
            [len(group) * int(max_block_ratio) for group in control_groups],
            dtype=np.int64,
        )
    )
    loads = np.zeros(len(control_groups), dtype=np.int64)
    k = min(int(knn_controls), control_features.shape[0])
    distances, control_locals = cKDTree(control_features).query(test_features, k=k)
    distances = np.asarray(distances, dtype=float)
    control_locals = np.asarray(control_locals, dtype=np.int64)
    if k == 1:
        distances = distances[:, None]
        control_locals = control_locals[:, None]
    neighbor_groups = group_by_control[control_locals]
    valid_neighbors = np.isfinite(distances)
    if caliper is not None:
        valid_neighbors &= distances <= float(caliper)
    finite_distances = np.where(valid_neighbors, distances, np.inf)
    best_distances = np.full(
        (test_features.shape[0], len(control_groups)), np.inf, dtype=float
    )
    row_ids = np.repeat(
        np.arange(test_features.shape[0], dtype=np.int64), finite_distances.shape[1]
    )
    np.minimum.at(
        best_distances,
        (row_ids, neighbor_groups.reshape(-1)),
        finite_distances.reshape(-1),
    )
    candidate_group_ids: list[np.ndarray] = []
    candidate_distances: list[np.ndarray] = []
    candidate_lengths = np.zeros(test_features.shape[0], dtype=np.int64)
    candidate_first_distances = np.full(test_features.shape[0], np.inf, dtype=float)
    for test_local in range(test_features.shape[0]):
        row = best_distances[test_local]
        groups = np.flatnonzero(np.isfinite(row))
        if groups.size == 0:
            candidate_group_ids.append(np.empty(0, dtype=np.int64))
            candidate_distances.append(np.empty(0, dtype=float))
            continue
        order = np.argsort(row[groups], kind="mergesort")
        ordered_groups = groups[order].astype(np.int64, copy=False)
        ordered_distances = row[ordered_groups].astype(float, copy=False)
        candidate_group_ids.append(ordered_groups)
        candidate_distances.append(ordered_distances)
        candidate_lengths[test_local] = int(ordered_groups.size)
        candidate_first_distances[test_local] = float(ordered_distances[0])
    test_order = sorted(
        range(test_features.shape[0]),
        key=lambda local: (
            int(candidate_lengths[local]) if candidate_lengths[local] else np.inf,
            float(candidate_first_distances[local]),
            int(test_original_indices[local]),
        ),
    )
    assignments: list[list[tuple[int, float]]] = [[] for _ in control_groups]
    group_sizes = np.asarray([len(group) for group in control_groups], dtype=float)
    use_penalized_assignment = (
        assignment_method == "load_balanced"
        or load_balance_penalty > 0
        or (soft_max_block_ratio is not None and soft_max_penalty > 0)
    )
    for test_local in test_order:
        groups = candidate_group_ids[test_local]
        if groups.size == 0:
            continue
        distances_for_test = candidate_distances[test_local]
        if not use_penalized_assignment:
            eligible = np.ones(groups.size, dtype=bool)
            if capacities is not None:
                eligible &= loads[groups] < capacities[groups]
            if not np.any(eligible):
                continue
            choice = int(np.flatnonzero(eligible)[0])
            group_id = int(groups[choice])
            assignments[group_id].append(
                (int(test_local), float(distances_for_test[choice]))
            )
            loads[group_id] += 1
            continue
        eligible = np.ones(groups.size, dtype=bool)
        if capacities is not None:
            eligible &= loads[groups] < capacities[groups]
        if not np.any(eligible):
            continue
        groups_eligible = groups[eligible]
        distances_eligible = distances_for_test[eligible]
        proposed_ratios = (loads[groups_eligible] + 1.0) / np.maximum(
            group_sizes[groups_eligible], 1.0
        )
        soft_penalties = np.zeros(groups_eligible.size, dtype=float)
        if soft_max_block_ratio is not None and soft_max_penalty > 0:
            excess = np.maximum(0.0, proposed_ratios - float(soft_max_block_ratio))
            soft_penalties = float(soft_max_penalty) * excess ** float(soft_max_power)
        scores = (
            distances_eligible
            + float(load_balance_penalty) * proposed_ratios
            + soft_penalties
        )
        choice = int(
            np.lexsort((groups_eligible, proposed_ratios, distances_eligible, scores))[
                0
            ]
        )
        group_id = int(groups_eligible[choice])
        distance = float(distances_eligible[choice])
        assignments[group_id].append((int(test_local), float(distance)))
        loads[group_id] += 1
    assignment_arrays = [
        (
            np.asarray(values, dtype=float).reshape(-1, 2)
            if values
            else np.empty((0, 2), dtype=float)
        )
        for values in assignments
    ]
    return (assignment_arrays, control_groups)


def _kd_control_groups(
    control_features: np.ndarray, *, max_group_size: int
) -> list[np.ndarray]:
    groups: list[np.ndarray] = []

    def split(indices: np.ndarray) -> None:
        if indices.size <= max_group_size:
            groups.append(indices.astype(np.int64))
            return
        spans = np.ptp(control_features[indices], axis=0)
        dim = int(np.argmax(spans))
        order = indices[np.argsort(control_features[indices, dim], kind="mergesort")]
        mid = order.size // 2
        split(order[:mid])
        split(order[mid:])

    split(np.arange(control_features.shape[0], dtype=np.int64))
    return groups


def _standardize_covariates(
    test_covariates: np.ndarray, control_covariates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([test_covariates, control_covariates]).astype(float)
    center = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return ((test_covariates - center) / scale, (control_covariates - center) / scale)


def _balance_diagnostics(
    test_covariates: np.ndarray,
    control_covariates: np.ndarray,
    matched_test_locals: np.ndarray,
    matched_control_locals: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, object]:
    if matched_test_locals.size == 0 or matched_control_locals.size == 0:
        return _with_empty_balance({}, names)
    pooled = np.vstack([test_covariates, control_covariates]).astype(float)
    pooled_sd = np.std(pooled, axis=0, ddof=0)
    pooled_sd = np.where(pooled_sd > 1e-12, pooled_sd, 1.0)
    test_mean = np.mean(test_covariates[matched_test_locals], axis=0)
    control_mean = np.mean(control_covariates[matched_control_locals], axis=0)
    smd = np.abs(test_mean - control_mean) / pooled_sd
    out: dict[str, object] = {
        "match_mean_abs_smd": float(np.mean(smd)),
        "match_max_abs_smd": float(np.max(smd)),
    }
    for name, value in zip(names, smd):
        out[f"match_abs_smd_{name}"] = float(value)
    return out


def _with_empty_optmatch_balance(
    row: dict[str, object], names: tuple[str, ...]
) -> dict[str, object]:
    out = _with_empty_balance(row, names)
    out.setdefault("shared_blocks", 0)
    out.setdefault("tests_per_block_min", np.nan)
    out.setdefault("tests_per_block_median", np.nan)
    out.setdefault("tests_per_block_mean", np.nan)
    out.setdefault("tests_per_block_max", np.nan)
    out.setdefault("controls_per_block_min", np.nan)
    out.setdefault("controls_per_block_median", np.nan)
    out.setdefault("controls_per_block_mean", np.nan)
    out.setdefault("controls_per_block_max", np.nan)
    out.setdefault("block_ratio_mean", np.nan)
    out.setdefault("block_ratio_max", np.nan)
    return out


def _with_empty_balance(
    row: dict[str, object], names: tuple[str, ...]
) -> dict[str, object]:
    out = dict(row)
    out.setdefault("distance_mean", np.nan)
    out.setdefault("distance_median", np.nan)
    out.setdefault("distance_max", np.nan)
    out["match_mean_abs_smd"] = np.nan
    out["match_max_abs_smd"] = np.nan
    for name in names:
        out[f"match_abs_smd_{name}"] = np.nan
    return out
