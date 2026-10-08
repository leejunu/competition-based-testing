"""Manuscript workflow helpers, extracted from scripts/crispr_shared_cbt_experiment.py."""

from __future__ import annotations
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSets:
    negative_controls_all: set[str]
    negative_controls_train: set[str]
    heldout_nulls: set[str]
    positive_controls_all: set[str]
    positive_controls_train: set[str]
    heldout_positives: set[str]


@dataclass(frozen=True)
class MatchingCovariates:
    gene_covariates: pd.DataFrame
    expression: pd.DataFrame | None
    covariate_names: tuple[str, ...]
    source_notes: dict[str, object]


def read_zip_csv(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with zf.open(name) as handle:
        return pd.read_csv(handle, **kwargs)


def read_zip_lines(
    zf: zipfile.ZipFile, name: str, *, skip_header: bool = False
) -> list[str]:
    with zf.open(name) as handle:
        lines = [
            line.strip() for line in handle.read().decode().splitlines() if line.strip()
        ]
    if skip_header and lines and (lines[0].lower() == "gene"):
        return lines[1:]
    return lines


def gene_effect_matrix(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    df = read_zip_csv(zf, name)
    first = df.columns[0]
    return df.set_index(first)


def infer_labels(zf: zipfile.ZipFile, common_essential_csv: Path) -> LabelSets:
    negative_controls_all = set(read_zip_lines(zf, "Achilles/negative_controls.txt"))
    negative_controls_train = set(
        read_zip_lines(zf, "Achilles/negative_controls_train.txt", skip_header=True)
    )
    controls_genes = set(
        gene_effect_matrix(zf, "Achilles/chronos_gene_effect_controls.csv").columns
    )
    no_positive_genes = set(
        gene_effect_matrix(zf, "Achilles/chronos_gene_effect_no_positives.csv").columns
    )
    positive_controls_all = controls_genes - no_positive_genes
    heldout_nulls = no_positive_genes - negative_controls_all
    positive_controls_train: set[str] = set()
    if common_essential_csv.exists() and common_essential_csv.stat().st_size > 0:
        common = pd.read_csv(common_essential_csv).iloc[:, 0].astype(str)
        positive_controls_train = set(common.str.split().str[0])
    heldout_positives = positive_controls_all - positive_controls_train
    if not heldout_positives:
        heldout_positives = set(positive_controls_all)
    return LabelSets(
        negative_controls_all=negative_controls_all,
        negative_controls_train=negative_controls_train,
        heldout_nulls=heldout_nulls,
        positive_controls_all=positive_controls_all,
        positive_controls_train=positive_controls_train,
        heldout_positives=heldout_positives,
    )


def score_long_from_matrix(
    matrix: pd.DataFrame,
    *,
    method: str,
    score_sign: float = 1.0,
    exclude_genes: set[str] | None = None,
) -> pd.DataFrame:
    exclude_genes = exclude_genes or set()
    use = matrix.loc[:, [g for g in matrix.columns if g not in exclude_genes]].copy()
    long = use.stack().dropna().rename("score").reset_index()
    long.columns = ["block", "gene", "score"]
    long["score"] = score_sign * long["score"].astype(float)
    long["method"] = method
    return long[["gene", "block", "method", "score"]]


def control_score_long_from_matrix(
    matrix: pd.DataFrame, *, control_genes: set[str], score_sign: float = 1.0
) -> pd.DataFrame:
    genes = [g for g in matrix.columns if g in control_genes]
    use = matrix.loc[:, genes].copy()
    long = use.stack().dropna().rename("score").reset_index()
    long.columns = ["block", "gene", "score"]
    long["score"] = score_sign * long["score"].astype(float)
    return long[["gene", "block", "score"]]


def _looks_like_real_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    prefix = path.read_bytes()[:256].lstrip().lower()
    if prefix.startswith(b"<!doctype") or prefix.startswith(b"<html"):
        return False
    return b"," in prefix or b"\t" in prefix


def prepare_matching_covariates(
    *, avana_guide_map: Path, expression_csv: Path, blocks: Iterable[str]
) -> MatchingCovariates | None:
    if not avana_guide_map.exists():
        print(f"[matching] missing {avana_guide_map}; skipping CEM/optmatch methods")
        return None
    guide = pd.read_csv(avana_guide_map)
    required = {"sgRNA", "Gene", "nAlignments", "UsedByChronos"}
    missing = sorted(required - set(guide.columns))
    if missing:
        raise ValueError(f"{avana_guide_map} is missing required columns: {missing}")
    guide = guide.copy()
    guide["gene"] = guide["Gene"].astype(str).str.split().str[0]
    guide["used_by_chronos"] = (
        guide["UsedByChronos"].astype(str).str.lower().isin({"true", "1"})
    )
    guide["n_alignments_numeric"] = pd.to_numeric(guide["nAlignments"], errors="coerce")
    guide["guide_length"] = guide["sgRNA"].astype(str).str.len().astype(float)
    guide["guide_gc"] = (
        guide["sgRNA"].astype(str).str.upper().str.count("G")
        + guide["sgRNA"].astype(str).str.upper().str.count("C")
    ) / guide["guide_length"].replace(0, np.nan)
    grouped = guide.groupby("gene", sort=True)
    cov = pd.DataFrame(
        {
            "guide_total_log1p": np.log1p(grouped.size().astype(float)),
            "guide_used_log1p": np.log1p(
                grouped["used_by_chronos"].sum().astype(float)
            ),
            "guide_used_fraction": grouped["used_by_chronos"].mean().astype(float),
            "guide_alignment_mean_log1p": np.log1p(
                grouped["n_alignments_numeric"].mean()
            ),
            "guide_alignment_max_log1p": np.log1p(
                grouped["n_alignments_numeric"].max()
            ),
            "guide_gc_mean": grouped["guide_gc"].mean(),
            "guide_gc_sd": grouped["guide_gc"].std(ddof=0),
            "guide_length_mean": grouped["guide_length"].mean(),
        }
    )
    cov = cov.replace([np.inf, -np.inf], np.nan)
    expression: pd.DataFrame | None = None
    expression_notes: dict[str, object] = {
        "path": str(expression_csv),
        "available": False,
    }
    if _looks_like_real_csv(expression_csv):
        expression = pd.read_csv(expression_csv, index_col=0)
        expression.columns = [str(c).split()[0] for c in expression.columns]
        expression = expression.apply(pd.to_numeric, errors="coerce")
        selected = expression.reindex(index=list(blocks))
        if selected.notna().to_numpy().any():
            selected = selected.reindex(columns=cov.index)
            cov["expr_selected_mean"] = selected.mean(axis=0)
            cov["expr_selected_sd"] = selected.std(axis=0, ddof=0)
            cov["expr_selected_min"] = selected.min(axis=0)
            cov["expr_selected_max"] = selected.max(axis=0)
            expression_notes.update(
                {
                    "available": True,
                    "rows": int(expression.shape[0]),
                    "columns": int(expression.shape[1]),
                    "matched_blocks": sorted(set(blocks) & set(expression.index)),
                }
            )
        else:
            expression = None
            expression_notes["reason"] = (
                "expression CSV did not contain the selected Achilles cell lines"
            )
    else:
        expression_notes["reason"] = (
            "missing, empty, or challenge/error HTML instead of CSV"
        )
    cov = cov.fillna(cov.median(numeric_only=True)).fillna(0.0)
    covariate_names = tuple(cov.columns)
    if expression is not None:
        covariate_names = covariate_names + ("expr_current",)
    return MatchingCovariates(
        gene_covariates=cov.astype(float),
        expression=expression,
        covariate_names=covariate_names,
        source_notes={
            "avana_guide_map": str(avana_guide_map),
            "guide_covariates": list(cov.columns),
            "guide_genes": int(cov.shape[0]),
            "expression": expression_notes,
        },
    )


def matching_covariate_matrix(
    records: pd.DataFrame, covariates: MatchingCovariates
) -> np.ndarray:
    base = covariates.gene_covariates.reindex(records["gene"].astype(str)).to_numpy(
        dtype=float
    )
    if covariates.expression is None:
        return base
    expression_by_block = {
        str(block): covariates.expression.loc[str(block)]
        for block in records["block"].astype(str).unique()
        if str(block) in covariates.expression.index
    }
    current = np.empty(len(records), dtype=float)
    current.fill(np.nan)
    for pos, (block, gene) in enumerate(
        zip(records["block"].astype(str), records["gene"].astype(str))
    ):
        values = expression_by_block.get(block)
        if values is not None and gene in values.index:
            current[pos] = float(values.loc[gene])
    return np.column_stack([base, current])


def _matching_summary_frame(
    *, task: str, method: str, inputs: Any, n_tests_total: int, n_controls_total: int
) -> pd.DataFrame:
    rows = []
    for row in inputs.match_cluster_summaries or ():
        out = dict(row)
        out["task"] = task
        out["method"] = method
        out["matched_test_total"] = int(inputs.test_scores.size)
        out["available_test_total"] = int(n_tests_total)
        out["available_controls_total"] = int(n_controls_total)
        out["matched_fraction_total"] = float(
            inputs.test_scores.size / max(n_tests_total, 1)
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(
            [
                {
                    "task": task,
                    "method": method,
                    "matched_test_total": 0,
                    "available_test_total": int(n_tests_total),
                    "available_controls_total": int(n_controls_total),
                    "matched_fraction_total": 0.0,
                }
            ]
        )
    return pd.DataFrame(rows)


def matched_shared_cbt_pvalues(
    *,
    scores: pd.DataFrame,
    controls: pd.DataFrame,
    covariates: MatchingCovariates,
    task: str,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    from support.matching import build_optmatch_shared_compound_inputs
    from support.arc import build_adaptive_cem_compound_inputs

    scores = scores.reset_index(drop=True).copy()
    controls = controls.reset_index(drop=True).copy()
    block_order = list(dict.fromkeys(scores["block"].astype(str)))
    block_to_id = {block: idx for idx, block in enumerate(block_order)}
    test_yhat = scores["score"].to_numpy(dtype=float)
    test_clusters = scores["block"].astype(str).map(block_to_id).to_numpy(dtype=int)
    test_covariates = matching_covariate_matrix(scores, covariates)
    test_y = np.zeros(scores.shape[0], dtype=float)
    calib_yhat = controls["score"].to_numpy(dtype=float)
    calib_clusters = controls["block"].astype(str).map(block_to_id).to_numpy(dtype=int)
    calib_covariates = matching_covariate_matrix(controls, covariates)
    calib_y = np.zeros(controls.shape[0], dtype=float)
    eligible_clusters = tuple(range(len(block_order)))
    covariate_names = covariates.covariate_names
    specs = [
        (
            "CEM CBT q=1",
            lambda: build_adaptive_cem_compound_inputs(
                test_yhat=test_yhat,
                test_y=test_y,
                test_clusters=test_clusters,
                test_covariates=test_covariates,
                calib_yhat=calib_yhat,
                calib_y=calib_y,
                calib_clusters=calib_clusters,
                calib_covariates=calib_covariates,
                null_number=0.5,
                eligible_clusters=eligible_clusters,
                target_controls=15,
                min_controls=None,
                max_depth=18,
                guided_ratio=None,
                covariate_names=covariate_names,
                control_pool="all",
            ),
        ),
        (
            "Optmatch CBT q=1",
            lambda: build_optmatch_shared_compound_inputs(
                test_yhat=test_yhat,
                test_clusters=test_clusters,
                test_covariates=test_covariates,
                calib_yhat=calib_yhat,
                calib_y=calib_y,
                calib_clusters=calib_clusters,
                calib_covariates=calib_covariates,
                null_number=0.5,
                max_block_ratio=None,
                controls_per_block=10,
                control_pool="all",
                caliper=None,
                knn_controls=128,
                assignment_method="load_balanced",
                load_balance_penalty=0.0,
                soft_max_block_ratio=5.0,
                soft_max_penalty=0.2,
                soft_max_power=2.0,
                test_y=test_y,
                eligible_clusters=eligible_clusters,
                covariate_names=covariate_names,
            ),
        ),
    ]
    pvalue_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for method, build in specs:
        print(f"[matching] {task}: building {method}", flush=True)
        inputs = build()
        summary_frames.append(
            _matching_summary_frame(
                task=task,
                method=method,
                inputs=inputs,
                n_tests_total=len(scores),
                n_controls_total=len(controls),
            )
        )
        if inputs.test_scores.size == 0:
            continue
        q_by_block = {int(block): 1 for block in inputs.control_scores_by_block}
        pvalues = shared_cbt_pvalues_fast_arrays(
            inputs.test_scores,
            inputs.idx_to_block,
            inputs.control_scores_by_block,
            q_by_block=q_by_block,
            common_offset=True,
        )
        matched_rows = scores.iloc[
            np.asarray(inputs.matched_test_indices, dtype=int)
        ].copy()
        out = matched_rows[["gene", "block"]].reset_index(drop=True)
        out["method"] = method
        out["pvalue"] = pvalues
        out["matched_block"] = np.asarray(inputs.idx_to_block, dtype=int)
        pvalue_frames.append(out)
    summaries = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )
    return (pvalue_frames, summaries)


def across_conformal_pvalues(
    scores: pd.DataFrame, control_scores_by_block: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for block, sub in scores.groupby("block", sort=False):
        controls = np.asarray(control_scores_by_block[block], dtype=float)
        test_scores = sub["score"].to_numpy(dtype=float)
        pvalues = (1.0 + (controls[None, :] >= test_scores[:, None]).sum(axis=1)) / (
            controls.size + 1.0
        )
        tmp = sub[["gene", "block"]].copy()
        tmp["method"] = "Across-cluster conformal"
        tmp["pvalue"] = pvalues
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def shared_cbt_pvalues_fast_arrays(
    test_scores: np.ndarray,
    idx_to_block: np.ndarray,
    control_scores_by_block: dict[Any, np.ndarray],
    *,
    q_by_block: dict[Any, int],
    common_offset: bool = False,
) -> np.ndarray:
    """Compute shared-CBT p-values, optionally using one family-wide offset."""
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1)
    idx_to_block = np.asarray(idx_to_block).reshape(-1)
    if test_scores.size != idx_to_block.size:
        raise ValueError("test_scores and idx_to_block must have the same length.")
    total_m = float(test_scores.size)
    offset_by_block: dict[Any, float] = {}
    finalist_by_block: dict[Any, np.ndarray] = {}
    ref_scores: list[np.ndarray] = []
    ref_weights: list[np.ndarray] = []
    for block, controls_raw in control_scores_by_block.items():
        block_mask = idx_to_block == block
        block_scores = test_scores[block_mask]
        if block_scores.size == 0:
            continue
        controls = np.asarray(controls_raw, dtype=float)
        controls = controls[np.isfinite(controls)]
        if controls.size == 0:
            raise ValueError(f"Block {block!r} has no finite controls.")
        q = int(q_by_block[block])
        if q < 1 or q > controls.size:
            raise ValueError(
                f"Block {block!r} has invalid q={q} for n={controls.size}."
            )
        sorted_controls = np.sort(controls)
        sorted_tests = np.sort(block_scores)
        offset_by_block[block] = float(block_scores.size) / float(controls.size)
        num_controls_greater_than_test = controls.size - np.searchsorted(
            sorted_controls, block_scores, side="right"
        )
        finalist_by_block[block] = num_controls_greater_than_test <= q - 1
        num_controls_greater_than_control = controls.size - np.searchsorted(
            sorted_controls, controls, side="right"
        )
        reference_counts = np.zeros(controls.size, dtype=float)
        always_reference = num_controls_greater_than_control <= q - 2
        reference_counts[always_reference] = float(block_scores.size)
        boundary_reference = num_controls_greater_than_control == q - 1
        if np.any(boundary_reference):
            reference_counts[boundary_reference] = np.searchsorted(
                sorted_tests, controls[boundary_reference], side="right"
            )
        weights = reference_counts / float(controls.size)
        keep = weights > 0
        if np.any(keep):
            ref_scores.append(controls[keep])
            ref_weights.append(weights[keep])
    if ref_scores:
        all_ref_scores = np.concatenate(ref_scores)
        all_ref_weights = np.concatenate(ref_weights)
        order = np.argsort(all_ref_scores, kind="mergesort")
        all_ref_scores = all_ref_scores[order]
        all_ref_weights = all_ref_weights[order]
        cumulative_weights = np.concatenate(
            [[0.0], np.cumsum(all_ref_weights, dtype=float)]
        )
        total_ref_weight = float(cumulative_weights[-1])
    else:
        all_ref_scores = np.asarray([], dtype=float)
        cumulative_weights = np.asarray([0.0], dtype=float)
        total_ref_weight = 0.0
    offset = max(offset_by_block.values()) if common_offset else None
    pvalues = np.ones(test_scores.size, dtype=float)
    for block in control_scores_by_block:
        block_mask = idx_to_block == block
        if not np.any(block_mask):
            continue
        block_scores = test_scores[block_mask]
        positions = np.searchsorted(all_ref_scores, block_scores, side="right")
        reference_sum = total_ref_weight - cumulative_weights[positions]
        finalist = finalist_by_block[block]
        block_offset = offset if offset is not None else offset_by_block[block]
        raw = (
            block_offset + reference_sum + (~finalist).astype(float) * total_m
        ) / total_m
        pvalues[block_mask] = np.minimum(raw, 1.0)
    return pvalues


def add_labels(df: pd.DataFrame, labels: LabelSets, *, task: str) -> pd.DataFrame:
    out = df.copy()
    if task == "achilles_allgenes":
        out["truth"] = "unknown"
        out.loc[out["gene"].isin(labels.heldout_nulls), "truth"] = "null"
        out.loc[out["gene"].isin(labels.heldout_positives), "truth"] = "positive"
    elif task == "achilles_no_positives":
        out["truth"] = "null"
    elif task == "psn1_nodiff":
        out["truth"] = "null"
    else:
        raise ValueError(f"unknown task {task}")
    return out
