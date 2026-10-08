"""Manuscript workflow helpers, extracted from scripts/utopia_train_models.py."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
from support.metadata import file_fingerprint, metadata_hash, timestamp_utc, write_json
from support.utopia_io import (
    concatenate_training_arrays,
    load_gene_roi_data,
    load_pickle,
    rescale_gene_images,
    save_pickle,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def fingerprint_inputs(
    data_dir: Path, config_path: Path, roi_indices: list[int], target_kind: str
) -> dict[str, Any]:
    files: dict[str, Path] = {"utopia_config": config_path}
    for roi_idx in roi_indices:
        files[f"roi_{roi_idx}_he_mask"] = (
            data_dir / "mask" / f"mask-small_roi_{roi_idx}_scale_1.png"
        )
        files[f"roi_{roi_idx}_embedding"] = (
            data_dir / "embeddings" / f"embeddings_roi_{roi_idx}_scale_1.npy"
        )
        files[f"roi_{roi_idx}_fold_masks"] = (
            data_dir / "calibration" / f"roi_{roi_idx}_fold_masks.pickle"
        )
        files[f"roi_{roi_idx}_center_masks"] = (
            data_dir / "calibration" / f"roi_{roi_idx}_center_masks.pickle"
        )
        if target_kind == "gene":
            files[f"roi_{roi_idx}_transcripts"] = (
                data_dir / "transcripts" / f"tr_image_roi_{roi_idx}_scale_1.npy"
            )
        else:
            files[f"roi_{roi_idx}_ct_image"] = (
                data_dir / "cell type" / f"ct_image_roi_{roi_idx}_scale_1.npy"
            )
            files[f"roi_{roi_idx}_ct_mask"] = (
                data_dir / "mask" / f"ct-mask-small_roi_{roi_idx}_scale_1.png"
            )
    return {key: file_fingerprint(path) for key, path in files.items()}


def identity_metadata(
    *,
    target_kind: str,
    role: str,
    data_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    roi_train_indices: list[int],
    roi_calib_indices: list[int],
    model_params: dict[str, Any],
    training_params: dict[str, Any],
    normalization: dict[str, Any] | None,
    fingerprints: dict[str, Any],
    fold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target_kind": target_kind,
        "role": role,
        "data_dir": str(data_dir),
        "config_path": str(config_path),
        "utopia_config": config,
        "roi_train_indices": roi_train_indices,
        "roi_calib_indices": roi_calib_indices,
        "model_params": model_params,
        "training_params": training_params,
        "normalization": normalization,
        "preprocessing_fingerprints": fingerprints,
        "fold": fold,
    }


def save_model_artifacts(
    *,
    save_dir: Path,
    target_kind: str,
    role: str,
    identity: dict[str, Any],
    model: Any,
    model_kind: str,
    history: dict[str, list[float]],
    suffix: str = "",
) -> dict[str, str]:
    from support.models import save_checkpoint

    digest = metadata_hash(identity)
    suffix_part = f"_{suffix}" if suffix else ""
    stem = f"{target_kind}_{role}{suffix_part}_{digest}"
    checkpoint_path = save_dir / f"{stem}.pt"
    metadata_path = save_dir / f"{stem}.json"
    history_path = save_dir / f"{stem}_history.json"
    metadata = dict(identity)
    metadata["metadata_hash"] = digest
    metadata["created_utc"] = timestamp_utc()
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["history_path"] = str(history_path)
    save_checkpoint(checkpoint_path, model, model_kind=model_kind, metadata=metadata)
    write_json(metadata_path, metadata)
    write_json(history_path, history)
    return {
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "history": str(history_path),
        "metadata_hash": digest,
    }


def artifact_exists(
    save_dir: Path,
    target_kind: str,
    role: str,
    identity: dict[str, Any],
    suffix: str = "",
) -> bool:
    digest = metadata_hash(identity)
    suffix_part = f"_{suffix}" if suffix else ""
    stem = f"{target_kind}_{role}{suffix_part}_{digest}"
    return (save_dir / f"{stem}.pt").exists() and (save_dir / f"{stem}.json").exists()


def run_gene_training(
    *,
    data_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    roi_train_indices: list[int],
    roi_calib_indices: list[int],
    save_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    from support.models import train_regressor

    roi_indices = sorted(set(roi_train_indices + roi_calib_indices))
    roi_data = load_gene_roi_data(data_dir, roi_indices)
    idx_to_gene = load_pickle(data_dir / "transcripts" / "idx_to_gene.pickle")
    thresholds = rescale_gene_images(roi_data, roi_train_indices, idx_to_gene)
    normalization = {
        "method": "per-gene percentile min-max",
        "lower_quantile": 0.0,
        "upper_quantile": 99.99,
        "thresholds": thresholds,
    }
    fingerprints = fingerprint_inputs(data_dir, config_path, roi_indices, "gene")
    model_params = {
        "architecture": "UTOPIA FeedForwardRegressor",
        "input_dim": int(next(iter(roi_data.values()))["x_image"].shape[-1]),
        "output_dim": int(next(iter(roi_data.values()))["y_image"].shape[-1]),
        "dropout_rate": args.gene_dropout,
    }
    training_params = {
        "profile": "utopia_defaults",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "fold_seed_policy": "same_seed_for_full_and_fold_models",
        "device": device,
        "loss": args.gene_loss,
        "beta": args.gene_beta,
        "asymmetric_loss_behavior": "UTOPIA model_gene double_sqrt",
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    full_identity = identity_metadata(
        target_kind="gene",
        role="full",
        data_dir=data_dir,
        config_path=config_path,
        config=config,
        roi_train_indices=roi_train_indices,
        roi_calib_indices=roi_calib_indices,
        model_params=model_params,
        training_params=training_params,
        normalization=normalization,
        fingerprints=fingerprints,
    )
    full_hash = metadata_hash(full_identity)
    save_pickle(save_dir / f"gene_thresholds_{full_hash}.pickle", thresholds)
    write_json(save_dir / f"gene_thresholds_{full_hash}.json", normalization)
    if args.resume and artifact_exists(save_dir, "gene", "full", full_identity):
        print("Skipping existing gene full-model checkpoint.")
    else:
        train_x, train_y = concatenate_training_arrays(roi_data, roi_train_indices)
        model, history = train_regressor(
            train_x,
            train_y,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta=args.gene_beta,
            loss=args.gene_loss,
            dropout_rate=args.gene_dropout,
            seed=args.seed,
            desc="gene full model",
        )
        artifacts.append(
            save_model_artifacts(
                save_dir=save_dir,
                target_kind="gene",
                role="full",
                identity=full_identity,
                model=model,
                model_kind="regressor",
                history=history,
            )
        )
    # Refit after excluding each spatial fold; use its center for calibration.
    fold_jobs: list[tuple[int, int, np.ndarray]] = []
    for roi_idx in roi_calib_indices:
        if roi_idx not in roi_train_indices:
            continue
        for fold_idx, fold_mask in enumerate(roi_data[roi_idx]["fold_masks"]):
            fold_jobs.append((roi_idx, fold_idx, fold_mask))
    if args.max_folds is not None:
        fold_jobs = fold_jobs[: args.max_folds]
    for roi_idx, fold_idx, fold_mask in progress(
        fold_jobs, desc="gene fold models", unit="fold"
    ):
        fold_identity = identity_metadata(
            target_kind="gene",
            role="fold",
            data_dir=data_dir,
            config_path=config_path,
            config=config,
            roi_train_indices=roi_train_indices,
            roi_calib_indices=roi_calib_indices,
            model_params=model_params,
            training_params=training_params,
            normalization=normalization,
            fingerprints=fingerprints,
            fold={"roi_idx": roi_idx, "fold_idx": fold_idx},
        )
        suffix = f"roi{roi_idx}_fold{fold_idx:03d}"
        if args.resume and artifact_exists(
            save_dir, "gene", "fold", fold_identity, suffix=suffix
        ):
            continue
        train_x, train_y = concatenate_training_arrays(
            roi_data, roi_train_indices, exclude_roi=roi_idx, exclude_mask=fold_mask
        )
        model, history = train_regressor(
            train_x,
            train_y,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta=args.gene_beta,
            loss=args.gene_loss,
            dropout_rate=args.gene_dropout,
            seed=args.seed,
            desc=f"gene roi {roi_idx} fold {fold_idx}",
        )
        artifacts.append(
            save_model_artifacts(
                save_dir=save_dir,
                target_kind="gene",
                role="fold",
                identity=fold_identity,
                model=model,
                model_kind="regressor",
                history=history,
                suffix=suffix,
            )
        )
    return artifacts
