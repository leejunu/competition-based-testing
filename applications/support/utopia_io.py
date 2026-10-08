"""Manuscript workflow helpers, extracted from code/utopia_compound/utopia_io.py."""

from __future__ import annotations
import configparser
import pickle
from pathlib import Path
from typing import Any
import numpy as np


def parse_utopia_config(config_path: str | Path) -> dict[str, Any]:
    parser = configparser.ConfigParser()
    parser.read(config_path)

    def ints(value: str) -> list[int]:
        return [int(item) for item in value.split()]

    def comma_ints(value: str) -> list[int]:
        return [int(item) for item in value.split(",")]

    return {
        "config_path": str(config_path),
        "data_dir": parser["global"].get("data_dir", "./"),
        "model_dir": parser["global"].get("model_dir", ""),
        "n_clusters": parser["clustering"].getint("n_clusters"),
        "scales": ints(parser["scale"]["scales"]),
        "num_roi": parser["roi"].getint("num_roi"),
        "roi_top_left": [
            [int(item) for item in pair.split(",")]
            for pair in parser["roi"]["top_left"].split(";")
        ],
        "roi_width": comma_ints(parser["roi"]["width"]),
        "roi_height": comma_ints(parser["roi"]["height"]),
        "roi_train_indices": ints(parser["calibration"]["roi_train_indices"]),
        "roi_calib_indices": ints(parser["calibration"]["roi_calib_indices"]),
        "n_cell_types": parser["calibration"].getint("n_cell_types", fallback=1),
        "predict_gene": parser["predict"].getboolean("gene", fallback=True),
        "predict_cell_type": parser["predict"].getboolean("cell_type", fallback=True),
        "infer_json": parser["inference"].get("infer_json", "./targets_to_infer.json"),
    }


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def load_mask(path: str | Path) -> np.ndarray:
    from PIL import Image

    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        return image.any(axis=2)
    return image > 0


def reshape_roi_embeddings(roi_embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    d = roi_embeddings.shape[1]
    reshaped = np.zeros((h, w, d), dtype=roi_embeddings.dtype)
    reshaped[mask] = roi_embeddings
    return reshaped


def load_fold_masks(
    data_dir: str | Path, roi_idx: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    data_dir = Path(data_dir)
    fold_masks = load_pickle(
        data_dir / "calibration" / f"roi_{roi_idx}_fold_masks.pickle"
    )
    center_masks = load_pickle(
        data_dir / "calibration" / f"roi_{roi_idx}_center_masks.pickle"
    )
    return (fold_masks, center_masks)


def load_gene_roi_data(
    data_dir: str | Path, roi_indices: list[int]
) -> dict[int, dict[str, Any]]:
    data_dir = Path(data_dir)
    out: dict[int, dict[str, Any]] = {}
    for roi_idx in roi_indices:
        he_mask = load_mask(data_dir / "mask" / f"mask-small_roi_{roi_idx}_scale_1.png")
        embeddings = np.load(
            data_dir / "embeddings" / f"embeddings_roi_{roi_idx}_scale_1.npy"
        )
        out[roi_idx] = {
            "x_image": reshape_roi_embeddings(embeddings, he_mask),
            "y_image": np.load(
                data_dir / "transcripts" / f"tr_image_roi_{roi_idx}_scale_1.npy"
            ).astype(np.uint16),
            "train_mask": he_mask,
            "calib_mask": he_mask,
            "fold_masks": load_fold_masks(data_dir, roi_idx)[0],
            "center_masks": load_fold_masks(data_dir, roi_idx)[1],
        }
    return out


def rescale_gene_images(
    roi_data: dict[int, dict[str, Any]],
    train_indices: list[int],
    idx_to_gene: dict[int, str] | None,
    lower_quantile: float = 0.0,
    upper_quantile: float = 99.99,
) -> dict[str, tuple[float, float]]:
    first = roi_data[train_indices[0]]["y_image"]
    n_genes = first.shape[2]
    thresholds: dict[str, tuple[float, float]] = {}
    for gene_idx in range(n_genes):
        values = np.concatenate(
            [
                roi_data[roi_idx]["y_image"][roi_data[roi_idx]["train_mask"], gene_idx]
                for roi_idx in train_indices
            ]
        )
        lower = float(np.percentile(values, lower_quantile))
        upper = float(np.percentile(values, upper_quantile))
        key = idx_to_gene[gene_idx] if idx_to_gene is not None else str(gene_idx)
        thresholds[key] = (lower, upper)
        for roi in roi_data.values():
            if roi["y_image"].dtype != np.float32:
                roi["y_image"] = roi["y_image"].astype(np.float32)
            if upper != lower:
                clipped = np.clip(roi["y_image"][..., gene_idx], lower, upper)
                scaled = (clipped - lower) / (upper - lower)
                roi["y_image"][..., gene_idx] = scaled
            else:
                roi["y_image"][..., gene_idx] = 0.0
    return thresholds


def concatenate_training_arrays(
    roi_data: dict[int, dict[str, Any]],
    train_indices: list[int],
    *,
    exclude_roi: int | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for roi_idx in train_indices:
        mask = roi_data[roi_idx]["train_mask"].copy()
        if exclude_roi == roi_idx and exclude_mask is not None:
            mask &= ~exclude_mask
        xs.append(roi_data[roi_idx]["x_image"][mask])
        ys.append(roi_data[roi_idx]["y_image"][mask])
    return (np.concatenate(xs, axis=0), np.concatenate(ys, axis=0))
