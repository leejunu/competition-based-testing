"""Score the whole slide and held-out calibration folds, preserving UTOPIA normalization."""

import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np
from support.scoring import load_model_from_checkpoint
from support.utopia_io import load_mask, parse_utopia_config, reshape_roi_embeddings

ROOT = Path(__file__).resolve().parents[1]


def score(data_dir, model_dir, output_dir, scale, device):
    DATA = data_dir
    MODEL_DIR = model_dir
    OUT_ROOT = output_dir
    ROI_TRAIN_INDICES = (0, 1)
    ROI_CALIB_INDICES = (0, 1)
    SCALE = scale
    N_CLUSTERS = 15
    TARGET_GROUP_NAME = "TLS_meta"
    NULL_NUMBER_OVERRIDE = 0.05
    DEVICE = device
    BATCH_SIZE = 8192
    MODEL_RUN_TAG = "train01_calib01"
    TEST_REGION_MODE = "out_of_train_roi"
    NORMALIZATION_MODE = "utopia_incremental"
    REFRESH_SCORE_CACHE = False

    def safe_name(value):
        return re.sub("[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()

    def block_sum2d(array, scale):
        array = np.asarray(array)
        h = array.shape[0] // scale * scale
        w = array.shape[1] // scale * scale
        return (
            array[:h, :w].reshape(h // scale, scale, w // scale, scale).sum(axis=(1, 3))
        )

    def block_any2d(array, scale):
        return block_sum2d(np.asarray(array, dtype=np.uint8), scale) > 0

    def out_of_train_roi_mask(data_dir, scale, train_roi_indices):
        tissue = load_mask(data_dir / f"mask/mask-small_scale_{scale}.png")
        config = parse_utopia_config(data_dir / "utopia_config")
        train_roi_indices = tuple((int(roi) for roi in train_roi_indices))
        if not train_roi_indices:
            raise ValueError("ROI_TRAIN_INDICES must contain at least one ROI.")
        train_roi_mask = np.zeros_like(tissue, dtype=bool)
        for roi in train_roi_indices:
            row, col = config["roi_top_left"][roi]
            row //= scale
            col //= scale
            height = config["roi_height"][roi] // scale
            width = config["roi_width"][roi] // scale
            train_roi_mask[row : row + height, col : col + width] = True
        return tissue & ~train_roi_mask

    def load_gene_group(data_dir, target_name, null_number_override=None):
        with open(data_dir / "transcripts/idx_to_gene.pickle", "rb") as handle:
            idx_to_gene = pickle.load(handle)
        gene_to_idx = {gene: idx for idx, gene in idx_to_gene.items()}
        with open(data_dir / "targets_to_infer.json") as handle:
            inference_config = json.load(handle)
        for null_group in inference_config.get("gene_groups", []):
            null_number = float(null_group["null_number"])
            for group in null_group.get("groups", []):
                genes = list(group["genes"])
                name = group.get(
                    "name", genes[0] if len(genes) == 1 else "_".join(genes)
                )
                if target_name == name or target_name in genes:
                    return {
                        "name": name,
                        "genes": genes,
                        "gene_indices": np.asarray(
                            [gene_to_idx[gene] for gene in genes], dtype=int
                        ),
                        "null_number": float(
                            null_number
                            if null_number_override is None
                            else null_number_override
                        ),
                        "idx_to_gene": idx_to_gene,
                        "gene_to_idx": gene_to_idx,
                    }
        if target_name not in gene_to_idx:
            raise ValueError(
                f"{target_name!r} was not found in targets_to_infer.json or idx_to_gene.pickle."
            )
        if null_number_override is None:
            raise ValueError(
                "Set NULL_NUMBER_OVERRIDE for a target that is not listed in targets_to_infer.json."
            )
        return {
            "name": target_name,
            "genes": [target_name],
            "gene_indices": np.asarray([gene_to_idx[target_name]], dtype=int),
            "null_number": float(null_number_override),
            "idx_to_gene": idx_to_gene,
            "gene_to_idx": gene_to_idx,
        }

    def latest_checkpoint(model_dir, pattern):
        paths = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not paths:
            raise FileNotFoundError(
                f"No checkpoint matching {pattern!r} found in {model_dir}"
            )
        return paths[-1]

    def parse_fold_index(path):
        match = re.search("fold(\\d+)_", path.name)
        if match is None:
            raise ValueError(f"Cannot parse fold index from {path.name}")
        return int(match.group(1))

    def require_checkpoint_metadata(
        metadata, *, expected_target_kind, expected_role, roi=None, fold_idx=None
    ):
        target_kind = metadata.get("target_kind")
        role = metadata.get("role")
        metadata_train = tuple(metadata.get("roi_train_indices", ()))
        metadata_calib = tuple(metadata.get("roi_calib_indices", ()))
        if target_kind and target_kind != expected_target_kind:
            raise RuntimeError(
                f"Checkpoint target_kind {target_kind!r} does not match {expected_target_kind!r}."
            )
        if role and role != expected_role:
            raise RuntimeError(
                f"Checkpoint role {role!r} does not match {expected_role!r}."
            )
        if metadata_train and metadata_train != ROI_TRAIN_INDICES:
            raise RuntimeError(
                f"Checkpoint roi_train_indices {metadata_train} do not match notebook config {ROI_TRAIN_INDICES}."
            )
        if metadata_calib and metadata_calib != ROI_CALIB_INDICES:
            raise RuntimeError(
                f"Checkpoint roi_calib_indices {metadata_calib} do not match notebook config {ROI_CALIB_INDICES}."
            )
        if expected_role == "fold":
            fold_info = metadata.get("fold") or {}
            if int(fold_info.get("roi_idx", -1)) != int(roi) or int(
                fold_info.get("fold_idx", -1)
            ) != int(fold_idx):
                raise RuntimeError(
                    f"Fold checkpoint metadata {fold_info} does not match ROI {roi}, fold {fold_idx}."
                )

    def score_weighted_genes(
        model, x, gene_indices, gene_weights, *, device, batch_size
    ):
        import torch

        model.to(device)
        model.eval()
        gene_indices_t = torch.as_tensor(gene_indices, dtype=torch.long, device=device)
        gene_weights_t = torch.as_tensor(
            gene_weights, dtype=torch.float32, device=device
        )
        out = np.empty(x.shape[0], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                batch = torch.as_tensor(
                    np.asarray(x[start:stop]), dtype=torch.float32, device=device
                )
                pred = model(batch)
                weighted = pred.index_select(1, gene_indices_t).matmul(gene_weights_t)
                out[start:stop] = weighted.detach().cpu().numpy().astype(np.float32)
        return out

    config = parse_utopia_config(DATA / "utopia_config")
    target = load_gene_group(DATA, TARGET_GROUP_NAME, NULL_NUMBER_OVERRIDE)
    null_number = target["null_number"]

    full_ckpt = latest_checkpoint(MODEL_DIR, "gene_full_*.pt")
    fold_ckpts_by_roi = {
        roi: sorted(MODEL_DIR.glob(f"gene_fold_roi{roi}_fold*.pt"))
        for roi in ROI_CALIB_INDICES
    }

    model, model_metadata = load_model_from_checkpoint(full_ckpt, device=DEVICE)
    require_checkpoint_metadata(
        model_metadata, expected_target_kind="gene", expected_role="full"
    )
    thresholds = model_metadata["normalization"]["thresholds"]
    gene_weights = np.asarray(
        [thresholds[gene][1] for gene in target["genes"]], dtype=np.float32
    )

    print("target:", target["name"], target["genes"])
    print("null_number:", null_number)
    print("gene_indices:", target["gene_indices"].tolist())
    print("gene_weights:", gene_weights.tolist())
    print("full checkpoint:", full_ckpt)
    print(
        "fold checkpoints by ROI:",
        {roi: len(paths) for roi, paths in fold_ckpts_by_roi.items()},
    )
    safe_target = safe_name(target["name"])
    OUT = OUT_ROOT / f"{safe_target}_scale{SCALE}_{TEST_REGION_MODE}_{MODEL_RUN_TAG}"
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_HASH = safe_name(model_metadata.get("metadata_hash", full_ckpt.stem))
    CACHE = (
        OUT
        / f"score_cache_{safe_target}_scale{SCALE}_{MODEL_RUN_TAG}_{MODEL_METADATA_HASH}_{NORMALIZATION_MODE}.npz"
    )

    def build_or_load_score_arrays():
        if CACHE.exists() and not REFRESH_SCORE_CACHE:
            with np.load(CACHE, allow_pickle=False) as data:
                return {
                    key: data[key].item() if data[key].shape == () else data[key]
                    for key in data.files
                }

        print("Scoring full WSI...")
        tissue1 = load_mask(DATA / "mask/mask-small_scale_1.png")
        x_wsi = np.load(DATA / "embeddings/embeddings_scale_1.npy", mmap_mode="r")
        weighted_wsi_flat = score_weighted_genes(
            model,
            x_wsi,
            target["gene_indices"],
            gene_weights,
            device=DEVICE,
            batch_size=BATCH_SIZE,
        )
        pred_wsi1 = np.zeros(tissue1.shape, dtype=np.float32)
        pred_wsi1[tissue1] = weighted_wsi_flat
        del weighted_wsi_flat

        true_wsi1 = np.sum(
            np.load(DATA / "transcripts/tr_image_scale_1.npy", mmap_mode="r")[
                ..., target["gene_indices"]
            ],
            axis=-1,
        ).astype(np.float32)
        pred_wsi = block_sum2d(pred_wsi1, SCALE).astype(np.float32)
        true_wsi = block_sum2d(true_wsi1, SCALE).astype(np.float32)

        test_mask = out_of_train_roi_mask(DATA, SCALE, ROI_TRAIN_INDICES)
        clusters_wsi = np.load(
            DATA
            / f"histology_clustering/he_clusters_image_scale_{SCALE}_num_{N_CLUSTERS}.npy"
        )
        test_yhat_raw = pred_wsi[test_mask]
        test_y_raw = true_wsi[test_mask]
        test_clusters = clusters_wsi[test_mask]
        test_coords = np.argwhere(test_mask).astype(np.float32)

        calib_pred_parts = []
        calib_true_parts = []
        calib_cluster_parts = []
        calib_coord_parts = []
        calib_roi_parts = []
        calib_percentiles = {}

        for roi in ROI_CALIB_INDICES:
            print(f"Scoring calibration ROI {roi}...")
            roi_mask1 = load_mask(DATA / f"mask/mask-small_roi_{roi}_scale_1.png")
            x_roi_flat = np.load(
                DATA / f"embeddings/embeddings_roi_{roi}_scale_1.npy", mmap_mode="r"
            )
            x_roi1 = reshape_roi_embeddings(x_roi_flat, roi_mask1)
            with open(
                DATA / f"calibration/roi_{roi}_center_masks.pickle", "rb"
            ) as handle:
                center_masks = pickle.load(handle)
            fold_ckpts = fold_ckpts_by_roi[roi]
            if len(fold_ckpts) != len(center_masks):
                raise RuntimeError(
                    f"ROI {roi}: expected {len(center_masks)} fold checkpoints, found {len(fold_ckpts)}. "
                    "Finish the configured ROI training first, then rerun this notebook."
                )
            fold_indices = [parse_fold_index(path) for path in fold_ckpts]
            if sorted(fold_indices) != list(range(len(center_masks))):
                raise RuntimeError(
                    f"ROI {roi}: fold checkpoint indices do not match center masks: {fold_indices}."
                )

            pred_roi1 = np.zeros(roi_mask1.shape, dtype=np.float32)
            for pos, ckpt in enumerate(fold_ckpts, start=1):
                fold_idx = parse_fold_index(ckpt)
                fold_model, fold_metadata = load_model_from_checkpoint(
                    ckpt, device=DEVICE
                )
                require_checkpoint_metadata(
                    fold_metadata,
                    expected_target_kind="gene",
                    expected_role="fold",
                    roi=roi,
                    fold_idx=fold_idx,
                )
                fold_mask = center_masks[fold_idx] & roi_mask1
                if np.any(fold_mask):
                    pred_roi1[fold_mask] = score_weighted_genes(
                        fold_model,
                        x_roi1[fold_mask],
                        target["gene_indices"],
                        gene_weights,
                        device=DEVICE,
                        batch_size=BATCH_SIZE,
                    )
                if pos % 25 == 0 or pos == len(fold_ckpts):
                    print(f"  ROI {roi}: scored folds {pos}/{len(fold_ckpts)}")

            true_roi1 = np.sum(
                np.load(
                    DATA / f"transcripts/tr_image_roi_{roi}_scale_1.npy", mmap_mode="r"
                )[..., target["gene_indices"]],
                axis=-1,
            ).astype(np.float32)
            pred_roi = block_sum2d(pred_roi1, SCALE).astype(np.float32)
            true_roi = block_sum2d(true_roi1, SCALE).astype(np.float32)
            calib_mask = load_mask(
                DATA / f"mask/mask-small_roi_{roi}_scale_{SCALE}.png"
            )
            clusters_roi = np.load(
                DATA
                / f"histology_clustering/he_clusters_image_roi_{roi}_scale_{SCALE}_num_{N_CLUSTERS}.npy"
            )
            roi_origin = np.asarray(
                config["roi_top_left"][roi], dtype=np.float32
            ) / float(SCALE)
            coords = np.argwhere(calib_mask).astype(np.float32)
            coords[:, 0] += roi_origin[0]
            coords[:, 1] += roi_origin[1]

            calib_pred_parts.append(pred_roi[calib_mask])
            calib_true_parts.append(true_roi[calib_mask])
            calib_cluster_parts.append(clusters_roi[calib_mask])
            calib_coord_parts.append(coords)
            calib_roi_parts.append(np.full(int(calib_mask.sum()), roi, dtype=np.int16))
            calib_percentiles[roi] = float(np.percentile(true_roi[calib_mask], 99.99))

        if NORMALIZATION_MODE == "global":
            calib_max_for_test = max(1.0, max(calib_percentiles.values()))
            calib_yhat_parts = [part / calib_max_for_test for part in calib_pred_parts]
            calib_y_parts = [part / calib_max_for_test for part in calib_true_parts]
        elif NORMALIZATION_MODE == "utopia_incremental":
            running = 1.0
            calib_yhat_parts = []
            calib_y_parts = []
            for roi, pred_part, true_part in zip(
                ROI_CALIB_INDICES, calib_pred_parts, calib_true_parts
            ):
                running = max(running, calib_percentiles[roi])
                calib_yhat_parts.append(pred_part / running)
                calib_y_parts.append(true_part / running)
            calib_max_for_test = running
        else:
            raise ValueError(
                "NORMALIZATION_MODE must be 'utopia_incremental' or 'global'."
            )

        payload = {
            "calib_yhat": np.concatenate(calib_yhat_parts).astype(np.float32),
            "calib_y": np.concatenate(calib_y_parts).astype(np.float32),
            "calib_clusters": np.concatenate(calib_cluster_parts).astype(np.int16),
            "calib_coords": np.vstack(calib_coord_parts).astype(np.float32),
            "calib_roi": np.concatenate(calib_roi_parts).astype(np.int16),
            "test_yhat": (test_yhat_raw / calib_max_for_test).astype(np.float32),
            "test_y": (test_y_raw / calib_max_for_test).astype(np.float32),
            "test_clusters": test_clusters.astype(np.int16),
            "test_coords": test_coords.astype(np.float32),
            "test_mask": test_mask.astype(bool),
            "calib_max_for_test": np.asarray(float(calib_max_for_test)),
            "normalization_mode": np.asarray(NORMALIZATION_MODE),
            "scale": np.asarray(int(SCALE)),
            "model_run_tag": np.asarray(MODEL_RUN_TAG),
            "test_region_mode": np.asarray(TEST_REGION_MODE),
            "model_metadata_hash": np.asarray(model_metadata.get("metadata_hash", "")),
            "full_checkpoint": np.asarray(str(full_ckpt)),
            "roi_train_indices": np.asarray(ROI_TRAIN_INDICES, dtype=np.int16),
            "roi_calib_indices": np.asarray(ROI_CALIB_INDICES, dtype=np.int16),
        }
        np.savez_compressed(CACHE, **payload)
        return payload

    score_arrays = build_or_load_score_arrays()
    print("cache:", CACHE)
    print("calibration points:", score_arrays["calib_yhat"].shape[0])
    print("out-of-train-ROI test points:", score_arrays["test_yhat"].shape[0])
    print("calib_max_for_test:", score_arrays["calib_max_for_test"])
    return CACHE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/utopia/demo")
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "results/utopia/models"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/utopia/scores")
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    for scale in args.scales:
        print(score(args.data_dir, args.model_dir, args.output, scale, args.device))


if __name__ == "__main__":
    main()
