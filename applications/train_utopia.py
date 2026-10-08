"""Train the full TLS regression model and all calibration fold models."""

import argparse
from pathlib import Path

from support.utopia_io import parse_utopia_config
from support.training import run_gene_training
from support.metadata import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/utopia/demo")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results/utopia/models")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.epochs, args.batch_size, args.learning_rate, args.seed = 20, 8096, 0.001, 42
    args.gene_loss, args.gene_beta, args.gene_dropout = "amse", 2.0, 0.1
    args.max_folds = None
    data = args.data_dir.resolve()
    config_path = data / "utopia_config"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    # The manuscript trained on both ROIs, with held-out spatial folds for calibration.
    artifacts = run_gene_training(
        data_dir=data,
        config_path=config_path,
        config=parse_utopia_config(config_path),
        roi_train_indices=[0, 1],
        roi_calib_indices=[0, 1],
        save_dir=args.save_dir,
        device=args.device,
        args=args,
    )
    write_json(
        args.save_dir / "training_manifest.json",
        {"device": args.device, "artifacts": artifacts},
    )


if __name__ == "__main__":
    main()
