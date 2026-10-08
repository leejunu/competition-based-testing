"""Manuscript workflow helpers, extracted from code/utopia_compound/scoring.py."""

from __future__ import annotations
from pathlib import Path
from typing import Any


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Load a metadata-addressed UTOPIA compound checkpoint for notebook inference."""
    import torch
    from support.models import FeedForwardRegressor

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    params = metadata["model_params"]
    model_kind = checkpoint["model_kind"]
    if model_kind == "regressor":
        model = FeedForwardRegressor(
            input_dim=int(params["input_dim"]),
            output_dim=int(params["output_dim"]),
            dropout_rate=float(params.get("dropout_rate", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind!r}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return (model, metadata)
