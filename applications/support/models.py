"""Manuscript workflow helpers, extracted from code/utopia_compound/torch_models.py."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "utopia_compound.torch_models requires PyTorch. Install torch before running training."
    ) from exc
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class NumpyDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        if self.y is None:
            return self.x[index]
        return (self.x[index], self.y[index])


class ShiftedELU(nn.Module):

    def __init__(self, alpha: float = 0.01, beta: float = 0.01):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class FeedForwardRegressor(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim),
            ShiftedELU(alpha=0.01, beta=0.01),
        )

    def forward(self, x):
        return self.network(x)


class AsymmetricMSELoss(nn.Module):

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        diff = target - pred
        mask = diff > 0
        loss = torch.mean(mask * (self.beta * diff**2) + ~mask * diff**2)
        return torch.sqrt(loss)


def set_utopia_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: str,
    epochs: int = 20,
    batch_size: int = 8096,
    learning_rate: float = 0.001,
    beta: float = 2.0,
    loss: str = "amse",
    dropout_rate: float = 0.1,
    seed: int = 42,
    desc: str = "gene model",
) -> tuple[FeedForwardRegressor, dict[str, list[float]]]:
    # Preserve UTOPIA's seeded initialization and asymmetric training objective.
    set_utopia_seeds(seed)
    model = FeedForwardRegressor(
        x_train.shape[1], y_train.shape[1], dropout_rate=dropout_rate
    ).to(device)
    dataset = NumpyDataset(x_train, y_train.astype(np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion: nn.Module = (
        nn.MSELoss() if loss == "mse" else AsymmetricMSELoss(beta=beta)
    )
    history = {"train_loss": []}
    epochs_iter = _progress(range(epochs), total=epochs, desc=desc, unit="epoch")
    for _ in epochs_iter:
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred = model(batch_x)
            train_loss = torch.sqrt(criterion(pred, batch_y))
            train_loss.backward()
            optimizer.step()
            losses.append(float(train_loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(avg_loss)
        if hasattr(epochs_iter, "set_postfix"):
            epochs_iter.set_postfix(train_loss=f"{avg_loss:.4g}")
    return (model, history)


def save_checkpoint(
    path: str | Path, model: nn.Module, *, model_kind: str, metadata: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": model_kind,
            "metadata": metadata,
            "state_dict": model.state_dict(),
        },
        path,
    )
