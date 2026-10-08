"""Manuscript workflow helpers, extracted from code/utopia_compound/compound_inputs.py."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CompoundInputs:
    test_scores: np.ndarray
    idx_to_block: np.ndarray
    control_scores_by_block: dict[int, np.ndarray]
    control_null_membership: dict[int, np.ndarray]
    block_labels: np.ndarray
    test_y: np.ndarray | None = None
    test_truth: np.ndarray | None = None
    kept_test_mask: np.ndarray | None = None


def _finite_cluster_mask(scores: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    return np.isfinite(scores) & np.isfinite(clusters) & (clusters >= 0)
