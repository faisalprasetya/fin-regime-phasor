"""Vector-quantize (sigma, theta) jointly into a finite alphabet for the discrete-observation HQMM.

PLAN.md "Discretizing the phasor": quantize the polar pair, not z's raw real/imag
parts, to avoid distorting polar semantics. Codebook must be fit on the training
split only (leakage control).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class VQCodebook:
    centers: np.ndarray  # (n_symbols, 2) in (sigma, theta) space
    feature_mean: np.ndarray  # (2,) used to standardize before fitting/predicting
    feature_std: np.ndarray  # (2,)

    @property
    def n_symbols(self) -> int:
        return self.centers.shape[0]


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def fit_vq_codebook(
    sigma: np.ndarray, theta: np.ndarray, n_symbols: int, random_state: int = 0
) -> VQCodebook:
    """Fit a k-means codebook on (sigma, theta) pairs from the training fold only."""
    features = np.column_stack(
        [np.asarray(sigma, dtype=np.float64), np.asarray(theta, dtype=np.float64)]
    )
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0

    standardized = _standardize(features, mean, std)
    kmeans = KMeans(n_clusters=n_symbols, random_state=random_state, n_init=10)
    kmeans.fit(standardized)

    return VQCodebook(centers=kmeans.cluster_centers_, feature_mean=mean, feature_std=std)


def apply_vq_codebook(codebook: VQCodebook, sigma: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Assign each (sigma, theta) pair to its nearest codebook symbol (int array)."""
    features = np.column_stack(
        [np.asarray(sigma, dtype=np.float64), np.asarray(theta, dtype=np.float64)]
    )
    standardized = _standardize(features, codebook.feature_mean, codebook.feature_std)
    # (n_points, n_symbols) squared distances
    dists = ((standardized[:, None, :] - codebook.centers[None, :, :]) ** 2).sum(axis=-1)
    return dists.argmin(axis=1)
