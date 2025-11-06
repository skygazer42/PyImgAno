"""
Histogram Comparison for Anomaly Detection

Compares color/intensity histograms between normal samples and test samples.
Simple, fast, and effective for detecting color/brightness anomalies.

Reference:
    Swain, M. J., & Ballard, D. H. (1991). "Color indexing"
    IJCV 1991.

Usage:
    >>> from pyimgano.models import HistogramComparison
    >>> model = HistogramComparison(n_bins=64, method='chi_square')
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Literal
from skimage import color
import cv2

from ..base import BaseVisionClassicalDetector


class HistogramComparison(BaseVisionClassicalDetector):
    """
    Histogram comparison for anomaly detection.

    Compares histograms of test images against reference histograms from normal samples.
    Effective for detecting color deviations, brightness changes, and overall appearance
    anomalies. Widely used in quality control for consistent product appearance.

    Parameters
    ----------
    n_bins : int, default=64
        Number of histogram bins per channel
    method : str, default='chi_square'
        Comparison method: 'chi_square', 'correlation', 'intersection',
        'bhattacharyya', 'hellinger', 'kl_divergence'
    color_space : str, default='RGB'
        Color space: 'RGB', 'HSV', 'LAB', 'GRAY'
    spatial : bool, default=False
        If True, compute spatial histograms (grid-based)
    grid_size : tuple, default=(4, 4)
        Grid size for spatial histograms
    percentile_thresh : float, default=95
        Percentile for anomaly threshold
    resize_shape : tuple, optional
        Resize images to this shape

    Attributes
    ----------
    reference_histograms_ : list
        Reference histograms from normal samples
    threshold_ : float
        Anomaly detection threshold

    Examples
    --------
    >>> # Color consistency check
    >>> model = HistogramComparison(n_bins=64, color_space='HSV')
    >>> model.fit(X_train_normal)
    >>> scores = model.predict(X_test)

    >>> # Grayscale with spatial information
    >>> model = HistogramComparison(
    ...     n_bins=32, color_space='GRAY',
    ...     spatial=True, grid_size=(8, 8)
    ... )
    >>> model.fit(X_train)
    >>> anomaly_scores = model.predict(X_test)
    """

    def __init__(
        self,
        n_bins: int = 64,
        method: Literal['chi_square', 'correlation', 'intersection',
                       'bhattacharyya', 'hellinger', 'kl_divergence'] = 'chi_square',
        color_space: Literal['RGB', 'HSV', 'LAB', 'GRAY'] = 'RGB',
        spatial: bool = False,
        grid_size: Tuple[int, int] = (4, 4),
        percentile_thresh: float = 95,
        resize_shape: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.n_bins = n_bins
        self.method = method
        self.color_space = color_space
        self.spatial = spatial
        self.grid_size = grid_size
        self.percentile_thresh = percentile_thresh
        self.resize_shape = resize_shape

        self.reference_histograms_ = None
        self.threshold_ = None

    def _convert_color_space(self, image: NDArray) -> NDArray:
        """Convert image to specified color space."""
        if self.color_space == 'GRAY':
            if len(image.shape) == 3:
                return color.rgb2gray(image)
            return image

        # Assume input is RGB
        if self.color_space == 'RGB':
            return image
        elif self.color_space == 'HSV':
            return color.rgb2hsv(image)
        elif self.color_space == 'LAB':
            return color.rgb2lab(image)

        return image

    def _compute_histogram(self, image: NDArray) -> NDArray:
        """Compute histogram features."""
        # Resize if specified
        if self.resize_shape is not None:
            from skimage.transform import resize
            image = resize(image, self.resize_shape, anti_aliasing=True)

        # Convert color space
        image = self._convert_color_space(image)

        if self.spatial:
            return self._compute_spatial_histogram(image)
        else:
            return self._compute_global_histogram(image)

    def _compute_global_histogram(self, image: NDArray) -> NDArray:
        """Compute global histogram."""
        if len(image.shape) == 2:
            # Grayscale
            hist, _ = np.histogram(image.ravel(), bins=self.n_bins, range=(0, 1), density=True)
            return hist
        else:
            # Multi-channel
            histograms = []
            for c in range(image.shape[2]):
                hist, _ = np.histogram(
                    image[:, :, c].ravel(),
                    bins=self.n_bins,
                    range=(0, 1),
                    density=True
                )
                histograms.append(hist)
            return np.concatenate(histograms)

    def _compute_spatial_histogram(self, image: NDArray) -> NDArray:
        """Compute spatial histogram."""
        h, w = image.shape[:2]
        grid_h, grid_w = self.grid_size
        cell_h = h // grid_h
        cell_w = w // grid_w

        features = []

        for i in range(grid_h):
            for j in range(grid_w):
                # Extract cell
                if len(image.shape) == 2:
                    cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    hist, _ = np.histogram(cell.ravel(), bins=self.n_bins, range=(0, 1), density=True)
                    features.extend(hist)
                else:
                    cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :]
                    for c in range(cell.shape[2]):
                        hist, _ = np.histogram(
                            cell[:, :, c].ravel(),
                            bins=self.n_bins,
                            range=(0, 1),
                            density=True
                        )
                        features.extend(hist)

        return np.array(features)

    def _compare_histograms(self, hist1: NDArray, hist2: NDArray) -> float:
        """Compare two histograms."""
        # Normalize
        hist1 = hist1 / (np.sum(hist1) + 1e-10)
        hist2 = hist2 / (np.sum(hist2) + 1e-10)

        if self.method == 'chi_square':
            # Chi-square distance
            return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))

        elif self.method == 'correlation':
            # Correlation (return 1 - correlation for dissimilarity)
            mean1 = np.mean(hist1)
            mean2 = np.mean(hist2)
            num = np.sum((hist1 - mean1) * (hist2 - mean2))
            den = np.sqrt(np.sum((hist1 - mean1) ** 2) * np.sum((hist2 - mean2) ** 2))
            corr = num / (den + 1e-10)
            return 1 - corr

        elif self.method == 'intersection':
            # Histogram intersection (return 1 - intersection for dissimilarity)
            return 1 - np.sum(np.minimum(hist1, hist2))

        elif self.method == 'bhattacharyya':
            # Bhattacharyya distance
            return -np.log(np.sum(np.sqrt(hist1 * hist2)) + 1e-10)

        elif self.method == 'hellinger':
            # Hellinger distance
            return np.sqrt(1 - np.sum(np.sqrt(hist1 * hist2)))

        elif self.method == 'kl_divergence':
            # Kullback-Leibler divergence
            return np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))

        return 0.0

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'HistogramComparison':
        """
        Fit histogram comparison model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : HistogramComparison
            Fitted estimator
        """
        # Compute histograms for all training samples
        print("Computing reference histograms...")
        self.reference_histograms_ = []
        for i in range(len(X)):
            hist = self._compute_histogram(X[i])
            self.reference_histograms_.append(hist)

        # Compute threshold based on pairwise distances
        print("Computing anomaly threshold...")
        distances = []
        for i in range(len(self.reference_histograms_)):
            for j in range(i + 1, len(self.reference_histograms_)):
                dist = self._compare_histograms(
                    self.reference_histograms_[i],
                    self.reference_histograms_[j]
                )
                distances.append(dist)

        if distances:
            self.threshold_ = np.percentile(distances, self.percentile_thresh)
        else:
            self.threshold_ = 0.0

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (histogram distance)
        """
        self._check_is_fitted()

        scores = []
        for i in range(len(X)):
            # Compute histogram
            test_hist = self._compute_histogram(X[i])

            # Compare with all reference histograms and take minimum distance
            min_dist = float('inf')
            for ref_hist in self.reference_histograms_:
                dist = self._compare_histograms(test_hist, ref_hist)
                min_dist = min(min_dist, dist)

            scores.append(min_dist)

        return np.array(scores)

    def predict_label(self, X: NDArray) -> NDArray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Test images

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels (1 = anomaly, 0 = normal)
        """
        scores = self.predict(X)
        return (scores > self.threshold_).astype(int)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_bins': self.n_bins,
            'method': self.method,
            'color_space': self.color_space,
            'spatial': self.spatial,
            'grid_size': self.grid_size,
            'percentile_thresh': self.percentile_thresh,
            'resize_shape': self.resize_shape,
        }
