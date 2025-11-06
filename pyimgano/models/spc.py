"""
Statistical Process Control (SPC) for Anomaly Detection

Classic industrial quality control method using control charts (Shewhart, CUSUM, EWMA).
The gold standard for manufacturing process monitoring.

Reference:
    Montgomery, D. C. (2009). "Introduction to Statistical Quality Control"
    Wiley, 6th Edition.

Usage:
    >>> from pyimgano.models import SPC
    >>> model = SPC(chart_type='shewhart', n_sigma=3)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Literal
from sklearn.decomposition import PCA

from ..base import BaseVisionClassicalDetector


class SPC(BaseVisionClassicalDetector):
    """
    Statistical Process Control for anomaly detection.

    Implements classic control charts (Shewhart, CUSUM, EWMA) for industrial
    quality control. Monitors process statistics and detects deviations from
    normal behavior. The industry standard for manufacturing defect detection.

    Parameters
    ----------
    chart_type : str, default='shewhart'
        Control chart type: 'shewhart', 'cusum', 'ewma'
    n_sigma : float, default=3
        Number of standard deviations for control limits (Shewhart)
    h : float, default=5
        Decision interval for CUSUM
    k : float, default=0.5
        Reference value for CUSUM (fraction of std)
    lambda_ewma : float, default=0.2
        Smoothing parameter for EWMA (0 < λ ≤ 1)
    L : float, default=3
        Control limit multiplier for EWMA
    feature_extraction : str, default='pca'
        Feature extraction method: 'pca', 'mean', 'std', 'mean_std'
    n_components : int, default=10
        Number of PCA components
    resize_shape : tuple, optional
        Resize images to this shape

    Attributes
    ----------
    pca_ : PCA
        PCA transformer (if using PCA)
    mu_ : ndarray
        Process mean
    sigma_ : ndarray
        Process standard deviation
    ucl_ : ndarray
        Upper control limit
    lcl_ : ndarray
        Lower control limit

    Examples
    --------
    >>> # Shewhart control chart (3-sigma rule)
    >>> model = SPC(chart_type='shewhart', n_sigma=3)
    >>> model.fit(X_train_normal)
    >>> scores = model.predict(X_test)

    >>> # CUSUM for detecting small shifts
    >>> model = SPC(chart_type='cusum', h=5, k=0.5)
    >>> model.fit(X_train)
    >>> anomalies = model.predict_label(X_test)

    >>> # EWMA for smooth tracking
    >>> model = SPC(chart_type='ewma', lambda_ewma=0.2, L=3)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
    """

    def __init__(
        self,
        chart_type: Literal['shewhart', 'cusum', 'ewma'] = 'shewhart',
        n_sigma: float = 3.0,
        h: float = 5.0,
        k: float = 0.5,
        lambda_ewma: float = 0.2,
        L: float = 3.0,
        feature_extraction: Literal['pca', 'mean', 'std', 'mean_std'] = 'pca',
        n_components: int = 10,
        resize_shape: Optional[Tuple[int, int]] = (64, 64)
    ):
        super().__init__()
        self.chart_type = chart_type
        self.n_sigma = n_sigma
        self.h = h
        self.k = k
        self.lambda_ewma = lambda_ewma
        self.L = L
        self.feature_extraction = feature_extraction
        self.n_components = n_components
        self.resize_shape = resize_shape

        self.pca_ = None
        self.mu_ = None
        self.sigma_ = None
        self.ucl_ = None
        self.lcl_ = None
        self.cusum_pos_ = None
        self.cusum_neg_ = None

    def _extract_features(self, X: NDArray) -> NDArray:
        """Extract statistical features from images."""
        # Resize if specified
        if self.resize_shape is not None:
            from skimage.transform import resize
            X_resized = []
            for i in range(len(X)):
                img = resize(X[i], self.resize_shape, anti_aliasing=True)
                X_resized.append(img)
            X = np.array(X_resized)

        # Flatten images
        n_samples = len(X)
        X_flat = X.reshape(n_samples, -1)

        if self.feature_extraction == 'pca':
            return X_flat
        elif self.feature_extraction == 'mean':
            return np.mean(X_flat, axis=1, keepdims=True)
        elif self.feature_extraction == 'std':
            return np.std(X_flat, axis=1, keepdims=True)
        elif self.feature_extraction == 'mean_std':
            means = np.mean(X_flat, axis=1, keepdims=True)
            stds = np.std(X_flat, axis=1, keepdims=True)
            return np.hstack([means, stds])

        return X_flat

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'SPC':
        """
        Fit SPC model to establish control limits.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Training images (normal, in-control process)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : SPC
            Fitted estimator
        """
        # Extract features
        print("Extracting features...")
        features = self._extract_features(X)

        # Apply PCA if specified
        if self.feature_extraction == 'pca':
            self.pca_ = PCA(n_components=min(self.n_components, features.shape[1]))
            features = self.pca_.fit_transform(features)

        # Compute process statistics
        self.mu_ = np.mean(features, axis=0)
        self.sigma_ = np.std(features, axis=0)

        # Compute control limits based on chart type
        if self.chart_type == 'shewhart':
            # Shewhart chart: μ ± n*σ
            self.ucl_ = self.mu_ + self.n_sigma * self.sigma_
            self.lcl_ = self.mu_ - self.n_sigma * self.sigma_

        elif self.chart_type == 'cusum':
            # CUSUM: initialize cumulative sums
            self.cusum_pos_ = np.zeros_like(self.mu_)
            self.cusum_neg_ = np.zeros_like(self.mu_)

        elif self.chart_type == 'ewma':
            # EWMA: compute control limits
            # UCL/LCL = μ ± L * σ * sqrt(λ / (2 - λ))
            factor = self.L * self.sigma_ * np.sqrt(
                self.lambda_ewma / (2 - self.lambda_ewma)
            )
            self.ucl_ = self.mu_ + factor
            self.lcl_ = self.mu_ - factor

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
            Anomaly scores
        """
        self._check_is_fitted()

        # Extract features
        features = self._extract_features(X)

        # Apply PCA if used in training
        if self.pca_ is not None:
            features = self.pca_.transform(features)

        scores = []

        if self.chart_type == 'shewhart':
            # Shewhart: distance from control limits
            for i in range(len(features)):
                # Check how many features are out of control
                upper_violations = np.sum(features[i] > self.ucl_)
                lower_violations = np.sum(features[i] < self.lcl_)
                total_violations = upper_violations + lower_violations

                # Compute normalized distance
                upper_dist = np.maximum(0, features[i] - self.ucl_)
                lower_dist = np.maximum(0, self.lcl_ - features[i])
                total_dist = np.sum(upper_dist + lower_dist)

                # Score combines violations and distance
                score = total_violations + total_dist
                scores.append(score)

        elif self.chart_type == 'cusum':
            # CUSUM: cumulative sum of deviations
            # Reset CUSUM for new predictions
            cusum_pos = self.cusum_pos_.copy()
            cusum_neg = self.cusum_neg_.copy()

            for i in range(len(features)):
                # Update positive CUSUM
                cusum_pos = np.maximum(
                    0,
                    features[i] - (self.mu_ + self.k * self.sigma_) + cusum_pos
                )

                # Update negative CUSUM
                cusum_neg = np.maximum(
                    0,
                    (self.mu_ - self.k * self.sigma_) - features[i] + cusum_neg
                )

                # Score is max CUSUM magnitude
                score = np.max(cusum_pos) + np.max(cusum_neg)
                scores.append(score)

        elif self.chart_type == 'ewma':
            # EWMA: exponentially weighted moving average
            ewma = self.mu_.copy()

            for i in range(len(features)):
                # Update EWMA
                ewma = self.lambda_ewma * features[i] + (1 - self.lambda_ewma) * ewma

                # Check violations
                upper_violations = np.sum(ewma > self.ucl_)
                lower_violations = np.sum(ewma < self.lcl_)

                # Compute distance from limits
                upper_dist = np.maximum(0, ewma - self.ucl_)
                lower_dist = np.maximum(0, self.lcl_ - ewma)

                score = upper_violations + lower_violations + np.sum(upper_dist + lower_dist)
                scores.append(score)

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
            Binary labels (1 = out of control/anomaly, 0 = in control/normal)
        """
        scores = self.predict(X)

        # Threshold: any non-zero score indicates out-of-control
        if self.chart_type == 'cusum':
            # CUSUM: threshold at h
            labels = (scores > self.h).astype(int)
        else:
            # Shewhart and EWMA: any violation is anomaly
            labels = (scores > 0).astype(int)

        return labels

    def get_control_limits(self) -> dict:
        """
        Get control limits.

        Returns
        -------
        limits : dict
            Control limits (UCL, LCL, Center Line)
        """
        self._check_is_fitted()

        return {
            'center_line': self.mu_,
            'ucl': self.ucl_ if self.ucl_ is not None else None,
            'lcl': self.lcl_ if self.lcl_ is not None else None,
            'sigma': self.sigma_,
        }

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'chart_type': self.chart_type,
            'n_sigma': self.n_sigma,
            'h': self.h,
            'k': self.k,
            'lambda_ewma': self.lambda_ewma,
            'L': self.L,
            'feature_extraction': self.feature_extraction,
            'n_components': self.n_components,
            'resize_shape': self.resize_shape,
        }
