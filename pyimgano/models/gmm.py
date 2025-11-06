"""
Gaussian Mixture Model (GMM) for Anomaly Detection

A classical statistical approach using Gaussian Mixture Models to model
the distribution of normal samples. Anomalies are detected as samples
with low likelihood under the learned GMM.

Reference:
    Reynolds, D. A. (2009). "Gaussian Mixture Models"
    Encyclopedia of Biometrics, pp. 659-663.

Usage:
    >>> from pyimgano.models import GMM
    >>> model = GMM(n_components=5, covariance_type='full')
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from ..base import BaseVisionClassicalDetector


class GMM(BaseVisionClassicalDetector):
    """
    Gaussian Mixture Model for anomaly detection.

    This classical approach models the distribution of normal samples
    using a mixture of Gaussian distributions. Anomalies are identified
    as samples with low probability density under the learned model.

    Parameters
    ----------
    n_components : int, default=5
        Number of Gaussian components in the mixture
    covariance_type : str, default='full'
        Covariance type: 'full', 'tied', 'diag', 'spherical'
    n_pca_components : int, default=128
        Number of PCA components for dimensionality reduction
    max_iter : int, default=100
        Maximum number of EM iterations
    random_state : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    gmm_ : GaussianMixture
        Fitted GMM model
    pca_ : PCA
        PCA transformer for dimensionality reduction
    threshold_ : float
        Anomaly threshold (negative log-likelihood)

    Examples
    --------
    >>> model = GMM(n_components=5)
    >>> model.fit(X_train)
    >>> anomaly_scores = model.predict(X_test)
    >>> labels = model.predict_label(X_test)
    """

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = 'full',
        n_pca_components: int = 128,
        max_iter: int = 100,
        random_state: int = 42
    ):
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_pca_components = n_pca_components
        self.max_iter = max_iter
        self.random_state = random_state

        self.gmm_ = None
        self.pca_ = None
        self.threshold_ = None

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'GMM':
        """
        Fit GMM to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training samples (normal only)
        y : ndarray, optional
            Ignored, present for API consistency

        Returns
        -------
        self : GMM
            Fitted estimator
        """
        # Apply PCA for dimensionality reduction
        self.pca_ = PCA(
            n_components=min(self.n_pca_components, X.shape[1]),
            random_state=self.random_state
        )
        X_reduced = self.pca_.fit_transform(X)

        # Fit GMM
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.gmm_.fit(X_reduced)

        # Compute threshold based on training data
        log_probs = self.gmm_.score_samples(X_reduced)
        # Use negative log-likelihood as anomaly score
        scores = -log_probs
        self.threshold_ = np.percentile(scores, 95)

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (higher = more anomalous)
        """
        self._check_is_fitted()

        # Transform with PCA
        X_reduced = self.pca_.transform(X)

        # Compute negative log-likelihood
        log_probs = self.gmm_.score_samples(X_reduced)
        scores = -log_probs

        return scores

    def predict_label(self, X: NDArray) -> NDArray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

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
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_pca_components': self.n_pca_components,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
        }
