"""
HOG + SVM for Anomaly Detection

Histogram of Oriented Gradients (HOG) combined with Support Vector Machine (SVM)
for traditional computer vision-based anomaly detection. Widely used in industrial
quality inspection.

Reference:
    Dalal, N., & Triggs, B. (2005). "Histograms of oriented gradients for human detection"
    CVPR 2005.

Usage:
    >>> from pyimgano.models import HOG_SVM
    >>> model = HOG_SVM(orientations=9, pixels_per_cell=(8, 8))
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Literal
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import color

from ..base import BaseVisionClassicalDetector


class HOG_SVM(BaseVisionClassicalDetector):
    """
    HOG (Histogram of Oriented Gradients) + One-Class SVM for anomaly detection.

    This traditional computer vision approach extracts HOG features that capture
    edge and gradient information, then uses One-Class SVM to model the distribution
    of normal samples. Widely used in industrial quality inspection.

    Parameters
    ----------
    orientations : int, default=9
        Number of orientation bins for HOG
    pixels_per_cell : tuple, default=(8, 8)
        Size of cells for HOG computation
    cells_per_block : tuple, default=(2, 2)
        Number of cells per block for normalization
    block_norm : str, default='L2-Hys'
        Block normalization method: 'L1', 'L1-sqrt', 'L2', 'L2-Hys'
    nu : float, default=0.1
        Upper bound on fraction of outliers (SVM parameter)
    kernel : str, default='rbf'
        SVM kernel: 'rbf', 'linear', 'poly', 'sigmoid'
    gamma : str or float, default='scale'
        Kernel coefficient
    resize_shape : tuple, optional
        Resize images to this shape before HOG extraction

    Attributes
    ----------
    svm_ : OneClassSVM
        Fitted One-Class SVM
    scaler_ : StandardScaler
        Feature scaler

    Examples
    --------
    >>> model = HOG_SVM(orientations=9, pixels_per_cell=(8, 8))
    >>> model.fit(X_train)
    >>> anomaly_scores = model.predict(X_test)
    >>> labels = model.predict_label(X_test)
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: Literal['L1', 'L1-sqrt', 'L2', 'L2-Hys'] = 'L2-Hys',
        nu: float = 0.1,
        kernel: Literal['rbf', 'linear', 'poly', 'sigmoid'] = 'rbf',
        gamma: str = 'scale',
        resize_shape: Optional[Tuple[int, int]] = (128, 128)
    ):
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.resize_shape = resize_shape

        self.svm_ = None
        self.scaler_ = None

    def _extract_hog_features(self, image: NDArray) -> NDArray:
        """Extract HOG features from image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Resize if specified
        if self.resize_shape is not None:
            from skimage.transform import resize
            image = resize(image, self.resize_shape, anti_aliasing=True)

        # Extract HOG features
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True
        )

        return features

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'HOG_SVM':
        """
        Fit HOG + SVM model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored, present for API consistency

        Returns
        -------
        self : HOG_SVM
            Fitted estimator
        """
        # Extract HOG features for all images
        print("Extracting HOG features...")
        hog_features = []
        for i in range(len(X)):
            features = self._extract_hog_features(X[i])
            hog_features.append(features)

        hog_features = np.array(hog_features)

        # Standardize features
        self.scaler_ = StandardScaler()
        hog_features_scaled = self.scaler_.fit_transform(hog_features)

        # Fit One-Class SVM
        print("Training One-Class SVM...")
        self.svm_ = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.svm_.fit(hog_features_scaled)

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
            Anomaly scores (higher = more anomalous)
        """
        self._check_is_fitted()

        # Extract HOG features
        hog_features = []
        for i in range(len(X)):
            features = self._extract_hog_features(X[i])
            hog_features.append(features)

        hog_features = np.array(hog_features)

        # Scale features
        hog_features_scaled = self.scaler_.transform(hog_features)

        # Compute decision scores
        # SVM returns negative values for outliers, so negate
        scores = -self.svm_.decision_function(hog_features_scaled)

        return scores

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
        self._check_is_fitted()

        # Extract HOG features
        hog_features = []
        for i in range(len(X)):
            features = self._extract_hog_features(X[i])
            hog_features.append(features)

        hog_features = np.array(hog_features)

        # Scale features
        hog_features_scaled = self.scaler_.transform(hog_features)

        # Predict (-1 = outlier, 1 = inlier)
        predictions = self.svm_.predict(hog_features_scaled)

        # Convert to binary labels (0 = normal, 1 = anomaly)
        labels = (predictions == -1).astype(int)

        return labels

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'block_norm': self.block_norm,
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'resize_shape': self.resize_shape,
        }
