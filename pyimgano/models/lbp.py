"""
LBP (Local Binary Pattern) for Anomaly Detection

Local Binary Pattern is a classic texture descriptor widely used in industrial
quality inspection. Effective for detecting surface defects and texture anomalies.

Reference:
    Ojala, T., et al. (2002). "Multiresolution gray-scale and rotation invariant
    texture classification with local binary patterns"
    IEEE TPAMI 2002.

Usage:
    >>> from pyimgano.models import LBP
    >>> model = LBP(n_points=24, radius=3, method='uniform')
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Literal
from skimage.feature import local_binary_pattern
from skimage import color
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..base import BaseVisionClassicalDetector


class LBP(BaseVisionClassicalDetector):
    """
    Local Binary Pattern for texture-based anomaly detection.

    LBP encodes local texture patterns and is highly effective for detecting
    surface defects, texture anomalies, and quality issues in industrial inspection.
    Particularly good for fabric, wood, metal, and other textured surfaces.

    Parameters
    ----------
    n_points : int, default=24
        Number of circularly symmetric neighbor points
    radius : int, default=3
        Radius of circle for neighbor points
    method : str, default='uniform'
        LBP method: 'default', 'ror', 'uniform', 'var'
    detector : str, default='isolation_forest'
        Anomaly detector: 'isolation_forest', 'one_class_svm'
    contamination : float, default=0.1
        Expected proportion of outliers
    n_bins : int, default=256
        Number of histogram bins
    grid_size : tuple, default=(4, 4)
        Divide image into grid for spatial LBP histograms
    resize_shape : tuple, optional
        Resize images to this shape

    Attributes
    ----------
    detector_ : object
        Fitted anomaly detector
    scaler_ : StandardScaler
        Feature scaler

    Examples
    --------
    >>> # Surface defect detection
    >>> model = LBP(n_points=24, radius=3, method='uniform')
    >>> model.fit(X_train_normal)
    >>> scores = model.predict(X_test)

    >>> # Fine texture analysis
    >>> model = LBP(n_points=16, radius=2, grid_size=(8, 8))
    >>> model.fit(X_train)
    >>> anomaly_map = model.get_anomaly_map(X_test[0])
    """

    def __init__(
        self,
        n_points: int = 24,
        radius: int = 3,
        method: Literal['default', 'ror', 'uniform', 'var'] = 'uniform',
        detector: Literal['isolation_forest', 'one_class_svm'] = 'isolation_forest',
        contamination: float = 0.1,
        n_bins: int = 256,
        grid_size: Tuple[int, int] = (4, 4),
        resize_shape: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.detector_type = detector
        self.contamination = contamination
        self.n_bins = n_bins
        self.grid_size = grid_size
        self.resize_shape = resize_shape

        self.detector_ = None
        self.scaler_ = None

    def _extract_lbp_features(self, image: NDArray) -> NDArray:
        """Extract LBP features with spatial information."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Resize if specified
        if self.resize_shape is not None:
            from skimage.transform import resize
            image = resize(image, self.resize_shape, anti_aliasing=True)

        # Compute LBP
        lbp = local_binary_pattern(
            image,
            P=self.n_points,
            R=self.radius,
            method=self.method
        )

        # Extract spatial histograms
        h, w = lbp.shape
        grid_h, grid_w = self.grid_size
        cell_h = h // grid_h
        cell_w = w // grid_w

        features = []
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract cell
                cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                # Compute histogram
                hist, _ = np.histogram(
                    cell.ravel(),
                    bins=self.n_bins,
                    range=(0, self.n_bins),
                    density=True
                )
                features.extend(hist)

        return np.array(features)

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'LBP':
        """
        Fit LBP model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : LBP
            Fitted estimator
        """
        # Extract LBP features
        print("Extracting LBP features...")
        lbp_features = []
        for i in range(len(X)):
            features = self._extract_lbp_features(X[i])
            lbp_features.append(features)

        lbp_features = np.array(lbp_features)

        # Standardize features
        self.scaler_ = StandardScaler()
        lbp_features_scaled = self.scaler_.fit_transform(lbp_features)

        # Fit detector
        print(f"Training {self.detector_type}...")
        if self.detector_type == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.detector_ = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        elif self.detector_type == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            self.detector_ = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )

        self.detector_.fit(lbp_features_scaled)

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

        # Extract LBP features
        lbp_features = []
        for i in range(len(X)):
            features = self._extract_lbp_features(X[i])
            lbp_features.append(features)

        lbp_features = np.array(lbp_features)

        # Scale features
        lbp_features_scaled = self.scaler_.transform(lbp_features)

        # Compute scores
        if self.detector_type == 'isolation_forest':
            # Isolation Forest: negative scores for anomalies
            scores = -self.detector_.score_samples(lbp_features_scaled)
        else:  # one_class_svm
            # SVM: negative decision values for anomalies
            scores = -self.detector_.decision_function(lbp_features_scaled)

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

        # Extract features
        lbp_features = []
        for i in range(len(X)):
            features = self._extract_lbp_features(X[i])
            lbp_features.append(features)

        lbp_features = np.array(lbp_features)

        # Scale features
        lbp_features_scaled = self.scaler_.transform(lbp_features)

        # Predict
        predictions = self.detector_.predict(lbp_features_scaled)

        # Convert to binary labels
        if self.detector_type == 'isolation_forest':
            # Isolation Forest: -1 = anomaly, 1 = normal
            labels = (predictions == -1).astype(int)
        else:
            # SVM: -1 = anomaly, 1 = normal
            labels = (predictions == -1).astype(int)

        return labels

    def get_anomaly_map(self, image: NDArray) -> NDArray:
        """
        Generate spatial anomaly map for single image.

        Parameters
        ----------
        image : ndarray of shape (height, width) or (height, width, channels)
            Input image

        Returns
        -------
        anomaly_map : ndarray of shape (grid_h, grid_w)
            Spatial anomaly map
        """
        self._check_is_fitted()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Resize if specified
        if self.resize_shape is not None:
            from skimage.transform import resize
            image = resize(image, self.resize_shape, anti_aliasing=True)

        # Compute LBP
        lbp = local_binary_pattern(
            image,
            P=self.n_points,
            R=self.radius,
            method=self.method
        )

        # Extract spatial scores
        h, w = lbp.shape
        grid_h, grid_w = self.grid_size
        cell_h = h // grid_h
        cell_w = w // grid_w

        anomaly_map = np.zeros((grid_h, grid_w))

        for i in range(grid_h):
            for j in range(grid_w):
                # Extract cell
                cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                # Compute histogram
                hist, _ = np.histogram(
                    cell.ravel(),
                    bins=self.n_bins,
                    range=(0, self.n_bins),
                    density=True
                )

                # Scale and predict
                hist_scaled = self.scaler_.transform(hist.reshape(1, -1))

                if self.detector_type == 'isolation_forest':
                    score = -self.detector_.score_samples(hist_scaled)[0]
                else:
                    score = -self.detector_.decision_function(hist_scaled)[0]

                anomaly_map[i, j] = score

        return anomaly_map

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_points': self.n_points,
            'radius': self.radius,
            'method': self.method,
            'detector': self.detector_type,
            'contamination': self.contamination,
            'n_bins': self.n_bins,
            'grid_size': self.grid_size,
            'resize_shape': self.resize_shape,
        }
