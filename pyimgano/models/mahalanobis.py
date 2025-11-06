"""
Mahalanobis Distance for Anomaly Detection

Uses Mahalanobis distance in feature space to detect anomalies.
Particularly effective when combined with pre-trained feature extractors.

Reference:
    Lee, K., et al. (2018). "A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks"
    NeurIPS 2018.

Usage:
    >>> from pyimgano.models import MahalanobisDetector
    >>> model = MahalanobisDetector(backbone='resnet18')
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Literal, List
import torchvision.models as models
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

from ..base import BaseVisionDeepDetector


class MahalanobisDetector(BaseVisionDeepDetector):
    """
    Mahalanobis distance-based anomaly detector.

    Extracts features using a pre-trained network and models the distribution
    of normal samples using mean and covariance. Anomalies are detected based
    on their Mahalanobis distance from the normal distribution.

    Parameters
    ----------
    backbone : str, default='resnet18'
        Pre-trained backbone: 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50'
    layers : list, default=['layer3']
        Layers to extract features from
    pooling : str, default='avg'
        Pooling method: 'avg', 'max', or 'none'
    device : str, default='cuda'
        Device for inference

    Attributes
    ----------
    feature_extractor_ : nn.Module
        Pre-trained feature extractor
    mean_ : ndarray
        Mean of normal features
    cov_inv_ : ndarray
        Inverse covariance matrix

    Examples
    --------
    >>> model = MahalanobisDetector(backbone='resnet18')
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
    """

    def __init__(
        self,
        backbone: Literal['resnet18', 'resnet34', 'resnet50', 'wide_resnet50'] = 'resnet18',
        layers: List[str] = ['layer3'],
        pooling: Literal['avg', 'max', 'none'] = 'avg',
        device: str = 'cuda'
    ):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self.pooling = pooling
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.feature_extractor_ = None
        self.mean_ = None
        self.cov_inv_ = None

    def _build_feature_extractor(self) -> nn.Module:
        """Build pre-trained feature extractor."""
        if self.backbone_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif self.backbone_name == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif self.backbone_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif self.backbone_name == 'wide_resnet50':
            model = models.wide_resnet50_2(pretrained=True)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        # Remove classification layers
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()

        return model

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from specified layers."""
        features = []

        def hook_fn(module, input, output):
            features.append(output)

        # Register hooks
        handles = []
        for layer_name in self.layers:
            layer = dict(self.feature_extractor_.named_modules())[layer_name]
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)

        # Forward pass
        _ = self.feature_extractor_(x)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Concatenate features from multiple layers
        if len(features) > 1:
            # Pool each feature map to same size
            pooled_features = []
            for feat in features:
                if self.pooling == 'avg':
                    pooled = F.adaptive_avg_pool2d(feat, 1).squeeze()
                elif self.pooling == 'max':
                    pooled = F.adaptive_max_pool2d(feat, 1).squeeze()
                else:
                    pooled = feat.reshape(feat.size(0), -1)
                pooled_features.append(pooled)
            features = torch.cat(pooled_features, dim=1)
        else:
            feat = features[0]
            if self.pooling == 'avg':
                features = F.adaptive_avg_pool2d(feat, 1).squeeze()
            elif self.pooling == 'max':
                features = F.adaptive_max_pool2d(feat, 1).squeeze()
            else:
                features = feat.reshape(feat.size(0), -1)

        return features

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'MahalanobisDetector':
        """
        Fit Mahalanobis detector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : MahalanobisDetector
            Fitted estimator
        """
        # Convert to torch tensor
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        X = np.transpose(X, (0, 3, 1, 2))
        X_tensor = torch.from_numpy(X).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        X_tensor = (X_tensor - mean) / std

        # Build feature extractor
        self.feature_extractor_ = self._build_feature_extractor().to(self.device)
        self.feature_extractor_.eval()

        # Extract features
        all_features = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), 32):
                batch = X_tensor[i:i + 32].to(self.device)
                features = self._extract_features(batch)
                all_features.append(features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)

        # Compute mean
        self.mean_ = np.mean(all_features, axis=0)

        # Compute covariance matrix
        centered = all_features - self.mean_
        cov = np.dot(centered.T, centered) / len(all_features)

        # Add regularization to ensure invertibility
        reg = 1e-3
        cov += reg * np.eye(cov.shape[0])

        # Compute inverse covariance
        self.cov_inv_ = np.linalg.inv(cov)

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Compute anomaly scores using Mahalanobis distance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Mahalanobis distances (higher = more anomalous)
        """
        self._check_is_fitted()

        # Preprocess
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        X = np.transpose(X, (0, 3, 1, 2))
        X_tensor = torch.from_numpy(X).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        X_tensor = (X_tensor - mean) / std

        # Extract features
        all_features = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), 32):
                batch = X_tensor[i:i + 32].to(self.device)
                features = self._extract_features(batch)
                all_features.append(features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)

        # Compute Mahalanobis distance
        scores = []
        for feat in all_features:
            diff = feat - self.mean_
            score = np.sqrt(np.dot(np.dot(diff, self.cov_inv_), diff.T))
            scores.append(score)

        return np.array(scores)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'backbone': self.backbone_name,
            'layers': self.layers,
            'pooling': self.pooling,
            'device': str(self.device),
        }
