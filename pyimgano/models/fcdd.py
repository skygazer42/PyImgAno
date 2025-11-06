"""
FCDD: Fully Convolutional Data Description

A fully convolutional extension of Deep SVDD that produces pixel-level
anomaly maps while maintaining global anomaly detection capabilities.

Reference:
    Liznerski, P., et al. (2021). "Explainable Deep One-Class Classification"
    ICLR 2021.

Usage:
    >>> from pyimgano.models import FCDD
    >>> model = FCDD(objective='hsc')
    >>> model.fit(X_train)
    >>> scores, maps = model.predict_with_map(X_test)
"""

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Literal, Tuple

from ..base import BaseVisionDeepDetector


class FCDDNetwork(nn.Module):
    """Fully Convolutional Data Description network."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(512, 512, 5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Final 1x1 conv for anomaly score map
        self.score_conv = nn.Conv2d(512, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        features : torch.Tensor
            Feature maps
        score_map : torch.Tensor
            Anomaly score map
        """
        features = self.encoder(x)
        score_map = self.score_conv(features)
        return features, score_map


class FCDD(BaseVisionDeepDetector):
    """
    Fully Convolutional Data Description.

    Extends Deep SVDD to produce pixel-level anomaly maps using fully
    convolutional architecture. Supports multiple training objectives.

    Parameters
    ----------
    objective : str, default='hsc'
        Training objective: 'hsc' (Hypersphere Compactness) or 'occ' (One-Class)
    nu : float, default=0.1
        Outlier fraction for soft-boundary
    learning_rate : float, default=1e-4
        Learning rate for Adam optimizer
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=100
        Number of training epochs
    device : str, default='cuda'
        Device for training

    Attributes
    ----------
    network_ : FCDDNetwork
        The fully convolutional network
    center_ : torch.Tensor
        Hypersphere center (for HSC objective)

    Examples
    --------
    >>> model = FCDD(objective='hsc', nu=0.1)
    >>> model.fit(X_train)
    >>> scores, maps = model.predict_with_map(X_test)
    """

    def __init__(
        self,
        objective: Literal['hsc', 'occ'] = 'hsc',
        nu: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'cuda'
    ):
        super().__init__()
        self.objective = objective
        self.nu = nu
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.network_ = None
        self.center_ = None
        self.radius_ = 0

    def _initialize_center(self, dataloader: DataLoader) -> torch.Tensor:
        """Initialize hypersphere center."""
        self.network_.eval()
        centers = []

        with torch.no_grad():
            for batch, in dataloader:
                batch = batch.to(self.device)
                features, _ = self.network_(batch)
                # Average pooling to get feature vector
                center_batch = F.adaptive_avg_pool2d(features, 1).squeeze()
                centers.append(center_batch)

        center = torch.cat(centers).mean(dim=0)

        # Avoid degenerate solution
        center[(abs(center) < 0.1) & (center < 0)] = -0.1
        center[(abs(center) < 0.1) & (center >= 0)] = 0.1

        return center

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'FCDD':
        """
        Fit FCDD model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : FCDD
            Fitted estimator
        """
        # Convert to torch tensor
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        X = np.transpose(X, (0, 3, 1, 2))
        X_tensor = torch.from_numpy(X).float() / 255.0

        # Initialize network
        self.network_ = FCDDNetwork(in_channels=X.shape[1]).to(self.device)

        # Setup data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Initialize center (for HSC objective)
        if self.objective == 'hsc':
            self.center_ = self._initialize_center(dataloader)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6
        )

        # Training loop
        self.network_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0

            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device)

                # Forward pass
                features, score_map = self.network_(batch)

                if self.objective == 'hsc':
                    # Hypersphere compactness loss
                    # Compute distance of each pixel feature to center
                    features_flat = features.permute(0, 2, 3, 1).reshape(-1, features.size(1))
                    dist = torch.sum((features_flat - self.center_) ** 2, dim=1)
                    loss = torch.mean(dist)

                elif self.objective == 'occ':
                    # One-class loss (minimize score map)
                    loss = torch.mean(score_map)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores
        """
        self._check_is_fitted()

        # Preprocess
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        X = np.transpose(X, (0, 3, 1, 2))
        X_tensor = torch.from_numpy(X).float() / 255.0

        # Compute scores
        self.network_.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size].to(self.device)
                _, score_map = self.network_(batch)

                # Average score map to get image-level score
                batch_scores = torch.mean(score_map, dim=(1, 2, 3))
                scores.append(batch_scores.cpu().numpy())

        return np.concatenate(scores)

    def predict_with_map(
        self,
        X: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute anomaly scores and pixel-level maps.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Image-level anomaly scores
        maps : ndarray of shape (n_samples, height, width)
            Pixel-level anomaly maps
        """
        self._check_is_fitted()

        # Preprocess
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        original_h, original_w = X.shape[1], X.shape[2]
        X = np.transpose(X, (0, 3, 1, 2))
        X_tensor = torch.from_numpy(X).float() / 255.0

        # Compute scores and maps
        self.network_.eval()
        scores = []
        maps = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size].to(self.device)
                _, score_map = self.network_(batch)

                # Upsample score map to original size
                score_map_up = F.interpolate(
                    score_map,
                    size=(original_h, original_w),
                    mode='bilinear',
                    align_corners=False
                )

                # Image-level score
                batch_scores = torch.mean(score_map, dim=(1, 2, 3))
                scores.append(batch_scores.cpu().numpy())

                # Pixel-level maps
                maps.append(score_map_up.squeeze(1).cpu().numpy())

        return np.concatenate(scores), np.concatenate(maps)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'objective': self.objective,
            'nu': self.nu,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
        }
