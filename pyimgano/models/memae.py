"""
MemAE: Memory-Augmented Autoencoder

Uses a memory module to store prototypical patterns of normal data.
During reconstruction, features are retrieved from memory, making it 
harder to reconstruct anomalies.

Reference:
    Gong, D., et al. (2019). "Memorizing Normality to Detect Anomaly:
    Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection"
    ICCV 2019.

Usage:
    >>> from pyimgano.models import MemAE
    >>> model = MemAE(mem_dim=2000, shrink_thres=0.0025)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from ..base import BaseVisionDeepDetector


class MemoryModule(nn.Module):
    """Memory module for storing prototypical patterns."""

    def __init__(self, mem_dim: int, fea_dim: int, shrink_thres: float = 0.0025):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres

        # Initialize memory
        self.register_buffer('memory', torch.randn(mem_dim, fea_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize memory with normal distribution."""
        stdv = 1. / np.sqrt(self.memory.size(1))
        self.memory.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor):
        """
        Retrieve from memory using attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input features (B, C, H, W)

        Returns
        -------
        output : torch.Tensor
            Retrieved features
        att_weight : torch.Tensor
            Attention weights
        """
        batch_size = x.size(0)
        
        # Reshape: (B, C, H, W) -> (B, HW, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.fea_dim)

        # Compute attention: (B, HW, mem_dim)
        att_weight = F.linear(x_flat, self.memory)  # (B, HW, mem_dim)
        att_weight = F.softmax(att_weight, dim=2)

        # Hard shrinkage
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, self.shrink_thres)
            # Re-normalize
            att_weight = F.normalize(att_weight, p=1, dim=2)

        # Retrieve from memory: (B, HW, C)
        output = F.linear(att_weight, self.memory.permute(1, 0))

        # Reshape back: (B, HW, C) -> (B, C, H, W)
        h, w = x.size(2), x.size(3)
        output = output.reshape(batch_size, h, w, self.fea_dim)
        output = output.permute(0, 3, 1, 2)

        return output, att_weight


def hard_shrink_relu(x: torch.Tensor, threshold: float = 0.5):
    """Hard shrinkage function."""
    return (F.relu(x - threshold) * x) / (x + 1e-12)


class MemAENetwork(nn.Module):
    """Memory-Augmented Autoencoder network."""

    def __init__(
        self,
        in_channels: int = 3,
        mem_dim: int = 2000,
        shrink_thres: float = 0.0025
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Memory module
        self.memory = MemoryModule(mem_dim, 256, shrink_thres)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # Encode
        z = self.encoder(x)

        # Memory read
        z_mem, att_weight = self.memory(z)

        # Decode
        recon = self.decoder(z_mem)

        return recon, z, z_mem, att_weight


class MemAE(BaseVisionDeepDetector):
    """
    Memory-Augmented Autoencoder for anomaly detection.

    Uses a memory module to store prototypical patterns of normal data.
    Anomalies are detected based on reconstruction error, as they cannot
    be well reconstructed using normal patterns from memory.

    Parameters
    ----------
    mem_dim : int, default=2000
        Memory dimension (number of memory items)
    shrink_thres : float, default=0.0025
        Shrinkage threshold for hard attention
    entropy_weight : float, default=0.0002
        Weight for entropy loss (encourages diversity)
    learning_rate : float, default=2e-4
        Learning rate for Adam optimizer
    batch_size : int, default=32
        Batch size for training
    epochs : int, default=100
        Number of training epochs
    device : str, default='cuda'
        Device for training

    Attributes
    ----------
    network_ : MemAENetwork
        The memory-augmented autoencoder network

    Examples
    --------
    >>> model = MemAE(mem_dim=2000, shrink_thres=0.0025)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
    """

    def __init__(
        self,
        mem_dim: int = 2000,
        shrink_thres: float = 0.0025,
        entropy_weight: float = 0.0002,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'cuda'
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.entropy_weight = entropy_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.network_ = None

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'MemAE':
        """
        Fit MemAE model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width, channels)
            Training images (normal only)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : MemAE
            Fitted estimator
        """
        # Convert to torch tensor
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        X = np.transpose(X, (0, 3, 1, 2))
        X_tensor = torch.from_numpy(X).float() / 255.0

        # Initialize network
        self.network_ = MemAENetwork(
            in_channels=X.shape[1],
            mem_dim=self.mem_dim,
            shrink_thres=self.shrink_thres
        ).to(self.device)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate
        )

        # Training loop
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.network_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0

            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device)

                # Forward pass
                recon, z, z_mem, att_weight = self.network_(batch)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, batch)

                # Entropy loss (encourages diverse memory usage)
                entropy_loss = -torch.sum(
                    att_weight * torch.log(att_weight + 1e-12)
                ) / att_weight.size(0)

                # Total loss
                loss = recon_loss + self.entropy_weight * entropy_loss

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
            Anomaly scores (reconstruction error)
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
                recon, _, _, _ = self.network_(batch)

                # MSE reconstruction error
                mse = torch.mean((recon - batch) ** 2, dim=(1, 2, 3))
                scores.append(mse.cpu().numpy())

        return np.concatenate(scores)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'mem_dim': self.mem_dim,
            'shrink_thres': self.shrink_thres,
            'entropy_weight': self.entropy_weight,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
        }
