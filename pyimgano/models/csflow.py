"""
CS-Flow: Continual Segmentation-based Normalizing Flow for Anomaly Detection

Paper: "Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection"
Conference: WACV 2022

Key Innovation:
- Combines normalizing flows with segmentation
- Cross-scale flow for multi-resolution modeling
- More expressive than standard normalizing flows
- Fast inference with good localization

Implementation includes:
- Multi-scale feature extraction
- Coupling layers for normalizing flows
- Likelihood-based anomaly scoring
- Pixel-level anomaly localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray

from pyimgano.models.base_dl import BaseVisionDeepDetector


class CrossScaleFlow(nn.Module):
    """Cross-scale normalizing flow module."""

    def __init__(self, in_channels: int, num_flows: int = 8):
        super().__init__()
        self.flows = nn.ModuleList([
            AffineCouplingLayer(in_channels) for _ in range(num_flows)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through flows.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            z: Latent representation [B, C, H, W]
            log_det: Log determinant of Jacobian [B]
        """
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x

        for flow in self.flows:
            z, ld = flow(z)
            log_det += ld

        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass through flows.

        Args:
            z: Latent representation [B, C, H, W]

        Returns:
            x: Reconstructed features [B, C, H, W]
        """
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.split_idx = in_channels // 2

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Conv2d(self.split_idx, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels - self.split_idx, 3, padding=1),
            nn.Tanh()
        )

        self.translation_net = nn.Sequential(
            nn.Conv2d(self.split_idx, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels - self.split_idx, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward coupling.

        Args:
            x: Input [B, C, H, W]

        Returns:
            y: Output [B, C, H, W]
            log_det: Log determinant [B]
        """
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]

        # Compute scale and translation
        s = self.scale_net(x1)
        t = self.translation_net(x1)

        # Affine transformation
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)

        # Log determinant
        log_det = s.sum(dim=[1, 2, 3])

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse coupling.

        Args:
            y: Input [B, C, H, W]

        Returns:
            x: Output [B, C, H, W]
        """
        y1, y2 = y[:, :self.split_idx], y[:, self.split_idx:]

        # Compute scale and translation
        s = self.scale_net(y1)
        t = self.translation_net(y1)

        # Inverse affine transformation
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)

        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)

            self.layer1 = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool, model.layer1
            )
            self.layer2 = model.layer2
            self.layer3 = model.layer3

            self.out_channels = [64, 128, 256]
        elif backbone == "wide_resnet50":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)

            self.layer1 = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool, model.layer1
            )
            self.layer2 = model.layer2
            self.layer3 = model.layer3

            self.out_channels = [256, 512, 1024]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of features at different scales
        """
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        return [feat1, feat2, feat3]


class CSFlowDetector(BaseVisionDeepDetector):
    """CS-Flow (Cross-Scale Flow) detector.

    Uses multi-scale normalizing flows for anomaly detection with
    improved expressiveness and localization.

    Args:
        backbone: Feature extraction backbone
        num_flows: Number of flow layers per scale
        epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        pretrained: Use pre-trained backbone
        device: Device to run on

    Example:
        >>> detector = CSFlowDetector(
        ...     backbone="resnet18",
        ...     num_flows=8,
        ...     epochs=50
        ... )
        >>> detector.fit(normal_images)
        >>> scores = detector.predict_proba(test_images)
        >>> anomaly_maps = detector.predict_anomaly_map(test_images)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_flows: int = 8,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        pretrained: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.backbone_name = backbone
        self.num_flows = num_flows
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained
        ).to(self.device)

        # Flow models for each scale
        self.flows = nn.ModuleList([
            CrossScaleFlow(in_channels=ch, num_flows=num_flows)
            for ch in self.feature_extractor.out_channels
        ]).to(self.device)

        self.fitted_ = False

    def _extract_features(self, images: NDArray) -> List[torch.Tensor]:
        """Extract multi-scale features from images.

        Args:
            images: Input images [N, H, W, C] or [N, C, H, W]

        Returns:
            List of features at different scales
        """
        # Convert to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).float()

        # Ensure [N, C, H, W] format
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:  # [N, H, W, C] -> [N, C, H, W]
            images = images.permute(0, 3, 1, 2)

        images = images.to(self.device)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images / 255.0 - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(images)

        return features

    def fit(self, X: NDArray, y: Optional[NDArray] = None):
        """Fit the detector on normal images.

        Args:
            X: Normal images [N, H, W, C]
            y: Ignored (unsupervised)
        """
        # Training mode
        for flow in self.flows:
            flow.train()

        # Optimizer
        optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.learning_rate)

        # Convert to tensor dataset
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()

        N = X.shape[0]

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, N, self.batch_size):
                batch = X[i:i + self.batch_size]

                # Extract features
                features = self._extract_features(batch)

                # Compute negative log likelihood
                total_loss = 0.0
                for feat, flow in zip(features, self.flows):
                    z, log_det = flow(feat)

                    # Negative log likelihood (assuming standard normal prior)
                    log_likelihood = -0.5 * torch.sum(z ** 2, dim=[1, 2, 3]) + log_det
                    loss = -log_likelihood.mean()

                    total_loss += loss

                # Backprop
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / num_batches:.4f}")

        # Evaluation mode
        for flow in self.flows:
            flow.eval()

        self.fitted_ = True
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        """Compute anomaly scores for images.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly scores [N]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract features
        features = self._extract_features(X)

        # Compute likelihood for each scale
        scores_list = []
        for feat, flow in zip(features, self.flows):
            with torch.no_grad():
                z, log_det = flow(feat)

                # Negative log likelihood
                log_likelihood = -0.5 * torch.sum(z ** 2, dim=[1, 2, 3]) + log_det
                scores = -log_likelihood  # Higher = more anomalous

                scores_list.append(scores.cpu().numpy())

        # Aggregate scores across scales
        final_scores = np.mean(scores_list, axis=0)

        return final_scores

    def predict_anomaly_map(self, X: NDArray) -> NDArray:
        """Generate pixel-level anomaly maps.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly maps [N, H, W]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get image size
        H_img, W_img = X.shape[1:3] if X.shape[-1] == 3 else X.shape[2:4]

        # Extract features
        features = self._extract_features(X)

        # Compute spatial likelihood maps
        anomaly_maps = []
        for feat, flow in zip(features, self.flows):
            with torch.no_grad():
                z, _ = flow(feat)

                # Spatial likelihood (per pixel)
                spatial_likelihood = -0.5 * torch.sum(z ** 2, dim=1)  # [B, H, W]

                # Upsample to image size
                upsampled = F.interpolate(
                    spatial_likelihood.unsqueeze(1),
                    size=(H_img, W_img),
                    mode='bilinear',
                    align_corners=False
                )

                anomaly_maps.append(-upsampled.squeeze(1).cpu().numpy())  # Negate for anomaly

        # Average across scales
        final_maps = np.mean(anomaly_maps, axis=0)

        return final_maps
