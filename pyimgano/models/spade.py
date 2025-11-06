"""
SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences.

Paper: https://arxiv.org/abs/2005.02357
Conference: ECCV 2020

SPADE uses deep pyramid feature extraction followed by k-NN matching
for sub-image anomaly detection with high localization accuracy.

Key Features:
- Multi-scale feature extraction
- K-NN based anomaly scoring
- Sub-image alignment
- Excellent localization
- Training-free (uses pretrained features)
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from torchvision import models

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class DeepPyramidExtractor(nn.Module):
    """Deep pyramid feature extractor for SPADE."""

    def __init__(self, backbone: str = "wide_resnet50"):
        super().__init__()

        if backbone == "wide_resnet50":
            resnet = models.wide_resnet50_2(pretrained=True)
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif backbone == "resnet18":
            resnet = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Extract features at different scales
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Low-level
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # Mid-level
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # High-level

        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Tuple of (layer1, layer2, layer3) features.
        """
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x1, x2, x3


@register_model("spade")
class SPADEDetector(BaseVisionDeepDetector):
    """SPADE anomaly detector.

    Uses deep pyramid features with k-NN matching for sub-image
    anomaly detection with excellent localization.

    Args:
        backbone: Feature extraction backbone ("wide_resnet50", "resnet50", "resnet18").
        k_neighbors: Number of nearest neighbors for scoring.
        feature_levels: Which feature levels to use (list of "layer1", "layer2", "layer3").
        align_features: Whether to align features before matching.
        gaussian_sigma: Sigma for Gaussian smoothing of anomaly maps.
        device: Device to use ("cuda" or "cpu").

    References:
        Cohen & Hoshen. "Sub-Image Anomaly Detection with Deep Pyramid Correspondences."
        ECCV 2020.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        k_neighbors: int = 50,
        feature_levels: list = ["layer1", "layer2", "layer3"],
        align_features: bool = True,
        gaussian_sigma: float = 4.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.k_neighbors = k_neighbors
        self.feature_levels = feature_levels
        self.align_features = align_features
        self.gaussian_sigma = gaussian_sigma

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

        # Feature memory bank
        self.memory_bank = None
        self.kd_trees = None

    def _build_model(self):
        """Build the SPADE model."""
        self.feature_extractor = DeepPyramidExtractor(self.backbone_name).to(self.device)
        self.feature_extractor.eval()

    def fit(self, X: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Build feature memory bank from training images.

        Args:
            X: Training images (N, H, W, C) or (N, C, H, W).
            y: Not used (unsupervised).
        """
        print("Building SPADE memory bank...")

        # Normalize to [0, 1]
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        # Extract features from all training images
        self.memory_bank = {level: [] for level in self.feature_levels}

        with torch.no_grad():
            for i, img in enumerate(X):
                if (i + 1) % 50 == 0:
                    print(f"  Processing image {i+1}/{len(X)}")

                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract multi-scale features
                f1, f2, f3 = self.feature_extractor(img_tensor)

                feature_dict = {"layer1": f1, "layer2": f2, "layer3": f3}

                # Store features for each level
                for level in self.feature_levels:
                    feat = feature_dict[level]
                    b, c, h, w = feat.shape

                    # Reshape to (HxW, C)
                    feat = feat.permute(0, 2, 3, 1).reshape(-1, c)

                    self.memory_bank[level].append(feat.cpu().numpy())

        # Concatenate all features
        for level in self.feature_levels:
            self.memory_bank[level] = np.vstack(self.memory_bank[level])
            print(f"  {level}: {self.memory_bank[level].shape}")

        # Build k-D trees for efficient k-NN search
        print("Building k-D trees...")
        self.kd_trees = {}
        for level in self.feature_levels:
            self.kd_trees[level] = cKDTree(self.memory_bank[level])

        print("Memory bank built successfully!")

    def predict_proba(self, X: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            Anomaly scores for each sample.
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        scores = []

        for img in X:
            # Get anomaly map
            anomaly_map = self._compute_anomaly_map(img)

            # Image-level score (max or mean of anomaly map)
            score = anomaly_map.max()
            scores.append(score)

        return np.array(scores)

    def predict_anomaly_map(self, X: NDArray) -> list:
        """Predict pixel-level anomaly maps.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            List of anomaly maps.
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        anomaly_maps = []

        for img in X:
            anomaly_map = self._compute_anomaly_map(img)
            anomaly_maps.append(anomaly_map)

        return anomaly_maps

    def _compute_anomaly_map(self, image: NDArray) -> NDArray:
        """Compute anomaly map for a single image.

        Args:
            image: Input image (H, W, C).

        Returns:
            Anomaly map (H, W).
        """
        with torch.no_grad():
            img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

            # Extract features
            f1, f2, f3 = self.feature_extractor(img_tensor)

            feature_dict = {"layer1": f1, "layer2": f2, "layer3": f3}

            # Compute anomaly map for each level
            anomaly_maps = []

            for level in self.feature_levels:
                feat = feature_dict[level]
                b, c, h, w = feat.shape

                # Reshape to (HxW, C)
                feat = feat.permute(0, 2, 3, 1).reshape(h * w, c).cpu().numpy()

                # Align features if requested
                if self.align_features:
                    feat = self._align_features(feat)

                # Query k-NN in memory bank
                distances, _ = self.kd_trees[level].query(
                    feat, k=self.k_neighbors, workers=-1
                )

                # Average distance to k nearest neighbors
                anomaly_scores = distances.mean(axis=1)

                # Reshape back to spatial dimensions
                anomaly_map = anomaly_scores.reshape(h, w)

                # Upsample to original image size
                anomaly_map = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
                anomaly_map = F.interpolate(
                    anomaly_map,
                    size=image.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map = anomaly_map.squeeze().numpy()

                anomaly_maps.append(anomaly_map)

            # Combine anomaly maps from different levels
            final_map = np.mean(anomaly_maps, axis=0)

            # Apply Gaussian smoothing
            if self.gaussian_sigma > 0:
                final_map = gaussian_filter(final_map, sigma=self.gaussian_sigma)

        return final_map

    def _align_features(self, features: NDArray) -> NDArray:
        """Align features using L2 normalization.

        Args:
            features: Input features (N, C).

        Returns:
            Aligned features.
        """
        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        return features

    def _preprocess(self, image: NDArray) -> torch.Tensor:
        """Preprocess image for feature extraction.

        Args:
            image: Input image (H, W, C) in [0, 1].

        Returns:
            Preprocessed tensor (C, H, W).
        """
        # Convert to tensor
        if image.ndim == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=2)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image

    def get_feature_shapes(self) -> dict:
        """Get feature shapes for each level.

        Returns:
            Dictionary of feature shapes.
        """
        if self.memory_bank is None:
            raise ValueError("Model not fitted. Call fit() first.")

        shapes = {}
        for level in self.feature_levels:
            shapes[level] = self.memory_bank[level].shape

        return shapes
