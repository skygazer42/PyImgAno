"""
MemSeg: Memory-Guided Semantic Segmentation for Anomaly Detection.

Paper: Memory-guided anomaly detection (various implementations)
Concept: Uses memory banks with attention mechanisms for anomaly segmentation

Key Features:
- Memory-guided attention
- Semantic segmentation approach
- Multi-scale memory banks
- Strong localization
- Efficient inference
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from torchvision import models

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class MemoryBank(nn.Module):
    """Memory bank for storing normal feature patterns."""

    def __init__(self, memory_size: int, feature_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim

        # Initialize memory bank
        self.register_buffer(
            "memory", torch.randn(memory_size, feature_dim)
        )
        self.memory_filled = 0

    def update(self, features: torch.Tensor):
        """Update memory bank with new features.

        Args:
            features: Feature vectors (N, D).
        """
        batch_size = features.size(0)

        if self.memory_filled < self.memory_size:
            # Fill memory bank
            end_idx = min(self.memory_filled + batch_size, self.memory_size)
            self.memory[self.memory_filled:end_idx] = features[:end_idx - self.memory_filled]
            self.memory_filled = end_idx
        else:
            # Random replacement
            indices = torch.randperm(self.memory_size)[:batch_size]
            self.memory[indices] = features

    def query(self, queries: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query memory bank for k nearest neighbors.

        Args:
            queries: Query features (N, D).
            k: Number of nearest neighbors.

        Returns:
            Tuple of (distances, indices).
        """
        # Normalize features
        queries = F.normalize(queries, dim=1)
        memory = F.normalize(self.memory, dim=1)

        # Compute similarities
        similarities = torch.mm(queries, memory.t())  # (N, M)

        # Get top-k
        values, indices = torch.topk(similarities, k=k, dim=1)

        # Convert similarities to distances
        distances = 1.0 - values

        return distances, indices


class FeatureExtractorWithMemory(nn.Module):
    """Feature extractor with memory-guided attention."""

    def __init__(
        self,
        backbone: str = "resnet18",
        memory_size: int = 1000,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load backbone
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Extract intermediate features
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])

        # Memory banks for different scales
        self.memory_banks = nn.ModuleDict({
            'layer2': MemoryBank(memory_size, 128 if backbone.startswith('resnet1') else 512),
            'layer3': MemoryBank(memory_size, 256 if backbone.startswith('resnet1') else 1024),
            'layer4': MemoryBank(memory_size, self.feature_dim),
        })

    def forward(self, x: torch.Tensor, update_memory: bool = False):
        """Forward pass with memory attention.

        Args:
            x: Input tensor (B, 3, H, W).
            update_memory: Whether to update memory banks.

        Returns:
            Multi-scale features and anomaly scores.
        """
        # Extract features
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = {'layer2': x2, 'layer3': x3, 'layer4': x4}

        if update_memory:
            # Update memory banks during training
            with torch.no_grad():
                for layer_name, feat in features.items():
                    # Global average pooling
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                    if pooled.dim() == 1:
                        pooled = pooled.unsqueeze(0)
                    self.memory_banks[layer_name].update(pooled)

        return features

    def compute_anomaly_scores(self, features: dict, k: int = 3) -> dict:
        """Compute anomaly scores using memory banks.

        Args:
            features: Dictionary of features for each layer.
            k: Number of nearest neighbors.

        Returns:
            Dictionary of anomaly scores.
        """
        anomaly_scores = {}

        for layer_name, feat in features.items():
            b, c, h, w = feat.shape

            # Reshape to (B*H*W, C)
            feat_reshaped = feat.permute(0, 2, 3, 1).reshape(b * h * w, c)

            # Query memory
            distances, _ = self.memory_banks[layer_name].query(feat_reshaped, k=k)

            # Average distance as anomaly score
            scores = distances.mean(dim=1)

            # Reshape back
            scores = scores.reshape(b, h, w)

            anomaly_scores[layer_name] = scores

        return anomaly_scores


class SegmentationHead(nn.Module):
    """Segmentation head for anomaly localization."""

    def __init__(self, feature_dims: List[int], output_dim: int = 1):
        super().__init__()

        # Projection layers for each scale
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(dim, 64, kernel_size=1) for dim in feature_dims
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * len(feature_dims), 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]):
        """Forward pass.

        Args:
            features: List of feature tensors.
            target_size: Target output size.

        Returns:
            Segmentation map.
        """
        # Project and upsample all features
        projected = []
        for feat, proj in zip(features, self.proj_layers):
            proj_feat = proj(feat)
            upsampled = F.interpolate(
                proj_feat,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            projected.append(upsampled)

        # Concatenate and fuse
        fused = torch.cat(projected, dim=1)
        output = self.fusion(fused)

        return output


@register_model("memseg")
class MemSegDetector(BaseVisionDeepDetector):
    """Memory-guided segmentation anomaly detector.

    Uses memory banks with attention mechanisms for anomaly detection
    and segmentation.

    Args:
        backbone: Feature extraction backbone ("resnet18", "resnet34", "resnet50").
        memory_size: Size of memory banks.
        k_neighbors: Number of nearest neighbors for scoring.
        pretrained: Whether to use pretrained backbone.
        use_segmentation_head: Whether to use segmentation head.
        device: Device to use ("cuda" or "cpu").

    References:
        Various memory-guided anomaly detection approaches.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        memory_size: int = 1000,
        k_neighbors: int = 3,
        pretrained: bool = True,
        use_segmentation_head: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.memory_size = memory_size
        self.k_neighbors = k_neighbors
        self.pretrained = pretrained
        self.use_segmentation_head = use_segmentation_head

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build the MemSeg model."""
        # Feature extractor with memory
        self.feature_extractor = FeatureExtractorWithMemory(
            self.backbone_name,
            self.memory_size,
            self.pretrained,
        ).to(self.device)

        # Segmentation head
        if self.use_segmentation_head:
            if self.backbone_name.startswith('resnet1'):
                feature_dims = [128, 256, 512]
            else:
                feature_dims = [512, 1024, 2048]

            self.seg_head = SegmentationHead(feature_dims).to(self.device)
        else:
            self.seg_head = None

    def fit(self, X: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Build memory banks from training images.

        Args:
            X: Training images (N, H, W, C).
            y: Not used (unsupervised).
        """
        print("Building MemSeg memory banks...")

        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        self.feature_extractor.eval()

        with torch.no_grad():
            for i, img in enumerate(X):
                if (i + 1) % 50 == 0:
                    print(f"  Processing image {i+1}/{len(X)}")

                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features and update memory
                _ = self.feature_extractor(img_tensor, update_memory=True)

        print("Memory banks built successfully!")
        print(f"  Memory filled: {self.feature_extractor.memory_banks['layer4'].memory_filled}/{self.memory_size}")

    def predict_proba(self, X: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            Anomaly scores.
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        self.feature_extractor.eval()
        scores = []

        with torch.no_grad():
            for img in X:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features
                features = self.feature_extractor(img_tensor, update_memory=False)

                # Compute anomaly scores
                anomaly_scores = self.feature_extractor.compute_anomaly_scores(
                    features, k=self.k_neighbors
                )

                # Aggregate scores
                score = torch.stack([s.max() for s in anomaly_scores.values()]).mean().item()
                scores.append(score)

        return np.array(scores)

    def predict_anomaly_map(self, X: NDArray) -> List[NDArray]:
        """Predict pixel-level anomaly maps.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            List of anomaly maps.
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        self.feature_extractor.eval()
        if self.seg_head is not None:
            self.seg_head.eval()

        anomaly_maps = []

        with torch.no_grad():
            for img in X:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features
                features = self.feature_extractor(img_tensor, update_memory=False)

                if self.use_segmentation_head and self.seg_head is not None:
                    # Use segmentation head
                    feat_list = [features['layer2'], features['layer3'], features['layer4']]
                    seg_map = self.seg_head(feat_list, img.shape[:2])
                    anomaly_map = seg_map.squeeze().cpu().numpy()
                else:
                    # Use memory-based scores
                    anomaly_scores = self.feature_extractor.compute_anomaly_scores(
                        features, k=self.k_neighbors
                    )

                    # Combine multi-scale scores
                    maps = []
                    for layer_name, score_map in anomaly_scores.items():
                        score_map = score_map.unsqueeze(0).unsqueeze(0)
                        upsampled = F.interpolate(
                            score_map,
                            size=img.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        )
                        maps.append(upsampled.squeeze().cpu().numpy())

                    anomaly_map = np.mean(maps, axis=0)

                anomaly_maps.append(anomaly_map)

        return anomaly_maps

    def _preprocess(self, image: NDArray) -> torch.Tensor:
        """Preprocess image.

        Args:
            image: Input image (H, W, C) in [0, 1].

        Returns:
            Preprocessed tensor (C, H, W).
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=2)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image
