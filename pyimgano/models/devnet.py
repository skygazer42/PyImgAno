"""
DevNet: Deviation Networks for Weakly-Supervised Anomaly Detection.

Paper: https://arxiv.org/abs/1911.08623
Conference: KDD 2019

DevNet learns anomaly scores using a small number of labeled anomalies
through deviation loss that measures how much samples deviate from normal.

Key Features:
- Weakly-supervised (needs few anomaly labels)
- Deviation loss for scoring
- Works with limited labels
- Flexible architecture
- Good generalization
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray
from torch.utils.data import DataLoader, TensorDataset

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


class DeviationLoss(nn.Module):
    """Deviation loss for anomaly score learning."""

    def __init__(self, margin: float = 5.0):
        """Initialize deviation loss.

        Args:
            margin: Margin for deviation loss.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        ref_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute deviation loss.

        Args:
            scores: Predicted anomaly scores (N,).
            labels: Binary labels (0=normal, 1=anomaly) (N,).
            ref_scores: Reference scores from normal samples.

        Returns:
            Loss value.
        """
        # Separate normal and anomaly samples
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        loss = 0.0

        # For normal samples: minimize score
        if normal_mask.sum() > 0:
            normal_scores = scores[normal_mask]
            normal_loss = normal_scores.mean()
            loss += normal_loss

        # For anomaly samples: maximize deviation from normal
        if anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
            anomaly_scores = scores[anomaly_mask]

            # Reference: mean normal score
            if ref_scores is not None:
                ref = ref_scores.mean()
            else:
                ref = scores[normal_mask].mean()

            # Deviation loss: encourage anomaly scores to be higher than normal + margin
            deviation = F.relu(self.margin - (anomaly_scores - ref))
            anomaly_loss = deviation.mean()
            loss += anomaly_loss

        return loss


class DevNetModel(nn.Module):
    """Deviation network model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2,
    ):
        """Initialize model.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer (anomaly score)
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, D).

        Returns:
            Anomaly scores (B, 1).
        """
        return self.model(x).squeeze(-1)


class FeatureExtractor(nn.Module):
    """Feature extractor for images."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        from torchvision import models

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

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Features (B, D).
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features


@register_model("devnet")
class DevNetDetector(BaseVisionDeepDetector):
    """DevNet anomaly detector.

    Weakly-supervised anomaly detection using deviation loss.
    Requires a small number of labeled anomaly samples.

    Args:
        backbone: Feature extraction backbone ("resnet18", "resnet34", "resnet50").
        hidden_dims: Hidden layer dimensions for DevNet.
        dropout: Dropout rate.
        margin: Margin for deviation loss.
        pretrained: Whether to use pretrained backbone.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        device: Device to use ("cuda" or "cpu").

    References:
        Pang et al. "Deep Anomaly Detection with Deviation Networks." KDD 2019.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        hidden_dims: list = [128, 64],
        dropout: float = 0.2,
        margin: float = 5.0,
        pretrained: bool = True,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.margin = margin
        self.pretrained = pretrained
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self._build_model()

    def _build_model(self):
        """Build the DevNet model."""
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            self.backbone_name,
            self.pretrained,
        ).to(self.device)

        # DevNet scoring model
        self.scoring_model = DevNetModel(
            self.feature_extractor.feature_dim,
            self.hidden_dims,
            self.dropout,
        ).to(self.device)

    def fit(self, X: NDArray, y: NDArray, **kwargs):
        """Train the DevNet model.

        Args:
            X: Training images (N, H, W, C).
            y: Labels (0=normal, 1=anomaly). Must include both normal and anomaly samples.
        """
        if y is None or len(np.unique(y)) < 2:
            raise ValueError(
                "DevNet requires labeled data with both normal (0) and anomaly (1) samples. "
                "For unsupervised learning, use other algorithms like CutPaste or SPADE."
            )

        print("Training DevNet (weakly-supervised)...")
        print(f"  Normal samples: {(y == 0).sum()}")
        print(f"  Anomaly samples: {(y == 1).sum()}")

        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        # Extract features
        print("Extracting features...")
        self.feature_extractor.eval()

        features_list = []
        with torch.no_grad():
            for img in X:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                features = self.feature_extractor(img_tensor)
                features_list.append(features.cpu())

        features = torch.cat(features_list, dim=0)
        labels = torch.from_numpy(y).long()

        # Create dataset
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.scoring_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        criterion = DeviationLoss(margin=self.margin)

        # Training loop
        self.scoring_model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward
                scores = self.scoring_model(batch_features)

                # Loss
                loss = criterion(scores, batch_labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

        self.scoring_model.eval()
        print("Training completed!")

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
        self.scoring_model.eval()

        scores = []

        with torch.no_grad():
            for img in X:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)

                # Extract features
                features = self.feature_extractor(img_tensor)

                # Compute score
                score = self.scoring_model(features)

                scores.append(score.item())

        return np.array(scores)

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

    def get_feature_importance(self, X: NDArray) -> NDArray:
        """Get feature importance scores.

        Args:
            X: Images (N, H, W, C).

        Returns:
            Feature importance scores (D,).
        """
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0

        self.feature_extractor.eval()
        self.scoring_model.eval()

        # Extract features
        features_list = []
        with torch.no_grad():
            for img in X:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                features = self.feature_extractor(img_tensor)
                features_list.append(features)

        features = torch.cat(features_list, dim=0)

        # Compute gradients w.r.t. features
        features.requires_grad = True
        scores = self.scoring_model(features)
        total_score = scores.sum()

        total_score.backward()

        # Feature importance = absolute gradient
        importance = features.grad.abs().mean(dim=0).cpu().numpy()

        return importance
