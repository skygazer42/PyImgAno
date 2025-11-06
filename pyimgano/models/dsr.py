"""
Deep Spectral Residual (DSR) for Anomaly Detection

Paper: "Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Anomaly Detection"
Conference: WACV 2023

Key Innovation:
- Combines deep features with spectral (frequency domain) analysis
- Uses FFT to detect anomalies in frequency space
- Saliency-based anomaly detection
- Fast and parameter-free inference

Implementation includes:
- Deep feature extraction from pre-trained models
- Fast Fourier Transform for spectral analysis
- Spectral residual computation
- Saliency map generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from pyimgano.models.base_dl import BaseVisionDeepDetector


class SpectralResidual:
    """Compute spectral residual for anomaly detection."""

    @staticmethod
    def compute(features: NDArray) -> NDArray:
        """Compute spectral residual of features.

        Args:
            features: Input features [H, W, C]

        Returns:
            Spectral residual [H, W]
        """
        if features.ndim == 3:
            # Process each channel and average
            residuals = []
            for c in range(features.shape[2]):
                channel = features[:, :, c]
                residual = SpectralResidual._compute_single_channel(channel)
                residuals.append(residual)
            return np.mean(residuals, axis=0)
        else:
            return SpectralResidual._compute_single_channel(features)

    @staticmethod
    def _compute_single_channel(channel: NDArray) -> NDArray:
        """Compute spectral residual for a single channel.

        Args:
            channel: Input channel [H, W]

        Returns:
            Spectral residual [H, W]
        """
        # FFT
        fft = np.fft.fft2(channel)
        amplitude = np.abs(fft)
        phase = np.angle(fft)

        # Log amplitude spectrum
        log_amplitude = np.log(amplitude + 1e-8)

        # Average filter (box filter)
        kernel_size = 3
        avg_filter = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        from scipy.signal import convolve2d
        log_amplitude_filtered = convolve2d(log_amplitude, avg_filter, mode='same')

        # Spectral residual
        spectral_residual = log_amplitude - log_amplitude_filtered

        # Inverse FFT
        residual_fft = np.exp(spectral_residual) * np.exp(1j * phase)
        saliency_map = np.abs(np.fft.ifft2(residual_fft)) ** 2

        # Normalize
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        return saliency_map


class FeatureExtractor(nn.Module):
    """Extract deep features for spectral analysis."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            # Extract features up to layer3
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
            )
            self.output_dim = 256
        elif backbone == "wide_resnet50":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
            )
            self.output_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Features [B, C', H', W']
        """
        return self.features(x)


class DSRDetector(BaseVisionDeepDetector):
    """Deep Spectral Residual (DSR) detector.

    Combines deep features with spectral analysis for anomaly detection.
    Key advantage: parameter-free, no training needed after feature extraction.

    Args:
        backbone: Feature extraction backbone
        gaussian_sigma: Sigma for smoothing saliency maps
        threshold_percentile: Percentile for automatic thresholding
        pretrained: Use pre-trained backbone
        device: Device to run on

    Example:
        >>> detector = DSRDetector(backbone="resnet18")
        >>> detector.fit(normal_images)
        >>> scores = detector.predict_proba(test_images)
        >>> anomaly_maps = detector.predict_anomaly_map(test_images)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        gaussian_sigma: float = 4.0,
        threshold_percentile: float = 95.0,
        pretrained: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.backbone_name = backbone
        self.gaussian_sigma = gaussian_sigma
        self.threshold_percentile = threshold_percentile

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained
        ).to(self.device)

        self.spectral_residual = SpectralResidual()
        self.fitted_ = False
        self.normal_scores_stats = None

    def _extract_features(self, images: NDArray) -> torch.Tensor:
        """Extract deep features from images.

        Args:
            images: Input images [N, H, W, C] or [N, C, H, W]

        Returns:
            Features [N, C, H, W]
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

    def _compute_spectral_anomaly_map(self, features: torch.Tensor) -> NDArray:
        """Compute spectral residual anomaly maps.

        Args:
            features: Deep features [N, C, H, W]

        Returns:
            Anomaly maps [N, H_img, W_img]
        """
        features_np = features.cpu().numpy()
        N, C, H, W = features_np.shape

        anomaly_maps = []
        for i in range(N):
            # Transpose to [H, W, C]
            feat = features_np[i].transpose(1, 2, 0)  # [H, W, C]

            # Compute spectral residual
            saliency = self.spectral_residual.compute(feat)

            # Smooth
            saliency = gaussian_filter(saliency, sigma=self.gaussian_sigma)

            anomaly_maps.append(saliency)

        return np.array(anomaly_maps)

    def fit(self, X: NDArray, y: Optional[NDArray] = None):
        """Fit the detector on normal images.

        For DSR, this computes statistics of normal spectral residuals
        for later normalization.

        Args:
            X: Normal images [N, H, W, C]
            y: Ignored (unsupervised)
        """
        # Extract features
        features = self._extract_features(X)

        # Compute anomaly maps for normal images
        anomaly_maps = self._compute_spectral_anomaly_map(features)

        # Store statistics
        self.normal_scores_stats = {
            'mean': np.mean(anomaly_maps),
            'std': np.std(anomaly_maps),
            'percentile_95': np.percentile(anomaly_maps, self.threshold_percentile)
        }

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

        # Compute anomaly maps
        anomaly_maps = self._compute_spectral_anomaly_map(features)

        # Aggregate to image-level scores (max)
        scores = np.max(anomaly_maps.reshape(anomaly_maps.shape[0], -1), axis=1)

        # Normalize using normal statistics
        if self.normal_scores_stats is not None:
            scores = (scores - self.normal_scores_stats['mean']) / (self.normal_scores_stats['std'] + 1e-8)

        return scores

    def predict_anomaly_map(self, X: NDArray) -> NDArray:
        """Generate pixel-level anomaly maps.

        Args:
            X: Test images [N, H, W, C]

        Returns:
            Anomaly maps [N, H, W]
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get original image size
        H_img, W_img = X.shape[1:3] if X.shape[-1] == 3 else X.shape[2:4]

        # Extract features
        features = self._extract_features(X)

        # Compute anomaly maps
        anomaly_maps = self._compute_spectral_anomaly_map(features)

        # Upsample to original image size
        N = anomaly_maps.shape[0]
        upsampled_maps = np.zeros((N, H_img, W_img))

        for i in range(N):
            map_tensor = torch.from_numpy(anomaly_maps[i]).unsqueeze(0).unsqueeze(0).float()
            upsampled = F.interpolate(
                map_tensor,
                size=(H_img, W_img),
                mode='bilinear',
                align_corners=False
            )
            upsampled_maps[i] = upsampled.squeeze().numpy()

        # Normalize
        if self.normal_scores_stats is not None:
            upsampled_maps = (upsampled_maps - self.normal_scores_stats['mean']) / (
                self.normal_scores_stats['std'] + 1e-8
            )

        return upsampled_maps

    def predict(self, X: NDArray, threshold: Optional[float] = None) -> NDArray:
        """Predict anomaly labels.

        Args:
            X: Test images [N, H, W, C]
            threshold: Anomaly threshold (if None, uses percentile from fit)

        Returns:
            Labels [N] (0=normal, 1=anomaly)
        """
        scores = self.predict_proba(X)

        if threshold is None:
            # Use auto threshold from normal data
            if self.normal_scores_stats is not None:
                threshold = (
                    self.normal_scores_stats['percentile_95'] - self.normal_scores_stats['mean']
                ) / (self.normal_scores_stats['std'] + 1e-8)
            else:
                threshold = 0.0

        predictions = (scores > threshold).astype(int)
        return predictions
