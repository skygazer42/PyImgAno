"""
Models module providing unified factory and registry interfaces.

This module auto-imports all available models and registers them
in the MODEL_REGISTRY for dynamic model creation.
"""

from importlib import import_module
from typing import Iterable
import warnings

from .baseml import BaseVisionDetector
from .baseCv import BaseVisionDeepDetector
from .registry import MODEL_REGISTRY, create_model, list_models, register_model


def _auto_import(modules: Iterable[str]) -> None:
    """
    Auto-import modules to trigger registry decorators.

    Parameters
    ----------
    modules : Iterable[str]
        Module names to import
    """
    for module_name in modules:
        try:
            import_module(f"{__name__}.{module_name}")
        except Exception as exc:  # noqa: BLE001 - Log import failures
            warnings.warn(
                f"Failed to load model module {module_name!r}: {exc}",
                RuntimeWarning,
            )


_auto_import(
    [
        # Classical ML algorithms
        "abod",  # Angle-Based Outlier Detection
        "cblof",  # Cluster-Based Local Outlier Factor
        "cof",  # Connectivity-based Outlier Factor
        "copod",  # Copula-based Outlier Detection (ICDM 2020)
        "crossmad",  # Cross-Modal Anomaly Detection (CVPR 2025)
        "dbscan",  # Density-Based Spatial Clustering
        "ecod",  # Empirical Cumulative Outlier Detection (TKDE 2022)
        "feature_bagging",  # Feature Bagging ensemble method
        "hbos",  # Histogram-Based Outlier Score
        "inne",  # Isolation using Nearest Neighbors
        "Isolationforest",  # Isolation Forest
        "knn",  # K-Nearest Neighbors
        "kpca",  # Kernel Principal Component Analysis
        "k_means",  # K-Means clustering-based detection
        "loci",  # Local Correlation Integral
        "loda",  # Lightweight On-line Detector of Anomalies
        "lof",  # Local Outlier Factor
        "lscp",  # Locally Selective Combination in Parallel
        "mcd",  # Minimum Covariance Determinant
        "ocsvm",  # One-Class Support Vector Machine
        "pca",  # Principal Component Analysis
        "suod",  # Scalable Unsupervised Outlier Detection
        "xgbod",  # Extreme Gradient Boosting Outlier Detection
        # Deep learning algorithms
        "ae",  # Autoencoder
        "ae1svm",  # Autoencoder with One-Class SVM
        "alad",  # Adversarially Learned Anomaly Detection
        "ast",  # Anomaly-aware Student-Teacher (ICCV 2023)
        "bayesianpf",  # Bayesian Prompt Flow (CVPR 2025)
        "bgad",  # Background-guided Anomaly Detection (CVPR 2023)
        "cflow",  # Conditional Normalizing Flows (WACV 2022)
        "csflow",  # Cross-scale Flows (WACV 2022)
        "cutpaste",  # CutPaste self-supervised learning (CVPR 2021)
        "deep_svdd",  # Deep Support Vector Data Description
        "devnet",  # Deviation Networks (KDD 2019)
        "dfm",  # Discriminative Feature Modeling
        "differnet",  # DifferNet learnable difference detector (WACV 2023)
        "draem",  # Discriminatively Reconstructed Embedding (ICCV 2021)
        "dsr",  # Deep Spectral Residual (WACV 2023)
        "dst",  # Double Student-Teacher (ICCV 2023)
        "efficientad",  # EfficientAD
        "fastflow",  # FastFlow normalizing flows
        "favae",  # Feature Adaptive Variational Autoencoder (ICCV 2023)
        "gcad",  # Graph Convolutional Anomaly Detection (ICCV 2023)
        "glad",  # Global-Local Adaptive Diffusion (ECCV 2024)
        "imdd",  # Image-level Multi-scale Discriminative Detector
        "inctrl",  # In-context Residual Learning (CVPR 2024)
        "intra",  # Industrial Transformer (ICCV 2023)
        "memseg",  # Memory-guided Segmentation
        "mo_gaal",  # Multi-Objective Generative Adversarial Active Learning
        "oddoneout",  # Odd-One-Out neighbor comparison (CVPR 2025)
        "one_svm_cnn",  # One-Class SVM with CNN features
        "oneformore",  # One-for-More continual diffusion (CVPR 2025)
        "padim",  # Patch Distribution Modeling
        "panda",  # Prototypical Anomaly Network (ICCV 2023)
        "patchcore",  # PatchCore patch-level detection (CVPR 2022)
        "promptad",  # Prompt-based Few-Shot Anomaly Detection (CVPR 2024)
        "pni",  # Pyramidal Normality Indexing (CVPR 2022)
        "rdplusplus",  # Reverse Distillation++ (CVPR 2022)
        "realnet",  # RealNet realistic synthetic anomalies (CVPR 2024)
        "regad",  # Registration-based Anomaly Detection (ICCV 2023)
        "reverse_distillation",  # Reverse Distillation
        "riad",  # Reconstruction from Adjacent Image Decomposition
        "simplenet",  # SimpleNet ultra-fast detector (CVPR 2023)
        "spade",  # Sub-image Anomaly Detection with SPADE (ECCV 2020)
        "ssim",  # Structural Similarity-based detection
        "ssim_struct",  # SSIM with structural features
        "stfpm",  # Student-Teacher Feature Pyramid Matching (BMVC 2021)
        "vae",  # Variational Autoencoder
        "winclip",  # WinCLIP zero-shot CLIP-based (CVPR 2023)
    ]
)

from .ae import OptimizedAEDetector  # noqa: E402  # Re-export commonly used models
from .loda import VisionLODA  # noqa: E402
from .vae import VAEAnomalyDetector  # noqa: E402

__all__ = [
    "BaseVisionDetector",
    "BaseVisionDeepDetector",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    "register_model",
    "VisionLODA",
    "OptimizedAEDetector",
    "VAEAnomalyDetector",
]
