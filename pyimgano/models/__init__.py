"""模型模块，提供统一的工厂与注册接口。"""

from importlib import import_module
from typing import Iterable
import warnings

from .baseml import BaseVisionDetector
from .baseCv import BaseVisionDeepDetector
from .registry import MODEL_REGISTRY, create_model, list_models, register_model


def _auto_import(modules: Iterable[str]) -> None:
    """按需导入并触发注册表装饰器。"""

    for module_name in modules:
        try:
            import_module(f"{__name__}.{module_name}")
        except Exception as exc:  # noqa: BLE001 - 记录导入失败信息
            warnings.warn(
                f"加载模型模块 {module_name!r} 失败: {exc}",
                RuntimeWarning,
            )


_auto_import(
    [
        # Classical ML algorithms
        "abod",
        "cblof",
        "cof",  # Connectivity-based outlier factor
        "copod",  # High-performance, parameter-free (ICDM 2020)
        "crossmad",  # NEW: Cross-Modal AD (CVPR 2025) ⭐⭐⭐⭐ 🚀
        "dbscan",
        "ecod",  # State-of-the-art, parameter-free (TKDE 2022)
        "feature_bagging",  # Ensemble method
        "hbos",
        "inne",  # Isolation using nearest neighbors
        "Isolationforest",
        "knn",  # K-Nearest Neighbors (classic)
        "kpca",
        "k_means",
        "loci",
        "loda",
        "lof",
        "lscp",
        "mcd",  # Minimum covariance determinant
        "ocsvm",
        "pca",  # Principal Component Analysis (classic)
        "suod",
        "xgbod",
        # Deep learning algorithms
        "ae",
        "ae1svm",
        "alad",
        "ast",  # NEW: Anomaly-aware Student-Teacher (2023) ⭐⭐ 🆕
        "bayesianpf",  # NEW: Bayesian Prompt Flow (CVPR 2025) ⭐⭐⭐⭐ 🚀
        "bgad",  # NEW: Background-guided detection (CVPR 2023) ⭐⭐ 🆕
        "cflow",  # NEW: Conditional normalizing flows (WACV 2022) ⭐
        "csflow",  # NEW: Cross-scale flows (WACV 2022) ⭐⭐ 🆕
        "cutpaste",  # NEW: Self-supervised learning (CVPR 2021) ⭐⭐
        "deep_svdd",
        "devnet",  # NEW: Deviation networks (KDD 2019) ⭐⭐ 🆕
        "dfm",  # NEW: Fast discriminative feature modeling ⭐
        "differnet",  # NEW: Learnable difference detector (WACV 2023) ⭐⭐
        "draem",  # NEW: Discriminative reconstruction (ICCV 2021) ⭐
        "dsr",  # NEW: Deep spectral residual (WACV 2023) ⭐⭐ 🆕
        "dst",  # NEW: Double Student-Teacher (2023) ⭐⭐ 🆕
        "efficientad",
        "fastflow",
        "favae",  # NEW: Feature Adaptive VAE (2023) ⭐⭐ 🆕
        "fcdd",  # NEW: Fully Convolutional Data Description (ICLR 2021) ⭐⭐⭐ 🆕
        "gcad",  # NEW: Graph Convolutional AD (2023) ⭐⭐ 🆕
        "glad",  # NEW: Global-Local Adaptive Diffusion (ECCV 2024) ⭐⭐⭐ 🔥
        "gmm",  # NEW: Gaussian Mixture Model (Classic) ⭐⭐ 🆕
        "imdd",
        "inctrl",  # NEW: In-context Residual Learning (CVPR 2024) ⭐⭐⭐ 🔥
        "intra",  # NEW: Industrial Transformer (ICCV 2023) ⭐⭐ 🆕
        "mahalanobis",  # NEW: Mahalanobis Distance (NeurIPS 2018) ⭐⭐⭐ 🆕
        "memae",  # NEW: Memory-Augmented Autoencoder (ICCV 2019) ⭐⭐⭐ 🆕
        "memseg",  # NEW: Memory-guided segmentation ⭐⭐ 🆕
        "mo_gaal",
        "oddoneout",  # NEW: Odd-One-Out (CVPR 2025) ⭐⭐⭐⭐ 🚀
        "one_svm_cnn",
        "oneformore",  # NEW: Continual Diffusion (#1 MVTec/VisA, CVPR 2025) ⭐⭐⭐⭐⭐ 🚀
        "padim",
        "panda",  # NEW: Prototypical Anomaly Network (2023) ⭐⭐ 🆕
        "patchcore",  # SOTA patch-level detection (CVPR 2022)
        "promptad",  # NEW: Prompt-based Few-Shot (CVPR 2024) ⭐⭐⭐ 🔥
        "pni",  # NEW: Pyramidal normality indexing (CVPR 2022) ⭐⭐ 🆕
        "rdplusplus",  # NEW: Reverse Distillation++ (Enhanced) ⭐⭐ 🆕
        "realnet",  # NEW: Realistic Synthetic Anomaly (CVPR 2024) ⭐⭐⭐ 🔥
        "regad",  # NEW: Registration-based AD (2023) ⭐⭐ 🆕
        "reverse_distillation",
        "riad",  # NEW: Reconstruction from adjacent decomposition ⭐⭐ 🆕
        "simplenet",  # Ultra-fast SOTA (CVPR 2023)
        "spade",  # NEW: Sub-image anomaly detection (ECCV 2020) ⭐⭐⭐ 🆕
        "ssim",
        "ssim_struct",
        "stfpm",  # Student-Teacher matching (BMVC 2021)
        "vae",
        "winclip",  # NEW: Zero-shot CLIP-based (CVPR 2023) ⭐⭐⭐
    ]
)

from .ae import OptimizedAEDetector  # noqa: E402  # re-export常用模型
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
