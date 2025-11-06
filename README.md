# PyImgAno

**Enterprise-Grade Visual Anomaly Detection Toolkit**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready Python toolkit for visual anomaly detection, integrating **47+ state-of-the-art algorithms** from classical machine learning to cutting-edge deep learning (CVPR 2023, CVPR 2022, ECCV 2020, ICCV 2021, WACV 2023, KDD 2019).

> **Translations:** [中文](README_cn.md) · [日本語](README_ja.md) · [한국어](README_ko.md)

---

## ✨ Key Features

- 🔥 **47+ Detection Algorithms** - From classical (ECOD, COPOD, KNN, PCA) to latest SOTA (PNI, DSR, CSFlow, SPADE, WinCLIP, SimpleNet)
- 🚀 **Production Ready** - Enterprise-grade code quality, comprehensive testing, CI/CD pipelines
- 📦 **Unified API** - Consistent interface across all algorithms with factory pattern
- ⚡ **High Performance** - Top-tier algorithms (ECOD, COPOD) optimized for speed and accuracy
- 🎯 **Flexible** - Works with any feature extractor or end-to-end deep learning
- 🖼️ **Image Preprocessing** - 80+ operations (edge detection, morphology, filters, FFT, texture analysis, segmentation, augmentation) with easy integration
- 📊 **Comprehensive Evaluation** - AUROC, AP, F1, confusion matrix, and more
- 🏆 **Built-in Benchmarking** - Compare multiple algorithms systematically
- 🎨 **Rich Visualization** - Anomaly heatmaps, ROC curves, score distributions
- 📖 **Well Documented** - Extensive docs, algorithm guide, and examples
- 🔧 **Easy to Extend** - Plugin architecture with model registry system

---

## 🏆 Highlights

### State-of-the-Art Algorithms

| Algorithm | Type | Year | Performance | Speed | Use Case |
|-----------|------|------|-------------|-------|----------|
| **WinCLIP** ⭐ | Deep Learning | 2023 | ⭐⭐⭐⭐⭐ | ⚡⚡ | Zero-shot, no training needed |
| **SimpleNet** ⭐ | Deep Learning | 2023 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Ultra-fast SOTA, production |
| **DifferNet** ⭐ | Deep Learning | 2023 | ⭐⭐⭐⭐ | ⚡⚡⚡ | Learnable differences, k-NN |
| **DSR** ⭐ 🆕 | Deep Learning | 2023 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Frequency domain, parameter-free |
| **PatchCore** ⭐ | Deep Learning | 2022 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Best accuracy, MVTec champion |
| **PNI** ⭐ 🆕 | Deep Learning | 2022 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Multi-scale pyramid, fast |
| **ECOD** | Classical | 2022 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Parameter-free, general purpose |
| **COPOD** | Classical | 2020 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Speed-critical applications |
| **SPADE** ⭐ | Deep Learning | 2020 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Excellent localization, k-NN |
| **CSFlow** 🆕 | Deep Learning | 2022 | ⭐⭐⭐⭐ | ⚡⚡⚡ | Expressive flows, good accuracy |
| **CutPaste** | Deep Learning | 2021 | ⭐⭐⭐⭐ | ⚡⚡⚡ | Self-supervised, no anomalies |
| **STFPM** | Deep Learning | 2021 | ⭐⭐⭐⭐ | ⚡⚡ | Student-Teacher, localization |
| **FastFlow** | Deep Learning | 2021 | ⭐⭐⭐⭐ | ⚡⚡⚡ | Normalizing flows, real-time |

> **See [Algorithm Selection Guide](docs/ALGORITHM_SELECTION_GUIDE.md) and [Deep Learning Guide](docs/DEEP_LEARNING_MODELS.md) for detailed comparison**

---

## 📦 Installation

### Basic Installation

```bash
pip install pyimgano
```

### Development Installation

```bash
git clone https://github.com/jhlu2019/pyimgano.git
cd pyimgano
pip install -e .[dev]
```

### With Optional Dependencies

```bash
# For diffusion models
pip install pyimgano[diffusion]

# For documentation
pip install pyimgano[docs]

# Everything
pip install pyimgano[all]
```

### Requirements

- Python >= 3.9
- PyTorch >= 1.9.0
- PyOD >= 1.1.0
- scikit-learn >= 0.22.0

---

## 🚀 Quick Start

### Example 1: Using ECOD (Recommended for First-Time Users)

```python
from pyimgano import models, utils

# 1. Create feature extractor
feature_extractor = utils.ImagePreprocessor(
    resize=(224, 224),
    output_tensor=False
)

# 2. Create detector (ECOD: parameter-free, top performance)
detector = models.create_model(
    "vision_ecod",
    feature_extractor=feature_extractor,
    contamination=0.1,  # Expected anomaly ratio
    n_jobs=-1  # Use all CPU cores
)

# 3. Train on normal images
train_paths = ["normal_1.jpg", "normal_2.jpg", ...]
detector.fit(train_paths)

# 4. Detect anomalies
test_paths = ["test_1.jpg", "test_2.jpg", ...]
predictions = detector.predict(test_paths)  # 0: normal, 1: anomaly
scores = detector.decision_function(test_paths)  # Anomaly scores

print(f"Detected {predictions.sum()} anomalies")
```

### Example 2: Using Deep Learning (SimpleNet - CVPR 2023)

```python
from pyimgano import models

# State-of-the-art deep learning (CVPR 2023)
# Ultra-fast training - only 10 epochs needed!
detector = models.create_model(
    "vision_simplenet",
    epochs=10,        # 10x faster than other DL methods
    batch_size=8,
    device='cuda'     # GPU acceleration
)

detector.fit(train_paths)
predictions = detector.predict(test_paths)
scores = detector.decision_function(test_paths)
```

### Example 3: Maximum Accuracy (PatchCore - CVPR 2022)

```python
# Best accuracy on MVTec AD benchmark (99.6% AUROC)
detector = models.create_model(
    "vision_patchcore",
    backbone='wide_resnet50',
    coreset_sampling_ratio=0.1,  # Memory-efficient
    device='cuda'
)

detector.fit(train_paths)
predictions = detector.predict(test_paths)

# Get pixel-level anomaly heatmap
anomaly_map = detector.get_anomaly_map('test_image.jpg')
```

### Example 4: Zero-Shot Detection (WinCLIP - CVPR 2023) ⭐ NEW

```python
# No training needed! Perfect for quick prototyping
detector = models.create_model(
    "vision_winclip",
    clip_model="ViT-B/32",
    k_shot=0  # Zero-shot mode
)

# Just set the class name and predict
detector.set_class_name("screw")
predictions = detector.predict(test_paths)
anomaly_maps = detector.predict_anomaly_map(test_paths)  # Pixel-level heatmaps
```

### Example 5: Best Localization (SPADE - ECCV 2020) ⭐ NEW

```python
# Excellent pixel-level anomaly localization
detector = models.create_model(
    "vision_spade",
    backbone="wide_resnet50",
    k_neighbors=50,
    feature_levels=["layer1", "layer2", "layer3"]
)

detector.fit(train_paths)
predictions = detector.predict(test_paths)
anomaly_maps = detector.predict_anomaly_map(test_paths)  # Precise localization
```

### Example 6: Self-Supervised Learning (CutPaste - CVPR 2021) ⭐ NEW

```python
# Train without any anomaly samples
detector = models.create_model(
    "vision_cutpaste",
    backbone="resnet18",
    augment_type="3way",  # normal, cutpaste, scar
    epochs=256
)

detector.fit(normal_images_only)  # Only normal images needed
predictions = detector.predict(test_paths)
```

### Example 7: Comparing Multiple Algorithms

```python
algorithms = ["vision_ecod", "vision_copod", "vision_simplenet", "vision_spade"]
results = {}

for algo_name in algorithms:
    detector = models.create_model(
        algo_name,
        feature_extractor=feature_extractor if "vision_ecod" in algo_name else None,
        contamination=0.1
    )
    detector.fit(train_paths)
    results[algo_name] = detector.predict(test_paths)

# Compare results
for name, preds in results.items():
    print(f"{name}: {preds.sum()} anomalies detected")
```

---

## 🖼️ Image Preprocessing

Comprehensive preprocessing module with **80+ operations** including augmentation for enhanced anomaly detection:

### Quick Example

```python
from pyimgano.preprocessing import AdvancedImageEnhancer, PreprocessingMixin
from pyimgano.models import ECOD

# Method 1: Using Advanced Enhancer
enhancer = AdvancedImageEnhancer()

# Basic operations
blurred = enhancer.gaussian_blur(image, ksize=(5, 5))
edges = enhancer.detect_edges(blurred, method='canny')

# Advanced operations
retinex = enhancer.retinex_multi(image, sigmas=[15, 80, 250])  # Illumination invariant
lbp = enhancer.compute_lbp(image, n_points=8, radius=1.0)  # Texture features
hog_features = enhancer.extract_hog(image, visualize=False)  # HOG features

# Method 2: Using Mixin with Detector
class ECODWithPreprocessing(PreprocessingMixin, ECOD):
    def __init__(self):
        super().__init__()
        self.setup_preprocessing(enable=True, use_pipeline=True)
        self.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        self.add_preprocessing_step('normalize', method='minmax')

    def fit(self, X, y=None):
        X_processed = self.preprocess_images(X)
        return super().fit([img.flatten() for img in X_processed], y)

detector = ECODWithPreprocessing()
detector.fit(train_images)
scores = detector.predict(test_images)

# Method 3: Data Augmentation for Training
from pyimgano.preprocessing import (
    Compose, OneOf, RandomRotate, RandomFlip,
    ColorJitter, GaussianNoise, get_medium_augmentation
)

# Custom augmentation pipeline
aug_pipeline = Compose([
    RandomFlip(mode="horizontal", p=0.5),
    RandomRotate(angle_range=(-20, 20), p=0.5),
    ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=0.5),
    OneOf([
        GaussianNoise(std_range=(10, 25), p=1.0),
        MotionBlur(kernel_size_range=(5, 15), p=1.0),
    ], p=0.3),
])

# Or use preset augmentation
aug_pipeline = get_medium_augmentation()

# Augment training data
augmented_images = [aug_pipeline(img) for img in train_images]
```

### Available Operations (80+)

#### Basic Operations (25)
| Category | Operations | Count |
|----------|------------|-------|
| **Edge Detection** | Canny, Sobel, Laplacian, Scharr, Prewitt, Sobel X/Y | 7 |
| **Morphology** | Erosion, Dilation, Opening, Closing, Gradient, TopHat, BlackHat | 7 |
| **Filters** | Gaussian, Bilateral, Median, Box | 4 |
| **Normalization** | MinMax, Z-Score, L2, Robust | 4 |
| **Enhancement** | Sharpen, Unsharp Mask, CLAHE | 3 |

#### Advanced Operations (25)
| Category | Operations | Count |
|----------|------------|-------|
| **Frequency Domain** | FFT, IFFT, Lowpass, Highpass, Bandpass, Bandstop | 6 |
| **Texture Analysis** | Gabor filters, LBP, GLCM features | 3 |
| **Color Space** | RGB, HSV, LAB, YCrCb, HLS, LUV + conversions | 8 |
| **Enhancement** | Gamma correction, Contrast stretching, Retinex (SSR, MSR) | 4 |
| **Denoising** | Non-local means, Anisotropic diffusion | 2 |
| **Feature Extraction** | HOG, Harris corners, Shi-Tomasi, FAST | 4 |
| **Adv. Morphology** | Skeleton, Thinning, Convex hull, Distance transform | 4 |
| **Segmentation** | Otsu, Adaptive, Triangle, Yen, Watershed | 6 |
| **Pyramids** | Gaussian pyramid, Laplacian pyramid | 2 |

#### Augmentation Operations (30+) ⭐ NEW!
| Category | Operations | Count |
|----------|------------|-------|
| **Geometric** | Rotate, Flip, Scale, Translate, Shear, Perspective, Affine | 7 |
| **Color** | Brightness, Contrast, Saturation, Hue, Color jitter | 5 |
| **Noise** | Gaussian, Salt-and-pepper, Poisson, Speckle | 4 |
| **Blur** | Motion blur, Defocus blur, Glass blur, Zoom blur | 4 |
| **Weather** | Rain, Fog, Snow, Shadow, Sunflare | 5 |
| **Occlusion** | Random cutout, Grid mask, Mixup, CutMix | 4 |
| **Distortion** | Elastic transform, Grid distortion | 2 |
| **Pipelines** | Compose, OneOf, RandomApply, 5 preset pipelines | 8 |

**See [Preprocessing Guide](docs/PREPROCESSING.md) for detailed usage and best practices**

---

## 📚 Available Algorithms

### Classical Machine Learning (19 algorithms)

| Algorithm | Model Name | Key Features |
|-----------|------------|--------------|
| **ECOD** ⭐ | `vision_ecod` | Parameter-free, top performance, TKDE 2022 |
| **COPOD** ⭐ | `vision_copod` | Very fast, parameter-free, ICDM 2020 |
| **Feature Bagging** ⭐ | `vision_feature_bagging` | Ensemble, high stability |
| KNN | `vision_knn` | Simple, interpretable |
| PCA | `vision_pca` | Classic dimensionality reduction |
| Isolation Forest | `vision_iforest` | Robust, widely used |
| INNE | `vision_inne` | Fast isolation-based |
| LOF | `vision_lof` | Density-based, local outliers |
| COF | `vision_cof` | Connectivity-based |
| MCD | `vision_mcd` | Robust covariance |
| One-Class SVM | `vision_ocsvm` | Kernel methods |
| ABOD | `vision_abod` | Angle-based detection |
| CBLOF | `vision_cblof` | Cluster-based |
| HBOS | `vision_hbos` | Histogram-based |
| KPCA | `vision_kpca` | Kernel PCA |
| LODA | `vision_loda` | Lightweight online |
| LOCI | `vision_loci` | Local correlation |
| SUOD | `vision_suod` | Scalable ensemble |
| XGBOD | `vision_xgbod` | XGBoost-based |

### Deep Learning (28 algorithms)

| Algorithm | Model Name | Key Features |
|-----------|------------|--------------|
| **WinCLIP** ⭐ | `vision_winclip` | Zero-shot CLIP-based (CVPR 2023), no training |
| **SimpleNet** ⭐ | `vision_simplenet` | Ultra-fast SOTA (CVPR 2023), 10x faster training |
| **DifferNet** ⭐ | `vision_differnet` | Learnable differences (WACV 2023), k-NN |
| **DSR** ⭐ 🆕 | `vision_dsr` | Deep spectral residual (WACV 2023), frequency domain |
| **PatchCore** ⭐ | `vision_patchcore` | Best accuracy (CVPR 2022), pixel localization |
| **PNI** ⭐ 🆕 | `vision_pni` | Pyramidal normality indexing (CVPR 2022), multi-scale |
| **SPADE** ⭐ | `vision_spade` | Deep pyramid k-NN (ECCV 2020), excellent localization |
| **CutPaste** ⭐ | `vision_cutpaste` | Self-supervised (CVPR 2021), no anomaly data |
| **DRAEM** ⭐ | `vision_draem` | Synthetic anomalies (ICCV 2021), robust |
| **CSFlow** 🆕 | `vision_csflow` | Cross-scale normalizing flows (WACV 2022), expressive |
| **MemSeg** | `vision_memseg` | Memory-guided segmentation (2022) |
| **RIAD** | `vision_riad` | Inpainting-based (2020), self-supervised |
| **DevNet** | `vision_devnet` | Weakly-supervised (KDD 2019), few labels |
| **CFlow-AD** ⭐ | `vision_cflow` | Conditional flows (WACV 2022), real-time |
| **DFM** ⭐ | `vision_dfm` | Fast discriminative features, training-free |
| **STFPM** | `vision_stfpm` | Student-Teacher (BMVC 2021), multi-scale |
| FastFlow | `vision_fastflow` | Normalizing flows, real-time |
| PaDiM | `vision_padim` | Patch distribution, edge devices |
| Deep SVDD | `vision_deep_svdd` | One-class deep learning |
| VAE | `vision_vae` | Variational autoencoder |
| AutoEncoder | `vision_ae` | Classic reconstruction |
| Reverse Distillation | `vision_reverse_dist` | Knowledge distillation |
| EfficientAD | `vision_efficientad` | Lightweight, resource-efficient |
| SSIM-based | `vision_ssim` | Structural similarity |
| ALAD | `vision_alad` | Adversarial learning |
| AE+SVM | `vision_ae1svm` | Hybrid approach |
| MO_GAAL | `vision_mo_gaal` | Multi-objective GAN |
| IMDD | `vision_imdd` | Iterative methods |

⭐ = Recommended for production use

---

## 🎯 Use Cases

### Industrial Quality Control
```python
# Detect defects in manufactured products
detector = models.create_model("vision_ecod", ...)
detector.fit(normal_product_images)
defects = detector.predict(inspection_images)
```

### Medical Imaging
```python
# Identify abnormal X-rays
detector = models.create_model("vision_deep_svdd", ...)
detector.fit(normal_xrays)
abnormal_cases = detector.predict(patient_xrays)
```

### Security & Surveillance
```python
# Detect unusual behavior
detector = models.create_model("vision_copod", ...)
detector.fit(normal_scene_frames)
anomalies = detector.predict(monitoring_frames)
```

---

## 📊 Algorithm Comparison & Selection

### Quick Selection Guide

| Your Need | Recommended Algorithms | Why |
|-----------|------------------------|-----|
| **Best Overall Accuracy** | PatchCore, SPADE, FastFlow | 99%+ AUROC on MVTec AD |
| **Fastest Training** | SimpleNet, ECOD, COPOD | 10× faster than competitors |
| **Zero-Shot (No Training)** | WinCLIP | CLIP-based, text prompts |
| **Best Localization** | SPADE, PatchCore, STFPM | Pixel-perfect anomaly maps |
| **Limited Data** | WinCLIP, CutPaste, RIAD | Zero-shot or self-supervised |
| **Real-Time Inference** | SimpleNet, FastFlow, COPOD | 100+ FPS on GPU |
| **No GPU Available** | ECOD, COPOD, Feature Bagging | CPU-optimized classical ML |
| **Few Anomaly Labels** | DevNet | Weakly-supervised learning |
| **Production Deployment** | SimpleNet, ECOD, PatchCore | Stable, well-tested, fast |

### Performance Comparison (MVTec AD Dataset)

| Algorithm | Image AUROC | Pixel AUROC | Training Time | Inference Speed | Memory |
|-----------|-------------|-------------|---------------|-----------------|---------|
| **PatchCore** | 99.6% ⭐ | 98.7% ⭐ | Medium | 30-50 FPS | High |
| **SPADE** | 98.0% | 99.0% ⭐ | Low | 40-60 FPS | Medium |
| **SimpleNet** | 99.0% | 98.0% | Very Low ⭐ | 100+ FPS ⭐ | Low |
| **FastFlow** | 99.0% | 98.0% | Low | 60-80 FPS | Medium |
| **DifferNet** | 97.0% | 97.0% | Medium | 20-40 FPS | Medium |
| **CutPaste** | 96.0% | N/A | Medium | 50+ FPS | Low |
| **STFPM** | 97.0% | 98.0% | Medium | 40-60 FPS | Medium |
| **WinCLIP** | 95.0% | 98.0% | None ⭐ | 5-10 FPS | Low |
| **ECOD** | 85-90% | N/A | None ⭐ | 200+ FPS ⭐ | Very Low ⭐ |
| **COPOD** | 85-90% | N/A | None ⭐ | 300+ FPS ⭐ | Very Low ⭐ |

### Decision Tree

```
Start Here
│
├─ Have GPU?
│  ├─ YES
│  │  ├─ Need best accuracy? → PatchCore, SPADE
│  │  ├─ Need speed? → SimpleNet, FastFlow
│  │  ├─ No training data? → WinCLIP (zero-shot)
│  │  └─ Need localization? → SPADE, PatchCore
│  │
│  └─ NO (CPU only)
│     ├─ Need speed? → COPOD, ECOD
│     ├─ Need accuracy? → ECOD, Feature Bagging
│     └─ General purpose → ECOD (best balance)
│
├─ Type of Data?
│  ├─ Only normal samples → CutPaste, RIAD, WinCLIP
│  ├─ Few anomaly labels → DevNet
│  └─ Mixed normal/anomaly → Any algorithm
│
└─ Deployment Scenario?
   ├─ Edge device → PaDiM, EfficientAD
   ├─ Cloud/Server → PatchCore, SimpleNet
   └─ Real-time critical → SimpleNet, COPOD
```

---

## 📖 Documentation

### Core Guides
- **[Quick Start Guide](docs/QUICK_START.md)** ⭐ - Get started in 5 minutes
- **[SOTA Algorithms Guide](docs/SOTA_ALGORITHMS.md)** ⭐ NEW! - Latest state-of-the-art algorithms (WinCLIP, SPADE, etc.)
- **[Deep Learning Models Guide](docs/DEEP_LEARNING_MODELS.md)** ⭐ - Comprehensive deep learning guide
- **[Algorithm Selection Guide](docs/ALGORITHM_SELECTION_GUIDE.md)** - Choose the right algorithm
- **[Preprocessing Guide](docs/PREPROCESSING.md)** ⭐ - Image enhancement and preprocessing
- **[Evaluation & Benchmarking Guide](docs/EVALUATION_AND_BENCHMARK.md)** ⭐ - Comprehensive evaluation tools

### Reference
- **[API Reference](docs/)** - Detailed API documentation
- **[Examples](examples/)** - Code examples and tutorials
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- **[Changelog](CHANGELOG.md)** - Version history

---

## 🏗️ Project Structure

```
pyimgano/
├── pyimgano/
│   ├── models/          # 47+ anomaly detection algorithms
│   │   ├── Classical ML (19 algorithms)
│   │   │   ├── ecod.py          # ECOD (TKDE 2022)
│   │   │   ├── copod.py         # COPOD (ICDM 2020)
│   │   │   ├── feature_bagging.py
│   │   │   ├── knn.py, pca.py, lof.py, ...
│   │   │   └── ...
│   │   ├── Deep Learning (28 algorithms)
│   │   │   ├── winclip.py       # WinCLIP (CVPR 2023) ⭐
│   │   │   ├── simplenet.py     # SimpleNet (CVPR 2023) ⭐
│   │   │   ├── differnet.py     # DifferNet (WACV 2023) ⭐
│   │   │   ├── dsr.py           # DSR (WACV 2023) ⭐ 🆕
│   │   │   ├── patchcore.py     # PatchCore (CVPR 2022) ⭐
│   │   │   ├── pni.py           # PNI (CVPR 2022) ⭐ 🆕
│   │   │   ├── stfpm.py         # STFPM (BMVC 2021) ⭐
│   │   │   ├── cutpaste.py      # CutPaste (CVPR 2021) ⭐
│   │   │   ├── draem.py         # DRAEM (ICCV 2021) ⭐
│   │   │   ├── cflow.py         # CFlow-AD (WACV 2022) ⭐
│   │   │   ├── csflow.py        # CSFlow (WACV 2022) 🆕
│   │   │   ├── spade.py         # SPADE (ECCV 2020) ⭐
│   │   │   ├── memseg.py        # MemSeg (2022)
│   │   │   ├── riad.py          # RIAD (2020)
│   │   │   ├── devnet.py        # DevNet (KDD 2019)
│   │   │   ├── dfm.py           # DFM (training-free) ⭐
│   │   │   ├── fastflow.py, padim.py, ...
│   │   │   └── ...
│   │   ├── registry.py  # Model registry system
│   │   └── baseml.py    # Base classes
│   ├── preprocessing/   # Image preprocessing module ⭐ NEW!
│   │   ├── enhancer.py  # 20+ enhancement operations
│   │   ├── mixin.py     # Easy integration mixin
│   │   └── __init__.py
│   ├── utils/           # Image processing utilities
│   ├── datasets/        # Data loading utilities
│   ├── evaluation.py    # Evaluation metrics ⭐
│   ├── benchmark.py     # Algorithm benchmarking ⭐
│   └── visualization/   # Visualization tools
├── tests/               # Comprehensive test suite
│   ├── test_pyod_models.py      # Classical ML tests
│   ├── test_dl_models.py        # Deep learning tests
│   ├── test_preprocessing.py    # Preprocessing tests ⭐ NEW!
│   ├── test_evaluation.py       # Evaluation tests
│   └── ...
├── examples/            # Usage examples
│   ├── preprocessing_example.py # Preprocessing guide ⭐ NEW!
│   ├── benchmark_example.py
│   └── ...
├── docs/                # Documentation
│   ├── DEEP_LEARNING_MODELS.md  # DL algorithms guide ⭐
│   ├── PREPROCESSING.md         # Preprocessing guide ⭐ NEW!
│   ├── EVALUATION_AND_BENCHMARK.md
│   ├── ALGORITHM_SELECTION_GUIDE.md
│   └── ...
└── .github/             # CI/CD workflows
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyimgano --cov-report=html

# Run specific tests
pytest tests/test_pyod_models.py -v

# Run with multiple Python versions
tox
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/jhlu2019/pyimgano.git
cd pyimgano

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Code Quality

We maintain high code quality standards:

```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type

# Run all checks
make all
```

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📊 Benchmarks

Based on [ADBench](https://github.com/Minqi824/ADBench) (30 algorithms on 57 datasets):

| Algorithm | Average Rank | Relative Speed |
|-----------|--------------|----------------|
| ECOD | 3.5/30 | 1.2× |
| COPOD | 4.2/30 | 1.0× (fastest) |
| IForest | 5.8/30 | 2.0× |
| LODA | 6.1/30 | 1.8× |
| Deep SVDD | 8.5/30 | 10× |

---

## 🔗 Related Projects

- **[PyOD](https://github.com/yzhao062/pyod)** - Python Outlier Detection Library
- **[Anomalib](https://github.com/openvinotoolkit/anomalib)** - Deep Learning Anomaly Detection
- **[ADBench](https://github.com/Minqi824/ADBench)** - Anomaly Detection Benchmark

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📮 Citation

If you use PyImgAno in your research, please cite:

```bibtex
@software{pyimgano2025,
  author = {PyImgAno Contributors},
  title = {PyImgAno: Enterprise-Grade Visual Anomaly Detection Toolkit},
  year = {2025},
  url = {https://github.com/jhlu2019/pyimgano}
}
```

---

## 🤝 Acknowledgments

- [PyOD](https://github.com/yzhao062/pyod) for the excellent outlier detection library
- [Anomalib](https://github.com/openvinotoolkit/anomalib) for deep learning inspiration
- All contributors and users of PyImgAno

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/jhlu2019/pyimgano/issues)
- **Discussions:** [GitHub Discussions](https://github.com/jhlu2019/pyimgano/discussions)
- **Documentation:** [Read the Docs](https://pyimgano.readthedocs.io/)

---

**Made with ❤️ by the PyImgAno Team**
