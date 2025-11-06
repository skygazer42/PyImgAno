# PyImgAno

**Enterprise-Grade Visual Anomaly Detection Toolkit**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready Python toolkit for visual anomaly detection, integrating **37+ state-of-the-art algorithms** from classical machine learning to cutting-edge deep learning (CVPR 2023, ICCV 2021, WACV 2022).

> **Translations:** [中文](README_cn.md) · [日本語](README_ja.md) · [한국어](README_ko.md)

---

## ✨ Key Features

- 🔥 **37+ Detection Algorithms** - From classical (ECOD, COPOD, KNN, PCA) to latest SOTA (SimpleNet, PatchCore, DRAEM, CFlow)
- 🚀 **Production Ready** - Enterprise-grade code quality, comprehensive testing, CI/CD pipelines
- 📦 **Unified API** - Consistent interface across all algorithms with factory pattern
- ⚡ **High Performance** - Top-tier algorithms (ECOD, COPOD) optimized for speed and accuracy
- 🎯 **Flexible** - Works with any feature extractor or end-to-end deep learning
- 🖼️ **Image Preprocessing** - 50+ operations (edge detection, morphology, filters, FFT, texture analysis, segmentation) with easy integration
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
| **SimpleNet** ⭐ | Deep Learning | 2023 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Ultra-fast SOTA, production |
| **PatchCore** ⭐ | Deep Learning | 2022 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Best accuracy, MVTec champion |
| **ECOD** | Classical | 2022 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Parameter-free, general purpose |
| **COPOD** | Classical | 2020 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Speed-critical applications |
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

### Example 4: Comparing Multiple Algorithms

```python
algorithms = ["vision_ecod", "vision_copod", "vision_knn"]
results = {}

for algo_name in algorithms:
    detector = models.create_model(
        algo_name,
        feature_extractor=feature_extractor,
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

Comprehensive preprocessing module with **50+ operations** for enhanced anomaly detection:

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
```

### Available Operations (50+)

#### Basic Operations (25)
| Category | Operations | Count |
|----------|------------|-------|
| **Edge Detection** | Canny, Sobel, Laplacian, Scharr, Prewitt, Sobel X/Y | 7 |
| **Morphology** | Erosion, Dilation, Opening, Closing, Gradient, TopHat, BlackHat | 7 |
| **Filters** | Gaussian, Bilateral, Median, Box | 4 |
| **Normalization** | MinMax, Z-Score, L2, Robust | 4 |
| **Enhancement** | Sharpen, Unsharp Mask, CLAHE | 3 |

#### Advanced Operations (25+)
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

### Deep Learning (18 algorithms)

| Algorithm | Model Name | Key Features |
|-----------|------------|--------------|
| **SimpleNet** ⭐ | `vision_simplenet` | Ultra-fast SOTA (CVPR 2023), 10x faster training |
| **PatchCore** ⭐ | `vision_patchcore` | Best accuracy (CVPR 2022), pixel localization |
| **DRAEM** ⭐ | `vision_draem` | Synthetic anomalies (ICCV 2021), robust |
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

## 📖 Documentation

- **[Deep Learning Models Guide](docs/DEEP_LEARNING_MODELS.md)** ⭐ - SOTA deep learning algorithms
- **[Preprocessing Guide](docs/PREPROCESSING.md)** ⭐ - NEW! Image enhancement and preprocessing
- **[Evaluation & Benchmarking Guide](docs/EVALUATION_AND_BENCHMARK.md)** ⭐ - Comprehensive evaluation tools
- **[Algorithm Selection Guide](docs/ALGORITHM_SELECTION_GUIDE.md)** - Choose the right algorithm
- **[API Reference](docs/)** - Detailed API documentation
- **[Examples](examples/)** - Code examples and tutorials
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- **[Changelog](CHANGELOG.md)** - Version history

---

## 🏗️ Project Structure

```
pyimgano/
├── pyimgano/
│   ├── models/          # 37+ anomaly detection algorithms
│   │   ├── Classical ML (19 algorithms)
│   │   │   ├── ecod.py          # ECOD (TKDE 2022)
│   │   │   ├── copod.py         # COPOD (ICDM 2020)
│   │   │   ├── feature_bagging.py
│   │   │   ├── knn.py, pca.py, lof.py, ...
│   │   │   └── ...
│   │   ├── Deep Learning (18 algorithms)
│   │   │   ├── simplenet.py     # SimpleNet (CVPR 2023) ⭐
│   │   │   ├── patchcore.py     # PatchCore (CVPR 2022) ⭐
│   │   │   ├── stfpm.py         # STFPM (BMVC 2021) ⭐
│   │   │   ├── draem.py         # DRAEM (ICCV 2021) ⭐
│   │   │   ├── cflow.py         # CFlow-AD (WACV 2022) ⭐
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
