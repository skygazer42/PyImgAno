# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Image Preprocessing Module** (`pyimgano.preprocessing`) ⭐ NEW!
  - **ImageEnhancer** - High-level interface for basic operations (25 operations)
  - **AdvancedImageEnhancer** - Extended with 25+ advanced operations (50+ total)
  - **PreprocessingPipeline** - Sequential operation composition with method chaining
  - **PreprocessingMixin** - Easy integration with detectors via mixin pattern

  **Basic Operations (25)**:
  - **Edge Detection** (7 methods): Canny, Sobel, Sobel X/Y, Laplacian, Scharr, Prewitt
  - **Morphological Operations** (7 types): Erosion, Dilation, Opening, Closing, Gradient, TopHat, BlackHat
  - **Filters** (4 types): Gaussian, Bilateral, Median, Box
  - **Normalization** (4 methods): MinMax, Z-Score, L2, Robust (IQR-based)
  - **Enhancement** (3): Sharpen, Unsharp Mask, CLAHE

  **Advanced Operations (25+)**:
  - **Frequency Domain** (6): FFT, IFFT, Lowpass, Highpass, Bandpass, Bandstop filters
  - **Texture Analysis** (3): Gabor filters, Local Binary Pattern (LBP), GLCM texture features
  - **Color Space Transformations** (8): RGB, HSV, LAB, YCrCb, HLS, LUV + conversions
  - **Advanced Enhancement** (4): Gamma correction, Contrast stretching, Single/Multi-Scale Retinex
  - **Advanced Denoising** (2): Non-local means denoising, Anisotropic diffusion
  - **Feature Extraction** (4): HOG features, Harris corners, Shi-Tomasi, FAST detector
  - **Advanced Morphology** (4): Skeletonization, Thinning, Convex hull, Distance transform
  - **Segmentation** (6): Otsu, Adaptive (mean/Gaussian), Triangle, Yen, Watershed
  - **Image Pyramids** (2): Gaussian pyramid, Laplacian pyramid

  - Comprehensive preprocessing documentation (docs/PREPROCESSING.md)
  - Complete test suite (tests/test_preprocessing.py - 500+ lines)
  - Usage examples (examples/preprocessing_example.py, advanced_preprocessing_example.py)
- **3 New High-Performance Deep Learning Algorithms** ⚡:
  - **DRAEM** (ICCV 2021) - Discriminatively trained reconstruction with synthetic anomalies
  - **CFlow-AD** (WACV 2022) - Real-time conditional normalizing flows
  - **DFM** - Fast discriminative feature modeling (training-free)
- Total algorithms now: **37+** (19 classical ML + 18 deep learning)
- Enterprise-grade package configuration
- Modern `pyproject.toml` build system configuration
- Comprehensive GitHub Actions CI/CD workflows
- Pre-commit hooks for code quality
- Code quality tools integration (Black, isort, flake8, mypy, ruff)
- Contributing guidelines (CONTRIBUTING.md)
- This changelog file
- Support for Python 3.9-3.12
- Type hints marker (py.typed)
- MANIFEST.in for proper package distribution
- .editorconfig for consistent coding style
- tox configuration for multi-version testing
- **34+ total algorithms** (19 classical ML + 15 deep learning)
- New PyOD classical ML integrations (19 total):
  - **ECOD** (Empirical CDF-based, TKDE 2022) - State-of-the-art, parameter-free
  - **COPOD** (Copula-based, ICDM 2020) - High-performance, parameter-free
  - **KNN** (K-Nearest Neighbors) - Classic, simple and effective
  - **PCA** (Principal Component Analysis) - Classic dimensionality reduction
  - **COF** (Connectivity-Based Outlier Factor, PAKDD 2002) - Density-based detection
  - **MCD** (Minimum Covariance Determinant, 1999) - Robust statistical method
  - **Feature Bagging** (KDD 2005) - Ensemble method for stability
  - **INNE** (Isolation Nearest Neighbors, ICDM 2014) - Fast isolation-based
- New state-of-the-art deep learning algorithms (15 total):
  - **SimpleNet** (CVPR 2023) ⭐ - Ultra-fast SOTA, 10x faster training, ~99% AUROC
  - **PatchCore** (CVPR 2022) ⭐ - Best accuracy (99.6% AUROC), memory-based, pixel localization
  - **STFPM** (BMVC 2021) ⭐ - Student-Teacher matching, multi-scale features, ~97% AUROC
- Enhanced error handling and logging for all detectors
- Improved PyOD version compatibility (>=1.1.0, <3.0.0)
- Comprehensive algorithm selection guide
- **Deep Learning Models Guide** - Detailed documentation for SOTA DL algorithms
- Complete test suite with comprehensive coverage:
  - Classical ML tests (test_pyod_models.py)
  - Deep learning tests (test_dl_models.py) - 400+ lines covering all DL models
- Enhanced documentation structure with dedicated DL guide
- **Comprehensive Evaluation Module** (`pyimgano.evaluation`) ⭐
  - `compute_auroc()` - ROC-AUC computation
  - `compute_average_precision()` - Average Precision for imbalanced datasets
  - `compute_classification_metrics()` - Precision, Recall, F1, Specificity, Accuracy
  - `find_optimal_threshold()` - Automatic threshold optimization (F1, Youden, etc.)
  - `evaluate_detector()` - One-stop comprehensive evaluation
  - `compute_pro_score()` - Pixel-level PRO score for localization
  - `print_evaluation_summary()` - Formatted result display
- **Benchmarking Tools** (`pyimgano.benchmark`) ⭐
  - `AlgorithmBenchmark` class - Systematic multi-algorithm comparison
  - `quick_benchmark()` - Fast benchmarking with defaults
  - Automatic timing measurements (training and inference)
  - Performance ranking by any metric
  - JSON export for results
- **Visualization Module** (`pyimgano.visualization`) ⭐
  - `plot_anomaly_map()` - Heatmap overlay on images
  - `plot_roc_curve()` - ROC curve with AUC
  - `plot_precision_recall_curve()` - PR curve with AP
  - `plot_score_distribution()` - Score histograms for normal/anomaly
  - `compare_detectors()` - Side-by-side detector comparison
  - `save_anomaly_overlay()` - Fast OpenCV-based overlay (no matplotlib)
- **Example Scripts** (examples/) ⭐
  - `quick_start.py` - Basic usage example
  - `benchmark_example.py` - Multi-algorithm benchmarking
  - `visualization_example.py` - Visualization demonstrations
- **Expanded Test Suite** ⭐
  - `test_evaluation.py` - 400+ lines testing all metrics
  - `test_integration.py` - End-to-end workflow tests
  - Synthetic dataset fixtures for testing
  - Edge case and error handling tests
- **Documentation**:
  - `EVALUATION_AND_BENCHMARK.md` - Complete evaluation guide

### Changed
- Enhanced package metadata and classifiers
- Improved dependency version specifications
- Updated development workflow documentation
- Updated `__init__.py` to export new modules and functions
- Version bump to 0.2.0

### Improved
- Test configuration with coverage reporting
- Documentation structure
- Code quality and consistency

## [0.1.0] - 2025-01-XX

### Added
- Initial release of PyImgAno
- Visual anomaly detection utilities
- Support for 15+ classical ML anomaly detectors:
  - ABOD (Angle-Based Outlier Detection)
  - CBLOF (Cluster-Based Local Outlier Factor)
  - DBSCAN
  - Isolation Forest
  - HBOS (Histogram-Based Outlier Score)
  - K-Means
  - Kernel PCA
  - LOCI (Local Correlation Integral)
  - LODA (Lightweight On-line Detector)
  - LOF (Local Outlier Factor)
  - LSCP (Locally Selective Combination)
  - MO_GAAL
  - One-Class SVM
  - SUOD (Scalable Unsupervised Outlier Detection)
  - XGBOD (XGBoost Outlier Detection)

- Support for 15 deep learning anomaly detectors:
  - **SimpleNet** (CVPR 2023) - Ultra-fast SOTA ⭐
  - **PatchCore** (CVPR 2022) - Best accuracy ⭐
  - **STFPM** (BMVC 2021) - Student-Teacher ⭐
  - AutoEncoder (AE)
  - AE + SVM
  - ALAD (Adversarial Learning)
  - Anomalib integration
  - Deep SVDD
  - EfficientAD
  - FastFlow
  - IMDD
  - PaDiM
  - Reverse Distillation
  - SSIM-based methods
  - VAE (Variational AutoEncoder)

- Model registry system with factory pattern
- Flexible data loading utilities
- Image preprocessing and transformation pipeline
- Augmentation registry system
- Defect detection operations
- Support for diffusion models (optional)
- Multi-language documentation (English, Chinese, Japanese, Korean)
- Example scripts and notebooks
- MIT License

### Features
- Unified API for all anomaly detection models
- PyTorch Lightning data modules
- Extensible architecture
- Easy model registration system
- Comprehensive image operations
- OpenCV-based augmentation
- Visualization utilities

---

## Release Notes Guidelines

### Types of Changes
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes
- **Improved** for enhancements to existing features

### Version Format
- **Major.Minor.Patch** (e.g., 1.0.0)
- **Major**: Incompatible API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

[Unreleased]: https://github.com/jhlu2019/pyimgano/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jhlu2019/pyimgano/releases/tag/v0.1.0
