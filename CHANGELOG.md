# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- Enhanced package metadata and classifiers
- Improved dependency version specifications
- Updated development workflow documentation

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

- Support for 12+ deep learning anomaly detectors:
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
