"""
Image preprocessing and enhancement utilities.

This module provides comprehensive image preprocessing capabilities including:
- Edge detection (Canny, Sobel, Laplacian, etc.)
- Morphological operations (erosion, dilation, opening, closing)
- Filters (Gaussian, bilateral, median, etc.)
- Color space conversions
- Normalization and standardization
- Preprocessing pipelines
- Advanced operations (FFT, texture analysis, segmentation, etc.)
"""

from .enhancer import (
    ImageEnhancer,
    AdvancedImageEnhancer,
    PreprocessingPipeline,
    edge_detection,
    morphological_operation,
    apply_filter,
    normalize_image,
)
from .mixin import PreprocessingMixin
from .advanced_operations import (
    # Frequency domain
    apply_fft,
    apply_ifft,
    frequency_filter,
    # Texture analysis
    apply_gabor_filter,
    compute_lbp,
    compute_glcm_features,
    # Color space
    convert_color_space,
    equalize_color_histogram,
    # Enhancement
    gamma_correction,
    contrast_stretching,
    retinex_ssr,
    retinex_msr,
    # Denoising
    non_local_means_denoising,
    anisotropic_diffusion,
    # Feature extraction
    extract_hog_features,
    detect_corners,
    # Advanced morphology
    apply_advanced_morphology,
    # Segmentation
    apply_threshold,
    watershed_segmentation,
    # Pyramids
    gaussian_pyramid,
    laplacian_pyramid,
    # Enums
    ColorSpace,
    ThresholdMethod,
    CornerDetector,
    MorphologicalAdvanced,
)

__all__ = [
    # Main classes
    "ImageEnhancer",
    "AdvancedImageEnhancer",
    "PreprocessingPipeline",
    "PreprocessingMixin",
    # Basic operations
    "edge_detection",
    "morphological_operation",
    "apply_filter",
    "normalize_image",
    # Frequency domain
    "apply_fft",
    "apply_ifft",
    "frequency_filter",
    # Texture analysis
    "apply_gabor_filter",
    "compute_lbp",
    "compute_glcm_features",
    # Color space
    "convert_color_space",
    "equalize_color_histogram",
    # Enhancement
    "gamma_correction",
    "contrast_stretching",
    "retinex_ssr",
    "retinex_msr",
    # Denoising
    "non_local_means_denoising",
    "anisotropic_diffusion",
    # Feature extraction
    "extract_hog_features",
    "detect_corners",
    # Advanced morphology
    "apply_advanced_morphology",
    # Segmentation
    "apply_threshold",
    "watershed_segmentation",
    # Pyramids
    "gaussian_pyramid",
    "laplacian_pyramid",
    # Enums
    "ColorSpace",
    "ThresholdMethod",
    "CornerDetector",
    "MorphologicalAdvanced",
]
