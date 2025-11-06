"""
Image enhancement and preprocessing operations.

This module provides a comprehensive set of image preprocessing operations
using OpenCV and PyTorch for anomaly detection tasks.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EdgeDetectionMethod(Enum):
    """Edge detection methods."""
    CANNY = "canny"
    SOBEL = "sobel"
    SOBEL_X = "sobel_x"
    SOBEL_Y = "sobel_y"
    LAPLACIAN = "laplacian"
    SCHARR = "scharr"
    PREWITT = "prewitt"


class MorphOperation(Enum):
    """Morphological operations."""
    EROSION = "erosion"
    DILATION = "dilation"
    OPENING = "opening"
    CLOSING = "closing"
    GRADIENT = "gradient"
    TOPHAT = "tophat"
    BLACKHAT = "blackhat"


class FilterType(Enum):
    """Filter types."""
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    MEDIAN = "median"
    BOX = "box"
    MEAN = "mean"


def edge_detection(
    image: NDArray,
    method: Union[str, EdgeDetectionMethod] = "canny",
    **kwargs
) -> NDArray:
    """
    Apply edge detection to image.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or color)
    method : str or EdgeDetectionMethod
        Edge detection method to use
    **kwargs : dict
        Method-specific parameters

    Returns
    -------
    edges : ndarray
        Edge-detected image

    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread('image.jpg', 0)
    >>> edges = edge_detection(img, 'canny', threshold1=50, threshold2=150)
    >>> edges = edge_detection(img, 'sobel', ksize=3)
    """
    if isinstance(method, str):
        method = EdgeDetectionMethod(method.lower())

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == EdgeDetectionMethod.CANNY:
        threshold1 = kwargs.get('threshold1', 50)
        threshold2 = kwargs.get('threshold2', 150)
        aperture_size = kwargs.get('aperture_size', 3)
        edges = cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size)

    elif method == EdgeDetectionMethod.SOBEL:
        ksize = kwargs.get('ksize', 3)
        dx = kwargs.get('dx', 1)
        dy = kwargs.get('dy', 1)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=ksize)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=ksize)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.uint8(np.clip(edges, 0, 255))

    elif method == EdgeDetectionMethod.SOBEL_X:
        ksize = kwargs.get('ksize', 3)
        edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        edges = np.uint8(np.absolute(edges))

    elif method == EdgeDetectionMethod.SOBEL_Y:
        ksize = kwargs.get('ksize', 3)
        edges = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        edges = np.uint8(np.absolute(edges))

    elif method == EdgeDetectionMethod.LAPLACIAN:
        ksize = kwargs.get('ksize', 3)
        edges = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        edges = np.uint8(np.absolute(edges))

    elif method == EdgeDetectionMethod.SCHARR:
        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        edges = np.sqrt(scharr_x**2 + scharr_y**2)
        edges = np.uint8(np.clip(edges, 0, 255))

    elif method == EdgeDetectionMethod.PREWITT:
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
        edges = np.uint8(np.clip(edges, 0, 255))

    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    logger.debug("Applied edge detection: %s", method.value)
    return edges


def morphological_operation(
    image: NDArray,
    operation: Union[str, MorphOperation] = "erosion",
    kernel_size: Tuple[int, int] = (3, 3),
    kernel_shape: str = "rect",
    iterations: int = 1,
) -> NDArray:
    """
    Apply morphological operation to image.

    Parameters
    ----------
    image : ndarray
        Input image
    operation : str or MorphOperation
        Morphological operation to apply
    kernel_size : tuple of int, default=(3, 3)
        Size of the structuring element
    kernel_shape : str, default='rect'
        Shape of kernel ('rect', 'ellipse', 'cross')
    iterations : int, default=1
        Number of times to apply operation

    Returns
    -------
    result : ndarray
        Processed image

    Examples
    --------
    >>> img = cv2.imread('image.jpg', 0)
    >>> eroded = morphological_operation(img, 'erosion', kernel_size=(5, 5))
    >>> opened = morphological_operation(img, 'opening', kernel_size=(3, 3))
    """
    if isinstance(operation, str):
        operation = MorphOperation(operation.lower())

    # Create kernel
    if kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        raise ValueError(f"Unknown kernel shape: {kernel_shape}")

    # Apply operation
    if operation == MorphOperation.EROSION:
        result = cv2.erode(image, kernel, iterations=iterations)

    elif operation == MorphOperation.DILATION:
        result = cv2.dilate(image, kernel, iterations=iterations)

    elif operation == MorphOperation.OPENING:
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    elif operation == MorphOperation.CLOSING:
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    elif operation == MorphOperation.GRADIENT:
        result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    elif operation == MorphOperation.TOPHAT:
        result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    elif operation == MorphOperation.BLACKHAT:
        result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    else:
        raise ValueError(f"Unknown morphological operation: {operation}")

    logger.debug("Applied morphological operation: %s", operation.value)
    return result


def apply_filter(
    image: NDArray,
    filter_type: Union[str, FilterType] = "gaussian",
    **kwargs
) -> NDArray:
    """
    Apply filter to image.

    Parameters
    ----------
    image : ndarray
        Input image
    filter_type : str or FilterType
        Type of filter to apply
    **kwargs : dict
        Filter-specific parameters

    Returns
    -------
    filtered : ndarray
        Filtered image

    Examples
    --------
    >>> img = cv2.imread('image.jpg')
    >>> blurred = apply_filter(img, 'gaussian', ksize=(5, 5), sigma=1.5)
    >>> denoised = apply_filter(img, 'bilateral', d=9, sigmaColor=75, sigmaSpace=75)
    """
    if isinstance(filter_type, str):
        filter_type = FilterType(filter_type.lower())

    if filter_type == FilterType.GAUSSIAN:
        ksize = kwargs.get('ksize', (5, 5))
        sigma = kwargs.get('sigma', 0)
        result = cv2.GaussianBlur(image, ksize, sigma)

    elif filter_type == FilterType.BILATERAL:
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigmaColor', 75)
        sigma_space = kwargs.get('sigmaSpace', 75)
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    elif filter_type == FilterType.MEDIAN:
        ksize = kwargs.get('ksize', 5)
        result = cv2.medianBlur(image, ksize)

    elif filter_type in [FilterType.BOX, FilterType.MEAN]:
        ksize = kwargs.get('ksize', (5, 5))
        result = cv2.blur(image, ksize)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    logger.debug("Applied filter: %s", filter_type.value)
    return result


def normalize_image(
    image: NDArray,
    method: str = "minmax",
    **kwargs
) -> NDArray:
    """
    Normalize image.

    Parameters
    ----------
    image : ndarray
        Input image
    method : str, default='minmax'
        Normalization method ('minmax', 'zscore', 'l2', 'robust')
    **kwargs : dict
        Method-specific parameters

    Returns
    -------
    normalized : ndarray
        Normalized image

    Examples
    --------
    >>> img = cv2.imread('image.jpg')
    >>> norm = normalize_image(img, 'minmax', min_val=0, max_val=1)
    >>> norm = normalize_image(img, 'zscore')
    """
    image = image.astype(np.float32)

    if method == "minmax":
        min_val = kwargs.get('min_val', 0)
        max_val = kwargs.get('max_val', 1)
        img_min = image.min()
        img_max = image.max()

        if img_max - img_min > 0:
            normalized = (image - img_min) / (img_max - img_min)
            normalized = normalized * (max_val - min_val) + min_val
        else:
            normalized = np.full_like(image, min_val)

    elif method == "zscore":
        mean = image.mean()
        std = image.std()
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean

    elif method == "l2":
        norm = np.linalg.norm(image)
        if norm > 0:
            normalized = image / norm
        else:
            normalized = image

    elif method == "robust":
        # Use median and IQR for robust normalization
        median = np.median(image)
        q75, q25 = np.percentile(image, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            normalized = (image - median) / iqr
        else:
            normalized = image - median

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    logger.debug("Applied normalization: %s", method)
    return normalized


class ImageEnhancer:
    """
    Comprehensive image enhancement and preprocessing.

    This class provides a high-level interface for image enhancement operations.

    Examples
    --------
    >>> enhancer = ImageEnhancer()
    >>>
    >>> # Edge detection
    >>> edges = enhancer.detect_edges(img, method='canny')
    >>>
    >>> # Morphological operations
    >>> eroded = enhancer.erode(img, kernel_size=(5, 5))
    >>> dilated = enhancer.dilate(img, kernel_size=(5, 5))
    >>>
    >>> # Filters
    >>> blurred = enhancer.gaussian_blur(img, ksize=(5, 5))
    >>> denoised = enhancer.bilateral_filter(img)
    >>>
    >>> # Normalization
    >>> normalized = enhancer.normalize(img, method='minmax')
    """

    def __init__(self):
        """Initialize ImageEnhancer."""
        logger.info("Initialized ImageEnhancer")

    # Edge detection methods
    def detect_edges(
        self,
        image: NDArray,
        method: str = "canny",
        **kwargs
    ) -> NDArray:
        """Detect edges in image."""
        return edge_detection(image, method, **kwargs)

    def canny_edges(
        self,
        image: NDArray,
        threshold1: int = 50,
        threshold2: int = 150,
    ) -> NDArray:
        """Apply Canny edge detection."""
        return edge_detection(
            image, "canny",
            threshold1=threshold1,
            threshold2=threshold2
        )

    def sobel_edges(self, image: NDArray, ksize: int = 3) -> NDArray:
        """Apply Sobel edge detection."""
        return edge_detection(image, "sobel", ksize=ksize)

    def laplacian_edges(self, image: NDArray, ksize: int = 3) -> NDArray:
        """Apply Laplacian edge detection."""
        return edge_detection(image, "laplacian", ksize=ksize)

    # Morphological operations
    def erode(
        self,
        image: NDArray,
        kernel_size: Tuple[int, int] = (3, 3),
        iterations: int = 1,
    ) -> NDArray:
        """Apply erosion."""
        return morphological_operation(
            image, "erosion",
            kernel_size=kernel_size,
            iterations=iterations
        )

    def dilate(
        self,
        image: NDArray,
        kernel_size: Tuple[int, int] = (3, 3),
        iterations: int = 1,
    ) -> NDArray:
        """Apply dilation."""
        return morphological_operation(
            image, "dilation",
            kernel_size=kernel_size,
            iterations=iterations
        )

    def opening(
        self,
        image: NDArray,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> NDArray:
        """Apply opening (erosion followed by dilation)."""
        return morphological_operation(image, "opening", kernel_size=kernel_size)

    def closing(
        self,
        image: NDArray,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> NDArray:
        """Apply closing (dilation followed by erosion)."""
        return morphological_operation(image, "closing", kernel_size=kernel_size)

    def morphological_gradient(
        self,
        image: NDArray,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> NDArray:
        """Apply morphological gradient."""
        return morphological_operation(image, "gradient", kernel_size=kernel_size)

    # Filters
    def gaussian_blur(
        self,
        image: NDArray,
        ksize: Tuple[int, int] = (5, 5),
        sigma: float = 0,
    ) -> NDArray:
        """Apply Gaussian blur."""
        return apply_filter(image, "gaussian", ksize=ksize, sigma=sigma)

    def bilateral_filter(
        self,
        image: NDArray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> NDArray:
        """Apply bilateral filter (edge-preserving)."""
        return apply_filter(
            image, "bilateral",
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

    def median_blur(self, image: NDArray, ksize: int = 5) -> NDArray:
        """Apply median blur."""
        return apply_filter(image, "median", ksize=ksize)

    # Normalization
    def normalize(self, image: NDArray, method: str = "minmax", **kwargs) -> NDArray:
        """Normalize image."""
        return normalize_image(image, method, **kwargs)

    # Advanced operations
    def sharpen(self, image: NDArray, amount: float = 1.0) -> NDArray:
        """
        Sharpen image.

        Parameters
        ----------
        image : ndarray
            Input image
        amount : float, default=1.0
            Sharpening amount (0-2)

        Returns
        -------
        sharpened : ndarray
            Sharpened image
        """
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * amount / 9

        if len(image.shape) == 2:
            sharpened = cv2.filter2D(image, -1, kernel)
        else:
            sharpened = cv2.filter2D(image, -1, kernel)

        return np.clip(sharpened, 0, 255).astype(image.dtype)

    def unsharp_mask(
        self,
        image: NDArray,
        sigma: float = 1.0,
        amount: float = 1.0,
    ) -> NDArray:
        """
        Apply unsharp mask for sharpening.

        Parameters
        ----------
        image : ndarray
            Input image
        sigma : float, default=1.0
            Gaussian blur sigma
        amount : float, default=1.0
            Sharpening amount

        Returns
        -------
        sharpened : ndarray
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened

    def adaptive_histogram_equalization(
        self,
        image: NDArray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> NDArray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Parameters
        ----------
        image : ndarray
            Input image (grayscale or will be converted)
        clip_limit : float, default=2.0
            Threshold for contrast limiting
        tile_grid_size : tuple, default=(8, 8)
            Size of grid for histogram equalization

        Returns
        -------
        equalized : ndarray
            Equalized image
        """
        if len(image.shape) == 3:
            # Convert to LAB and equalize L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            equalized = clahe.apply(image)

        return equalized


class PreprocessingPipeline:
    """
    Build and execute preprocessing pipelines.

    Examples
    --------
    >>> pipeline = PreprocessingPipeline()
    >>> pipeline.add_step('gaussian_blur', ksize=(5, 5))
    >>> pipeline.add_step('edge_detection', method='canny')
    >>> pipeline.add_step('normalize', method='minmax')
    >>>
    >>> result = pipeline.transform(image)
    """

    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.steps: List[Tuple[str, Callable, Dict]] = []
        self.enhancer = ImageEnhancer()
        logger.info("Initialized PreprocessingPipeline")

    def add_step(
        self,
        operation: str,
        **kwargs
    ) -> "PreprocessingPipeline":
        """
        Add a preprocessing step.

        Parameters
        ----------
        operation : str
            Operation name (method of ImageEnhancer)
        **kwargs : dict
            Operation parameters

        Returns
        -------
        self : PreprocessingPipeline
            For method chaining
        """
        if not hasattr(self.enhancer, operation):
            raise ValueError(f"Unknown operation: {operation}")

        func = getattr(self.enhancer, operation)
        self.steps.append((operation, func, kwargs))
        logger.debug("Added step: %s with params %s", operation, kwargs)
        return self

    def transform(self, image: NDArray) -> NDArray:
        """
        Apply pipeline to image.

        Parameters
        ----------
        image : ndarray
            Input image

        Returns
        -------
        result : ndarray
            Processed image
        """
        result = image.copy()

        for operation, func, kwargs in self.steps:
            logger.debug("Applying: %s", operation)
            result = func(result, **kwargs)

        return result

    def transform_batch(self, images: List[NDArray]) -> List[NDArray]:
        """
        Apply pipeline to multiple images.

        Parameters
        ----------
        images : list of ndarray
            Input images

        Returns
        -------
        results : list of ndarray
            Processed images
        """
        return [self.transform(img) for img in images]

    def clear(self) -> None:
        """Clear all steps from pipeline."""
        self.steps.clear()
        logger.debug("Cleared pipeline")

    def __len__(self) -> int:
        """Get number of steps in pipeline."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation."""
        step_names = [name for name, _, _ in self.steps]
        return f"PreprocessingPipeline(steps={step_names})"


# Import advanced operations
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
)


# Extend ImageEnhancer with advanced operations
class AdvancedImageEnhancer(ImageEnhancer):
    """
    Extended ImageEnhancer with advanced image processing operations.

    Includes all basic operations plus:
    - Frequency domain operations (FFT, filters)
    - Texture analysis (Gabor, LBP, GLCM)
    - Color space transformations
    - Advanced enhancement (Gamma, Retinex)
    - Feature extraction (HOG, corners)
    - Advanced morphology (skeleton, distance transform)
    - Image segmentation
    - Image pyramids
    """

    # Frequency Domain Operations

    def apply_fft(self, image: NDArray) -> Tuple[NDArray, NDArray]:
        """Apply Fast Fourier Transform."""
        return apply_fft(image)

    def apply_ifft(self, magnitude: NDArray, phase: NDArray) -> NDArray:
        """Apply Inverse Fast Fourier Transform."""
        return apply_ifft(magnitude, phase)

    def frequency_filter(
        self,
        image: NDArray,
        filter_type: str = "lowpass",
        cutoff_frequency: float = 30.0
    ) -> NDArray:
        """Apply frequency domain filter."""
        return frequency_filter(image, filter_type, cutoff_frequency)

    # Texture Analysis

    def gabor_filter(
        self,
        image: NDArray,
        frequency: float = 0.1,
        theta: float = 0,
        sigma_x: float = 3.0,
        sigma_y: float = 3.0
    ) -> NDArray:
        """Apply Gabor filter for texture analysis."""
        return apply_gabor_filter(image, frequency, theta, sigma_x, sigma_y)

    def compute_lbp(
        self,
        image: NDArray,
        n_points: int = 8,
        radius: float = 1.0,
        method: str = "uniform"
    ) -> NDArray:
        """Compute Local Binary Pattern features."""
        return compute_lbp(image, n_points, radius, method)

    def compute_glcm(
        self,
        image: NDArray,
        distances: list = [1],
        angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    ) -> dict:
        """Compute GLCM texture features."""
        return compute_glcm_features(image, distances, angles)

    # Color Space Operations

    def convert_color(
        self,
        image: NDArray,
        from_space: str,
        to_space: str
    ) -> NDArray:
        """Convert between color spaces."""
        return convert_color_space(image, from_space, to_space)

    def equalize_color_hist(
        self,
        image: NDArray,
        method: str = "hsv"
    ) -> NDArray:
        """Equalize color histogram."""
        return equalize_color_histogram(image, method)

    # Advanced Enhancement

    def gamma_correct(self, image: NDArray, gamma: float = 1.0) -> NDArray:
        """Apply gamma correction."""
        return gamma_correction(image, gamma)

    def contrast_stretch(
        self,
        image: NDArray,
        lower_percentile: float = 2,
        upper_percentile: float = 98
    ) -> NDArray:
        """Apply contrast stretching."""
        return contrast_stretching(image, lower_percentile, upper_percentile)

    def retinex_single(self, image: NDArray, sigma: float = 15.0) -> NDArray:
        """Apply Single-Scale Retinex."""
        return retinex_ssr(image, sigma)

    def retinex_multi(
        self,
        image: NDArray,
        sigmas: list = [15, 80, 250]
    ) -> NDArray:
        """Apply Multi-Scale Retinex."""
        return retinex_msr(image, sigmas)

    # Denoising

    def nlm_denoise(
        self,
        image: NDArray,
        h: float = 10,
        template_window_size: int = 7,
        search_window_size: int = 21
    ) -> NDArray:
        """Apply non-local means denoising."""
        return non_local_means_denoising(
            image, h, template_window_size, search_window_size
        )

    def anisotropic_diffusion(
        self,
        image: NDArray,
        niter: int = 10,
        kappa: float = 50,
        gamma: float = 0.1,
        option: int = 1
    ) -> NDArray:
        """Apply anisotropic diffusion."""
        return anisotropic_diffusion(image, niter, kappa, gamma, option)

    # Feature Extraction

    def extract_hog(
        self,
        image: NDArray,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        visualize: bool = False
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Extract HOG features."""
        return extract_hog_features(
            image, orientations, pixels_per_cell, cells_per_block, visualize
        )

    def detect_corners(
        self,
        image: NDArray,
        method: str = "harris",
        **kwargs
    ) -> NDArray:
        """Detect corners in image."""
        return detect_corners(image, method, **kwargs)

    # Advanced Morphology

    def skeleton(self, image: NDArray) -> NDArray:
        """Extract skeleton of binary image."""
        return apply_advanced_morphology(image, "skeleton")

    def thin(self, image: NDArray) -> NDArray:
        """Apply thinning to binary image."""
        return apply_advanced_morphology(image, "thin")

    def convex_hull(self, image: NDArray) -> NDArray:
        """Compute convex hull of binary image."""
        return apply_advanced_morphology(image, "convex_hull")

    def distance_transform(self, image: NDArray) -> NDArray:
        """Apply distance transform."""
        return apply_advanced_morphology(image, "distance_transform")

    # Segmentation

    def threshold(
        self,
        image: NDArray,
        method: str = "otsu",
        **kwargs
    ) -> NDArray:
        """Apply thresholding for segmentation."""
        return apply_threshold(image, method, **kwargs)

    def watershed(
        self,
        image: NDArray,
        markers: Optional[NDArray] = None
    ) -> NDArray:
        """Apply watershed segmentation."""
        return watershed_segmentation(image, markers)

    # Image Pyramids

    def build_gaussian_pyramid(
        self,
        image: NDArray,
        levels: int = 3
    ) -> list:
        """Create Gaussian pyramid."""
        return gaussian_pyramid(image, levels)

    def build_laplacian_pyramid(
        self,
        image: NDArray,
        levels: int = 3
    ) -> list:
        """Create Laplacian pyramid."""
        return laplacian_pyramid(image, levels)
