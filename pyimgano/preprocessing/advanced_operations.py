"""
Advanced image processing operations.

This module provides advanced image processing techniques including:
- Frequency domain operations (FFT, filters)
- Texture analysis (Gabor, LBP, GLCM)
- Color space transformations
- Advanced enhancement (Gamma, Retinex)
- Feature extraction (HOG, corners)
- Morphological extensions (skeletonization, distance transform)
- Image segmentation
"""

from enum import Enum
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import feature, filters, morphology
from skimage.feature import greycomatrix, greycoprops, hog, local_binary_pattern
from skimage.filters import gabor


# Enums for type-safe operation selection

class ColorSpace(Enum):
    """Color space options."""
    RGB = "rgb"
    BGR = "bgr"
    HSV = "hsv"
    LAB = "lab"
    YCrCb = "ycrcb"
    GRAY = "gray"
    HLS = "hls"
    LUV = "luv"


class ThresholdMethod(Enum):
    """Thresholding methods."""
    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    TRIANGLE = "triangle"
    YEN = "yen"
    ISODATA = "isodata"


class CornerDetector(Enum):
    """Corner detection methods."""
    HARRIS = "harris"
    SHI_TOMASI = "shi_tomasi"
    FAST = "fast"
    GOOD_FEATURES = "good_features"


class MorphologicalAdvanced(Enum):
    """Advanced morphological operations."""
    SKELETON = "skeleton"
    THIN = "thin"
    CONVEX_HULL = "convex_hull"
    DISTANCE_TRANSFORM = "distance_transform"


# Frequency Domain Operations

def apply_fft(image: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Apply Fast Fourier Transform to image.

    Args:
        image: Input image (grayscale)

    Returns:
        Tuple of (magnitude_spectrum, phase_spectrum)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Get magnitude and phase
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)

    return magnitude, phase


def apply_ifft(magnitude: NDArray, phase: NDArray) -> NDArray:
    """
    Apply Inverse Fast Fourier Transform.

    Args:
        magnitude: Magnitude spectrum
        phase: Phase spectrum

    Returns:
        Reconstructed image
    """
    # Reconstruct complex spectrum
    fshift = magnitude * np.exp(1j * phase)
    f = np.fft.ifftshift(fshift)

    # Apply inverse FFT
    img_back = np.fft.ifft2(f)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)


def frequency_filter(
    image: NDArray,
    filter_type: str = "lowpass",
    cutoff_frequency: float = 30.0
) -> NDArray:
    """
    Apply frequency domain filter.

    Args:
        image: Input image
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        cutoff_frequency: Cutoff frequency for filter

    Returns:
        Filtered image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Apply FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create mask
    mask = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)

            if filter_type == "lowpass":
                if distance <= cutoff_frequency:
                    mask[i, j] = 1
            elif filter_type == "highpass":
                if distance > cutoff_frequency:
                    mask[i, j] = 1
            elif filter_type == "bandpass":
                if cutoff_frequency <= distance <= cutoff_frequency * 2:
                    mask[i, j] = 1
            elif filter_type == "bandstop":
                if distance < cutoff_frequency or distance > cutoff_frequency * 2:
                    mask[i, j] = 1

    # Apply mask
    fshift = fshift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)


# Texture Analysis

def apply_gabor_filter(
    image: NDArray,
    frequency: float = 0.1,
    theta: float = 0,
    sigma_x: float = 3.0,
    sigma_y: float = 3.0
) -> NDArray:
    """
    Apply Gabor filter for texture analysis.

    Args:
        image: Input image
        frequency: Frequency of the sinusoidal wave
        theta: Orientation in radians
        sigma_x: Standard deviation in x direction
        sigma_y: Standard deviation in y direction

    Returns:
        Filtered image (real part)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Apply Gabor filter
    real, imag = gabor(image, frequency=frequency, theta=theta,
                       sigma_x=sigma_x, sigma_y=sigma_y)

    # Normalize and convert to uint8
    real = np.abs(real)
    real = (real - real.min()) / (real.max() - real.min()) * 255

    return real.astype(np.uint8)


def compute_lbp(
    image: NDArray,
    n_points: int = 8,
    radius: float = 1.0,
    method: str = "uniform"
) -> NDArray:
    """
    Compute Local Binary Pattern features.

    Args:
        image: Input image
        n_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        method: 'default', 'ror', 'uniform', 'var'

    Returns:
        LBP feature image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method=method)

    # Normalize to 0-255
    lbp = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)

    return lbp


def compute_glcm_features(
    image: NDArray,
    distances: list = [1],
    angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
) -> dict:
    """
    Compute Gray-Level Co-occurrence Matrix features.

    Args:
        image: Input image
        distances: List of pixel pair distances
        angles: List of pixel pair angles

    Returns:
        Dictionary of GLCM features (contrast, dissimilarity, homogeneity, energy, correlation)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute GLCM
    glcm = greycomatrix(image, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    # Compute properties
    features = {
        'contrast': greycoprops(glcm, 'contrast').mean(),
        'dissimilarity': greycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': greycoprops(glcm, 'homogeneity').mean(),
        'energy': greycoprops(glcm, 'energy').mean(),
        'correlation': greycoprops(glcm, 'correlation').mean(),
    }

    return features


# Color Space Operations

def convert_color_space(
    image: NDArray,
    from_space: Union[str, ColorSpace],
    to_space: Union[str, ColorSpace]
) -> NDArray:
    """
    Convert image between color spaces.

    Args:
        image: Input image
        from_space: Source color space
        to_space: Target color space

    Returns:
        Converted image
    """
    if isinstance(from_space, str):
        from_space = ColorSpace(from_space.lower())
    if isinstance(to_space, str):
        to_space = ColorSpace(to_space.lower())

    # Mapping of color space conversions
    conversions = {
        (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
        (ColorSpace.BGR, ColorSpace.HSV): cv2.COLOR_BGR2HSV,
        (ColorSpace.BGR, ColorSpace.LAB): cv2.COLOR_BGR2LAB,
        (ColorSpace.BGR, ColorSpace.YCrCb): cv2.COLOR_BGR2YCrCb,
        (ColorSpace.BGR, ColorSpace.GRAY): cv2.COLOR_BGR2GRAY,
        (ColorSpace.BGR, ColorSpace.HLS): cv2.COLOR_BGR2HLS,
        (ColorSpace.BGR, ColorSpace.LUV): cv2.COLOR_BGR2LUV,
        (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
        (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
        (ColorSpace.RGB, ColorSpace.LAB): cv2.COLOR_RGB2LAB,
        (ColorSpace.RGB, ColorSpace.GRAY): cv2.COLOR_RGB2GRAY,
        (ColorSpace.HSV, ColorSpace.BGR): cv2.COLOR_HSV2BGR,
        (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
        (ColorSpace.LAB, ColorSpace.BGR): cv2.COLOR_LAB2BGR,
        (ColorSpace.LAB, ColorSpace.RGB): cv2.COLOR_LAB2RGB,
        (ColorSpace.GRAY, ColorSpace.BGR): cv2.COLOR_GRAY2BGR,
        (ColorSpace.GRAY, ColorSpace.RGB): cv2.COLOR_GRAY2RGB,
    }

    key = (from_space, to_space)
    if key in conversions:
        return cv2.cvtColor(image, conversions[key])
    else:
        raise ValueError(f"Conversion from {from_space} to {to_space} not supported")


def equalize_color_histogram(image: NDArray, method: str = "hsv") -> NDArray:
    """
    Equalize histogram in color space.

    Args:
        image: Input color image
        method: 'hsv' (equalize V), 'lab' (equalize L), 'ycrcb' (equalize Y)

    Returns:
        Histogram equalized image
    """
    if method == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif method == "lab":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    elif method == "ycrcb":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    else:
        raise ValueError(f"Unknown method: {method}")


# Advanced Enhancement

def gamma_correction(image: NDArray, gamma: float = 1.0) -> NDArray:
    """
    Apply gamma correction.

    Args:
        image: Input image
        gamma: Gamma value (< 1: brighter, > 1: darker)

    Returns:
        Gamma corrected image
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction
    return cv2.LUT(image, table)


def contrast_stretching(
    image: NDArray,
    lower_percentile: float = 2,
    upper_percentile: float = 98
) -> NDArray:
    """
    Apply contrast stretching.

    Args:
        image: Input image
        lower_percentile: Lower percentile for stretching
        upper_percentile: Upper percentile for stretching

    Returns:
        Contrast stretched image
    """
    if len(image.shape) == 3:
        # Process each channel separately
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i]
            p_low = np.percentile(channel, lower_percentile)
            p_high = np.percentile(channel, upper_percentile)
            result[:, :, i] = np.clip((channel - p_low) * 255.0 / (p_high - p_low), 0, 255)
        return result.astype(np.uint8)
    else:
        p_low = np.percentile(image, lower_percentile)
        p_high = np.percentile(image, upper_percentile)
        return np.clip((image - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)


def retinex_ssr(image: NDArray, sigma: float = 15.0) -> NDArray:
    """
    Single-Scale Retinex for illumination invariant processing.

    Args:
        image: Input image
        sigma: Gaussian kernel sigma

    Returns:
        Retinex enhanced image
    """
    if len(image.shape) == 3:
        # Process each channel separately
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            channel = image[:, :, i].astype(np.float32) + 1.0  # Add 1 to avoid log(0)
            gaussian = cv2.GaussianBlur(channel, (0, 0), sigma)
            retinex = np.log10(channel) - np.log10(gaussian)
            result[:, :, i] = retinex

        # Normalize
        result = (result - result.min()) / (result.max() - result.min()) * 255
        return result.astype(np.uint8)
    else:
        image_float = image.astype(np.float32) + 1.0
        gaussian = cv2.GaussianBlur(image_float, (0, 0), sigma)
        retinex = np.log10(image_float) - np.log10(gaussian)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
        return retinex.astype(np.uint8)


def retinex_msr(
    image: NDArray,
    sigmas: list = [15, 80, 250]
) -> NDArray:
    """
    Multi-Scale Retinex for illumination invariant processing.

    Args:
        image: Input image
        sigmas: List of Gaussian kernel sigmas

    Returns:
        Retinex enhanced image
    """
    result = np.zeros_like(image, dtype=np.float32)

    for sigma in sigmas:
        ssr = retinex_ssr(image, sigma).astype(np.float32)
        result += ssr

    result = result / len(sigmas)
    result = (result - result.min()) / (result.max() - result.min()) * 255

    return result.astype(np.uint8)


# Denoising

def non_local_means_denoising(
    image: NDArray,
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> NDArray:
    """
    Apply non-local means denoising.

    Args:
        image: Input image
        h: Filter strength
        template_window_size: Size of template patch
        search_window_size: Size of search area

    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h,
            h,
            template_window_size,
            search_window_size
        )
    else:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h,
            template_window_size,
            search_window_size
        )


def anisotropic_diffusion(
    image: NDArray,
    niter: int = 10,
    kappa: float = 50,
    gamma: float = 0.1,
    option: int = 1
) -> NDArray:
    """
    Apply anisotropic diffusion for edge-preserving smoothing.

    Args:
        image: Input image
        niter: Number of iterations
        kappa: Conduction coefficient
        gamma: Integration constant (0 < gamma <= 0.25)
        option: 1 or 2, different diffusion functions

    Returns:
        Smoothed image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = image.astype(np.float32)

    for _ in range(niter):
        # Calculate gradients
        nabla_n = np.roll(img, -1, axis=0) - img  # North
        nabla_s = np.roll(img, 1, axis=0) - img   # South
        nabla_e = np.roll(img, -1, axis=1) - img  # East
        nabla_w = np.roll(img, 1, axis=1) - img   # West

        # Calculate diffusion coefficients
        if option == 1:
            c_n = np.exp(-(nabla_n / kappa) ** 2)
            c_s = np.exp(-(nabla_s / kappa) ** 2)
            c_e = np.exp(-(nabla_e / kappa) ** 2)
            c_w = np.exp(-(nabla_w / kappa) ** 2)
        else:
            c_n = 1.0 / (1.0 + (nabla_n / kappa) ** 2)
            c_s = 1.0 / (1.0 + (nabla_s / kappa) ** 2)
            c_e = 1.0 / (1.0 + (nabla_e / kappa) ** 2)
            c_w = 1.0 / (1.0 + (nabla_w / kappa) ** 2)

        # Update image
        img += gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)

    return np.clip(img, 0, 255).astype(np.uint8)


# Feature Extraction

def extract_hog_features(
    image: NDArray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """
    Extract Histogram of Oriented Gradients features.

    Args:
        image: Input image
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell
        cells_per_block: Number of cells in each block
        visualize: Return visualization image

    Returns:
        HOG features or (features, visualization)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if visualize:
        features, hog_image = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True
        )
        # Normalize visualization
        hog_image = (hog_image * 255).astype(np.uint8)
        return features, hog_image
    else:
        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False
        )
        return features


def detect_corners(
    image: NDArray,
    method: Union[str, CornerDetector] = "harris",
    **kwargs
) -> NDArray:
    """
    Detect corners in image.

    Args:
        image: Input image
        method: Corner detection method
        **kwargs: Method-specific parameters

    Returns:
        Corner response image or corner coordinates
    """
    if isinstance(method, str):
        method = CornerDetector(method.lower())

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if method == CornerDetector.HARRIS:
        block_size = kwargs.get('block_size', 2)
        ksize = kwargs.get('ksize', 3)
        k = kwargs.get('k', 0.04)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        return dst

    elif method == CornerDetector.SHI_TOMASI:
        max_corners = kwargs.get('max_corners', 100)
        quality_level = kwargs.get('quality_level', 0.01)
        min_distance = kwargs.get('min_distance', 10)
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        return corners

    elif method == CornerDetector.FAST:
        threshold = kwargs.get('threshold', 10)
        non_max_suppression = kwargs.get('non_max_suppression', True)
        fast = cv2.FastFeatureDetector_create(threshold=threshold,
                                               nonmaxSuppression=non_max_suppression)
        keypoints = fast.detect(gray, None)
        # Convert to coordinates
        corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return corners

    else:
        raise ValueError(f"Unknown corner detection method: {method}")


# Advanced Morphological Operations

def apply_advanced_morphology(
    image: NDArray,
    operation: Union[str, MorphologicalAdvanced]
) -> NDArray:
    """
    Apply advanced morphological operations.

    Args:
        image: Input binary image
        operation: Morphological operation

    Returns:
        Processed image
    """
    if isinstance(operation, str):
        operation = MorphologicalAdvanced(operation.lower())

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary.astype(bool)

    if operation == MorphologicalAdvanced.SKELETON:
        skeleton = morphology.skeletonize(binary_bool)
        return (skeleton * 255).astype(np.uint8)

    elif operation == MorphologicalAdvanced.THIN:
        thinned = morphology.thin(binary_bool)
        return (thinned * 255).astype(np.uint8)

    elif operation == MorphologicalAdvanced.CONVEX_HULL:
        hull = morphology.convex_hull_image(binary_bool)
        return (hull * 255).astype(np.uint8)

    elif operation == MorphologicalAdvanced.DISTANCE_TRANSFORM:
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Normalize to 0-255
        dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        return dist_transform.astype(np.uint8)

    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


# Segmentation

def apply_threshold(
    image: NDArray,
    method: Union[str, ThresholdMethod] = "otsu",
    **kwargs
) -> NDArray:
    """
    Apply thresholding for segmentation.

    Args:
        image: Input image
        method: Thresholding method
        **kwargs: Method-specific parameters

    Returns:
        Binary image
    """
    if isinstance(method, str):
        method = ThresholdMethod(method.lower())

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if method == ThresholdMethod.OTSU:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    elif method == ThresholdMethod.ADAPTIVE_MEAN:
        block_size = kwargs.get('block_size', 11)
        c = kwargs.get('c', 2)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, block_size, c)
        return binary

    elif method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
        block_size = kwargs.get('block_size', 11)
        c = kwargs.get('c', 2)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, c)
        return binary

    elif method == ThresholdMethod.TRIANGLE:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return binary

    elif method == ThresholdMethod.YEN:
        threshold_value = filters.threshold_yen(gray)
        binary = (gray > threshold_value).astype(np.uint8) * 255
        return binary

    elif method == ThresholdMethod.ISODATA:
        threshold_value = filters.threshold_isodata(gray)
        binary = (gray > threshold_value).astype(np.uint8) * 255
        return binary

    else:
        raise ValueError(f"Unknown thresholding method: {method}")


def watershed_segmentation(
    image: NDArray,
    markers: Optional[NDArray] = None
) -> NDArray:
    """
    Apply watershed segmentation.

    Args:
        image: Input image
        markers: Marker image (if None, automatic markers are generated)

    Returns:
        Segmented image
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if markers is None:
        # Automatic marker generation
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)

    # Create output image
    result = np.zeros_like(gray)
    result[markers == -1] = 255  # Boundaries

    return result


# Image Pyramids

def gaussian_pyramid(image: NDArray, levels: int = 3) -> list:
    """
    Create Gaussian pyramid.

    Args:
        image: Input image
        levels: Number of pyramid levels

    Returns:
        List of pyramid levels
    """
    pyramid = [image]

    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)

    return pyramid


def laplacian_pyramid(image: NDArray, levels: int = 3) -> list:
    """
    Create Laplacian pyramid.

    Args:
        image: Input image
        levels: Number of pyramid levels

    Returns:
        List of pyramid levels
    """
    # Build Gaussian pyramid
    gaussian_pyr = gaussian_pyramid(image, levels)

    # Build Laplacian pyramid
    laplacian_pyr = []

    for i in range(levels - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(laplacian)

    # Add the smallest level
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr
