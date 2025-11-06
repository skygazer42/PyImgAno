# Image Preprocessing Module

Comprehensive image preprocessing and enhancement capabilities for anomaly detection.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Components](#components)
  - [ImageEnhancer](#imageenhancer)
  - [PreprocessingPipeline](#preprocessingpipeline)
  - [PreprocessingMixin](#preprocessingmixin)
- [Operations](#operations)
  - [Edge Detection](#edge-detection)
  - [Morphological Operations](#morphological-operations)
  - [Filters](#filters)
  - [Normalization](#normalization)
  - [Advanced Operations](#advanced-operations)
- [Integration with Detectors](#integration-with-detectors)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The preprocessing module provides comprehensive image enhancement capabilities designed to improve anomaly detection performance. It offers:

- **7 edge detection methods** (Canny, Sobel, Laplacian, Scharr, Prewitt, etc.)
- **7 morphological operations** (Erosion, Dilation, Opening, Closing, etc.)
- **4 filter types** (Gaussian, Bilateral, Median, Box)
- **4 normalization methods** (MinMax, Z-Score, L2, Robust)
- **Pipeline system** for sequential operations
- **Easy integration** with anomaly detectors via Mixin pattern

## Quick Start

### Basic Usage

```python
from pyimgano.preprocessing import ImageEnhancer
import cv2

# Create enhancer
enhancer = ImageEnhancer()

# Load image
img = cv2.imread('image.jpg')

# Apply operations
blurred = enhancer.gaussian_blur(img, ksize=(5, 5))
edges = enhancer.detect_edges(blurred, method='canny')
normalized = enhancer.normalize(edges, method='minmax')
```

### Using Pipelines

```python
from pyimgano.preprocessing import PreprocessingPipeline

# Build pipeline
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(5, 5))
pipeline.add_step('detect_edges', method='canny')
pipeline.add_step('normalize', method='minmax')

# Apply to image
result = pipeline.transform(img)
```

### Integration with Detectors

```python
from pyimgano.preprocessing import PreprocessingMixin
from pyimgano.models import ECOD

class ECODWithPreprocessing(PreprocessingMixin, ECOD):
    def __init__(self):
        super().__init__()
        self.setup_preprocessing(enable=True, use_pipeline=True)
        self.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        self.add_preprocessing_step('normalize', method='minmax')

    def fit(self, X, y=None):
        X_processed = self.preprocess_images(X)
        return super().fit(X_processed, y)
```

## Components

### ImageEnhancer

High-level interface for image enhancement operations.

```python
from pyimgano.preprocessing import ImageEnhancer

enhancer = ImageEnhancer()

# Edge detection
edges = enhancer.detect_edges(img, method='canny', threshold1=50, threshold2=150)

# Blur
blurred = enhancer.gaussian_blur(img, ksize=(5, 5), sigma=1.0)

# Morphology
eroded = enhancer.erode(img, kernel_size=(5, 5))
dilated = enhancer.dilate(img, kernel_size=(5, 5))

# Normalization
normalized = enhancer.normalize(img, method='minmax')
```

**Available Methods:**

| Method | Description |
|--------|-------------|
| `detect_edges()` | Edge detection (7 methods) |
| `gaussian_blur()` | Gaussian smoothing |
| `bilateral_filter()` | Edge-preserving smoothing |
| `median_blur()` | Median filtering |
| `erode()` | Morphological erosion |
| `dilate()` | Morphological dilation |
| `opening()` | Morphological opening |
| `closing()` | Morphological closing |
| `morphological_gradient()` | Morphological gradient |
| `tophat()` | Top-hat transform |
| `blackhat()` | Black-hat transform |
| `sharpen()` | Image sharpening |
| `unsharp_mask()` | Unsharp masking |
| `clahe()` | Contrast enhancement |
| `normalize()` | Normalization (4 methods) |

### PreprocessingPipeline

Build and execute sequential preprocessing operations.

```python
from pyimgano.preprocessing import PreprocessingPipeline

# Create pipeline
pipeline = PreprocessingPipeline()

# Add steps (method chaining supported)
pipeline.add_step('gaussian_blur', ksize=(5, 5))
       .add_step('detect_edges', method='canny')
       .add_step('normalize', method='minmax')

# Apply to image
result = pipeline.transform(img)

# Pipeline info
print(f"Pipeline has {len(pipeline)} steps")

# Clear pipeline
pipeline.clear()
```

**Features:**
- Sequential operation execution
- Method chaining
- Reusable across multiple images
- Easy to modify and extend

### PreprocessingMixin

Mixin class for easy integration with anomaly detectors.

```python
from pyimgano.preprocessing import PreprocessingMixin

class MyDetector(PreprocessingMixin):
    def __init__(self):
        # Setup preprocessing
        self.setup_preprocessing(enable=True, use_pipeline=True)

        # Configure pipeline
        self.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        self.add_preprocessing_step('normalize', method='minmax')

    def fit(self, X, y=None):
        # Preprocess training images
        X_processed = self.preprocess_images(X)
        # ... your fitting logic

    def predict(self, X):
        # Preprocess test images
        X_processed = self.preprocess_images(X)
        # ... your prediction logic
```

**Available Methods:**

| Method | Description |
|--------|-------------|
| `setup_preprocessing()` | Initialize preprocessing |
| `add_preprocessing_step()` | Add step to pipeline |
| `preprocess_image()` | Preprocess single image |
| `preprocess_images()` | Preprocess multiple images |
| `preprocess_with_edges()` | Convenience: edge detection |
| `preprocess_with_blur()` | Convenience: blur |
| `preprocess_with_morphology()` | Convenience: morphology |
| `clear_preprocessing_pipeline()` | Clear all steps |
| `get_preprocessing_info()` | Get configuration info |

## Operations

### Edge Detection

Detect edges in images using various methods.

```python
from pyimgano.preprocessing import edge_detection

# Canny edge detection
edges = edge_detection(img, method='canny', threshold1=50, threshold2=150)

# Sobel edge detection
edges = edge_detection(img, method='sobel', ksize=3)

# Laplacian edge detection
edges = edge_detection(img, method='laplacian', ksize=3)
```

**Available Methods:**

| Method | Description | Parameters |
|--------|-------------|------------|
| `canny` | Canny edge detector | threshold1, threshold2, aperture_size |
| `sobel` | Sobel operator | ksize |
| `sobel_x` | Sobel X derivative | ksize |
| `sobel_y` | Sobel Y derivative | ksize |
| `laplacian` | Laplacian operator | ksize |
| `scharr` | Scharr operator | - |
| `prewitt` | Prewitt operator | - |

**When to Use:**
- **Canny**: Best for general edge detection, multi-stage algorithm
- **Sobel**: Good for gradient-based detection, directional edges
- **Laplacian**: Detects edges using second derivative, good for blobs
- **Scharr**: More accurate than Sobel for small kernels
- **Prewitt**: Similar to Sobel, sometimes better for noisy images

### Morphological Operations

Apply morphological transformations to images.

```python
from pyimgano.preprocessing import morphological_operation

# Erosion - shrinks bright regions
eroded = morphological_operation(
    img,
    operation='erosion',
    kernel_size=(5, 5),
    kernel_shape='rect',
    iterations=1
)

# Dilation - expands bright regions
dilated = morphological_operation(img, operation='dilation', kernel_size=(5, 5))

# Opening - removes small objects
opened = morphological_operation(img, operation='opening', kernel_size=(5, 5))

# Closing - fills small holes
closed = morphological_operation(img, operation='closing', kernel_size=(5, 5))
```

**Available Operations:**

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `erosion` | Shrink bright regions | Remove small bright noise |
| `dilation` | Expand bright regions | Fill small dark holes |
| `opening` | Erosion → Dilation | Remove small objects |
| `closing` | Dilation → Erosion | Fill holes in objects |
| `gradient` | Dilation - Erosion | Outline objects |
| `tophat` | Original - Opening | Bright objects on dark background |
| `blackhat` | Closing - Original | Dark objects on bright background |

**Kernel Shapes:**
- `rect`: Rectangular structuring element
- `ellipse`: Elliptical structuring element
- `cross`: Cross-shaped structuring element

**When to Use:**
- **Opening**: Remove small bright noise while preserving shape
- **Closing**: Fill small holes in objects
- **Gradient**: Highlight object boundaries
- **Top-hat**: Extract small bright features
- **Black-hat**: Extract small dark features

### Filters

Apply various filters for smoothing and noise reduction.

```python
from pyimgano.preprocessing import apply_filter

# Gaussian blur - general smoothing
blurred = apply_filter(img, filter_type='gaussian', ksize=(5, 5), sigma=1.0)

# Bilateral filter - edge-preserving smoothing
bilateral = apply_filter(img, filter_type='bilateral', d=9, sigma_color=75, sigma_space=75)

# Median blur - salt-and-pepper noise removal
median = apply_filter(img, filter_type='median', ksize=5)

# Box filter - simple averaging
box = apply_filter(img, filter_type='box', ksize=(5, 5))
```

**Available Filters:**

| Filter | Description | Parameters | Best For |
|--------|-------------|------------|----------|
| `gaussian` | Gaussian smoothing | ksize, sigma | General noise reduction |
| `bilateral` | Edge-preserving smoothing | d, sigma_color, sigma_space | Preserve edges while smoothing |
| `median` | Median filtering | ksize | Salt-and-pepper noise |
| `box` | Box/Mean filter | ksize | Fast averaging |

**When to Use:**
- **Gaussian**: General-purpose noise reduction, before edge detection
- **Bilateral**: When you need to preserve edges (e.g., texture analysis)
- **Median**: Specifically for salt-and-pepper noise
- **Box**: When speed is critical and quality is less important

### Normalization

Normalize image values to standard ranges.

```python
from pyimgano.preprocessing import normalize_image

# MinMax normalization to [0, 1]
normalized = normalize_image(img, method='minmax')

# Z-score normalization (mean=0, std=1)
normalized = normalize_image(img, method='zscore')

# L2 normalization
normalized = normalize_image(img, method='l2')

# Robust normalization (IQR-based, outlier-resistant)
normalized = normalize_image(img, method='robust')
```

**Available Methods:**

| Method | Description | Output Range | Use Case |
|--------|-------------|--------------|----------|
| `minmax` | Linear scaling | [0, 1] | General use, neural networks |
| `zscore` | Standardization | Mean=0, Std=1 | Statistical analysis |
| `l2` | L2 normalization | L2 norm = 1 | Feature vectors |
| `robust` | IQR-based scaling | Approx. [-1, 1] | Data with outliers |

**When to Use:**
- **MinMax**: Default choice, good for neural networks
- **Z-Score**: When you need zero-mean unit-variance data
- **L2**: For distance-based algorithms (KNN, etc.)
- **Robust**: When data contains outliers

### Advanced Operations

Additional enhancement operations.

```python
from pyimgano.preprocessing import ImageEnhancer

enhancer = ImageEnhancer()

# Sharpening - enhance details
sharpened = enhancer.sharpen(img, kernel_type='standard')

# Unsharp masking - controllable sharpening
unsharp = enhancer.unsharp_mask(img, sigma=1.0, amount=1.5)

# CLAHE - adaptive contrast enhancement
clahe = enhancer.clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
```

**When to Use:**
- **Sharpen**: Enhance edges and details
- **Unsharp Mask**: More control over sharpening amount
- **CLAHE**: Improve contrast in low-contrast regions

## Integration with Detectors

### Method 1: Inheritance with Mixin

Most flexible approach using the Mixin pattern:

```python
from pyimgano.preprocessing import PreprocessingMixin
from pyimgano.models import ECOD

class ECODWithPreprocessing(PreprocessingMixin, ECOD):
    def __init__(self, **kwargs):
        # Initialize ECOD
        super().__init__(**kwargs)

        # Setup preprocessing
        self.setup_preprocessing(enable=True, use_pipeline=True)

        # Configure pipeline
        self.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        self.add_preprocessing_step('normalize', method='minmax')

    def fit(self, X, y=None):
        """Fit with preprocessing."""
        X_processed = self.preprocess_images(X)
        X_flat = [img.flatten() for img in X_processed]
        return super().fit(X_flat, y)

    def predict(self, X):
        """Predict with preprocessing."""
        X_processed = self.preprocess_images(X)
        X_flat = [img.flatten() for img in X_processed]
        return super().predict(X_flat)

# Usage
detector = ECODWithPreprocessing()
detector.fit(train_images)
scores = detector.predict(test_images)
```

### Method 2: External Pipeline

Keep preprocessing separate from detector:

```python
from pyimgano.preprocessing import PreprocessingPipeline
from pyimgano.models import ECOD

# Create preprocessing pipeline
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(5, 5))
pipeline.add_step('normalize', method='minmax')

# Preprocess data
train_processed = [pipeline.transform(img) for img in train_images]
test_processed = [pipeline.transform(img) for img in test_images]

# Use with detector
detector = ECOD()
detector.fit(train_processed)
scores = detector.predict(test_processed)
```

### Method 3: Manual Preprocessing

Use ImageEnhancer directly:

```python
from pyimgano.preprocessing import ImageEnhancer
from pyimgano.models import ECOD

enhancer = ImageEnhancer()

# Preprocess manually
def preprocess(images):
    processed = []
    for img in images:
        img = enhancer.gaussian_blur(img, ksize=(5, 5))
        img = enhancer.normalize(img, method='minmax')
        processed.append(img)
    return processed

train_processed = preprocess(train_images)
test_processed = preprocess(test_images)

detector = ECOD()
detector.fit(train_processed)
scores = detector.predict(test_processed)
```

## Best Practices

### 1. Choose Appropriate Preprocessing Strategy

Different anomaly types benefit from different preprocessing:

```python
# For surface defects (texture anomalies)
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(3, 3))
pipeline.add_step('unsharp_mask', sigma=1.0, amount=1.5)
pipeline.add_step('normalize', method='minmax')

# For structural anomalies (shape/edge)
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(5, 5))
pipeline.add_step('detect_edges', method='canny')
pipeline.add_step('dilate', kernel_size=(3, 3))
pipeline.add_step('normalize', method='minmax')

# For contrast issues
pipeline = PreprocessingPipeline()
pipeline.add_step('clahe', clip_limit=2.0)
pipeline.add_step('normalize', method='robust')
```

### 2. Order Matters

Apply operations in the right order:

```python
# Good: Denoise before edge detection
pipeline.add_step('gaussian_blur', ksize=(5, 5))  # Remove noise first
pipeline.add_step('detect_edges', method='canny')  # Then detect edges

# Bad: Edge detection on noisy image
# pipeline.add_step('detect_edges', method='canny')  # Edges include noise
# pipeline.add_step('gaussian_blur', ksize=(5, 5))  # Too late!
```

**Recommended Order:**
1. **Denoising** (Gaussian, bilateral, median)
2. **Enhancement** (sharpen, CLAHE)
3. **Feature extraction** (edges, morphology)
4. **Normalization** (always last)

### 3. Parameter Tuning

Start with conservative parameters and adjust:

```python
# Edge detection - start conservative
edges = enhancer.detect_edges(img, method='canny', threshold1=50, threshold2=150)

# If too many edges (noisy):
# - Increase thresholds: threshold1=100, threshold2=200
# - Increase blur before edge detection

# If too few edges (missing details):
# - Decrease thresholds: threshold1=30, threshold2=100
# - Reduce blur or skip it
```

### 4. Validate on Sample Data

Always test preprocessing on sample images:

```python
# Test pipeline on samples
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(5, 5))
pipeline.add_step('normalize', method='minmax')

# Check results
sample_results = [pipeline.transform(img) for img in sample_images[:5]]

# Verify:
# - Values in expected range
# - No black/white images (over-processing)
# - Features still visible
```

### 5. Consider Computational Cost

Balance quality and speed:

```python
# Fast preprocessing (real-time applications)
pipeline.add_step('gaussian_blur', ksize=(3, 3))
pipeline.add_step('normalize', method='minmax')

# High-quality preprocessing (offline analysis)
pipeline.add_step('bilateral_filter', d=9, sigma_color=75, sigma_space=75)
pipeline.add_step('clahe', clip_limit=2.0)
pipeline.add_step('unsharp_mask', sigma=1.0, amount=1.5)
pipeline.add_step('normalize', method='robust')
```

## Examples

### Example 1: Texture Defect Detection

```python
from pyimgano.preprocessing import PreprocessingPipeline

# Enhance texture features
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(3, 3))
pipeline.add_step('unsharp_mask', sigma=1.0, amount=2.0)
pipeline.add_step('clahe', clip_limit=3.0)
pipeline.add_step('normalize', method='minmax')

processed = pipeline.transform(texture_image)
```

### Example 2: Edge-Based Anomaly Detection

```python
# Detect structural anomalies
pipeline = PreprocessingPipeline()
pipeline.add_step('bilateral_filter', d=9, sigma_color=75, sigma_space=75)
pipeline.add_step('detect_edges', method='canny', threshold1=50, threshold2=150)
pipeline.add_step('dilate', kernel_size=(3, 3), iterations=2)
pipeline.add_step('normalize', method='minmax')

processed = pipeline.transform(structure_image)
```

### Example 3: Noise Removal

```python
# Remove noise while preserving features
pipeline = PreprocessingPipeline()
pipeline.add_step('median_blur', ksize=5)  # Remove salt-and-pepper
pipeline.add_step('bilateral_filter', d=9)  # Smooth while preserving edges
pipeline.add_step('normalize', method='robust')  # Outlier-resistant

processed = pipeline.transform(noisy_image)
```

### Example 4: Feature Enhancement

```python
# Enhance specific features using morphology
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(5, 5))
pipeline.add_step('tophat', kernel_size=(9, 9))  # Bright features
pipeline.add_step('normalize', method='minmax')

processed = pipeline.transform(feature_image)
```

### Example 5: Multi-Scale Processing

```python
# Process at different scales and combine
from pyimgano.preprocessing import ImageEnhancer

enhancer = ImageEnhancer()

# Fine details
fine = enhancer.gaussian_blur(img, ksize=(3, 3))
fine = enhancer.detect_edges(fine, method='canny')

# Coarse structures
coarse = enhancer.gaussian_blur(img, ksize=(7, 7))
coarse = enhancer.detect_edges(coarse, method='canny')

# Combine (simple average)
combined = (fine + coarse) / 2.0
```

## Advanced Operations

PyImgAno now includes 25+ advanced image processing operations via the `AdvancedImageEnhancer` class.

### Using Advanced Enhancer

```python
from pyimgano.preprocessing import AdvancedImageEnhancer

enhancer = AdvancedImageEnhancer()
```

### Frequency Domain Operations

```python
# Fast Fourier Transform
magnitude, phase = enhancer.apply_fft(image)

# Inverse FFT
reconstructed = enhancer.apply_ifft(magnitude, phase)

# Frequency filters
lowpass = enhancer.frequency_filter(image, filter_type='lowpass', cutoff_frequency=30)
highpass = enhancer.frequency_filter(image, filter_type='highpass', cutoff_frequency=30)
bandpass = enhancer.frequency_filter(image, filter_type='bandpass', cutoff_frequency=20)
```

**When to use:**
- **Lowpass**: Remove high-frequency noise
- **Highpass**: Emphasize edges and details
- **Bandpass**: Select specific frequency range

### Texture Analysis

```python
# Gabor filters for oriented texture detection
gabor = enhancer.gabor_filter(image, frequency=0.1, theta=np.pi/4)

# Local Binary Pattern (LBP) for texture description
lbp = enhancer.compute_lbp(image, n_points=8, radius=1.0, method='uniform')

# GLCM texture features
glcm_features = enhancer.compute_glcm(image)
# Returns: contrast, dissimilarity, homogeneity, energy, correlation
```

**When to use:**
- **Gabor**: Detect oriented textures (fabric, wood grain)
- **LBP**: Fast texture classification, illumination invariant
- **GLCM**: Statistical texture analysis, defect detection

### Color Space Transformations

```python
# Convert between color spaces
hsv = enhancer.convert_color(image, from_space='bgr', to_space='hsv')
lab = enhancer.convert_color(image, from_space='bgr', to_space='lab')

# Color histogram equalization
eq_hsv = enhancer.equalize_color_hist(image, method='hsv')  # Equalize V channel
eq_lab = enhancer.equalize_color_hist(image, method='lab')  # Equalize L channel
```

**Supported color spaces**: RGB, BGR, HSV, LAB, YCrCb, HLS, LUV, GRAY

### Advanced Enhancement

```python
# Gamma correction (brightness adjustment)
brightened = enhancer.gamma_correct(image, gamma=0.5)  # Brighter
darkened = enhancer.gamma_correct(image, gamma=2.0)   # Darker

# Contrast stretching
stretched = enhancer.contrast_stretch(image, lower_percentile=2, upper_percentile=98)

# Retinex (illumination invariant)
ssr = enhancer.retinex_single(image, sigma=15.0)  # Single-scale
msr = enhancer.retinex_multi(image, sigmas=[15, 80, 250])  # Multi-scale (better)
```

**When to use:**
- **Gamma**: Adjust overall brightness
- **Contrast stretch**: Enhance low-contrast images
- **Retinex**: Handle varying illumination, shadowy images

### Advanced Denoising

```python
# Non-local means denoising
denoised = enhancer.nlm_denoise(noisy_image, h=10, template_window_size=7)

# Anisotropic diffusion (edge-preserving smoothing)
smoothed = enhancer.anisotropic_diffusion(noisy_image, niter=10, kappa=50)
```

**When to use:**
- **NLM**: Best quality denoising, slower
- **Anisotropic diffusion**: Edge-preserving, good for gradual smoothing

### Feature Extraction

```python
# Histogram of Oriented Gradients (HOG)
hog_features, hog_image = enhancer.extract_hog(image, visualize=True)

# Corner detection
harris = enhancer.detect_corners(image, method='harris')
corners = enhancer.detect_corners(image, method='shi_tomasi', max_corners=100)
fast_corners = enhancer.detect_corners(image, method='fast', threshold=10)
```

**When to use:**
- **HOG**: Object shape description
- **Harris/Shi-Tomasi**: Precise corner localization
- **FAST**: Real-time corner detection

### Advanced Morphological Operations

```python
# Skeletonization (medial axis)
skeleton = enhancer.skeleton(binary_image)

# Thinning
thinned = enhancer.thin(binary_image)

# Convex hull
hull = enhancer.convex_hull(binary_image)

# Distance transform
dist = enhancer.distance_transform(binary_image)
```

**When to use:**
- **Skeleton**: Extract centerlines, shape analysis
- **Thinning**: Similar to skeleton, alternative algorithm
- **Convex hull**: Fill concavities, shape completion
- **Distance transform**: Find object centers, watershed markers

### Image Segmentation

```python
# Thresholding
otsu = enhancer.threshold(image, method='otsu')
adaptive = enhancer.threshold(image, method='adaptive_gaussian', block_size=11, c=2)
triangle = enhancer.threshold(image, method='triangle')
yen = enhancer.threshold(image, method='yen')

# Watershed segmentation
segmented = enhancer.watershed(image)
# Or with custom markers:
segmented = enhancer.watershed(image, markers=custom_markers)
```

**When to use:**
- **Otsu**: Bimodal histograms, automatic threshold
- **Adaptive**: Varying illumination
- **Triangle**: Skewed histograms
- **Yen**: Maximum correlation
- **Watershed**: Separate touching objects

### Image Pyramids

```python
# Gaussian pyramid (multi-scale representation)
gaussian_pyr = enhancer.build_gaussian_pyramid(image, levels=4)

# Laplacian pyramid (band-pass decomposition)
laplacian_pyr = enhancer.build_laplacian_pyramid(image, levels=4)
```

**When to use:**
- **Gaussian pyramid**: Multi-scale processing, image blending
- **Laplacian pyramid**: Edge detection at multiple scales, image compression

### Example: Complete Advanced Workflow

```python
from pyimgano.preprocessing import AdvancedImageEnhancer

enhancer = AdvancedImageEnhancer()

# 1. Illumination normalization
normalized = enhancer.retinex_multi(image, sigmas=[15, 80, 250])

# 2. Color space conversion
lab = enhancer.convert_color(normalized, from_space='bgr', to_space='lab')

# 3. Denoising
denoised = enhancer.nlm_denoise(lab, h=10)

# 4. Texture analysis
lbp = enhancer.compute_lbp(denoised, n_points=8, radius=1.0)

# 5. Segmentation
segmented = enhancer.threshold(lbp, method='otsu')

# 6. Feature extraction
hog_features = enhancer.extract_hog(segmented, visualize=False)
```

## API Reference

See the [API documentation](API.md) for complete method signatures and parameters.

## See Also

- [Deep Learning Models Guide](DEEP_LEARNING_MODELS.md)
- [Evaluation and Benchmarking](EVALUATION_AND_BENCHMARK.md)
- [Quick Start Guide](../README.md)
