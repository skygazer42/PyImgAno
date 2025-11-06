# PyImgAno Quick Start Guide

Get started with PyImgAno in minutes! This guide covers installation, basic usage, and common workflows.

## Table of Contents

1. [Installation](#installation)
2. [Your First Anomaly Detection](#your-first-anomaly-detection)
3. [Preprocessing Images](#preprocessing-images)
4. [Data Augmentation](#data-augmentation)
5. [Complete Workflow](#complete-workflow)
6. [Common Use Cases](#common-use-cases)
7. [Next Steps](#next-steps)

## Installation

### Basic Installation

```bash
pip install pyimgano
```

### With Optional Dependencies

```bash
# For diffusion models
pip install pyimgano[diffusion]

# For development tools
pip install pyimgano[dev]

# Install everything
pip install pyimgano[all]
```

### Verify Installation

```python
import pyimgano
print(f"PyImgAno version: {pyimgano.__version__}")
```

## Your First Anomaly Detection

### 1. Classical Method (Fast, Good for Getting Started)

```python
import numpy as np
from pyimgano.detectors import IsolationForestDetector

# Generate sample data
# In practice, you would load and preprocess your images
n_normal = 1000
n_anomaly = 50

# Normal samples (feature vectors extracted from images)
normal_data = np.random.randn(n_normal, 100) * 0.5

# Anomalous samples (different distribution)
anomaly_data = np.random.randn(n_anomaly, 100) * 2.0

# Training data (normal samples only)
X_train = normal_data

# Test data (mix of normal and anomalous)
X_test = np.vstack([normal_data[:100], anomaly_data])
y_test = np.hstack([np.zeros(100), np.ones(n_anomaly)])

# Create and train detector
detector = IsolationForestDetector(n_estimators=100)
detector.fit(X_train)

# Predict anomaly scores
scores = detector.predict_proba(X_test)
predictions = detector.predict(X_test)  # Binary: 0=normal, 1=anomaly

# Evaluate
from sklearn.metrics import roc_auc_score, confusion_matrix

auc = roc_auc_score(y_test, scores)
cm = confusion_matrix(y_test, predictions)

print(f"AUC-ROC: {auc:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

**Expected output:**
```
AUC-ROC: 0.8523
Confusion Matrix:
[[95  5]
 [ 8 42]]
```

### 2. Deep Learning Method (Better Accuracy)

```python
import numpy as np
from pyimgano.detectors import AutoencoderDetector

# Same data as above
X_train = normal_data
X_test = np.vstack([normal_data[:100], anomaly_data])

# Create and train autoencoder
detector = AutoencoderDetector(
    input_dim=100,
    encoding_dim=32,
    hidden_dims=[64],
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# Train (may take a minute)
detector.fit(X_train)

# Predict
scores = detector.predict_proba(X_test)
predictions = detector.predict(X_test)

# Evaluate
auc = roc_auc_score(y_test, scores)
print(f"Autoencoder AUC-ROC: {auc:.4f}")
```

## Preprocessing Images

PyImgAno provides 80+ preprocessing operations.

### Basic Preprocessing

```python
import cv2
import numpy as np
from pyimgano.preprocessing import ImageEnhancer

# Load an image
image = cv2.imread('path/to/image.jpg')

# Create enhancer
enhancer = ImageEnhancer()

# Edge detection
edges = enhancer.detect_edges(image, method='canny',
                               threshold1=100, threshold2=200)

# Gaussian blur
blurred = enhancer.apply_filter(image, filter_type='gaussian',
                                kernel_size=5)

# Morphological operations
dilated = enhancer.morph_operation(image, operation='dilate',
                                   kernel_size=5)

# Normalize
normalized = enhancer.normalize(image, method='minmax')

print(f"Original: {image.shape}")
print(f"Edges: {edges.shape}")
print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
```

### Advanced Preprocessing

```python
from pyimgano.preprocessing import AdvancedImageEnhancer

# Create advanced enhancer
enhancer = AdvancedImageEnhancer()

# Color space conversion
lab_image = enhancer.convert_color(image, from_space='bgr', to_space='lab')

# Multi-scale Retinex (illumination invariant)
enhanced = enhancer.retinex_multi(image, sigmas=[15, 80, 250])

# Texture analysis with Local Binary Pattern
lbp = enhancer.compute_lbp(image, n_points=8, radius=1.0)

# HOG features for anomaly detection
hog_features, hog_viz = enhancer.extract_hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    visualize=True
)

print(f"HOG features: {len(hog_features)} dimensions")
```

### Preprocessing Pipeline

```python
from pyimgano.preprocessing import PreprocessingPipeline

# Define preprocessing steps
pipeline = PreprocessingPipeline([
    ('edge', 'canny', {'threshold1': 50, 'threshold2': 150}),
    ('filter', 'gaussian', {'kernel_size': 5}),
    ('normalize', 'minmax', {}),
])

# Apply pipeline
processed = pipeline.transform(image)

# Or apply to batch
images = [cv2.imread(f'image_{i}.jpg') for i in range(10)]
processed_batch = pipeline.transform_batch(images)
```

## Data Augmentation

PyImgAno includes 30+ augmentation operations for training robust models.

### Simple Augmentation

```python
from pyimgano.preprocessing import (
    RandomRotate, RandomFlip, ColorJitter, GaussianNoise
)

# Single augmentations
rotate = RandomRotate(angle_range=(-30, 30), p=1.0)
flip = RandomFlip(mode='horizontal', p=1.0)
jitter = ColorJitter(brightness=0.2, contrast=0.2, p=1.0)
noise = GaussianNoise(mean=0, std=0.02, p=1.0)

# Apply to image
rotated = rotate(image)
flipped = flip(image)
jittered = jitter(image)
noisy = noise(image)
```

### Augmentation Pipeline

```python
from pyimgano.preprocessing import Compose, OneOf, RandomApply

# Create complex pipeline
augmentation = Compose([
    RandomFlip(mode='horizontal', p=0.5),
    RandomRotate(angle_range=(-15, 15), p=0.5),
    OneOf([
        GaussianNoise(std=0.01, p=1.0),
        ColorJitter(brightness=0.2, p=1.0),
    ], p=0.3),
    RandomApply([
        RandomRotate(angle_range=(-5, 5), p=1.0)
    ], p=0.2),
])

# Apply to training data
augmented_images = [augmentation(img) for img in training_images]
```

### Preset Pipelines

```python
from pyimgano.preprocessing import (
    get_light_augmentation,
    get_medium_augmentation,
    get_heavy_augmentation,
    get_weather_augmentation,
    get_anomaly_augmentation,
)

# Light augmentation (minimal changes)
light_aug = get_light_augmentation()
augmented = light_aug(image)

# Medium augmentation (balanced)
medium_aug = get_medium_augmentation()
augmented = medium_aug(image)

# Heavy augmentation (aggressive)
heavy_aug = get_heavy_augmentation()
augmented = heavy_aug(image)

# Weather effects (rain, fog, snow)
weather_aug = get_weather_augmentation()
augmented = weather_aug(image)

# Anomaly-specific augmentation
anomaly_aug = get_anomaly_augmentation()
augmented = anomaly_aug(image)
```

### Augmentation Pipeline with Statistics

```python
from pyimgano.preprocessing import AugmentationPipeline

# Create pipeline with tracking
pipeline = AugmentationPipeline(get_medium_augmentation(), track_stats=True)

# Apply to batch
augmented_batch = [pipeline(img) for img in training_images]

# Get statistics
stats = pipeline.get_stats()
print(f"Total applications: {stats['total_applications']}")
print(f"Average time: {stats['avg_time_ms']:.2f}ms")
print(f"Operation counts: {stats['operation_counts']}")

# Reset statistics
pipeline.reset_stats()
```

## Complete Workflow

### Industrial Defect Detection Example

```python
import cv2
import numpy as np
from pyimgano.preprocessing import (
    AdvancedImageEnhancer,
    get_medium_augmentation,
)
from pyimgano.detectors import IsolationForestDetector
from sklearn.metrics import classification_report

# 1. Load and preprocess training data (normal samples only)
def load_images(folder_path):
    """Load images from folder."""
    import glob
    images = []
    for path in glob.glob(f"{folder_path}/*.jpg"):
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images

# Load normal training images
normal_images = load_images('data/normal/')
print(f"Loaded {len(normal_images)} normal images")

# 2. Preprocessing
enhancer = AdvancedImageEnhancer()

def preprocess_image(img):
    """Preprocess single image."""
    # Convert to LAB color space
    lab = enhancer.convert_color(img, 'bgr', 'lab')

    # Apply Retinex for illumination invariance
    enhanced = enhancer.retinex_multi(lab)

    # Extract texture features with LBP
    lbp = enhancer.compute_lbp(enhanced)

    # Flatten to feature vector
    features = lbp.flatten()

    return features

# Preprocess training data
X_train = np.array([preprocess_image(img) for img in normal_images])
print(f"Training features shape: {X_train.shape}")

# 3. Data augmentation (optional, for more robust training)
augmentation = get_medium_augmentation()
augmented_images = [augmentation(img) for img in normal_images]
X_train_aug = np.array([preprocess_image(img) for img in augmented_images])
X_train = np.vstack([X_train, X_train_aug])
print(f"Training features with augmentation: {X_train.shape}")

# 4. Train detector
detector = IsolationForestDetector(
    n_estimators=100,
    max_samples='auto',
    contamination=0.1,
    random_state=42
)
detector.fit(X_train)
print("Detector trained successfully")

# 5. Test on mixed data (normal + anomalous)
test_normal = load_images('data/test/normal/')
test_anomaly = load_images('data/test/anomaly/')

X_test_normal = np.array([preprocess_image(img) for img in test_normal])
X_test_anomaly = np.array([preprocess_image(img) for img in test_anomaly])

X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test = np.hstack([
    np.zeros(len(test_normal)),
    np.ones(len(test_anomaly))
])

# 6. Predict
predictions = detector.predict(X_test)
scores = detector.predict_proba(X_test)

# 7. Evaluate
from sklearn.metrics import roc_auc_score, classification_report

auc = roc_auc_score(y_test, scores)
print(f"\nAUC-ROC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions,
                          target_names=['Normal', 'Anomaly']))

# 8. Visualize results
import matplotlib.pyplot as plt

# Plot ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("\nROC curve saved to roc_curve.png")
```

## Common Use Cases

### Use Case 1: Surface Defect Inspection

```python
from pyimgano.detectors import LOFDetector
from pyimgano.preprocessing import AdvancedImageEnhancer

# For surface defects, use texture features
enhancer = AdvancedImageEnhancer()

def extract_surface_features(image):
    # GLCM texture features
    glcm = enhancer.compute_glcm(image)

    # LBP texture
    lbp = enhancer.compute_lbp(image)

    # Combine features
    features = np.hstack([
        list(glcm.values()),
        lbp.flatten()[:100]  # First 100 LBP features
    ])
    return features

# Train LOF detector (good for local anomalies)
detector = LOFDetector(n_neighbors=20)
detector.fit(X_train_features)
```

### Use Case 2: PCB Inspection

```python
from pyimgano.detectors import AutoencoderDetector
from pyimgano.preprocessing import ImageEnhancer

# For PCB, use edge-based features
enhancer = ImageEnhancer()

def extract_pcb_features(image):
    # Edge detection
    edges = enhancer.detect_edges(image, method='canny')

    # Morphological operations to clean up
    cleaned = enhancer.morph_operation(edges, 'close', kernel_size=3)

    return cleaned.flatten()

# Autoencoder works well for complex patterns
detector = AutoencoderDetector(
    input_dim=image_size[0] * image_size[1],
    encoding_dim=128,
    hidden_dims=[512, 256],
    epochs=100
)
detector.fit(X_train_features)
```

### Use Case 3: Fabric Defect Detection

```python
from pyimgano.detectors import VAEDetector
from pyimgano.preprocessing import AdvancedImageEnhancer

# For fabric, combine frequency and texture analysis
enhancer = AdvancedImageEnhancer()

def extract_fabric_features(image):
    # FFT for periodic patterns
    magnitude, phase = enhancer.apply_fft(image)

    # Gabor filters for texture orientation
    gabor1 = enhancer.gabor_filter(image, theta=0)
    gabor2 = enhancer.gabor_filter(image, theta=np.pi/4)

    # Combine
    features = np.hstack([
        magnitude.flatten()[:500],
        gabor1.flatten()[:250],
        gabor2.flatten()[:250]
    ])
    return features

# VAE for complex texture patterns
detector = VAEDetector(
    input_dim=1000,
    latent_dim=64,
    hidden_dims=[512, 256],
    epochs=100
)
detector.fit(X_train_features)
```

### Use Case 4: Real-time Inspection

```python
from pyimgano.detectors import KNNDetector
from pyimgano.preprocessing import ImageEnhancer

# For real-time, use fast methods
enhancer = ImageEnhancer()
detector = KNNDetector(n_neighbors=5)  # Very fast inference

def fast_preprocess(image):
    # Minimal preprocessing
    resized = cv2.resize(image, (64, 64))
    normalized = enhancer.normalize(resized)
    return normalized.flatten()

# Quick training
X_train = np.array([fast_preprocess(img) for img in normal_images])
detector.fit(X_train)

# Fast inference (<10ms per image)
import time
start = time.time()
score = detector.predict_proba([test_features])[0]
inference_time = (time.time() - start) * 1000
print(f"Inference time: {inference_time:.2f}ms")
```

## Next Steps

### 1. Explore More Algorithms

Check out all 37+ algorithms:

```python
from pyimgano import detectors

# Statistical
detectors.IQRDetector
detectors.MADDetector
detectors.HistogramBasedDetector

# Distance-based
detectors.KNNDetector
detectors.LOFDetector

# Density-based
detectors.ECODDetector
detectors.COPODDetector

# Deep learning
detectors.AutoencoderDetector
detectors.VAEDetector
detectors.DeepSVDDDetector

# And many more...
```

### 2. Read Full Documentation

- [API Reference](https://github.com/jhlu2019/pyimgano#api-reference)
- [Algorithm Comparison](./COMPARISON.md)
- [Advanced Examples](../examples/)

### 3. Run Benchmarks

```bash
cd benchmarks
python run_all_benchmarks.py
```

See [Benchmark README](../benchmarks/README.md) for details.

### 4. Contribute

- [Contributing Guide](../CONTRIBUTING.md)
- [Development Setup](../CONTRIBUTING.md#development-setup)
- [Code of Conduct](../CODE_OF_CONDUCT.md)

### 5. Join the Community

- 🐛 [Report Issues](https://github.com/jhlu2019/pyimgano/issues)
- 💡 [Request Features](https://github.com/jhlu2019/pyimgano/issues/new)
- ⭐ [Star on GitHub](https://github.com/jhlu2019/pyimgano)

## FAQ

### Q: Which algorithm should I use?

**A:** It depends on your use case:

- **Real-time applications**: KNN, LOF (< 10ms inference)
- **High accuracy**: Autoencoder, VAE, Deep SVDD
- **Limited data**: IForest, ECOD, COPOD
- **Interpretability**: Statistical methods (IQR, MAD)

### Q: How much training data do I need?

**A:**

- **Classical methods**: 100-1000 normal samples
- **Deep learning**: 1000+ normal samples
- **Use augmentation**: Can reduce requirement by 2-3x

### Q: Can I use GPU?

**A:** Yes! Deep learning methods automatically use GPU if available:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

### Q: How do I save/load trained models?

**A:**

```python
# Save
detector.save_model('my_detector.pkl')

# Load
from pyimgano.detectors import IsolationForestDetector
detector = IsolationForestDetector.load_model('my_detector.pkl')
```

### Q: What image formats are supported?

**A:** Any format supported by OpenCV:
- Common: JPG, PNG, BMP, TIFF
- Use `cv2.imread()` to load images

### Q: Can I use grayscale images?

**A:** Yes! Most operations work with grayscale:

```python
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
features = preprocess(gray)
```

## Troubleshooting

### Import Error

```bash
# Make sure PyImgAno is installed
pip install pyimgano

# Or install from source
git clone https://github.com/jhlu2019/pyimgano
cd pyimgano
pip install -e .
```

### CUDA Out of Memory

```python
# Reduce batch size
detector = AutoencoderDetector(batch_size=16)  # Default is 32

# Or use CPU
import torch
torch.cuda.is_available = lambda: False
```

### Slow Training

```python
# Reduce epochs for deep learning
detector = AutoencoderDetector(epochs=10)  # Default is 50

# Or use classical methods
detector = IsolationForestDetector()  # Trains in seconds
```

## Get Help

Need help? Here's how to get support:

1. 📖 Check the [documentation](https://github.com/jhlu2019/pyimgano)
2. 🔍 Search [existing issues](https://github.com/jhlu2019/pyimgano/issues)
3. 💬 Ask a [new question](https://github.com/jhlu2019/pyimgano/issues/new)
4. 📧 Contact: pyimgano@example.com

Happy anomaly detecting! 🎯
