# PyImgAno Benchmarks

Comprehensive benchmark suite for PyImgAno's anomaly detection algorithms and preprocessing operations.

## Overview

This directory contains performance benchmarks for:

1. **Classical Algorithms** - Statistical and ML-based methods
2. **Deep Learning Algorithms** - Neural network-based methods
3. **Preprocessing & Augmentation** - Image processing operations

## Quick Start

### Run All Benchmarks

```bash
cd benchmarks
python run_all_benchmarks.py
```

This will:
- Run all benchmark suites sequentially
- Generate CSV files with raw data
- Create visualization plots (PNG)
- Generate a comprehensive markdown report

### Run Individual Benchmarks

```bash
# Classical algorithms only
python benchmark_classical.py

# Deep learning algorithms only
python benchmark_deeplearning.py

# Preprocessing operations only
python benchmark_preprocessing.py
```

## Requirements

Install dependencies:

```bash
pip install pyimgano[dev]
pip install psutil  # For memory monitoring
```

## Benchmark Details

### 1. Classical Algorithms (`benchmark_classical.py`)

**Algorithms tested:**
- Statistical: IQR, MAD, Histogram-based
- Distance-based: KNN, LOF
- Density-based: ECOD, COPOD
- Isolation-based: Isolation Forest

**Metrics:**
- Training time (seconds)
- Inference time per image (seconds)
- Memory usage (MB)
- AUC-ROC score

**Output:**
- `benchmark_classical_results.csv` - Raw data
- `benchmark_classical_results.png` - Comparison plots

**Expected runtime:** ~30-60 seconds

### 2. Deep Learning Algorithms (`benchmark_deeplearning.py`)

**Algorithms tested:**
- Autoencoder (AE)
- Variational Autoencoder (VAE)
- Deep SVDD

**Metrics:**
- Training time per epoch (seconds)
- Total training time (seconds)
- Inference time per image (seconds)
- Model size (MB)
- AUC-ROC score

**Output:**
- `benchmark_deeplearning_results.csv` - Raw data
- `benchmark_deeplearning_results.png` - Comparison plots

**Expected runtime:** ~5-10 minutes (depends on epochs and hardware)

**Note:** Uses GPU if available, otherwise CPU.

### 3. Preprocessing & Augmentation (`benchmark_preprocessing.py`)

**Operations tested:**

*Basic:*
- Canny edge detection
- Gaussian blur
- Morphological operations
- Normalization

*Advanced:*
- Fast Fourier Transform (FFT)
- Gabor filters
- Local Binary Pattern (LBP)
- Multi-scale Retinex (MSR)
- HOG feature extraction

*Augmentation:*
- Light pipeline
- Medium pipeline
- Heavy pipeline

**Metrics:**
- Average processing time per image (ms)
- Standard deviation of processing time (ms)
- Throughput (images/second)

**Output:**
- `benchmark_preprocessing_results.csv` - Raw data
- `benchmark_preprocessing_results.png` - Comparison plots

**Expected runtime:** ~2-3 minutes

## Generated Files

After running benchmarks, you will find:

```
benchmarks/
├── benchmark_classical_results.csv
├── benchmark_classical_results.png
├── benchmark_deeplearning_results.csv
├── benchmark_deeplearning_results.png
├── benchmark_preprocessing_results.csv
├── benchmark_preprocessing_results.png
└── benchmark_report.md              # Comprehensive report
```

## Interpreting Results

### Classical Algorithms

**When to use:**
- Real-time applications (fast inference)
- Limited computational resources
- Interpretable results needed
- Small to medium datasets

**Performance characteristics:**
- ✅ **Fast training**: Most train in < 1 second
- ✅ **Fast inference**: Typically < 10ms per image
- ✅ **Low memory**: Usually < 100MB
- ⚠️ **Accuracy**: Good but may struggle with complex patterns

**Best performers:**
- **Speed**: IQR, MAD (instant training)
- **Accuracy**: ECOD, COPOD, IForest (AUC-ROC > 0.85)
- **Balance**: KNN with k=5 (fast + accurate)

### Deep Learning Algorithms

**When to use:**
- Complex visual anomalies
- Large datasets available
- Offline training acceptable
- High accuracy critical

**Performance characteristics:**
- ⚠️ **Slower training**: 10-100+ seconds per epoch
- ✅ **Reasonable inference**: Typically 10-50ms per image
- ⚠️ **Higher memory**: Models 10-100MB+
- ✅ **High accuracy**: Excels on complex patterns

**Best performers:**
- **Speed**: Autoencoder (faster than VAE)
- **Accuracy**: VAE, Deep SVDD (AUC-ROC > 0.90)
- **Size**: Autoencoder with small encoding_dim

### Preprocessing Operations

**When to use:**
- As preprocessing before detection
- Data augmentation during training
- Feature engineering

**Performance characteristics:**
- ✅ **Basic ops**: < 10ms per image (very fast)
- ⚠️ **Advanced ops**: 50-200ms per image (moderate)
- ⚠️ **Heavy augmentation**: 100-500ms per image (slow)

**Speed ranking (fastest to slowest):**
1. Normalize, Blur (< 5ms)
2. Edge detection, Morphology (5-15ms)
3. LBP, Gabor (20-50ms)
4. HOG, Retinex (50-100ms)
5. FFT (100-200ms)

## Customization

### Modify Test Parameters

Edit the benchmark scripts to customize:

```python
# In benchmark_classical.py
X_train, y_train, X_test, y_test = generate_synthetic_data(
    n_normal=1000,      # Increase for more training data
    n_anomaly=100,      # Increase anomaly ratio
    n_features=100,     # Feature dimensionality
)

# In benchmark_deeplearning.py
detector_params['epochs'] = 10  # Increase for better training

# In benchmark_preprocessing.py
n_iterations = 100  # Increase for more stable timing
```

### Add New Algorithms

To benchmark a new algorithm:

```python
# In benchmark_classical.py or benchmark_deeplearning.py

def benchmark_my_new_algorithm(X_train, y_train, X_test, y_test):
    """Benchmark my new algorithm."""
    result = benchmark_algorithm(
        MyNewDetector,
        {'param1': value1, 'param2': value2},
        X_train, y_train, X_test, y_test,
        "My Algorithm Name"
    )
    return [result]
```

## Hardware Considerations

### CPU vs GPU

Deep learning benchmarks automatically detect and use GPU if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Expected speedup with GPU:**
- Training: 3-10x faster
- Inference: 2-5x faster

### Memory Requirements

**Minimum requirements:**
- Classical: 2GB RAM
- Deep Learning (CPU): 4GB RAM
- Deep Learning (GPU): 2GB VRAM

### Parallel Processing

For multiple benchmarks:

```bash
# Run in parallel (Unix/Linux/Mac)
python benchmark_classical.py &
python benchmark_preprocessing.py &
wait

# Or use GNU parallel
parallel python ::: benchmark_*.py
```

## Troubleshooting

### Out of Memory

Reduce dataset size or batch size:

```python
# In benchmark scripts
X_train, y_train, X_test, y_test = generate_synthetic_data(
    n_normal=500,   # Reduced from 1000
    n_anomaly=50,   # Reduced from 100
)
```

### Slow Execution

Skip expensive operations or reduce iterations:

```python
# In benchmark_preprocessing.py
n_iterations = 20  # Reduced from 100
```

### Missing Dependencies

```bash
pip install scikit-learn torch torchvision pyod matplotlib psutil
```

## Continuous Integration

To run benchmarks in CI/CD:

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install psutil
      - name: Run benchmarks
        run: |
          cd benchmarks
          python run_all_benchmarks.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/*.csv
```

## Comparison with Other Libraries

### vs PyOD

PyImgAno focuses on **visual anomaly detection** with:
- ✅ Built-in image preprocessing
- ✅ Data augmentation pipelines
- ✅ Visual-specific algorithms

PyOD focuses on **general outlier detection** with:
- ✅ More classical algorithms (30+)
- ✅ Better documentation
- ✅ Larger community

**Performance comparison:**
- Training speed: Similar for shared algorithms (KNN, LOF, IForest)
- Inference speed: Similar for shared algorithms
- Memory usage: Similar for shared algorithms
- Preprocessing: PyImgAno has 80+ operations vs PyOD's minimal preprocessing

### vs Alibi-Detect

Alibi-Detect focuses on **production ML monitoring** with:
- ✅ Distribution drift detection
- ✅ Outlier detection
- ✅ Adversarial detection

PyImgAno focuses on **visual anomaly detection** with:
- ✅ Wider range of visual-specific algorithms
- ✅ Comprehensive preprocessing
- ✅ Industrial inspection use cases

## Contributing

To add new benchmarks:

1. Create a new benchmark script following the existing structure
2. Use the `BenchmarkResult` class for consistent output
3. Generate both CSV and visualization
4. Update this README
5. Submit a pull request

## References

- [PyOD Benchmarks](https://github.com/yzhao062/pyod/tree/master/notebooks)
- [Scikit-learn Performance Tips](https://scikit-learn.org/stable/developers/performance.html)
- [PyTorch Benchmarking](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{pyimgano2024,
  title={PyImgAno: Enterprise-Grade Visual Anomaly Detection Toolkit},
  author={PyImgAno Contributors},
  year={2024},
  url={https://github.com/jhlu2019/pyimgano}
}
```

## License

MIT License - See LICENSE file for details
