"""
Benchmark classical anomaly detection algorithms.

This script benchmarks statistical and classical machine learning algorithms
for visual anomaly detection, including:
- Statistical methods (IQR, MAD, Histogram)
- Distance-based methods (KNN, LOF)
- Density-based methods (ECOD, COPOD)
- Isolation-based methods (IForest)
- Ensemble methods

Metrics measured:
- Training time
- Inference time (per image)
- Memory usage
- Detection accuracy (AUC-ROC)
"""

import os
import sys
import time
import psutil
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyimgano.detectors import (
    IQRDetector,
    MADDetector,
    HistogramBasedDetector,
    KNNDetector,
    LOFDetector,
    IsolationForestDetector,
    ECODDetector,
    COPODDetector,
)

warnings.filterwarnings('ignore')


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.train_time = 0.0
        self.inference_time = 0.0
        self.memory_usage = 0.0
        self.auc_roc = 0.0
        self.error = None

    def __repr__(self):
        if self.error:
            return f"{self.algorithm_name}: ERROR - {self.error}"
        return (f"{self.algorithm_name}: "
                f"Train={self.train_time:.3f}s, "
                f"Inference={self.inference_time:.4f}s/img, "
                f"Memory={self.memory_usage:.1f}MB, "
                f"AUC-ROC={self.auc_roc:.4f}")


def generate_synthetic_data(
    n_normal: int = 1000,
    n_anomaly: int = 100,
    n_features: int = 100,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic anomaly detection dataset."""
    np.random.seed(random_state)

    # Normal samples: multivariate Gaussian
    normal_data = np.random.randn(n_normal, n_features) * 0.5

    # Anomalous samples: outliers with larger variance
    anomaly_data = np.random.randn(n_anomaly, n_features) * 2.0
    anomaly_data += np.random.uniform(-3, 3, size=(n_anomaly, n_features))

    # Combine data
    X_train = normal_data
    y_train = np.zeros(n_normal)

    X_test = np.vstack([normal_data[:n_normal//5], anomaly_data])
    y_test = np.hstack([np.zeros(n_normal//5), np.ones(n_anomaly)])

    return X_train, y_train, X_test, y_test


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_algorithm(
    detector_class,
    detector_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    algorithm_name: str
) -> BenchmarkResult:
    """Benchmark a single algorithm."""
    result = BenchmarkResult(algorithm_name)

    try:
        # Initialize detector
        detector = detector_class(**detector_params)

        # Measure training time and memory
        mem_before = get_memory_usage()
        start_time = time.time()
        detector.fit(X_train)
        train_time = time.time() - start_time
        mem_after = get_memory_usage()

        result.train_time = train_time
        result.memory_usage = mem_after - mem_before

        # Measure inference time
        start_time = time.time()
        scores = detector.predict_proba(X_test)
        inference_time = (time.time() - start_time) / len(X_test)

        result.inference_time = inference_time

        # Calculate AUC-ROC
        if len(np.unique(y_test)) > 1:
            result.auc_roc = roc_auc_score(y_test, scores)
        else:
            result.auc_roc = 0.0

    except Exception as e:
        result.error = str(e)

    return result


def benchmark_statistical_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> List[BenchmarkResult]:
    """Benchmark statistical anomaly detection methods."""
    print("\n" + "="*60)
    print("Benchmarking Statistical Methods")
    print("="*60)

    results = []

    # IQR Detector
    print("\n1. IQR Detector...")
    result = benchmark_algorithm(
        IQRDetector,
        {},
        X_train, y_train, X_test, y_test,
        "IQR"
    )
    results.append(result)
    print(f"   {result}")

    # MAD Detector
    print("\n2. MAD Detector...")
    result = benchmark_algorithm(
        MADDetector,
        {},
        X_train, y_train, X_test, y_test,
        "MAD"
    )
    results.append(result)
    print(f"   {result}")

    # Histogram-based Detector
    print("\n3. Histogram-based Detector...")
    result = benchmark_algorithm(
        HistogramBasedDetector,
        {'bins': 50},
        X_train, y_train, X_test, y_test,
        "Histogram"
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_distance_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> List[BenchmarkResult]:
    """Benchmark distance-based anomaly detection methods."""
    print("\n" + "="*60)
    print("Benchmarking Distance-Based Methods")
    print("="*60)

    results = []

    # KNN Detector
    print("\n1. KNN Detector (k=5)...")
    result = benchmark_algorithm(
        KNNDetector,
        {'n_neighbors': 5},
        X_train, y_train, X_test, y_test,
        "KNN"
    )
    results.append(result)
    print(f"   {result}")

    # LOF Detector
    print("\n2. LOF Detector (k=20)...")
    result = benchmark_algorithm(
        LOFDetector,
        {'n_neighbors': 20},
        X_train, y_train, X_test, y_test,
        "LOF"
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_density_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> List[BenchmarkResult]:
    """Benchmark density-based anomaly detection methods."""
    print("\n" + "="*60)
    print("Benchmarking Density-Based Methods")
    print("="*60)

    results = []

    # ECOD Detector
    print("\n1. ECOD Detector...")
    result = benchmark_algorithm(
        ECODDetector,
        {},
        X_train, y_train, X_test, y_test,
        "ECOD"
    )
    results.append(result)
    print(f"   {result}")

    # COPOD Detector
    print("\n2. COPOD Detector...")
    result = benchmark_algorithm(
        COPODDetector,
        {},
        X_train, y_train, X_test, y_test,
        "COPOD"
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_isolation_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> List[BenchmarkResult]:
    """Benchmark isolation-based anomaly detection methods."""
    print("\n" + "="*60)
    print("Benchmarking Isolation-Based Methods")
    print("="*60)

    results = []

    # Isolation Forest
    print("\n1. Isolation Forest (100 trees)...")
    result = benchmark_algorithm(
        IsolationForestDetector,
        {'n_estimators': 100},
        X_train, y_train, X_test, y_test,
        "IForest"
    )
    results.append(result)
    print(f"   {result}")

    return results


def plot_benchmark_results(all_results: List[BenchmarkResult], output_path: str = "benchmark_classical_results.png"):
    """Plot benchmark results."""
    # Filter out error results
    valid_results = [r for r in all_results if r.error is None]

    if not valid_results:
        print("No valid results to plot")
        return

    algorithms = [r.algorithm_name for r in valid_results]
    train_times = [r.train_time for r in valid_results]
    inference_times = [r.inference_time * 1000 for r in valid_results]  # Convert to ms
    memory_usages = [r.memory_usage for r in valid_results]
    auc_rocs = [r.auc_roc for r in valid_results]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Classical Algorithms Benchmark Results', fontsize=16)

    # Training time
    axes[0, 0].barh(algorithms, train_times, color='skyblue')
    axes[0, 0].set_xlabel('Training Time (seconds)')
    axes[0, 0].set_title('Training Time Comparison')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Inference time
    axes[0, 1].barh(algorithms, inference_times, color='lightgreen')
    axes[0, 1].set_xlabel('Inference Time (ms per image)')
    axes[0, 1].set_title('Inference Time Comparison')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Memory usage
    axes[1, 0].barh(algorithms, memory_usages, color='lightcoral')
    axes[1, 0].set_xlabel('Memory Usage (MB)')
    axes[1, 0].set_title('Memory Usage Comparison')
    axes[1, 0].grid(axis='x', alpha=0.3)

    # AUC-ROC
    axes[1, 1].barh(algorithms, auc_rocs, color='plum')
    axes[1, 1].set_xlabel('AUC-ROC Score')
    axes[1, 1].set_title('Detection Accuracy Comparison')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBenchmark plot saved to {output_path}")


def save_results_csv(all_results: List[BenchmarkResult], output_path: str = "benchmark_classical_results.csv"):
    """Save benchmark results to CSV."""
    with open(output_path, 'w') as f:
        f.write("Algorithm,Train_Time(s),Inference_Time(s),Memory_Usage(MB),AUC_ROC,Error\n")
        for result in all_results:
            if result.error:
                f.write(f"{result.algorithm_name},,,,,{result.error}\n")
            else:
                f.write(f"{result.algorithm_name},{result.train_time:.4f},"
                       f"{result.inference_time:.6f},{result.memory_usage:.2f},"
                       f"{result.auc_roc:.4f},\n")
    print(f"Results saved to {output_path}")


def main():
    """Run all classical algorithm benchmarks."""
    print("\n" + "="*60)
    print("PyImgAno Classical Algorithms Benchmark")
    print("="*60)

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_normal=1000,
        n_anomaly=100,
        n_features=100
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples ({np.sum(y_test)} anomalies)")

    # Run benchmarks
    all_results = []

    all_results.extend(benchmark_statistical_methods(X_train, y_train, X_test, y_test))
    all_results.extend(benchmark_distance_methods(X_train, y_train, X_test, y_test))
    all_results.extend(benchmark_density_methods(X_train, y_train, X_test, y_test))
    all_results.extend(benchmark_isolation_methods(X_train, y_train, X_test, y_test))

    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)

    valid_results = [r for r in all_results if r.error is None]
    if valid_results:
        # Best by metric
        best_train = min(valid_results, key=lambda r: r.train_time)
        best_inference = min(valid_results, key=lambda r: r.inference_time)
        best_memory = min(valid_results, key=lambda r: r.memory_usage)
        best_accuracy = max(valid_results, key=lambda r: r.auc_roc)

        print(f"\n🏆 Fastest Training: {best_train.algorithm_name} ({best_train.train_time:.3f}s)")
        print(f"🏆 Fastest Inference: {best_inference.algorithm_name} ({best_inference.inference_time*1000:.2f}ms)")
        print(f"🏆 Lowest Memory: {best_memory.algorithm_name} ({best_memory.memory_usage:.1f}MB)")
        print(f"🏆 Best Accuracy: {best_accuracy.algorithm_name} (AUC-ROC: {best_accuracy.auc_roc:.4f})")

    # Save results
    save_results_csv(all_results)
    plot_benchmark_results(all_results)

    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
