"""
Benchmark deep learning anomaly detection algorithms.

This script benchmarks neural network-based algorithms for visual anomaly detection:
- Autoencoder (AE)
- Variational Autoencoder (VAE)
- Deep SVDD

Metrics measured:
- Training time (per epoch)
- Inference time (per image)
- Memory usage (GPU/CPU)
- Detection accuracy (AUC-ROC)
- Model size
"""

import os
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyimgano.detectors import (
    AutoencoderDetector,
    VAEDetector,
    DeepSVDDDetector,
)

warnings.filterwarnings('ignore')


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.train_time_per_epoch = 0.0
        self.total_train_time = 0.0
        self.inference_time = 0.0
        self.memory_usage = 0.0
        self.model_size_mb = 0.0
        self.auc_roc = 0.0
        self.error = None

    def __repr__(self):
        if self.error:
            return f"{self.algorithm_name}: ERROR - {self.error}"
        return (f"{self.algorithm_name}: "
                f"Train={self.total_train_time:.1f}s ({self.train_time_per_epoch:.2f}s/epoch), "
                f"Inference={self.inference_time*1000:.2f}ms/img, "
                f"Model={self.model_size_mb:.1f}MB, "
                f"AUC-ROC={self.auc_roc:.4f}")


def generate_image_data(
    n_normal: int = 500,
    n_anomaly: int = 50,
    image_size: Tuple[int, int] = (64, 64),
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic image data for benchmarking."""
    np.random.seed(random_state)

    # Normal images: Simple patterns
    normal_images = []
    for _ in range(n_normal):
        img = np.random.randn(1, *image_size) * 0.1
        # Add structured patterns
        img[:, ::4, :] += 0.5
        img[:, :, ::4] += 0.5
        normal_images.append(img)

    # Anomalous images: Different patterns
    anomaly_images = []
    for _ in range(n_anomaly):
        img = np.random.randn(1, *image_size) * 0.3
        # Add random structures
        img[:, ::3, :] += 1.0
        img[:, :, ::5] -= 1.0
        anomaly_images.append(img)

    # Prepare train/test splits
    X_train = np.array(normal_images[:int(0.8*n_normal)])
    y_train = np.zeros(len(X_train))

    X_test = np.array(normal_images[int(0.8*n_normal):] + anomaly_images)
    y_test = np.hstack([
        np.zeros(n_normal - int(0.8*n_normal)),
        np.ones(n_anomaly)
    ])

    return X_train, y_train, X_test, y_test


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def benchmark_algorithm(
    detector_class,
    detector_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    algorithm_name: str,
    epochs: int = 10
) -> BenchmarkResult:
    """Benchmark a single deep learning algorithm."""
    result = BenchmarkResult(algorithm_name)

    try:
        # Initialize detector
        detector_params['epochs'] = epochs
        detector = detector_class(**detector_params)

        # Measure training time
        start_time = time.time()
        detector.fit(X_train)
        total_train_time = time.time() - start_time

        result.total_train_time = total_train_time
        result.train_time_per_epoch = total_train_time / epochs

        # Get model size
        if hasattr(detector, 'model'):
            result.model_size_mb = get_model_size(detector.model)

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

        # Memory usage (approximate)
        if torch.cuda.is_available():
            result.memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            result.memory_usage = result.model_size_mb

    except Exception as e:
        result.error = str(e)

    return result


def benchmark_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> List[BenchmarkResult]:
    """Benchmark Autoencoder-based methods."""
    print("\n" + "="*60)
    print("Benchmarking Autoencoder Methods")
    print("="*60)

    results = []
    input_dim = np.prod(X_train.shape[1:])

    # Standard Autoencoder
    print("\n1. Standard Autoencoder...")
    result = benchmark_algorithm(
        AutoencoderDetector,
        {
            'input_dim': input_dim,
            'encoding_dim': 32,
            'hidden_dims': [128, 64],
            'batch_size': 32,
            'learning_rate': 0.001,
        },
        X_train, y_train, X_test, y_test,
        "Autoencoder",
        epochs=10
    )
    results.append(result)
    print(f"   {result}")

    # Variational Autoencoder
    print("\n2. Variational Autoencoder (VAE)...")
    result = benchmark_algorithm(
        VAEDetector,
        {
            'input_dim': input_dim,
            'latent_dim': 32,
            'hidden_dims': [128, 64],
            'batch_size': 32,
            'learning_rate': 0.001,
        },
        X_train, y_train, X_test, y_test,
        "VAE",
        epochs=10
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_deep_svdd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> List[BenchmarkResult]:
    """Benchmark Deep SVDD method."""
    print("\n" + "="*60)
    print("Benchmarking Deep SVDD")
    print("="*60)

    results = []
    input_dim = np.prod(X_train.shape[1:])

    print("\n1. Deep SVDD...")
    result = benchmark_algorithm(
        DeepSVDDDetector,
        {
            'input_dim': input_dim,
            'hidden_dims': [128, 64, 32],
            'batch_size': 32,
            'learning_rate': 0.001,
        },
        X_train, y_train, X_test, y_test,
        "Deep SVDD",
        epochs=10
    )
    results.append(result)
    print(f"   {result}")

    return results


def plot_benchmark_results(
    all_results: List[BenchmarkResult],
    output_path: str = "benchmark_deeplearning_results.png"
):
    """Plot benchmark results."""
    # Filter out error results
    valid_results = [r for r in all_results if r.error is None]

    if not valid_results:
        print("No valid results to plot")
        return

    algorithms = [r.algorithm_name for r in valid_results]
    train_times = [r.train_time_per_epoch for r in valid_results]
    inference_times = [r.inference_time * 1000 for r in valid_results]  # Convert to ms
    model_sizes = [r.model_size_mb for r in valid_results]
    auc_rocs = [r.auc_roc for r in valid_results]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Deep Learning Algorithms Benchmark Results', fontsize=16)

    # Training time per epoch
    axes[0, 0].barh(algorithms, train_times, color='skyblue')
    axes[0, 0].set_xlabel('Training Time per Epoch (seconds)')
    axes[0, 0].set_title('Training Time Comparison')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Inference time
    axes[0, 1].barh(algorithms, inference_times, color='lightgreen')
    axes[0, 1].set_xlabel('Inference Time (ms per image)')
    axes[0, 1].set_title('Inference Time Comparison')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Model size
    axes[1, 0].barh(algorithms, model_sizes, color='lightcoral')
    axes[1, 0].set_xlabel('Model Size (MB)')
    axes[1, 0].set_title('Model Size Comparison')
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


def save_results_csv(
    all_results: List[BenchmarkResult],
    output_path: str = "benchmark_deeplearning_results.csv"
):
    """Save benchmark results to CSV."""
    with open(output_path, 'w') as f:
        f.write("Algorithm,Train_Time_Per_Epoch(s),Total_Train_Time(s),"
               "Inference_Time(s),Model_Size(MB),AUC_ROC,Error\n")
        for result in all_results:
            if result.error:
                f.write(f"{result.algorithm_name},,,,,,{result.error}\n")
            else:
                f.write(f"{result.algorithm_name},{result.train_time_per_epoch:.4f},"
                       f"{result.total_train_time:.2f},{result.inference_time:.6f},"
                       f"{result.model_size_mb:.2f},{result.auc_roc:.4f},\n")
    print(f"Results saved to {output_path}")


def main():
    """Run all deep learning algorithm benchmarks."""
    print("\n" + "="*60)
    print("PyImgAno Deep Learning Algorithms Benchmark")
    print("="*60)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Generate synthetic image data
    print("\nGenerating synthetic image dataset...")
    X_train, y_train, X_test, y_test = generate_image_data(
        n_normal=500,
        n_anomaly=50,
        image_size=(64, 64)
    )
    print(f"Training set: {X_train.shape[0]} samples, shape: {X_train.shape}")
    print(f"Test set: {X_test.shape[0]} samples ({np.sum(y_test)} anomalies)")

    # Flatten images for detectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Run benchmarks
    all_results = []

    all_results.extend(benchmark_autoencoder(X_train_flat, y_train, X_test_flat, y_test))
    all_results.extend(benchmark_deep_svdd(X_train_flat, y_train, X_test_flat, y_test))

    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)

    valid_results = [r for r in all_results if r.error is None]
    if valid_results:
        # Best by metric
        best_train = min(valid_results, key=lambda r: r.train_time_per_epoch)
        best_inference = min(valid_results, key=lambda r: r.inference_time)
        best_size = min(valid_results, key=lambda r: r.model_size_mb)
        best_accuracy = max(valid_results, key=lambda r: r.auc_roc)

        print(f"\n🏆 Fastest Training: {best_train.algorithm_name} "
              f"({best_train.train_time_per_epoch:.2f}s/epoch)")
        print(f"🏆 Fastest Inference: {best_inference.algorithm_name} "
              f"({best_inference.inference_time*1000:.2f}ms)")
        print(f"🏆 Smallest Model: {best_size.algorithm_name} ({best_size.model_size_mb:.1f}MB)")
        print(f"🏆 Best Accuracy: {best_accuracy.algorithm_name} "
              f"(AUC-ROC: {best_accuracy.auc_roc:.4f})")

    # Save results
    save_results_csv(all_results)
    plot_benchmark_results(all_results)

    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
