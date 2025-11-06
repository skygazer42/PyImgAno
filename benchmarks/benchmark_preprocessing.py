"""
Benchmark preprocessing and augmentation operations.

This script benchmarks image preprocessing and augmentation operations:
- Basic preprocessing (resize, normalize, filter)
- Advanced operations (FFT, texture analysis, enhancement)
- Augmentation operations (geometric, color, noise, blur)
- Pipeline operations

Metrics measured:
- Processing time (per image)
- Memory overhead
- Throughput (images/second)
"""

import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyimgano.preprocessing import (
    ImageEnhancer,
    AdvancedImageEnhancer,
    AugmentationPipeline,
    get_light_augmentation,
    get_medium_augmentation,
    get_heavy_augmentation,
)

warnings.filterwarnings('ignore')


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.avg_time = 0.0
        self.std_time = 0.0
        self.throughput = 0.0
        self.error = None

    def __repr__(self):
        if self.error:
            return f"{self.operation_name}: ERROR - {self.error}"
        return (f"{self.operation_name}: "
                f"Time={self.avg_time*1000:.2f}±{self.std_time*1000:.2f}ms, "
                f"Throughput={self.throughput:.1f} img/s")


def create_test_image(size=(256, 256)):
    """Create a test image with various features."""
    img = np.zeros((*size, 3), dtype=np.uint8)

    # Add shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (128, 128), 40, (128, 128, 128), -1)

    # Add texture
    noise = np.random.randint(0, 30, (*size, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def benchmark_operation(
    operation_func,
    test_image: np.ndarray,
    n_iterations: int = 100,
    operation_name: str = "Operation"
) -> BenchmarkResult:
    """Benchmark a single operation."""
    result = BenchmarkResult(operation_name)

    try:
        times = []

        # Warmup
        for _ in range(5):
            _ = operation_func(test_image.copy())

        # Benchmark
        for _ in range(n_iterations):
            img_copy = test_image.copy()
            start_time = time.time()
            _ = operation_func(img_copy)
            elapsed = time.time() - start_time
            times.append(elapsed)

        result.avg_time = np.mean(times)
        result.std_time = np.std(times)
        result.throughput = 1.0 / result.avg_time if result.avg_time > 0 else 0

    except Exception as e:
        result.error = str(e)

    return result


def benchmark_basic_preprocessing(test_image: np.ndarray) -> List[BenchmarkResult]:
    """Benchmark basic preprocessing operations."""
    print("\n" + "="*60)
    print("Benchmarking Basic Preprocessing Operations")
    print("="*60)

    enhancer = ImageEnhancer()
    results = []

    # Edge detection
    print("\n1. Canny edge detection...")
    result = benchmark_operation(
        lambda img: enhancer.detect_edges(img, method='canny'),
        test_image,
        operation_name="Canny Edge"
    )
    results.append(result)
    print(f"   {result}")

    # Gaussian blur
    print("\n2. Gaussian blur...")
    result = benchmark_operation(
        lambda img: enhancer.apply_filter(img, filter_type='gaussian', kernel_size=5),
        test_image,
        operation_name="Gaussian Blur"
    )
    results.append(result)
    print(f"   {result}")

    # Morphological operation
    print("\n3. Morphological dilation...")
    result = benchmark_operation(
        lambda img: enhancer.morph_operation(img, operation='dilate', kernel_size=5),
        test_image,
        operation_name="Morphology"
    )
    results.append(result)
    print(f"   {result}")

    # Normalization
    print("\n4. Min-max normalization...")
    result = benchmark_operation(
        lambda img: enhancer.normalize(img, method='minmax'),
        test_image,
        operation_name="Normalize"
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_advanced_operations(test_image: np.ndarray) -> List[BenchmarkResult]:
    """Benchmark advanced preprocessing operations."""
    print("\n" + "="*60)
    print("Benchmarking Advanced Operations")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    results = []

    # FFT
    print("\n1. Fast Fourier Transform...")
    result = benchmark_operation(
        lambda img: enhancer.apply_fft(img),
        test_image,
        n_iterations=50,
        operation_name="FFT"
    )
    results.append(result)
    print(f"   {result}")

    # Gabor filter
    print("\n2. Gabor filter...")
    result = benchmark_operation(
        lambda img: enhancer.gabor_filter(img, frequency=0.1),
        test_image,
        operation_name="Gabor"
    )
    results.append(result)
    print(f"   {result}")

    # LBP
    print("\n3. Local Binary Pattern...")
    result = benchmark_operation(
        lambda img: enhancer.compute_lbp(img),
        test_image,
        operation_name="LBP"
    )
    results.append(result)
    print(f"   {result}")

    # Retinex
    print("\n4. Multi-scale Retinex...")
    result = benchmark_operation(
        lambda img: enhancer.retinex_multi(img),
        test_image,
        n_iterations=20,
        operation_name="Retinex MSR"
    )
    results.append(result)
    print(f"   {result}")

    # HOG features
    print("\n5. HOG feature extraction...")
    result = benchmark_operation(
        lambda img: enhancer.extract_hog(img, visualize=False),
        test_image,
        n_iterations=50,
        operation_name="HOG"
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_augmentation_operations(test_image: np.ndarray) -> List[BenchmarkResult]:
    """Benchmark augmentation operations."""
    print("\n" + "="*60)
    print("Benchmarking Augmentation Operations")
    print("="*60)

    results = []

    # Light augmentation
    print("\n1. Light augmentation pipeline...")
    light_aug = get_light_augmentation()
    result = benchmark_operation(
        lambda img: light_aug(img),
        test_image,
        operation_name="Light Aug"
    )
    results.append(result)
    print(f"   {result}")

    # Medium augmentation
    print("\n2. Medium augmentation pipeline...")
    medium_aug = get_medium_augmentation()
    result = benchmark_operation(
        lambda img: medium_aug(img),
        test_image,
        operation_name="Medium Aug"
    )
    results.append(result)
    print(f"   {result}")

    # Heavy augmentation
    print("\n3. Heavy augmentation pipeline...")
    heavy_aug = get_heavy_augmentation()
    result = benchmark_operation(
        lambda img: heavy_aug(img),
        test_image,
        n_iterations=50,
        operation_name="Heavy Aug"
    )
    results.append(result)
    print(f"   {result}")

    return results


def benchmark_image_sizes(operation_func, operation_name: str) -> List[BenchmarkResult]:
    """Benchmark operation across different image sizes."""
    print(f"\n{operation_name} - Image Size Scaling:")
    results = []

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    for size in sizes:
        test_img = create_test_image(size=size)
        result = benchmark_operation(
            operation_func,
            test_img,
            n_iterations=50,
            operation_name=f"{operation_name} {size[0]}x{size[1]}"
        )
        results.append(result)
        print(f"   {size[0]}x{size[1]}: {result.avg_time*1000:.2f}ms, {result.throughput:.1f} img/s")

    return results


def plot_benchmark_results(
    basic_results: List[BenchmarkResult],
    advanced_results: List[BenchmarkResult],
    aug_results: List[BenchmarkResult],
    output_path: str = "benchmark_preprocessing_results.png"
):
    """Plot benchmark results."""
    # Filter valid results
    all_results = basic_results + advanced_results + aug_results
    valid_results = [r for r in all_results if r.error is None]

    if not valid_results:
        print("No valid results to plot")
        return

    # Split by category
    basic_valid = [r for r in basic_results if r.error is None]
    advanced_valid = [r for r in advanced_results if r.error is None]
    aug_valid = [r for r in aug_results if r.error is None]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Preprocessing & Augmentation Benchmark Results', fontsize=16)

    # Basic operations
    if basic_valid:
        names = [r.operation_name for r in basic_valid]
        times = [r.avg_time * 1000 for r in basic_valid]
        axes[0].barh(names, times, color='skyblue')
        axes[0].set_xlabel('Processing Time (ms)')
        axes[0].set_title('Basic Preprocessing')
        axes[0].grid(axis='x', alpha=0.3)

    # Advanced operations
    if advanced_valid:
        names = [r.operation_name for r in advanced_valid]
        times = [r.avg_time * 1000 for r in advanced_valid]
        axes[1].barh(names, times, color='lightgreen')
        axes[1].set_xlabel('Processing Time (ms)')
        axes[1].set_title('Advanced Operations')
        axes[1].grid(axis='x', alpha=0.3)

    # Augmentation operations
    if aug_valid:
        names = [r.operation_name for r in aug_valid]
        times = [r.avg_time * 1000 for r in aug_valid]
        axes[2].barh(names, times, color='lightcoral')
        axes[2].set_xlabel('Processing Time (ms)')
        axes[2].set_title('Augmentation Pipelines')
        axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBenchmark plot saved to {output_path}")


def save_results_csv(
    all_results: List[BenchmarkResult],
    output_path: str = "benchmark_preprocessing_results.csv"
):
    """Save benchmark results to CSV."""
    with open(output_path, 'w') as f:
        f.write("Operation,Avg_Time(ms),Std_Time(ms),Throughput(img/s),Error\n")
        for result in all_results:
            if result.error:
                f.write(f"{result.operation_name},,,, {result.error}\n")
            else:
                f.write(f"{result.operation_name},{result.avg_time*1000:.4f},"
                       f"{result.std_time*1000:.4f},{result.throughput:.2f},\n")
    print(f"Results saved to {output_path}")


def main():
    """Run all preprocessing benchmarks."""
    print("\n" + "="*60)
    print("PyImgAno Preprocessing & Augmentation Benchmark")
    print("="*60)

    # Create test image
    test_image = create_test_image(size=(256, 256))
    print(f"\nTest image size: {test_image.shape}")

    # Run benchmarks
    basic_results = benchmark_basic_preprocessing(test_image)
    advanced_results = benchmark_advanced_operations(test_image)
    aug_results = benchmark_augmentation_operations(test_image)

    # Benchmark image size scaling
    print("\n" + "="*60)
    print("Image Size Scaling Analysis")
    print("="*60)

    enhancer = ImageEnhancer()
    size_results = benchmark_image_sizes(
        lambda img: enhancer.detect_edges(img, method='canny'),
        "Canny Edge Detection"
    )

    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)

    all_results = basic_results + advanced_results + aug_results
    valid_results = [r for r in all_results if r.error is None]

    if valid_results:
        fastest = min(valid_results, key=lambda r: r.avg_time)
        slowest = max(valid_results, key=lambda r: r.avg_time)
        highest_throughput = max(valid_results, key=lambda r: r.throughput)

        print(f"\n🏆 Fastest Operation: {fastest.operation_name} ({fastest.avg_time*1000:.2f}ms)")
        print(f"⏱️  Slowest Operation: {slowest.operation_name} ({slowest.avg_time*1000:.2f}ms)")
        print(f"🚀 Highest Throughput: {highest_throughput.operation_name} "
              f"({highest_throughput.throughput:.1f} img/s)")

    # Save results
    save_results_csv(all_results + size_results)
    plot_benchmark_results(basic_results, advanced_results, aug_results)

    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
