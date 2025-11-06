"""
Advanced preprocessing operations examples.

This example demonstrates advanced image processing techniques including:
1. Frequency domain operations (FFT, filters)
2. Texture analysis (Gabor, LBP, GLCM)
3. Color space transformations
4. Advanced enhancement (Gamma, Retinex)
5. Denoising techniques
6. Feature extraction (HOG, corners)
7. Advanced morphology
8. Image segmentation
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyimgano.preprocessing import AdvancedImageEnhancer


def create_test_image():
    """Create a test image with various features."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (100, 100), 30, (128, 128, 128), -1)

    # Add texture
    noise = np.random.randint(0, 50, (200, 200, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def example_frequency_domain():
    """Example 1: Frequency domain operations."""
    print("\n" + "="*60)
    print("Example 1: Frequency Domain Operations")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Fast Fourier Transform (FFT):")
    magnitude, phase = enhancer.apply_fft(img)
    print(f"   Magnitude shape: {magnitude.shape}")
    print(f"   Phase shape: {phase.shape}")

    print("\n2. Inverse FFT:")
    reconstructed = enhancer.apply_ifft(magnitude, phase)
    print(f"   Reconstructed shape: {reconstructed.shape}")

    print("\n3. Frequency Filters:")
    print("   a) Lowpass filter (remove high frequencies):")
    lowpass = enhancer.frequency_filter(img, filter_type='lowpass', cutoff_frequency=30)
    print(f"      Result shape: {lowpass.shape}")

    print("   b) Highpass filter (remove low frequencies):")
    highpass = enhancer.frequency_filter(img, filter_type='highpass', cutoff_frequency=30)
    print(f"      Result shape: {highpass.shape}")

    print("   c) Bandpass filter:")
    bandpass = enhancer.frequency_filter(img, filter_type='bandpass', cutoff_frequency=20)
    print(f"      Result shape: {bandpass.shape}")


def example_texture_analysis():
    """Example 2: Texture analysis."""
    print("\n" + "="*60)
    print("Example 2: Texture Analysis")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Gabor Filters (oriented texture detection):")
    print("   Testing different orientations:")
    for i, theta in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
        gabor = enhancer.gabor_filter(img, frequency=0.1, theta=theta)
        print(f"   Orientation {i+1} (theta={theta:.2f}): shape={gabor.shape}")

    print("\n2. Local Binary Pattern (LBP):")
    lbp = enhancer.compute_lbp(img, n_points=8, radius=1.0, method='uniform')
    print(f"   LBP shape: {lbp.shape}")
    print(f"   LBP range: [{lbp.min()}, {lbp.max()}]")

    print("\n3. Gray-Level Co-occurrence Matrix (GLCM) Features:")
    glcm_features = enhancer.compute_glcm(img)
    print("   GLCM texture features:")
    for feature_name, value in glcm_features.items():
        print(f"   - {feature_name}: {value:.4f}")


def example_color_space():
    """Example 3: Color space transformations."""
    print("\n" + "="*60)
    print("Example 3: Color Space Transformations")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Color Space Conversions:")
    color_spaces = ['hsv', 'lab', 'ycrcb', 'hls']

    for space in color_spaces:
        converted = enhancer.convert_color(img, from_space='bgr', to_space=space)
        print(f"   BGR to {space.upper()}: shape={converted.shape}")

    print("\n2. Color Histogram Equalization:")
    print("   a) HSV-based (equalize V channel):")
    eq_hsv = enhancer.equalize_color_hist(img, method='hsv')
    print(f"      Result shape: {eq_hsv.shape}")

    print("   b) LAB-based (equalize L channel):")
    eq_lab = enhancer.equalize_color_hist(img, method='lab')
    print(f"      Result shape: {eq_lab.shape}")

    print("   c) YCrCb-based (equalize Y channel):")
    eq_ycrcb = enhancer.equalize_color_hist(img, method='ycrcb')
    print(f"      Result shape: {eq_ycrcb.shape}")


def example_advanced_enhancement():
    """Example 4: Advanced enhancement techniques."""
    print("\n" + "="*60)
    print("Example 4: Advanced Enhancement Techniques")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Gamma Correction:")
    print("   a) Brighten (gamma < 1):")
    bright = enhancer.gamma_correct(img, gamma=0.5)
    print(f"      Result shape: {bright.shape}")

    print("   b) Darken (gamma > 1):")
    dark = enhancer.gamma_correct(img, gamma=2.0)
    print(f"      Result shape: {dark.shape}")

    print("\n2. Contrast Stretching:")
    stretched = enhancer.contrast_stretch(img, lower_percentile=2, upper_percentile=98)
    print(f"   Result shape: {stretched.shape}")

    print("\n3. Single-Scale Retinex (SSR):")
    ssr = enhancer.retinex_single(img, sigma=15.0)
    print(f"   SSR shape: {ssr.shape}")

    print("\n4. Multi-Scale Retinex (MSR) - illumination invariant:")
    msr = enhancer.retinex_multi(img, sigmas=[15, 80, 250])
    print(f"   MSR shape: {msr.shape}")


def example_denoising():
    """Example 5: Advanced denoising techniques."""
    print("\n" + "="*60)
    print("Example 5: Advanced Denoising")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    # Add noise
    noise = np.random.randint(-50, 50, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print(f"\nOriginal noise level: {np.std(noise):.2f}")

    print("\n1. Non-Local Means Denoising:")
    nlm = enhancer.nlm_denoise(noisy, h=10, template_window_size=7, search_window_size=21)
    print(f"   Denoised shape: {nlm.shape}")
    residual = noisy.astype(np.float32) - nlm.astype(np.float32)
    print(f"   Residual std: {np.std(residual):.2f}")

    print("\n2. Anisotropic Diffusion (edge-preserving smoothing):")
    diffused = enhancer.anisotropic_diffusion(noisy, niter=10, kappa=50, gamma=0.1)
    print(f"   Diffused shape: {diffused.shape}")


def example_feature_extraction():
    """Example 6: Feature extraction."""
    print("\n" + "="*60)
    print("Example 6: Feature Extraction")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Histogram of Oriented Gradients (HOG):")
    hog_features, hog_image = enhancer.extract_hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    print(f"   HOG features shape: {hog_features.shape}")
    print(f"   HOG visualization shape: {hog_image.shape}")
    print(f"   Number of features: {len(hog_features)}")

    print("\n2. Corner Detection:")
    print("   a) Harris corner detector:")
    harris = enhancer.detect_corners(img, method='harris', block_size=2, ksize=3, k=0.04)
    print(f"      Harris response shape: {harris.shape}")

    print("   b) Shi-Tomasi corner detector:")
    corners = enhancer.detect_corners(img, method='shi_tomasi', max_corners=100, quality_level=0.01)
    if corners is not None:
        print(f"      Found {len(corners)} corners")

    print("   c) FAST feature detector:")
    fast_corners = enhancer.detect_corners(img, method='fast', threshold=10)
    if fast_corners is not None:
        print(f"      Found {len(fast_corners)} FAST corners")


def example_advanced_morphology():
    """Example 7: Advanced morphological operations."""
    print("\n" + "="*60)
    print("Example 7: Advanced Morphological Operations")
    print("="*60)

    enhancer = AdvancedImageEnhancer()

    # Create binary image
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.circle(img, (100, 100), 30, 0, -1)  # Hole

    print("\nStarting with binary image with rectangle and hole")

    print("\n1. Skeletonization:")
    skeleton = enhancer.skeleton(img)
    print(f"   Skeleton shape: {skeleton.shape}")
    print(f"   Skeleton pixels: {np.sum(skeleton > 0)}")

    print("\n2. Thinning:")
    thinned = enhancer.thin(img)
    print(f"   Thinned shape: {thinned.shape}")
    print(f"   Thinned pixels: {np.sum(thinned > 0)}")

    print("\n3. Convex Hull:")
    hull = enhancer.convex_hull(img)
    print(f"   Hull shape: {hull.shape}")
    print(f"   Hull pixels: {np.sum(hull > 0)}")

    print("\n4. Distance Transform:")
    dist = enhancer.distance_transform(img)
    print(f"   Distance transform shape: {dist.shape}")
    print(f"   Max distance: {dist.max():.2f}")


def example_segmentation():
    """Example 8: Image segmentation."""
    print("\n" + "="*60)
    print("Example 8: Image Segmentation")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Thresholding Methods:")
    methods = ['otsu', 'adaptive_mean', 'adaptive_gaussian', 'triangle', 'yen', 'isodata']

    for method in methods:
        try:
            binary = enhancer.threshold(img, method=method)
            foreground_pixels = np.sum(binary == 255)
            print(f"   {method:20s}: foreground pixels = {foreground_pixels}")
        except Exception as e:
            print(f"   {method:20s}: Error - {e}")

    print("\n2. Watershed Segmentation:")
    segmented = enhancer.watershed(img)
    unique_labels = len(np.unique(segmented))
    print(f"   Number of segments: {unique_labels}")
    print(f"   Segmented shape: {segmented.shape}")


def example_pyramids():
    """Example 9: Image pyramids."""
    print("\n" + "="*60)
    print("Example 9: Image Pyramids")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\n1. Gaussian Pyramid:")
    gaussian_pyr = enhancer.build_gaussian_pyramid(img, levels=4)
    print(f"   Number of levels: {len(gaussian_pyr)}")
    for i, level in enumerate(gaussian_pyr):
        print(f"   Level {i}: shape={level.shape}")

    print("\n2. Laplacian Pyramid:")
    laplacian_pyr = enhancer.build_laplacian_pyramid(img, levels=4)
    print(f"   Number of levels: {len(laplacian_pyr)}")
    for i, level in enumerate(laplacian_pyr):
        print(f"   Level {i}: shape={level.shape}")


def example_complete_workflow():
    """Example 10: Complete preprocessing workflow."""
    print("\n" + "="*60)
    print("Example 10: Complete Preprocessing Workflow")
    print("="*60)

    enhancer = AdvancedImageEnhancer()
    img = create_test_image()

    print("\nComplete workflow for anomaly detection preprocessing:")

    print("\n1. Color space conversion (BGR to LAB):")
    lab = enhancer.convert_color(img, from_space='bgr', to_space='lab')
    print(f"   LAB shape: {lab.shape}")

    print("\n2. Contrast enhancement (Retinex):")
    enhanced = enhancer.retinex_multi(lab, sigmas=[15, 80, 250])
    print(f"   Enhanced shape: {enhanced.shape}")

    print("\n3. Denoising (Non-Local Means):")
    denoised = enhancer.nlm_denoise(enhanced, h=10)
    print(f"   Denoised shape: {denoised.shape}")

    print("\n4. Edge detection (Canny):")
    edges = enhancer.detect_edges(denoised, method='canny')
    print(f"   Edges shape: {edges.shape}")

    print("\n5. Texture analysis (LBP):")
    lbp = enhancer.compute_lbp(denoised, n_points=8, radius=1.0)
    print(f"   LBP shape: {lbp.shape}")

    print("\n6. Normalization:")
    normalized = enhancer.normalize(lbp, method='minmax')
    print(f"   Normalized shape: {normalized.shape}")
    print(f"   Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    print("\n7. Feature extraction (HOG):")
    hog_features = enhancer.extract_hog(normalized, visualize=False)
    print(f"   HOG features: {len(hog_features)} dimensions")

    print("\nWorkflow complete! Ready for anomaly detection.")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PyImgAno Advanced Preprocessing Examples")
    print("="*60)

    try:
        example_frequency_domain()
    except Exception as e:
        print(f"\nError in frequency domain example: {e}")

    try:
        example_texture_analysis()
    except Exception as e:
        print(f"\nError in texture analysis example: {e}")

    try:
        example_color_space()
    except Exception as e:
        print(f"\nError in color space example: {e}")

    try:
        example_advanced_enhancement()
    except Exception as e:
        print(f"\nError in advanced enhancement example: {e}")

    try:
        example_denoising()
    except Exception as e:
        print(f"\nError in denoising example: {e}")

    try:
        example_feature_extraction()
    except Exception as e:
        print(f"\nError in feature extraction example: {e}")

    try:
        example_advanced_morphology()
    except Exception as e:
        print(f"\nError in advanced morphology example: {e}")

    try:
        example_segmentation()
    except Exception as e:
        print(f"\nError in segmentation example: {e}")

    try:
        example_pyramids()
    except Exception as e:
        print(f"\nError in pyramids example: {e}")

    try:
        example_complete_workflow()
    except Exception as e:
        print(f"\nError in complete workflow example: {e}")

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
