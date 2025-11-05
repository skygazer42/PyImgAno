"""
Pytest configuration and fixtures for PyImgAno tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image for testing."""
    # Create a simple 100x100 RGB image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_image_path(temp_dir: Path, sample_image: np.ndarray) -> Path:
    """Create and save a sample image, return its path."""
    image_path = temp_dir / "test_image.jpg"
    pil_image = Image.fromarray(sample_image)
    pil_image.save(image_path)
    return image_path


@pytest.fixture
def sample_image_paths(temp_dir: Path) -> list[Path]:
    """Create multiple sample images and return their paths."""
    paths = []
    for i in range(5):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = temp_dir / f"test_image_{i}.jpg"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        paths.append(image_path)
    return paths


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a sample grayscale image for testing."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    return image


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    # Add torch seed if torch is available
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    except ImportError:
        pass


@pytest.fixture
def mock_dataset_paths(temp_dir: Path) -> dict[str, list[Path]]:
    """Create a mock dataset structure with train and test images."""
    train_dir = temp_dir / "train"
    test_dir = temp_dir / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    train_paths = []
    test_paths = []

    # Create training images
    for i in range(10):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = train_dir / f"train_{i}.jpg"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        train_paths.append(image_path)

    # Create test images
    for i in range(5):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = test_dir / f"test_{i}.jpg"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        test_paths.append(image_path)

    return {"train": train_paths, "test": test_paths}


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "requires_torch: marks tests that require PyTorch")


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if CUDA is not available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "requires_gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
