"""
Tests for PyOD-based anomaly detection models.
"""

import numpy as np
import pytest

from pyimgano import models


class MockFeatureExtractor:
    """Mock feature extractor for testing."""

    def __init__(self, n_features=50):
        self.n_features = n_features

    def extract(self, X):
        """Return mock features."""
        n_samples = len(list(X)) if hasattr(X, '__len__') else 10
        np.random.seed(42)
        return np.random.rand(n_samples, self.n_features)


@pytest.fixture
def feature_extractor():
    """Fixture for mock feature extractor."""
    return MockFeatureExtractor()


@pytest.fixture
def mock_image_paths():
    """Fixture for mock image paths."""
    train_paths = [f"train_{i}.jpg" for i in range(100)]
    test_paths = [f"test_{i}.jpg" for i in range(20)]
    return train_paths, test_paths


class TestECOD:
    """Tests for ECOD detector."""

    def test_create(self, feature_extractor):
        """Test ECOD creation."""
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor,
            contamination=0.1
        )
        assert detector is not None
        assert hasattr(detector, 'fit')
        assert hasattr(detector, 'predict')

    def test_fit_predict(self, feature_extractor, mock_image_paths):
        """Test ECOD fit and predict."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor,
            contamination=0.1
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)

        assert predictions.shape == (len(test_paths),)
        assert set(predictions).issubset({0, 1})
        assert hasattr(detector, 'decision_scores_')
        assert hasattr(detector, 'threshold_')

    def test_decision_function(self, feature_extractor, mock_image_paths):
        """Test ECOD decision function."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor
        )

        detector.fit(train_paths)
        scores = detector.decision_function(test_paths)

        assert scores.shape == (len(test_paths),)
        assert np.all(np.isfinite(scores))

    def test_invalid_contamination(self, feature_extractor):
        """Test ECOD with invalid contamination."""
        with pytest.raises(ValueError, match="contamination must be in"):
            models.create_model(
                "vision_ecod",
                feature_extractor=feature_extractor,
                contamination=0.6  # Invalid
            )

    def test_parallel_jobs(self, feature_extractor, mock_image_paths):
        """Test ECOD with parallel jobs."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor,
            n_jobs=-1
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)
        assert predictions.shape == (len(test_paths),)


class TestCOPOD:
    """Tests for COPOD detector."""

    def test_create(self, feature_extractor):
        """Test COPOD creation."""
        detector = models.create_model(
            "vision_copod",
            feature_extractor=feature_extractor
        )
        assert detector is not None

    def test_fit_predict(self, feature_extractor, mock_image_paths):
        """Test COPOD fit and predict."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_copod",
            feature_extractor=feature_extractor,
            contamination=0.1
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)

        assert predictions.shape == (len(test_paths),)
        assert set(predictions).issubset({0, 1})


class TestKNN:
    """Tests for KNN detector."""

    def test_create(self, feature_extractor):
        """Test KNN creation."""
        detector = models.create_model(
            "vision_knn",
            feature_extractor=feature_extractor,
            n_neighbors=5
        )
        assert detector is not None

    def test_fit_predict(self, feature_extractor, mock_image_paths):
        """Test KNN fit and predict."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_knn",
            feature_extractor=feature_extractor,
            n_neighbors=10
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)

        assert predictions.shape == (len(test_paths),)

    @pytest.mark.parametrize("method", ["largest", "mean", "median"])
    def test_knn_methods(self, feature_extractor, mock_image_paths, method):
        """Test different KNN scoring methods."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_knn",
            feature_extractor=feature_extractor,
            n_neighbors=5,
            method=method
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)
        assert predictions.shape == (len(test_paths),)


class TestPCA:
    """Tests for PCA detector."""

    def test_create(self, feature_extractor):
        """Test PCA creation."""
        detector = models.create_model(
            "vision_pca",
            feature_extractor=feature_extractor,
            n_components=10
        )
        assert detector is not None

    def test_fit_predict(self, feature_extractor, mock_image_paths):
        """Test PCA fit and predict."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_pca",
            feature_extractor=feature_extractor,
            n_components=0.95  # Keep 95% variance
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)

        assert predictions.shape == (len(test_paths),)

    def test_pca_whiten(self, feature_extractor, mock_image_paths):
        """Test PCA with whitening."""
        train_paths, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_pca",
            feature_extractor=feature_extractor,
            n_components=10,
            whiten=True
        )

        detector.fit(train_paths)
        predictions = detector.predict(test_paths)
        assert predictions.shape == (len(test_paths),)


class TestModelRegistry:
    """Test model registry for PyOD models."""

    def test_models_registered(self):
        """Test that PyOD models are properly registered."""
        from pyimgano.models import list_models

        all_models = list_models()
        pyod_models = ["vision_ecod", "vision_copod", "vision_knn", "vision_pca"]

        for model_name in pyod_models:
            assert model_name in all_models, f"{model_name} not in registry"

    def test_list_by_tags(self):
        """Test listing models by tags."""
        from pyimgano.models import list_models

        # Test parameter-free models
        param_free = list_models(tags=["parameter-free"])
        assert "vision_ecod" in param_free
        assert "vision_copod" in param_free

        # Test high-performance models
        high_perf = list_models(tags=["high-performance"])
        assert "vision_ecod" in high_perf
        assert "vision_copod" in high_perf

    def test_model_metadata(self):
        """Test model metadata."""
        from pyimgano.models import MODEL_REGISTRY

        ecod_info = MODEL_REGISTRY.info("vision_ecod")
        assert ecod_info.metadata["year"] == 2022
        assert ecod_info.metadata["parameter_free"] is True
        assert ecod_info.metadata["benchmark_rank"] == "top-tier"

        copod_info = MODEL_REGISTRY.info("vision_copod")
        assert copod_info.metadata["year"] == 2020
        assert copod_info.metadata["fast"] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_before_fit(self, feature_extractor, mock_image_paths):
        """Test prediction before fitting raises error."""
        _, test_paths = mock_image_paths
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.decision_function(test_paths)

    def test_empty_input(self, feature_extractor):
        """Test handling of empty input."""
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor
        )

        # This should work but with warning/error handling
        # depending on implementation
        try:
            detector.fit([])
        except (ValueError, IndexError):
            # Expected behavior for empty input
            pass

    def test_single_sample(self, feature_extractor):
        """Test with single sample."""
        detector = models.create_model(
            "vision_ecod",
            feature_extractor=feature_extractor
        )

        # Single sample should work
        try:
            detector.fit(["single_image.jpg"])
        except ValueError:
            # Some algorithms may require minimum samples
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
