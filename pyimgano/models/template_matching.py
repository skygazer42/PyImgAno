"""
Template Matching for Anomaly Detection

Compares test images against reference templates using correlation-based matching.
Effective for detecting positional defects, missing parts, and structural anomalies.

Reference:
    Brunelli, R. (2009). "Template Matching Techniques in Computer Vision:
    Theory and Practice" Wiley.

Usage:
    >>> from pyimgano.models import TemplateMatching
    >>> model = TemplateMatching(method='ncc', threshold=0.9)
    >>> model.fit(X_train)
    >>> scores = model.predict(X_test)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Literal
from skimage import color
from scipy.signal import correlate2d
import cv2

from ..base import BaseVisionClassicalDetector


class TemplateMatching(BaseVisionClassicalDetector):
    """
    Template matching for anomaly detection.

    Uses normalized cross-correlation or other similarity measures to compare
    test images against reference templates. Effective for:
    - Positional defects
    - Missing or extra parts
    - Structural anomalies
    - Assembly verification

    Parameters
    ----------
    method : str, default='ncc'
        Matching method: 'ncc' (normalized cross-correlation),
        'ssd' (sum of squared differences), 'sad' (sum of absolute differences),
        'zncc' (zero-mean normalized cross-correlation)
    threshold : float, default=0.9
        Similarity threshold for normal samples (method-dependent)
    use_multiple_templates : bool, default=True
        Use multiple templates or single average template
    max_templates : int, default=10
        Maximum number of templates to store
    color_space : str, default='GRAY'
        Color space: 'GRAY', 'RGB', 'HSV'
    align_images : bool, default=False
        Apply image alignment before matching
    resize_shape : tuple, optional
        Resize images to this shape

    Attributes
    ----------
    templates_ : list
        Reference templates
    template_mean_ : ndarray
        Average template (if not using multiple)

    Examples
    --------
    >>> # Structural defect detection
    >>> model = TemplateMatching(method='ncc', threshold=0.95)
    >>> model.fit(X_train_normal)
    >>> scores = model.predict(X_test)

    >>> # Multi-template for variations
    >>> model = TemplateMatching(
    ...     method='ncc',
    ...     use_multiple_templates=True,
    ...     max_templates=20
    ... )
    >>> model.fit(X_train)
    >>> anomalies = model.predict_label(X_test)
    """

    def __init__(
        self,
        method: Literal['ncc', 'ssd', 'sad', 'zncc'] = 'ncc',
        threshold: float = 0.9,
        use_multiple_templates: bool = True,
        max_templates: int = 10,
        color_space: Literal['GRAY', 'RGB', 'HSV'] = 'GRAY',
        align_images: bool = False,
        resize_shape: Optional[Tuple[int, int]] = (128, 128)
    ):
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.use_multiple_templates = use_multiple_templates
        self.max_templates = max_templates
        self.color_space = color_space
        self.align_images = align_images
        self.resize_shape = resize_shape

        self.templates_ = None
        self.template_mean_ = None

    def _preprocess_image(self, image: NDArray) -> NDArray:
        """Preprocess image."""
        # Resize if specified
        if self.resize_shape is not None:
            from skimage.transform import resize
            image = resize(image, self.resize_shape, anti_aliasing=True)

        # Convert color space
        if self.color_space == 'GRAY':
            if len(image.shape) == 3:
                image = color.rgb2gray(image)
        elif self.color_space == 'HSV':
            if len(image.shape) == 3:
                image = color.rgb2hsv(image)
        # RGB: keep as is

        return image

    def _align_image(self, image: NDArray, template: NDArray) -> NDArray:
        """Align image to template using feature-based alignment."""
        try:
            # Convert to uint8 for OpenCV
            img_uint8 = (image * 255).astype(np.uint8)
            template_uint8 = (template * 255).astype(np.uint8)

            # Detect ORB features
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(template_uint8, None)
            kp2, des2 = orb.detectAndCompute(img_uint8, None)

            if des1 is None or des2 is None:
                return image

            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 4:
                return image

            # Get matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)

            # Find homography
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if M is None:
                return image

            # Warp image
            h, w = template.shape[:2]
            aligned = cv2.warpPerspective(img_uint8, M, (w, h))
            aligned = aligned.astype(np.float32) / 255.0

            return aligned

        except:
            return image

    def _compute_similarity(self, image: NDArray, template: NDArray) -> float:
        """Compute similarity between image and template."""
        if self.align_images:
            image = self._align_image(image, template)

        if self.method == 'ncc':
            # Normalized Cross-Correlation
            image_flat = image.ravel()
            template_flat = template.ravel()

            # Normalize
            image_norm = image_flat - np.mean(image_flat)
            template_norm = template_flat - np.mean(template_flat)

            # Compute correlation
            numerator = np.dot(image_norm, template_norm)
            denominator = np.linalg.norm(image_norm) * np.linalg.norm(template_norm)

            if denominator == 0:
                return 0.0

            similarity = numerator / denominator
            return similarity

        elif self.method == 'zncc':
            # Zero-mean Normalized Cross-Correlation
            image_zm = image - np.mean(image)
            template_zm = template - np.mean(template)

            numerator = np.sum(image_zm * template_zm)
            denominator = np.sqrt(np.sum(image_zm ** 2) * np.sum(template_zm ** 2))

            if denominator == 0:
                return 0.0

            similarity = numerator / denominator
            return similarity

        elif self.method == 'ssd':
            # Sum of Squared Differences (convert to similarity)
            ssd = np.sum((image - template) ** 2)
            # Normalize and convert to similarity (lower SSD = higher similarity)
            max_ssd = np.prod(image.shape)  # Maximum possible SSD
            similarity = 1.0 - (ssd / max_ssd)
            return similarity

        elif self.method == 'sad':
            # Sum of Absolute Differences
            sad = np.sum(np.abs(image - template))
            # Normalize and convert to similarity
            max_sad = np.prod(image.shape)
            similarity = 1.0 - (sad / max_sad)
            return similarity

        return 0.0

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'TemplateMatching':
        """
        Fit template matching model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Training images (normal templates)
        y : ndarray, optional
            Ignored

        Returns
        -------
        self : TemplateMatching
            Fitted estimator
        """
        # Preprocess all images
        print("Preprocessing templates...")
        preprocessed = []
        for i in range(len(X)):
            img = self._preprocess_image(X[i])
            preprocessed.append(img)

        preprocessed = np.array(preprocessed)

        if self.use_multiple_templates:
            # Select diverse templates using k-means or random sampling
            if len(preprocessed) <= self.max_templates:
                self.templates_ = preprocessed
            else:
                # Random sampling
                indices = np.random.choice(
                    len(preprocessed),
                    self.max_templates,
                    replace=False
                )
                self.templates_ = preprocessed[indices]
        else:
            # Use average template
            self.template_mean_ = np.mean(preprocessed, axis=0)
            self.templates_ = [self.template_mean_]

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Test images

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores (dissimilarity, higher = more anomalous)
        """
        self._check_is_fitted()

        scores = []

        for i in range(len(X)):
            # Preprocess test image
            test_img = self._preprocess_image(X[i])

            # Compute maximum similarity with any template
            max_similarity = -float('inf')
            for template in self.templates_:
                sim = self._compute_similarity(test_img, template)
                max_similarity = max(max_similarity, sim)

            # Convert similarity to dissimilarity score
            dissimilarity = 1.0 - max_similarity
            scores.append(dissimilarity)

        return np.array(scores)

    def predict_label(self, X: NDArray) -> NDArray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, height, width) or (n_samples, height, width, channels)
            Test images

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Binary labels (1 = anomaly, 0 = normal)
        """
        scores = self.predict(X)

        # For methods where higher similarity = normal:
        # scores represent dissimilarity, so threshold accordingly
        dissimilarity_threshold = 1.0 - self.threshold
        labels = (scores > dissimilarity_threshold).astype(int)

        return labels

    def get_similarity_map(self, image: NDArray) -> NDArray:
        """
        Generate similarity map showing matching quality across templates.

        Parameters
        ----------
        image : ndarray of shape (height, width) or (height, width, channels)
            Test image

        Returns
        -------
        similarity_map : ndarray
            Similarity scores for each template
        """
        self._check_is_fitted()

        # Preprocess
        test_img = self._preprocess_image(image)

        # Compute similarity with each template
        similarities = []
        for template in self.templates_:
            sim = self._compute_similarity(test_img, template)
            similarities.append(sim)

        return np.array(similarities)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'use_multiple_templates': self.use_multiple_templates,
            'max_templates': self.max_templates,
            'color_space': self.color_space,
            'align_images': self.align_images,
            'resize_shape': self.resize_shape,
        }
