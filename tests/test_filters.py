import numpy as np
import pytest

from imazing import Imazing


class TestBlur:
    @pytest.mark.parametrize("method", ["gaussian", "median", "box", "bilateral"])
    def test_all_methods_run_without_error(self, color_image, method):
        im = Imazing(color_image).blur(method=method, ksize=5)
        assert im.image.shape == color_image.shape

    def test_even_ksize_is_handled(self, color_image):
        """ksize must be odd for most OpenCV blur kernels; the method bumps
        it up by one internally rather than crashing."""
        im = Imazing(color_image).blur(method="gaussian", ksize=4)
        assert im.image.shape == color_image.shape


class TestSharpen:
    def test_sharpen_runs(self, color_image):
        im = Imazing(color_image).sharpen()
        assert im.image.shape == color_image.shape


class TestDetectEdges:
    def test_canny(self, color_image):
        im = Imazing(color_image).detect_edges(method="canny")
        assert im.image is not None

    def test_sobel(self, color_image):
        im = Imazing(color_image).detect_edges(method="sobel")
        assert im.image is not None

    def test_sobel_on_grayscale(self, gray_image):
        im = Imazing(gray_image).detect_edges(method="sobel")
        assert im.image is not None


class TestMorphological:
    @pytest.mark.parametrize("op", ["erode", "dilate", "open", "close"])
    def test_all_ops_run_without_error(self, color_image, op):
        im = Imazing(color_image).morphological(op=op)
        assert im.image.shape == color_image.shape


class TestDenoise:
    def test_denoise_color(self, color_image):
        im = Imazing(color_image).denoise(strength=5)
        assert im.image.shape == color_image.shape

    def test_denoise_grayscale(self, gray_image):
        im = Imazing(gray_image).denoise(strength=5)
        assert im.image.shape == gray_image.shape


class TestSegmentThreshold:
    @pytest.mark.parametrize("thresh_type", ["otsu", "adaptive", "fixed"])
    def test_all_types_run_without_error(self, color_image, thresh_type):
        im = Imazing(color_image).segment_threshold(type=thresh_type)
        assert im.image is not None


class TestRemoveBackgroundGrabcut:
    def test_runs_without_error(self, color_image):
        h, w = color_image.shape[:2]
        im = Imazing(color_image).remove_background_grabcut((5, 5, w - 10, h - 10))
        assert im.image.shape == color_image.shape


class TestAddNoise:
    def test_gaussian_on_color_image(self, color_image):
        im = Imazing(color_image).add_noise("gaussian")
        assert im.image.shape == color_image.shape

    def test_gaussian_on_grayscale_image_does_not_crash(self, gray_image):
        """Regression test for the confirmed bug: add_noise('gaussian') used
        to assume 3 channels and raise ValueError on grayscale input."""
        im = Imazing(gray_image).add_noise("gaussian")
        assert im.image.shape == gray_image.shape

    def test_salt_pepper_on_color_image(self, color_image):
        im = Imazing(color_image).add_noise("salt_pepper")
        assert im.image.shape == color_image.shape

    def test_salt_pepper_on_grayscale_image(self, gray_image):
        im = Imazing(gray_image).add_noise("salt_pepper")
        assert im.image.shape == gray_image.shape

    def test_salt_pepper_actually_introduces_extreme_values(self, color_image):
        """Sanity check the vectorized rewrite still does what it's supposed
        to: introduce a mix of 0s and 255s."""
        im = Imazing(color_image.copy()).add_noise("salt_pepper")
        assert (im.image == 0).any() or (im.image == 255).any()
