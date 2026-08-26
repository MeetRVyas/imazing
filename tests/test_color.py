import pytest

from imazing import Imazing


class TestConvertColor:
    @pytest.mark.parametrize("mode", ["GRAY", "HSV", "RGB", "LAB", "gray", "hsv"])
    def test_valid_modes_do_not_raise(self, color_image, mode):
        Imazing(color_image).convert_color(mode)

    def test_gray_actually_reduces_to_2d(self, color_image):
        im = Imazing(color_image).convert_color("GRAY")
        assert im.image.ndim == 2

    def test_invalid_mode_raises_value_error(self, color_image):
        """Regression test: used to silently no-op instead of raising."""
        with pytest.raises(ValueError):
            Imazing(color_image).convert_color("NOT_A_REAL_MODE")

    def test_bgr_on_grayscale_converts_back_to_3_channels(self, gray_image):
        im = Imazing(gray_image).convert_color("BGR")
        assert im.image.ndim == 3


class TestBrightnessContrast:
    def test_adjust_brightness_contrast_runs(self, color_image):
        im = Imazing(color_image).adjust_brightness_contrast(alpha=1.2, beta=20)
        assert im.image.shape == color_image.shape


class TestInvert:
    def test_invert_is_its_own_inverse(self, color_image):
        im = Imazing(color_image.copy())
        im.invert()
        im.invert()
        assert (im.image == color_image).all()


class TestHistogramEqualization:
    def test_on_color_image(self, color_image):
        im = Imazing(color_image).histogram_equalization()
        assert im.image.shape == color_image.shape

    def test_on_grayscale_image(self, gray_image):
        im = Imazing(gray_image).histogram_equalization()
        assert im.image.shape == gray_image.shape
