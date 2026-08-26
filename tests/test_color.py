import cv2
import numpy as np
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

    def test_color_space_attribute_updates_after_conversion(self, color_image):
        im = Imazing(color_image).convert_color("HSV")
        assert im.color_space == "HSV"

    def test_converting_to_current_mode_is_a_noop(self, color_image):
        im = Imazing(color_image)
        original_id = id(im.image)
        im.convert_color("BGR")  # already BGR
        assert id(im.image) == original_id

    def test_routes_correctly_from_a_non_bgr_current_space(self, color_image):
        """Regression test for the original bug: convert_color always
        assumed the source was BGR, so converting HSV->RGB (for example)
        would misinterpret the HSV values as if they were BGR instead of
        actually going HSV->BGR->RGB."""
        im = Imazing(color_image).convert_color("HSV")
        hsv_snapshot = im.image.copy()
        im.convert_color("RGB")
        assert im.color_space == "RGB"

        # The correct route (HSV -> BGR -> RGB) should match doing the
        # same two-step conversion directly with cv2.
        expected_bgr = cv2.cvtColor(hsv_snapshot, cv2.COLOR_HSV2BGR)
        expected_rgb = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
        assert np.array_equal(im.image, expected_rgb)

    def test_bgra_to_gray_drops_alpha_correctly(self, bgra_image):
        im = Imazing(bgra_image).convert_color("GRAY")
        assert im.image.ndim == 2
        assert im.color_space == "GRAY"

    def test_invalid_mode_still_raises_with_bgra_added(self, color_image):
        with pytest.raises(ValueError):
            Imazing(color_image).convert_color("NOT_A_REAL_MODE")


class TestHistogramEqualizationColorSpaceAwareness:
    def test_preserves_alpha_channel_on_bgra_image(self, bgra_image):
        im = Imazing(bgra_image)
        original_alpha = im.image[:, :, 3].copy()
        im.histogram_equalization()
        assert im.image.shape == bgra_image.shape
        assert np.array_equal(im.image[:, :, 3], original_alpha)

    def test_runs_on_hsv_image_without_misreading_channels(self, color_image):
        im = Imazing(color_image).convert_color("HSV")
        im.histogram_equalization()
        assert im.image.shape == color_image.shape
        assert im.color_space == "HSV"


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
