import pytest

from imazing import Imazing


class TestResize:
    def test_resize_exact_dimensions(self, color_image):
        im = Imazing(color_image).resize(width=80, height=60)
        assert im.image.shape[:2] == (60, 80)

    def test_resize_width_only_preserves_aspect_ratio(self, color_image):
        h, w = color_image.shape[:2]
        im = Imazing(color_image).resize(width=80)
        expected_height = round(h * (80 / w))
        assert im.image.shape[1] == 80
        assert abs(im.image.shape[0] - expected_height) <= 1

    def test_resize_height_only_preserves_aspect_ratio(self, color_image):
        h, w = color_image.shape[:2]
        im = Imazing(color_image).resize(height=30)
        expected_width = round(w * (30 / h))
        assert im.image.shape[0] == 30
        assert abs(im.image.shape[1] - expected_width) <= 1

    def test_resize_no_args_is_a_noop(self, color_image):
        im = Imazing(color_image).resize()
        assert im.image.shape == color_image.shape


class TestCrop:
    def test_crop_returns_expected_region_size(self, color_image):
        im = Imazing(color_image).crop(10, 10, 50, 40)
        assert im.image.shape[:2] == (40, 50)


class TestRotate:
    def test_rotate_does_not_crash_and_preserves_size(self, color_image):
        im = Imazing(color_image).rotate(45)
        assert im.image.shape == color_image.shape

    def test_rotate_zero_degrees_is_effectively_identity(self, color_image):
        im = Imazing(color_image).rotate(0)
        assert im.image.shape == color_image.shape


class TestFlip:
    def test_flip_horizontal(self, color_image):
        im = Imazing(color_image).flip(horizontal=True, vertical=False)
        assert im.image.shape == color_image.shape

    def test_flip_both(self, color_image):
        im = Imazing(color_image).flip(horizontal=True, vertical=True)
        assert im.image.shape == color_image.shape


class TestPad:
    def test_pad_increases_dimensions_by_expected_amount(self, color_image):
        h, w = color_image.shape[:2]
        im = Imazing(color_image).pad(top=5, bottom=5, left=10, right=10)
        assert im.image.shape[:2] == (h + 10, w + 20)


class TestWarpPerspective:
    def test_warp_perspective_runs_and_produces_requested_size(self, color_image):
        import numpy as np

        h, w = color_image.shape[:2]
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        # A mild perspective shift, not a degenerate transform
        dst = np.float32([[0, 0], [w, 0], [w * 0.1, h], [w * 0.9, h]])
        im = Imazing(color_image).warp_perspective(src, dst, (w, h))
        assert im.image.shape[:2] == (h, w)


class TestAugmentRandom:
    def test_augment_random_never_crashes_on_grayscale(self, gray_image):
        """Regression test: augment_random has a 1-in-4 chance of calling
        add_noise(), which used to crash on grayscale images."""
        for _ in range(50):
            im = Imazing(gray_image.copy()).augment_random()
            assert im.image is not None

    def test_augment_random_never_crashes_on_color(self, color_image):
        for _ in range(50):
            im = Imazing(color_image.copy()).augment_random()
            assert im.image is not None
