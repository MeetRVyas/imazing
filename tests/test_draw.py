import numpy as np
import pytest

from imazing import Imazing


class TestDrawPrimitives:
    def test_draw_rect(self, color_image):
        im = Imazing(color_image).draw_rect(5, 5, 20, 20)
        assert im.image.shape == color_image.shape

    def test_draw_circle(self, color_image):
        im = Imazing(color_image).draw_circle(50, 50, 10)
        assert im.image.shape == color_image.shape

    def test_draw_text(self, color_image):
        im = Imazing(color_image).draw_text("hi", 5, 70)
        assert im.image.shape == color_image.shape

    def test_drawing_calls_chain(self, color_image):
        im = Imazing(color_image).draw_rect(5, 5, 20, 20).draw_circle(50, 50, 10).draw_text("hi", 5, 70)
        assert im.image.shape == color_image.shape


class TestOverlayImage:
    def test_overlay_within_bounds(self, color_image, small_color_image):
        im = Imazing(color_image).overlay_image(small_color_image, x=5, y=5, alpha=0.4)
        assert im.image.shape == color_image.shape

    def test_overlay_negative_x_does_not_crash(self, color_image, small_color_image):
        """Regression test for the confirmed bug: only the bottom/right edge
        was bounds-checked, so a negative coordinate reached OpenCV's
        addWeighted() and crashed with a raw assertion error."""
        im = Imazing(color_image.copy()).overlay_image(small_color_image, x=-5, y=5, alpha=0.4)
        assert im.image.shape == color_image.shape

    def test_overlay_negative_y_does_not_crash(self, color_image, small_color_image):
        im = Imazing(color_image.copy()).overlay_image(small_color_image, x=5, y=-5, alpha=0.4)
        assert im.image.shape == color_image.shape

    def test_overlay_out_of_bounds_bottom_right_is_a_noop(self, color_image, small_color_image):
        h, w = color_image.shape[:2]
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(small_color_image, x=w - 2, y=h - 2, alpha=0.4)
        assert np.array_equal(im.image, original)

    def test_overlay_actually_blends_pixels(self, color_image):
        overlay = np.zeros((20, 20, 3), dtype=np.uint8)  # solid black overlay
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(overlay, x=5, y=5, alpha=1.0)
        # alpha=1.0 means the overlay fully replaces the region
        assert (im.image[5:25, 5:25] == 0).all()
        assert not np.array_equal(im.image, original)
