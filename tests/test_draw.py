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


def _assert_clipped_to(result, original, region, expected_value):
    """`region` must equal `expected_value` exactly, and every pixel outside
    it must be untouched. Using a solid overlay + alpha=1.0 in the tests
    below makes `expected_value` an unambiguous constant, so this doesn't
    depend on what the base fixture happens to contain."""
    assert (result[region] == expected_value).all(), "clipped region wasn't blended as expected"
    mask = np.ones_like(original, dtype=bool)
    mask[region] = False
    assert np.array_equal(result[mask], original[mask]), "pixels outside the overlap changed"


class TestOverlayImage:
    def test_overlay_within_bounds(self, color_image, small_color_image):
        im = Imazing(color_image).overlay_image(small_color_image, x=5, y=5, alpha=0.4)
        assert im.image.shape == color_image.shape

    def test_overlay_negative_x_is_clipped(self, color_image):
        """Regression test for the confirmed bug: only the bottom/right edge
        was bounds-checked, so a negative coordinate reached OpenCV's
        addWeighted() and crashed with a raw assertion error. A negative x
        should now clip the overlay to whichever part of it is still
        visible and blend only that.

        A solid overlay at alpha=1.0 is used (rather than the random
        small_color_image fixture) so the expected result is an exact,
        unambiguous constant instead of "differs from whatever the base
        image happened to contain there" — the latter can coincidentally
        hold even when clipping is correct, if the fixtures are generated
        deterministically."""
        overlay = np.full((20, 20, 3), 42, dtype=np.uint8)
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(overlay, x=-5, y=5, alpha=1.0)
        assert im.image.shape == color_image.shape
        # overlay's rightmost 15 columns (its own columns [5:20)) land at
        # image columns [0:15), rows [5:25) are untouched vertically.
        _assert_clipped_to(im.image, original, np.s_[5:25, 0:15], 42)

    def test_overlay_negative_y_is_clipped(self, color_image):
        overlay = np.full((20, 20, 3), 42, dtype=np.uint8)
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(overlay, x=5, y=-5, alpha=1.0)
        assert im.image.shape == color_image.shape
        # overlay's bottom 15 rows (its own rows [5:20)) land at image
        # rows [0:15), columns [5:25).
        _assert_clipped_to(im.image, original, np.s_[0:15, 5:25], 42)

    def test_overlay_partial_overlap_bottom_right_blends_visible_region(self, color_image):
        """When the overlay hangs off the bottom/right edge, only the
        overlapping corner should be blended; everything else stays as-is."""
        overlay = np.full((20, 20, 3), 42, dtype=np.uint8)
        h, w = color_image.shape[:2]
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(overlay, x=w - 2, y=h - 2, alpha=1.0)
        assert im.image.shape == color_image.shape
        _assert_clipped_to(im.image, original, np.s_[h-2:h, w-2:w], 42)

    def test_overlay_fully_out_of_bounds_is_a_noop(self, color_image, small_color_image):
        """If the overlay doesn't intersect the base image at all, nothing
        should be drawn."""
        h, w = color_image.shape[:2]
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(small_color_image, x=w + 10, y=h + 10, alpha=0.4)
        assert np.array_equal(im.image, original)

    def test_overlay_actually_blends_pixels(self, color_image):
        overlay = np.zeros((20, 20, 3), dtype=np.uint8)  # solid black overlay
        original = color_image.copy()
        im = Imazing(color_image.copy()).overlay_image(overlay, x=5, y=5, alpha=1.0)
        # alpha=1.0 means the overlay fully replaces the region
        assert (im.image[5:25, 5:25] == 0).all()
        assert not np.array_equal(im.image, original)