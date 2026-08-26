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


class TestDrawingOnBgraImages:
    """Regression tests: cv2's drawing functions zero-fill any channel a
    color tuple doesn't specify, so a plain 3-value BGR color drawn on a
    4-channel image used to leave alpha=0 -- the shape was drawn but fully
    transparent, with no error."""

    def test_draw_rect_default_color_is_not_transparent(self, bgra_image):
        im = Imazing(bgra_image).draw_rect(5, 5, 20, 20, thickness=-1)
        alpha_in_rect = im.image[10:20, 10:20, 3]  # safely inside the filled rect
        assert (alpha_in_rect == 255).all()

    def test_draw_circle_default_color_is_not_transparent(self, bgra_image):
        im = Imazing(bgra_image).draw_circle(50, 50, 10)
        assert im.image[50, 50, 3] == 255

    def test_draw_text_default_color_is_not_transparent(self, bgra_image):
        im = Imazing(bgra_image).draw_text("hi", 5, 70)
        # At least some pixel in the text's bounding area should have been
        # drawn opaque; scanning a small box below the baseline is enough
        # to catch it without depending on exact glyph rendering.
        assert (im.image[55:75, 5:60, 3] == 255).any()

    def test_explicit_4_value_color_is_respected_as_is(self, bgra_image):
        im = Imazing(bgra_image).draw_rect(5, 5, 20, 20, color=(10, 20, 30, 128), thickness=-1)
        assert tuple(im.image[10, 10]) == (10, 20, 30, 128)

    def test_3_value_color_on_3_channel_image_is_unaffected(self, color_image):
        """The normalization shouldn't change anything for the common
        BGR case that already worked correctly."""
        im = Imazing(color_image).draw_rect(5, 5, 20, 20, color=(1, 2, 3), thickness=-1)
        assert tuple(im.image[10, 10]) == (1, 2, 3)


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
