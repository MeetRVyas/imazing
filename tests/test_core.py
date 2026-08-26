from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import requests

from imazing import Imazing, ImazingError, ImageLoadError, NoImageLoadedError


class TestLoad:
    def test_load_from_numpy_array(self, color_image):
        im = Imazing(color_image)
        assert im.image.shape == color_image.shape
        assert im.image.dtype == color_image.dtype

    def test_load_copies_the_array(self, color_image):
        original = color_image.copy()
        im = Imazing(color_image)
        im.image[0, 0] = (1, 2, 3)
        assert np.array_equal(color_image, original), "loading should not mutate the caller's array"

    def test_load_nonexistent_path_raises_image_load_error(self):
        with pytest.raises(ImageLoadError):
            Imazing("this_path_does_not_exist_12345.jpg")

    def test_image_load_error_is_still_catchable_as_value_error(self):
        """Backward compatibility: old code catching ValueError must keep working."""
        with pytest.raises(ValueError):
            Imazing("this_path_does_not_exist_12345.jpg")

    def test_load_from_file(self, color_image, tmp_path):
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), color_image)
        im = Imazing(str(path))
        assert im.image.shape == color_image.shape

    def test_load_from_bytes(self, color_image):
        _, buf = cv2.imencode(".png", color_image)
        im = Imazing(buf.tobytes())
        assert im.image.shape == color_image.shape

    def test_load_from_base64(self, color_image):
        im_src = Imazing(color_image)
        data_uri = im_src.to_base64(format=".png")
        im = Imazing(data_uri)
        assert im.image.shape == color_image.shape

    def test_no_source_leaves_image_none(self):
        im = Imazing()
        assert im.image is None

    def test_load_can_be_called_after_construction(self, color_image):
        im = Imazing()
        im.load(color_image)
        assert im.image is not None

    def test_url_network_failure_raises_image_load_error_with_chained_cause(self):
        """The original bug: network errors were print()-ed and swallowed,
        losing the real cause behind a generic message."""
        with patch("imazing.core.requests.get", side_effect=requests.ConnectionError("boom")):
            with pytest.raises(ImageLoadError) as exc_info:
                Imazing("https://example.com/some_image.jpg")
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, requests.ConnectionError)

    def test_url_http_error_status_raises(self):
        """raise_for_status() means a 404/500 page body doesn't just silently
        fail to decode as an image with no explanation."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
        with patch("imazing.core.requests.get", return_value=mock_response):
            with pytest.raises(ImageLoadError):
                Imazing("https://example.com/missing.jpg")

    def test_url_success(self, color_image):
        _, buf = cv2.imencode(".png", color_image)
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.raw.read.return_value = buf.tobytes()
        with patch("imazing.core.requests.get", return_value=mock_response) as mock_get:
            im = Imazing("https://example.com/image.png")
            assert im.image.shape == color_image.shape
            _, kwargs = mock_get.call_args
            assert kwargs.get("timeout") is not None, "requests should always be called with a timeout"


class TestNoImageLoadedGuard:
    """The @requires_image decorator should turn a confusing AttributeError
    into a clear, specific exception."""

    def test_resize_before_load_raises_no_image_loaded_error(self):
        with pytest.raises(NoImageLoadedError):
            Imazing().resize(width=10)

    def test_no_image_loaded_error_is_catchable_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            Imazing().resize(width=10)

    def test_no_image_loaded_error_is_catchable_as_imazing_error(self):
        with pytest.raises(ImazingError):
            Imazing().blur()

    def test_error_message_names_the_method(self):
        with pytest.raises(NoImageLoadedError, match="resize"):
            Imazing().resize(width=10)


class TestColorSpaceInference:
    """Imazing.color_space should be inferred from channel count the moment
    an image is loaded, regardless of which load path was used."""

    def test_ndarray_3_channel_is_bgr(self, color_image):
        assert Imazing(color_image).color_space == "BGR"

    def test_ndarray_1_channel_is_gray(self, gray_image):
        assert Imazing(gray_image).color_space == "GRAY"

    def test_ndarray_4_channel_is_bgra(self, bgra_image):
        assert Imazing(bgra_image).color_space == "BGRA"

    def test_load_from_file_infers_from_the_decoded_array(self, color_image, tmp_path):
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), color_image)
        assert Imazing(str(path)).color_space == "BGR"

    def test_load_from_bytes_infers_correctly(self, color_image):
        _, buf = cv2.imencode(".png", color_image)
        assert Imazing(buf.tobytes()).color_space == "BGR"

    def test_clone_preserves_color_space(self, color_image):
        im = Imazing(color_image).convert_color("HSV")
        cloned = im.clone()
        assert cloned.color_space == "HSV"

    def test_reload_after_construction_updates_color_space(self, color_image, gray_image):
        im = Imazing(color_image)
        assert im.color_space == "BGR"
        im.load(gray_image)
        assert im.color_space == "GRAY"


class TestMetadata:
    def test_get_metadata_keys(self, color_image):
        meta = Imazing(color_image).get_metadata()
        assert set(meta.keys()) == {"width", "height", "channels", "color_space", "dtype", "has_alpha"}

    def test_get_metadata_values_for_color_image(self, color_image):
        h, w = color_image.shape[:2]
        meta = Imazing(color_image).get_metadata()
        assert meta["width"] == w
        assert meta["height"] == h
        assert meta["channels"] == 3
        assert meta["color_space"] == "BGR"
        assert meta["has_alpha"] is False

    def test_get_metadata_for_bgra_image(self, bgra_image):
        meta = Imazing(bgra_image).get_metadata()
        assert meta["channels"] == 4
        assert meta["color_space"] == "BGRA"
        assert meta["has_alpha"] is True

    def test_get_stats_keys_are_unchanged(self, color_image):
        """get_metadata() was added as a separate method specifically so
        get_stats()'s existing return shape doesn't change for callers."""
        stats = Imazing(color_image).get_stats()
        assert set(stats.keys()) == {"width", "height", "channels", "mean", "std", "min", "max"}

    def test_get_metadata_before_load_raises(self):
        with pytest.raises(NoImageLoadedError):
            Imazing().get_metadata()


class TestChannelsAndAlphaProperties:
    def test_channels_property_color_image(self, color_image):
        assert Imazing(color_image).channels == 3

    def test_channels_property_gray_image(self, gray_image):
        assert Imazing(gray_image).channels == 1

    def test_channels_property_bgra_image(self, bgra_image):
        assert Imazing(bgra_image).channels == 4

    def test_has_alpha_false_for_bgr(self, color_image):
        assert Imazing(color_image).has_alpha is False

    def test_has_alpha_true_for_bgra(self, bgra_image):
        assert Imazing(bgra_image).has_alpha is True

    def test_channels_before_load_raises(self):
        with pytest.raises(NoImageLoadedError):
            Imazing().channels

    def test_has_alpha_before_load_raises(self):
        with pytest.raises(NoImageLoadedError):
            Imazing().has_alpha


class TestCropDoesNotShareMemory:
    """Regression test: crop() used to return a numpy *view* into the
    original array (self.image[y:y+h, x:x+w] with no .copy()), so the
    full original buffer stayed alive in memory for as long as the crop
    was referenced, and mutating one could silently affect the other."""

    def test_crop_result_does_not_share_memory_with_original(self, color_image):
        im = Imazing(color_image).crop(10, 10, 50, 40)
        assert not np.shares_memory(im.image, color_image)

    def test_mutating_crop_does_not_affect_original_source_array(self, color_image):
        original = color_image.copy()
        im = Imazing(color_image).crop(10, 10, 50, 40)
        im.image[:] = 0
        assert np.array_equal(color_image, original)


class TestResizeInterpolation:
    """resize() should auto-pick INTER_AREA when shrinking and
    INTER_LINEAR when enlarging, rather than always using INTER_AREA."""

    def test_default_uses_area_when_downscaling(self, color_image, monkeypatch):
        import imazing.geometry as geometry_module

        captured = {}
        real_resize = geometry_module.cv2.resize

        def spy_resize(img, dim, interpolation=None):
            captured["interpolation"] = interpolation
            return real_resize(img, dim, interpolation=interpolation)

        monkeypatch.setattr(geometry_module.cv2, "resize", spy_resize)
        Imazing(color_image).resize(width=40)  # smaller than the 160px original
        assert captured["interpolation"] == cv2.INTER_AREA

    def test_default_uses_linear_when_upscaling(self, color_image, monkeypatch):
        import imazing.geometry as geometry_module

        captured = {}
        real_resize = geometry_module.cv2.resize

        def spy_resize(img, dim, interpolation=None):
            captured["interpolation"] = interpolation
            return real_resize(img, dim, interpolation=interpolation)

        monkeypatch.setattr(geometry_module.cv2, "resize", spy_resize)
        Imazing(color_image).resize(width=320)  # larger than the 160px original
        assert captured["interpolation"] == cv2.INTER_LINEAR

    def test_explicit_inter_overrides_auto_selection(self, color_image, monkeypatch):
        import imazing.geometry as geometry_module

        captured = {}
        real_resize = geometry_module.cv2.resize

        def spy_resize(img, dim, interpolation=None):
            captured["interpolation"] = interpolation
            return real_resize(img, dim, interpolation=interpolation)

        monkeypatch.setattr(geometry_module.cv2, "resize", spy_resize)
        Imazing(color_image).resize(width=320, inter=cv2.INTER_NEAREST)
        assert captured["interpolation"] == cv2.INTER_NEAREST



    def test_save_and_reload_roundtrip(self, color_image, tmp_path):
        path = tmp_path / "out.jpg"
        Imazing(color_image).save(str(path))
        assert path.exists()
        reloaded = Imazing(str(path))
        assert reloaded.image.shape == color_image.shape

    def test_save_png_roundtrip_preserves_pixels_exactly(self, color_image, tmp_path):
        path = tmp_path / "out.png"  # lossless, unlike jpg
        Imazing(color_image).save(str(path))
        reloaded = Imazing(str(path))
        assert np.array_equal(reloaded.image, color_image)

    def test_save_png_roundtrip_preserves_alpha_channel(self, bgra_image, tmp_path):
        path = tmp_path / "out.png"
        Imazing(bgra_image).save(str(path))
        reloaded = Imazing(str(path))
        assert reloaded.color_space == "BGRA"
        assert np.array_equal(reloaded.image, bgra_image)

    def test_to_base64_returns_data_uri(self, color_image):
        uri = Imazing(color_image).to_base64()
        assert uri.startswith("data:image/jpg;base64,")

    def test_to_base64_before_load_raises(self):
        with pytest.raises(NoImageLoadedError):
            Imazing().to_base64()

    def test_to_numpy_returns_a_copy(self, color_image):
        im = Imazing(color_image)
        arr = im.to_numpy()
        arr[0, 0] = 0
        assert not np.array_equal(im.image[0, 0], arr[0, 0])


class TestSaveFormats:
    def test_save_webp(self, color_image, tmp_path):
        path = tmp_path / "out.webp"
        Imazing(color_image).save(str(path))
        assert path.exists() and path.stat().st_size > 0

    def test_save_before_load_raises_no_image_loaded_error(self, tmp_path):
        """save() is now consistent with the rest of the API: it raises
        NoImageLoadedError like every other operation, instead of the old
        silent no-op."""
        path = tmp_path / "should_not_exist.jpg"
        with pytest.raises(NoImageLoadedError):
            Imazing().save(str(path))
        assert not path.exists()


class TestCaptureScreen:
    def test_converts_rgb_screenshot_to_bgr(self, monkeypatch):
        from PIL import Image as PILImage

        import imazing.core as core_module

        # A screenshot with distinct R/G/B values so channel-order bugs
        # (a classic source of CV bugs) would be caught.
        fake_screenshot = PILImage.new("RGB", (10, 10), color=(10, 20, 30))
        mock_pyautogui = Mock()
        mock_pyautogui.screenshot.return_value = fake_screenshot
        monkeypatch.setattr(core_module, "pyautogui", mock_pyautogui)

        im = Imazing.capture_screen()
        # PIL gives RGB (10, 20, 30); OpenCV/BGR should store it as (30, 20, 10)
        assert tuple(im.image[0, 0]) == (30, 20, 10)

    def test_raises_import_error_when_pyautogui_unavailable(self, monkeypatch):
        import imazing.core as core_module

        monkeypatch.setattr(core_module, "pyautogui", None)
        with pytest.raises(ImportError):
            Imazing.capture_screen()


class TestFromClipboard:
    def test_converts_clipboard_image_to_bgr(self, monkeypatch):
        from PIL import Image as PILImage

        import imazing.core as core_module

        fake_clip_image = PILImage.new("RGB", (10, 10), color=(10, 20, 30))
        mock_grab = Mock()
        mock_grab.grabclipboard.return_value = fake_clip_image
        monkeypatch.setattr(core_module, "ImageGrab", mock_grab)

        im = Imazing.from_clipboard()
        assert tuple(im.image[0, 0]) == (30, 20, 10)

    def test_returns_none_when_clipboard_has_no_image(self, monkeypatch):
        import imazing.core as core_module

        mock_grab = Mock()
        mock_grab.grabclipboard.return_value = None
        monkeypatch.setattr(core_module, "ImageGrab", mock_grab)

        assert Imazing.from_clipboard() is None

    def test_raises_import_error_when_pillow_unavailable(self, monkeypatch):
        import imazing.core as core_module

        monkeypatch.setattr(core_module, "Image", None)
        with pytest.raises(ImportError):
            Imazing.from_clipboard()


class TestCaptureWebcam:
    def test_returns_none_when_camera_cannot_open(self, monkeypatch):
        import imazing.core as core_module

        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        monkeypatch.setattr(core_module.cv2, "VideoCapture", Mock(return_value=mock_cap))

        assert Imazing.capture_webcam() is None

    def test_returns_none_when_read_fails(self, monkeypatch):
        import imazing.core as core_module

        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        monkeypatch.setattr(core_module.cv2, "VideoCapture", Mock(return_value=mock_cap))

        assert Imazing.capture_webcam() is None

    def test_returns_imazing_instance_on_success(self, monkeypatch, color_image):
        import imazing.core as core_module

        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, color_image)
        monkeypatch.setattr(core_module.cv2, "VideoCapture", Mock(return_value=mock_cap))

        im = Imazing.capture_webcam()
        assert isinstance(im, Imazing)
        mock_cap.release.assert_called_once()


class TestShow:
    """cv2.imshow needs a real display/GUI backend, which a CI runner
    doesn't have -- mocked at the cv2 call level rather than skipped
    entirely, so the wiring logic (wait flag, no-op with no image) is
    still verified."""

    def test_raises_when_no_image_loaded(self, monkeypatch):
        import imazing.core as core_module

        mock_imshow = Mock()
        monkeypatch.setattr(core_module.cv2, "imshow", mock_imshow)
        with pytest.raises(NoImageLoadedError):
            Imazing().show()
        mock_imshow.assert_not_called()

    def test_wait_true_calls_waitkey_and_destroy(self, monkeypatch, color_image):
        import imazing.core as core_module

        monkeypatch.setattr(core_module.cv2, "imshow", Mock())
        mock_waitkey = Mock(return_value=-1)
        mock_destroy = Mock()
        monkeypatch.setattr(core_module.cv2, "waitKey", mock_waitkey)
        monkeypatch.setattr(core_module.cv2, "destroyAllWindows", mock_destroy)

        Imazing(color_image).show(wait=True)
        mock_waitkey.assert_called_once()
        mock_destroy.assert_called_once()

    def test_wait_false_skips_waitkey(self, monkeypatch, color_image):
        import imazing.core as core_module

        monkeypatch.setattr(core_module.cv2, "imshow", Mock())
        mock_waitkey = Mock()
        monkeypatch.setattr(core_module.cv2, "waitKey", mock_waitkey)

        Imazing(color_image).show(wait=False)
        mock_waitkey.assert_not_called()


class TestHeadlessImportSafety:
    """Regression test for a real bug found while writing CI: pyautogui
    (via its mouseinfo dependency) raises a bare KeyError('DISPLAY') on
    headless Linux -- not ImportError -- so importing it inside a plain
    `except ImportError` let that exception crash the entire
    `import imazing` statement on any headless server/CI/Docker
    environment where pyautogui happened to be installed.

    This forces the exact failure condition via a fake broken pyautogui
    module, in a real subprocess, rather than relying on the test runner's
    own environment happening to be headless -- so it stays protected even
    if this suite is later run somewhere with a display available.
    """

    def test_import_survives_a_pyautogui_that_raises_keyerror_on_import(self, tmp_path):
        import os
        import subprocess
        import sys

        fake_pkg_dir = tmp_path / "fake_pyautogui_pkg"
        fake_pkg_dir.mkdir()
        (fake_pkg_dir / "pyautogui.py").write_text(
            "import os\n"
            "os.environ['DISPLAY']  # raises KeyError on headless systems, like mouseinfo does\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", "import imazing; print('OK')"],
            env={**os.environ, "PYTHONPATH": str(fake_pkg_dir)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"import imazing crashed with a broken pyautogui.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout


class TestChaining:
    def test_methods_return_self_for_chaining(self, color_image):
        im = Imazing(color_image)
        result = im.resize(width=80).convert_color("GRAY").invert()
        assert result is im
