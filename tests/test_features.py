import importlib.util
import shutil

import cv2
import numpy as np
import pytest

from imazing import Imazing

_HAS_PYTESSERACT = importlib.util.find_spec("pytesseract") is not None
_HAS_TESSERACT_BINARY = shutil.which("tesseract") is not None
_HAS_PYZBAR = importlib.util.find_spec("pyzbar") is not None
_HAS_QRCODE = importlib.util.find_spec("qrcode") is not None

requires_ocr = pytest.mark.skipif(
    not (_HAS_PYTESSERACT and _HAS_TESSERACT_BINARY),
    reason="pytesseract and/or the tesseract binary is not installed (pip install 'imazing[ocr]')",
)
requires_qr = pytest.mark.skipif(
    not _HAS_PYZBAR, reason="pyzbar is not installed (pip install 'imazing[qr]')"
)
requires_qrcode_lib = pytest.mark.skipif(
    not _HAS_QRCODE, reason="qrcode (test-only, dev extra) is not installed"
)


class TestDetectContours:
    def test_returns_a_list(self, color_image):
        contours = Imazing(color_image).detect_contours()
        assert isinstance(contours, list)

    def test_finds_the_synthetic_rectangle(self, color_image):
        # conftest's color_image fixture draws a solid white rectangle --
        # there should be at least one contour of meaningful size.
        contours = Imazing(color_image).detect_contours(min_area=100)
        assert len(contours) >= 1

    def test_runs_on_bgra_image_without_crashing(self, bgra_image):
        contours = Imazing(bgra_image).detect_contours()
        assert isinstance(contours, list)

    def test_runs_on_already_hsv_image_without_crashing(self, color_image):
        """Regression test: detect_contours used to assume any 3-channel
        image was BGR, misreading HSV values as BGR instead of converting
        from the image's actual current color space."""
        contours = Imazing(color_image).convert_color("HSV").detect_contours()
        assert isinstance(contours, list)


class TestMatchTemplate:
    def test_finds_a_template_cropped_from_the_same_image(self, color_image):
        template = color_image[30:60, 40:70].copy()
        points = Imazing(color_image).match_template(template, threshold=0.9)
        assert len(points) >= 1

    def test_works_on_grayscale_images(self, gray_image):
        template = gray_image[30:60, 40:70].copy()
        points = Imazing(gray_image).match_template(template, threshold=0.9)
        assert len(points) >= 1


class TestDetectFaces:
    def test_runs_without_error_and_returns_iterable(self, color_image):
        # Haar cascades are OpenCV's pretrained model, not this library's
        # own logic -- we're testing that the call succeeds and returns a
        # sane type, not detection accuracy on synthetic noise.
        faces = Imazing(color_image).detect_faces()
        assert hasattr(faces, "__len__")

    def test_runs_on_bgra_image_without_crashing(self, bgra_image):
        faces = Imazing(bgra_image).detect_faces()
        assert hasattr(faces, "__len__")


@requires_qr
@requires_qrcode_lib
class TestDecodeQrBarcode:
    def test_decodes_a_real_qr_code(self, tmp_path):
        import qrcode

        payload = "https://github.com/MeetRVyas/imazing"
        img = qrcode.make(payload)
        path = tmp_path / "qr.png"
        img.save(str(path))

        results = Imazing(str(path)).decode_qr_barcode()
        assert len(results) == 1
        assert results[0]["data"] == payload
        assert results[0]["type"] == "QRCODE"

    def test_no_qr_code_returns_empty_list(self, color_image):
        results = Imazing(color_image).decode_qr_barcode()
        assert results == []


class TestDecodeQrBarcodeMissingDependency:
    def test_raises_import_error_when_pyzbar_unavailable(self, color_image, monkeypatch):
        import imazing.features as features_module

        monkeypatch.setattr(features_module, "qr_decode", None)
        with pytest.raises(ImportError):
            Imazing(color_image).decode_qr_barcode()


@requires_ocr
class TestOcrText:
    def test_extracts_real_text(self):
        img = np.full((100, 400, 3), 255, dtype=np.uint8)
        cv2.putText(img, "HELLO IMAZING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        text = Imazing(img).ocr_text()
        assert "HELLO" in text.upper()


class TestOcrTextMissingDependency:
    def test_raises_import_error_when_pytesseract_unavailable(self, color_image, monkeypatch):
        import imazing.features as features_module

        monkeypatch.setattr(features_module, "pytesseract", None)
        with pytest.raises(ImportError):
            Imazing(color_image).ocr_text()
