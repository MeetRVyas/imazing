import importlib.util
import shutil

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from imazing.cli import app

runner = CliRunner()

_HAS_PYTESSERACT = importlib.util.find_spec("pytesseract") is not None
_HAS_TESSERACT_BINARY = shutil.which("tesseract") is not None
_HAS_PYZBAR = importlib.util.find_spec("pyzbar") is not None
_HAS_QRCODE = importlib.util.find_spec("qrcode") is not None

requires_ocr = pytest.mark.skipif(
    not (_HAS_PYTESSERACT and _HAS_TESSERACT_BINARY), reason="ocr extra / tesseract binary not installed"
)
requires_qr_stack = pytest.mark.skipif(
    not (_HAS_PYZBAR and _HAS_QRCODE), reason="qr extra or test-only qrcode lib not installed"
)


@pytest.fixture
def input_image_path(tmp_path, color_image):
    path = tmp_path / "input.jpg"
    cv2.imwrite(str(path), color_image)
    return path


class TestGlobalOptions:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "imazing" in result.stdout

    def test_help_lists_all_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ["info", "resize", "convert", "rotate", "blur", "edges", "faces", "ocr", "qr"]:
            assert command in result.stdout

    def test_no_args_shows_help(self):
        """no_args_is_help=True prints usage, but Click's convention is to
        still exit 2 (a usage error) rather than 0 -- printing help isn't
        the same as successfully doing something."""
        result = runner.invoke(app, [])
        assert result.exit_code == 2
        assert "Usage" in result.output


class TestInfoCommand:
    def test_prints_stats(self, input_image_path):
        result = runner.invoke(app, ["info", str(input_image_path)])
        assert result.exit_code == 0
        assert "width" in result.stdout
        assert "height" in result.stdout

    def test_nonexistent_file_fails_cleanly(self):
        result = runner.invoke(app, ["info", "/tmp/does_not_exist_xyz.jpg"])
        assert result.exit_code != 0

    def test_existing_but_invalid_image_file_fails_cleanly(self, tmp_path):
        """Exercises _load_or_exit's ImazingError branch specifically: the
        path exists (passes Typer's exists=True check) but isn't decodable
        as an image."""
        bogus = tmp_path / "not_an_image.jpg"
        bogus.write_text("this is definitely not image data")
        result = runner.invoke(app, ["info", str(bogus)])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestResizeCommand:
    def test_resize_by_width(self, input_image_path, tmp_path):
        out = tmp_path / "out.jpg"
        result = runner.invoke(app, ["resize", str(input_image_path), str(out), "--width", "80"])
        assert result.exit_code == 0
        assert out.exists()
        saved = cv2.imread(str(out))
        assert saved.shape[1] == 80

    def test_resize_without_dimensions_fails_cleanly(self, input_image_path, tmp_path):
        out = tmp_path / "out.jpg"
        result = runner.invoke(app, ["resize", str(input_image_path), str(out)])
        assert result.exit_code == 1
        assert not out.exists()


class TestConvertCommand:
    def test_convert_to_grayscale(self, input_image_path, tmp_path):
        out = tmp_path / "gray.jpg"
        result = runner.invoke(app, ["convert", str(input_image_path), str(out), "--mode", "gray"])
        assert result.exit_code == 0
        saved = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        assert saved.ndim == 2

    def test_invalid_mode_fails_cleanly_with_message(self, input_image_path, tmp_path):
        out = tmp_path / "out.jpg"
        result = runner.invoke(app, ["convert", str(input_image_path), str(out), "--mode", "not_a_mode"])
        assert result.exit_code == 1


class TestRotateCommand:
    def test_rotate(self, input_image_path, tmp_path):
        out = tmp_path / "rotated.jpg"
        result = runner.invoke(app, ["rotate", str(input_image_path), str(out), "45"])
        assert result.exit_code == 0
        assert out.exists()


class TestBlurCommand:
    def test_blur_with_custom_method_and_ksize(self, input_image_path, tmp_path):
        out = tmp_path / "blurred.jpg"
        result = runner.invoke(
            app, ["blur", str(input_image_path), str(out), "--method", "median", "--ksize", "7"]
        )
        assert result.exit_code == 0
        assert out.exists()


class TestEdgesCommand:
    def test_edges_sobel(self, input_image_path, tmp_path):
        out = tmp_path / "edges.jpg"
        result = runner.invoke(app, ["edges", str(input_image_path), str(out), "--method", "sobel"])
        assert result.exit_code == 0
        assert out.exists()


class TestFacesCommand:
    def test_faces_runs_and_reports_a_count(self, input_image_path, tmp_path):
        out = tmp_path / "faces.jpg"
        result = runner.invoke(app, ["faces", str(input_image_path), str(out)])
        assert result.exit_code == 0
        assert "face(s)" in result.stdout
        assert out.exists()

    def test_faces_draws_boxes_when_detections_are_found(self, input_image_path, tmp_path, monkeypatch):
        """The draw loop only runs when detect_faces() returns something --
        force a detection so that path is actually exercised."""
        import numpy as np

        import imazing.cli as cli_module

        fake_detection = np.array([[10, 10, 30, 30]])
        monkeypatch.setattr(
            cli_module.Imazing, "detect_faces", lambda self, cascade_path=None: fake_detection
        )
        out = tmp_path / "faces.jpg"
        result = runner.invoke(app, ["faces", str(input_image_path), str(out)])
        assert result.exit_code == 0
        assert "Found 1 face(s)" in result.stdout


class TestOcrCommand:
    @requires_ocr
    def test_ocr_extracts_real_text(self, tmp_path):
        img = np.full((100, 400, 3), 255, dtype=np.uint8)
        cv2.putText(img, "HELLO IMAZING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        path = tmp_path / "text.jpg"
        cv2.imwrite(str(path), img)
        result = runner.invoke(app, ["ocr", str(path)])
        assert result.exit_code == 0
        assert "HELLO" in result.stdout.upper()

    def test_ocr_missing_dependency_fails_cleanly(self, input_image_path, monkeypatch):
        import imazing.features as features_module

        monkeypatch.setattr(features_module, "pytesseract", None)
        result = runner.invoke(app, ["ocr", str(input_image_path)])
        assert result.exit_code == 1
        assert "not installed" in result.output.lower()


class TestQrCommand:
    @requires_qr_stack
    def test_qr_decodes_real_code(self, tmp_path):
        import qrcode

        payload = "https://github.com/MeetRVyas/imazing"
        img = qrcode.make(payload)
        path = tmp_path / "qr.png"
        img.save(str(path))
        result = runner.invoke(app, ["qr", str(path)])
        assert result.exit_code == 0
        assert payload in result.stdout

    def test_qr_no_code_found(self, input_image_path):
        if not _HAS_PYZBAR:
            pytest.skip("pyzbar not installed")
        result = runner.invoke(app, ["qr", str(input_image_path)])
        assert result.exit_code == 0
        assert "No QR codes" in result.stdout

    def test_qr_missing_dependency_fails_cleanly(self, input_image_path, monkeypatch):
        import imazing.features as features_module

        monkeypatch.setattr(features_module, "qr_decode", None)
        result = runner.invoke(app, ["qr", str(input_image_path)])
        assert result.exit_code == 1
        assert "not installed" in result.output.lower()
