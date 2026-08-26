import cv2
import numpy as np

from ._validation import requires_image, as_gray


class AnalysisMixin:
    """Handles Statistics and Metadata"""

    @requires_image
    def get_stats(self):
        """Returns basic statistics."""
        return {
            "width": self.image.shape[1],
            "height": self.image.shape[0],
            "channels": self.channels,
            "mean": np.mean(self.image),
            "std": np.std(self.image),
            "min": np.min(self.image),
            "max": np.max(self.image)
        }

    @requires_image
    def get_metadata(self):
        """Returns structural metadata about the loaded image: color space,
        channel count, dtype, and whether an alpha channel is present.

        Kept separate from get_stats() (which reports pixel statistics)
        rather than adding these keys there, so existing get_stats() callers
        see no change in its return shape.
        """
        return self.metadata

    @requires_image
    def compute_hash(self, size=8):
        """Perceptive hash for duplicate detection (aHash)."""
        resized = cv2.resize(self.image, (size, size))
        gray = as_gray(resized, self.color_space)
        avg = gray.mean()
        diff = gray > avg
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    @requires_image
    def inspect(self, out_dir : str = "inspect"):
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents = True, exist_ok = True)

        # stats = self.get_stats()
        
        report = {
            "source": self.source,
            **self.get_stats(),
            "color_space": self.color_space,
            "hash": self.compute_hash(),
        }

        # OCR: useful for checking whether expected UI text exists.
        try:
            text = self.ocr_text()
            report["ocr"] = {
                "text": text,
                "characters": len(text),
            }
        except Exception as exc:
            report["ocr"] = {
                "available": False,
                "error": str(exc),
            }

        # QR/barcode detection.
        try:
            codes = self.decode_qr_barcode()
            report["codes"] = codes
        except Exception:
            report["codes"] = []

        # Face detection, useful when validating screenshots/profile pages.
        try:
            faces = self.detect_faces()
            report["faces"] = len(faces)
        except Exception:
            report["faces"] = None

        import json
    
        # Machine-readable report
        (out / "report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )

        # Original diagnostics
        self.clone().convert_color("gray").save(out / "gray.png")
        self.clone().detect_edges(method="canny").save(out / "edges.png")
        self.clone().resize(width=1200).save(out / "annotated.png")

        print("\n=== IMAZING VISUAL DEBUGGER ===")
        print(f"File:       {self.source}")
        print(f"Channels:   {report['channels']}")
        print(f"Faces:      {report['faces']}")
        print(f"QR/Codes:   {len(report['codes'])}")
        print(f"OCR chars:  {report['ocr'].get('characters', 0)}")
        print(f"Hash:       {report['hash']}")

        print(f"\nDiagnostics written to: {str(out)}/")

        return self, report