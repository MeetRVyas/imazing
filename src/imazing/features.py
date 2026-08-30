from __future__ import annotations

import cv2
import numpy as np

from ._validation import requires_image, as_gray

# Optional imports handled nicely inside functions or here with checks
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pyzbar.pyzbar import decode as qr_decode
except ImportError:
    qr_decode = None

class FeatureMixin:
    """Handles Detection, AI, and Recognition"""

    @requires_image
    def detect_contours(self, min_area=100):
        """Finds contours in the image."""
        gray = as_gray(self.image, self.color_space)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > min_area]

    @requires_image
    def match_template(self, template_img, threshold=0.8):
        """Finds a smaller image inside the current image.

        `template_img` is assumed to be in the same color space as this image.
        """
        gray = as_gray(self.image, self.color_space)
        tpl = as_gray(template_img, self.color_space)

        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        points = []
        for pt in zip(*loc[::-1]):
            points.append(pt)
        return points

    @requires_image
    def detect_faces(self, cascade_path=None):
        """Uses Haar Cascades. Defaults to OpenCV's built-in frontal face."""
        if cascade_path is None:
            # Try to load from cv2 data path
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = as_gray(self.image, self.color_space)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces # Returns list of Rect (x,y,w,h)

    @requires_image
    def decode_qr_barcode(self):
        """Decodes QR codes or Barcodes."""
        if not qr_decode: raise ImportError("pyzbar not installed")
        decoded_objects = qr_decode(self.image)
        results = []
        for obj in decoded_objects:
            results.append({
                'type': obj.type,
                'data': obj.data.decode('utf-8'),
                'rect': obj.rect
            })
        return results

    @requires_image
    def ocr_text(self, lang='eng'):
        """Extracts text using Tesseract."""
        if not pytesseract: raise ImportError("pytesseract not installed")
        # Preprocessing for better OCR
        gray = as_gray(self.image, self.color_space)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(gray, lang=lang)