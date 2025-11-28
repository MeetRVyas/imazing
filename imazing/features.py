
import cv2
import numpy as np
from .utils import ensure_valid, check_dependency

class FeatureMixin:
    """Handles Advanced Detection (Faces, Corners, Blobs, OCR, QR)."""

    @ensure_valid
    def detect_contours(self, min_area=100, mode='external') -> list:
        """Finds contours in the image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image

        # Auto threshold if needed
        if len(np.unique(gray)) > 2:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            thresh = gray

        retr_mode = cv2.RETR_EXTERNAL if mode == 'external' else cv2.RETR_TREE
        contours, _ = cv2.findContours(thresh, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > min_area]

    @ensure_valid
    def detect_corners_harris(self, block_size=2, ksize=3, k=0.04):
        """Detects corners. Returns a heatmap image of corners."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        # Dilate for visibility
        dst = cv2.dilate(dst, None)
        # Mark red on original image where corners are found
        self.image[dst > 0.01 * dst.max()] = [0, 0, 255]
        return self

    @ensure_valid
    def detect_corners_shi_tomasi(self, max_corners=100, quality=0.01, min_dist=10):
        """Good Features to Track. Returns list of (x, y) points."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist)
        if corners is not None:
            return np.int0(corners)
        return []

    @ensure_valid
    def detect_blobs(self, min_area=100, circularity=0.1):
        """Detects circular blobs. Returns list of KeyPoints."""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        params.filterByCircularity = True
        params.minCircularity = circularity

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(self.image)
        return keypoints

    @ensure_valid
    def match_template(self, template_img, threshold=0.8):
        """Finds location of a smaller template image."""
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            tpl = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
            tpl = template_img

        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        points = []
        for pt in zip(*loc[::-1]):
            points.append(pt)
        return points

    @ensure_valid
    def detect_faces(self, cascade_path=None, scale=1.1, min_neighbors=4):
        """
        Robust Face Detection.
        Uses Histogram Equalization to improve detection in low/bad lighting.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 1. Convert to Gray
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        
        # 2. Equalize Histogram (The Logic Upgrade)
        # This spreads out contrast, making faces visible in dark/bright spots
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(gray, scale, min_neighbors)
        return faces

    @ensure_valid
    def decode_qr_barcode(self):
        """Decodes QR/Barcodes using pyzbar."""
        pyzbar = check_dependency("pyzbar.pyzbar")
        if not pyzbar:
            print("Warning: pyzbar not installed.")
            return []

        decoded_objects = pyzbar.decode(self.image)
        results = []
        for obj in decoded_objects:
            results.append({
                'type': obj.type,
                'data': obj.data.decode('utf-8'),
                'rect': obj.rect
            })
        return results

    @ensure_valid
    def ocr_text(self, lang='eng'):
        """Extracts text using Tesseract."""
        pytesseract = check_dependency("pytesseract")
        if not pytesseract:
            print("Warning: pytesseract not installed.")
            return ""

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        # Preprocessing: Thresholding helps OCR
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(gray, lang=lang)


    @ensure_valid
    def extract_features_orb(self, n_features=500):
        """
        Extracts ORB keypoints and descriptors.
        Returns: (keypoints, descriptors)
        Useful for object matching/recognition.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        orb = cv2.ORB_create(n_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors


    @staticmethod
    def analyze_contour(contour):
        """Returns dict: area, perimeter, center(cx,cy), bounding_rect."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        return {
            "area": area, 
            "perimeter": perimeter, 
            "center": (cx, cy), 
            "rect": (x, y, w, h)
        }

    @staticmethod
    def verify_selection_overlap(selection_rect, target_rect, threshold=0.5):
        """
        Calculates Intersection over Union (IoU).
        rect = (x, y, w, h)
        Returns True if IoU > threshold.
        """
        x1, y1, w1, h1 = selection_rect
        x2, y2, w2, h2 = target_rect

        # Determine intersection coordinates
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = w1 * h1
        boxBArea = w2 * h2

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou >= threshold
