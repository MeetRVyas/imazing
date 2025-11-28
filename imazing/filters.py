import cv2
import numpy as np
import random
from .utils import ensure_valid

class FilterMixin:
    """Handles Blurs, Edges, Noise, and Segmentation."""

    @ensure_valid
    def blur(self, method='gaussian', ksize=5):
        """Methods: gaussian, median, box, bilateral."""
        if ksize % 2 == 0: ksize += 1 # Kernel must be odd

        if method == 'gaussian':
            self.image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
        elif method == 'median':
            self.image = cv2.medianBlur(self.image, ksize)
        elif method == 'box':
            self.image = cv2.blur(self.image, (ksize, ksize))
        elif method == 'bilateral':
            # ksize here maps to 'd' (diameter), standard is 9
            self.image = cv2.bilateralFilter(self.image, 9, 75, 75)
        return self

    @ensure_valid
    def sharpen(self, strength=1.0):
        """Sharpen using unsharp masking logic."""
        blurred = cv2.GaussianBlur(self.image, (0, 0), 3)
        self.image = cv2.addWeighted(self.image, 1.0 + strength, blurred, -strength, 0)
        return self

    @ensure_valid
    def detect_edges(self, method='canny', t1=100, t2=200):
        if method == 'canny':
            self.image = cv2.Canny(self.image, t1, t2)
        elif method == 'sobel':
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            mag = cv2.magnitude(sobelx, sobely)
            self.image = np.uint8(np.clip(mag, 0, 255))
        elif method == 'laplacian':
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            self.image = np.uint8(np.absolute(lap))
        return self

    @ensure_valid
    def morphological(self, op='erode', ksize=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        ops = {
            'erode': cv2.MORPH_ERODE, 'dilate': cv2.MORPH_DILATE,
            'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT, 'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT
        }
        if op in ops:
            self.image = cv2.morphologyEx(self.image, ops[op], kernel, iterations=iterations)
        return self

    @ensure_valid
    def denoise(self, strength=10):
        """Removes noise while keeping details (Non-local Means)."""
        if len(self.image.shape) == 3:
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, strength, 10, 7, 21)
        else:
            self.image = cv2.fastNlMeansDenoising(self.image, None, strength, 7, 21)
        return self

    @ensure_valid
    def segment_threshold(self, type='otsu', val=127):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        if type == 'otsu':
            _, self.image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif type == 'adaptive':
            self.image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        else:
            _, self.image = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        return self

    @ensure_valid
    def remove_background_grabcut(self, rect):
        """
        Interactive Foreground Extraction.
        rect = (x, y, w, h) of the object.
        """
        mask = np.zeros(self.image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        self.image = self.image * mask2[:, :, np.newaxis]
        return self


    @ensure_valid
    def apply_custom_kernel(self, kernel):
        """Applies a user-defined convolution matrix."""
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    @ensure_valid
    def apply_mask_shape(self, shape='circle'):
        """Masks the image to a shape (circle/diamond), making rest black."""
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if shape == 'circle':
            cv2.circle(mask, (w//2, h//2), min(w, h)//2, 255, -1)
        elif shape == 'diamond':
            pts = np.array([[w//2, 0], [w, h//2], [w//2, h], [0, h//2]])
            cv2.fillPoly(mask, [pts], 255)

        self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
        return self

    @ensure_valid
    def watershed_segment(self, markers):
        """
        Applies Watershed segmentation. 
        Markers must be int32 array same size as image.
        """
        if len(self.image.shape) != 3:
             # Watershed needs color image
             img_color = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
             img_color = self.image

        cv2.watershed(img_color, markers)
        # Result is stored in markers (-1 is boundary)
        # We visualize the boundaries on the image
        self.image[markers == -1] = [0, 0, 255]
        return self

    @ensure_valid
    def detect_edges_prewitt(self):
        """Detects edges using Prewitt operator."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(gray, -1, kernelx)
        img_prewitty = cv2.filter2D(gray, -1, kernely)
        self.image = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
        return self
