import cv2
import numpy as np
import random

class FilterMixin:
    """Handles Blurs, Edges, Noise, and Segmentation"""

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
            self.image = cv2.bilateralFilter(self.image, 9, 75, 75)
        return self

    def sharpen(self):
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    def detect_edges(self, method='canny', t1=100, t2=200):
        if method == 'canny':
            self.image = cv2.Canny(self.image, t1, t2)
        elif method == 'sobel':
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            self.image = cv2.magnitude(sobelx, sobely)
            self.image = np.uint8(self.image)
        return self

    def morphological(self, op='erode', ksize=3, iterations=1):
        kernel = np.ones((ksize, ksize), np.uint8)
        ops = {
            'erode': cv2.MORPH_ERODE, 'dilate': cv2.MORPH_DILATE,
            'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE
        }
        if op in ops:
            self.image = cv2.morphologyEx(self.image, ops[op], kernel, iterations=iterations)
        return self

    def denoise(self, strength=10):
        """Removes noise while keeping details."""
        if len(self.image.shape) == 3:
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, strength, 10, 7, 21)
        else:
            self.image = cv2.fastNlMeansDenoising(self.image, None, strength, 7, 21)
        return self

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

    def remove_background_grabcut(self, rect):
        """rect = (x, y, w, h) of the foreground object."""
        mask = np.zeros(self.image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        self.image = self.image * mask2[:, :, np.newaxis]
        return self

    def add_noise(self, noise_type="gaussian"):
        if noise_type == "gaussian":
            row, col, ch = self.image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = self.image + (gauss * 255)
            self.image = np.clip(noisy, 0, 255).astype(np.uint8)
        elif noise_type == "salt_pepper":
            # Simple implementation
            prob = 0.02
            thres = 1 - prob
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    rdn = random.random()
                    if rdn < prob:
                        self.image[i][j] = 0
                    elif rdn > thres:
                        self.image[i][j] = 255
        return self
