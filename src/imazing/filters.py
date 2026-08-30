import cv2
import numpy as np

from ._validation import requires_image, as_gray


def _apply_bilateral(img, ksize):
    # cv2.bilateralFilter only supports 1- or 3-channel 8-bit images; a
    # 4-channel (BGRA) image would otherwise raise a cv2 error. Filter just
    # the color channels and reattach the original alpha unchanged.
    if img.ndim == 3 and img.shape[2] == 4:
        bgr, alpha = img[:, :, :3], img[:, :, 3]
        return np.dstack([cv2.bilateralFilter(bgr, 9, 75, 75), alpha])
    return cv2.bilateralFilter(img, 9, 75, 75)


_VALID_BLUR_MODES = {
    "gaussian" : lambda img, ksize : cv2.GaussianBlur(img, (ksize, ksize), 0),
    "median" : lambda img, ksize : cv2.medianBlur(img, ksize),
    "box" : lambda img, ksize : cv2.blur(img, (ksize, ksize)),
    "bilateral" : _apply_bilateral,
}

_MORPHS = {
    'erode': cv2.MORPH_ERODE,
    'dilate': cv2.MORPH_DILATE,
    'open': cv2.MORPH_OPEN,
    'close': cv2.MORPH_CLOSE
}

def _apply_sobel(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    img = cv2.magnitude(sobelx, sobely)
    return np.uint8(img)

_EDGE_DETECTION_METHODS = {
    "canny" : lambda img, t1, t2 : cv2.Canny(img, t1, t2),
    "sobel" : lambda img, t1, t2 : _apply_sobel(img)
}

_THRESHOLD_TYPES = {
    "otsu" : lambda img, val : cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    "adaptive" : lambda img, val : cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY, 11, 2),
    "default" : lambda img, val : cv2.threshold(img, val, 255, cv2.THRESH_BINARY)[1]
}

def _apply_gaussian_noise(img) :
    mean = 0
    var = 0.1
    sigma = var**0.5
    # img.shape works whether the image is (h, w) grayscale
    # or (h, w, c) color -- no need to unpack channel count.
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float64) + (gauss * 255)
    return np.clip(noisy, 0, 255).astype(np.uint8)
    
def _apply_salt_pepper_noise(img) :
    # Vectorized: a Python-level double loop here would be a real
    # bottleneck if this is ever called per-frame via VideoStream.
    prob = 0.02
    rnd = np.random.random(img.shape[:2])
    img[rnd < prob] = 0
    img[rnd > 1 - prob] = 255
    return img

_NOISE_TYPES = {
    "gaussian" : _apply_gaussian_noise,
    "salt_pepper" : _apply_salt_pepper_noise,
}


class FilterMixin:
    """Handles Blurs, Edges, Noise, and Segmentation"""

    @requires_image
    def blur(self, method='gaussian', ksize=5):
        """Methods: gaussian, median, box, bilateral."""
        if ksize % 2 == 0: ksize += 1 # Kernel must be odd

        if method not in _VALID_BLUR_MODES :
            raise ValueError(
                f"Blur method '{method}' is not supported. "
                f"Valid methods are: {', '.join(_VALID_BLUR_MODES.keys())}."
            )
        else :
            self.image = _VALID_BLUR_MODES.get(method)(self.image, ksize)
        return self

    @requires_image
    def sharpen(self):
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    @requires_image
    def detect_edges(self, method='canny', t1=100, t2=200):
        if method not in _EDGE_DETECTION_METHODS :
            raise ValueError(
                f"Edge detection method '{method}' is not supported. "
                f"Valid methods are: {', '.join(_EDGE_DETECTION_METHODS.keys())}."
            )
        else :
            self.image = _EDGE_DETECTION_METHODS.get(method)(self.image, t1, t2)
        return self

    @requires_image
    def morphological(self, op='erode', ksize=3, iterations=1):
        kernel = np.ones((ksize, ksize), np.uint8)
        if op not in _MORPHS :
            raise ValueError(
                f"Morphological operation '{op}' is not supported. "
                f"Valid operations are: {', '.join(_MORPHS.keys())}."
            )
        else :
            self.image = cv2.morphologyEx(self.image, _MORPHS.get(op), kernel, iterations=iterations)
        return self

    @requires_image
    def denoise(self, strength=10):
        """Removes noise while keeping details."""
        if self.image.ndim == 3 and self.image.shape[2] == 4:
            # fastNlMeansDenoisingColored requires exactly 3 channels;
            # denoise the color channels and keep alpha unchanged.
            bgr, alpha = self.image[:, :, :3], self.image[:, :, 3]
            denoised = cv2.fastNlMeansDenoisingColored(bgr, None, strength, 10, 7, 21)
            self.image = np.dstack([denoised, alpha])
        elif len(self.image.shape) == 3:
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, strength, 10, 7, 21)
        else:
            self.image = cv2.fastNlMeansDenoising(self.image, None, strength, 7, 21)
        return self

    @requires_image
    def segment_threshold(self, type='otsu', val=127):
        gray = as_gray(self.image, self.color_space)

        if type not in _THRESHOLD_TYPES :
            raise ValueError(
                f"Threshold type '{type}' is not supported. "
                f"Valid types are: {', '.join(_THRESHOLD_TYPES.keys())}."
            )
        self.image = _THRESHOLD_TYPES.get(type)(gray, val=val)
        return self

    @requires_image
    def remove_background_grabcut(self, rect):
        """rect = (x, y, w, h) of the foreground object."""
        mask = np.zeros(self.image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        self.image = self.image * mask2[:, :, np.newaxis]
        return self

    @requires_image
    def add_noise(self, noise_type="gaussian"):
        """Adds noise in-place. Works on both grayscale and color images."""
        if noise_type not in _NOISE_TYPES :
            raise ValueError(
                f"Noise type '{noise_type}' is not supported. "
                f"Valid types are: {', '.join(_NOISE_TYPES.keys())}."
            )
        else :
            self.image = _NOISE_TYPES.get(noise_type)(self.image)
        return self