import cv2
import numpy as np
import hashlib
from .utils import ensure_valid, check_dependency

class AnalysisMixin:
    """Handles Statistics, Hashing, Histograms, and EXIF."""

    @ensure_valid
    def get_stats(self):
        """Returns basic statistics."""
        return {
            "width": self.image.shape[1],
            "height": self.image.shape[0],
            "channels": self.image.shape[2] if len(self.image.shape) > 2 else 1,
            "mean": np.mean(self.image),
            "std": np.std(self.image),
            "min": np.min(self.image),
            "max": np.max(self.image),
            "dtype": str(self.image.dtype)
        }

    @ensure_valid
    def compute_hash_perceptual(self, size=8):
        """Perceptive hash (aHash) for visual similarity."""
        resized = cv2.resize(self.image, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 2 else resized
        avg = gray.mean()
        diff = gray > avg
        # Convert binary array to hex string
        return hex(sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v]))[2:]

    @ensure_valid
    def compute_hash_md5(self):
        """Cryptographic MD5 hash (Exact duplicate check)."""
        return hashlib.md5(self.image.tobytes()).hexdigest()

    @ensure_valid
    def compute_histogram(self, bins=256):
        """Returns histograms for each channel."""
        if len(self.image.shape) == 3:
            colors = ('b', 'g', 'r')
            hists = {}
            for i, col in enumerate(colors):
                hists[col] = cv2.calcHist([self.image], [i], None, [bins], [0, 256]).flatten()
            return hists
        else:
            return {'gray': cv2.calcHist([self.image], [0], None, [bins], [0, 256]).flatten()}

    def get_exif(self, file_path):
        """
        Extracts EXIF metadata.
        NOTE: Requires the original file path, as numpy arrays lose metadata.
        """
        Image = check_dependency("PIL.Image")
        ExifTags = check_dependency("PIL.ExifTags")

        if not Image or not ExifTags:
            print("Warning: Pillow not installed. Cannot extract EXIF.")
            return {}

        try:
            pil_img = Image.open(file_path)
            exif_data = {}
            if hasattr(pil_img, '_getexif') and pil_img._getexif():
                exif = pil_img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value) # Convert to string for safety
            return exif_data
        except Exception as e:
            print(f"Error reading EXIF: {e}")
            return {}

    @ensure_valid
    def compute_ssim(self, other_img):
        """
        Computes Structural Similarity Index (SSIM) between self and other_img.
        Returns float 0.0 to 1.0 (1.0 = identical).
        """
        # Resize other to match
        h, w = self.image.shape[:2]
        other_resized = cv2.resize(other_img, (w, h))

        # Convert to gray
        grayA = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(other_resized, cv2.COLOR_BGR2GRAY)

        # Compute stats
        mu1 = cv2.GaussianBlur(grayA, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(grayB, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(grayA ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(grayB ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(grayA * grayB, (11, 11), 1.5) - mu1_mu2

        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


    @ensure_valid
    def split_channels(self):
        """Returns list of [Blue, Green, Red] numpy arrays."""
        if len(self.image.shape) == 3:
            return cv2.split(self.image)
        return [self.image]

    @ensure_valid
    def merge_channels(self, channels):
        """Merges [B, G, R] channels into the current image."""
        self.image = cv2.merge(channels)
        return self

    @ensure_valid
    def get_stats_detailed(self):
        """Global AND Per-Channel Statistics."""
        stats = self.get_stats() # Call existing
        if len(self.image.shape) == 3:
            colors = ('blue', 'green', 'red')
            for i, col in enumerate(colors):
                c_data = self.image[:, :, i]
                stats[f'mean_{col}'] = np.mean(c_data)
                stats[f'std_{col}'] = np.std(c_data)
        return stats

    @ensure_valid
    def compute_hu_moments(self):
        """Calculates Hu Moments (Shape Descriptors)."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments)
        # Log scale for readability
        for i in range(0,7):
            hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]))
        return hu.flatten()

    @ensure_valid
    def auto_select_format(self):
        """Returns '.png' for graphic/flat art, '.jpg' for photos."""
        # Heuristic: High number of unique colors usually means photo
        pixels = self.image.reshape(-1, self.image.shape[-1])
        unique_colors = len(np.unique(pixels, axis=0))
        if unique_colors < 10000: # Arbitrary threshold
            return '.png'
        return '.jpg'
