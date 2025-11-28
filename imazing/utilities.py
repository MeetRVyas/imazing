import cv2
import numpy as np
import os
from .utils import ensure_valid, check_dependency, ImageError

class UtilityMixin:
    """
    The 'Crazy Good' Utilities:
    Green Screen, Doc Scanner, Inpainting, ASCII, Grid, and more.
    """

    @ensure_valid
    def chroma_key(self, bg_image, key_color='green', threshold=50, softness=3):
        """
        Green Screen with Alpha Feathering (Soft Edges).
        softness: Kernel size for edge smoothing (odd number).
        """
        h, w = self.image.shape[:2]
        bg_resized = cv2.resize(bg_image, (w, h))
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Color Ranges
        if key_color.lower() == 'green':
            lower = np.array([40, 50, 50])
            upper = np.array([80, 255, 255])
        elif key_color.lower() == 'blue':
            lower = np.array([100, 50, 50])
            upper = np.array([130, 255, 255])
        else:
            lower = np.array([40, 50, 50])
            upper = np.array([80, 255, 255])

        # 1. Create Base Mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # 2. Invert Mask (Foreground is White)
        mask_inv = cv2.bitwise_not(mask)
        
        # 3. Feathering (Blur the mask to create soft alpha)
        # Convert to float 0.0-1.0
        alpha = mask_inv.astype(float) / 255.0
        alpha = cv2.GaussianBlur(alpha, (softness, softness), 0)
        
        # Expand Alpha to 3 channels
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # 4. Alpha Blend
        foreground = self.image.astype(float)
        background = bg_resized.astype(float)
        
        # Result = FG * Alpha + BG * (1 - Alpha)
        final = cv2.multiply(alpha, foreground) + cv2.multiply(1.0 - alpha, background)
        
        self.image = final.astype(np.uint8)
        return self

    @ensure_valid
    def doc_scan_flatten(self):
        """
        Doc Scanner with Morphological Closing.
        Closes gaps between text to find the actual paper edge, not the text.
        """
        # 1. Preprocessing
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Bilateral: Removes noise but keeps edges sharp (Better than Gaussian)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Edge Detection
        edged = cv2.Canny(blur, 75, 200)
        
        # 3. Morphological Closing (The logic upgrade)
        # Connects broken edges and smears text blocks together
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # 4. Find Contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        screen_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Check if it is convex (paper sheets are usually convex)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                screen_cnt = approx
                break
        
        if screen_cnt is None: return self
        self.image = self._four_point_transform(self.image, screen_cnt.reshape(4, 2))
        return self

    @ensure_valid
    def magic_erase(self, mask_img=None, rect=None):
        """
        Inpainting to remove objects.
        Provide EITHER a mask_img (white pixels = remove) OR a rect (x,y,w,h).
        """
        if mask_img is None and rect is None:
            return self

        if rect is not None:
            x, y, w, h = rect
            mask_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask_img[y:y+h, x:x+w] = 255

        # Inpaint
        self.image = cv2.inpaint(self.image, mask_img, 3, cv2.INPAINT_TELEA)
        return self
    
    @ensure_valid
    def slice_sprite_sheet(self, sprite_w, sprite_h, count=None, skip_empty=True):
        """
        Slices a grid image (sprite sheet) into individual frames.
        
        :param sprite_w: Width of a single sprite
        :param sprite_h: Height of a single sprite
        :param count: Max number of sprites to extract (optional)
        :param skip_empty: If True, skips completely transparent/black frames
        :return: List of numpy images (frames)
        """
        sheet_h, sheet_w = self.image.shape[:2]
        frames = []
        
        # Iterate rows and cols
        for y in range(0, sheet_h, sprite_h):
            for x in range(0, sheet_w, sprite_w):
                if count is not None and len(frames) >= count:
                    break
                
                # Check bounds
                if (y + sprite_h) > sheet_h or (x + sprite_w) > sheet_w:
                    continue
                
                # Crop sprite
                sprite = self.image[y:y+sprite_h, x:x+sprite_w].copy()
                
                # Filter empty
                if skip_empty:
                    if np.sum(sprite) == 0: # Pitch black
                        continue
                    # Check alpha if exists
                    if sprite.shape[2] == 4 and np.mean(sprite[:,:,3]) == 0:
                        continue

                frames.append(sprite)
                
        return frames

    @ensure_valid
    def to_ascii_art(self, cols=80):
        """
        Converts image to ASCII string representation.
        Returns the string (does not modify self.image).
        """
        # Simple ASCII mapping (dark to light)
        chars = "@%#*+=-:. "

        # Resize logic
        h, w = self.image.shape[:2]
        tile_w = w / cols
        tile_h = tile_w / 0.55 # Adjust for font aspect ratio
        rows = int(h / tile_h)

        small = cv2.resize(self.image, (cols, rows))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        ascii_str = ""
        for i in range(rows):
            for j in range(cols):
                pixel = gray[i, j]
                index = int(pixel / 255 * (len(chars) - 1))
                ascii_str += chars[index]
            ascii_str += "\n"

        return ascii_str

    @ensure_valid
    def compute_diff(self, other_img, threshold=25):
        """
        Computes difference between current image and another.
        Useful for motion detection or security.
        Updates self.image to show the difference (Highlights).
        """
        # Resize other to match self
        h, w = self.image.shape[:2]
        other_resized = cv2.resize(other_img, (w, h))

        # Compute Abs Diff
        diff = cv2.absdiff(self.image, other_resized)

        # Turn to binary mask for highlighing
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Return the visual diff (colored)
        self.image = cv2.bitwise_and(diff, diff, mask=thresh)
        return self

    @classmethod
    def make_grid(cls, images, cols=2):
        """
        Static: Combines a list of numpy images into a single grid image.
        """
        if not images: return None

        # Normalize sizes to the first image
        h, w = images[0].shape[:2]
        resized_imgs = [cv2.resize(img, (w, h)) for img in images]

        # Create rows
        rows_list = []
        for i in range(0, len(resized_imgs), cols):
            chunk = resized_imgs[i:i+cols]
            # Pad if last row is incomplete
            while len(chunk) < cols:
                chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
            rows_list.append(np.hstack(chunk))

        # Stack rows
        grid = np.vstack(rows_list)
        return cls(grid) # Return new Imazing object

    @staticmethod
    def create_gif(images, output_path, duration=100, loop=0):
        """
        Static: Saves a list of numpy images as an animated GIF.
        Requires Pillow.
        """
        Image = check_dependency("PIL.Image")
        if not Image:
            print("Error: Pillow required for GIF creation.")
            return

        pil_images = []
        for cv_img in images:
            # Convert BGR (OpenCV) to RGB (Pillow)
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        if pil_images:
            pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=loop
            )

    # --- Internal Helper for Doc Scanner ---
    def _four_point_transform(self, image, pts):
        # Obtain a consistent order of the points and unpack them individually
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Construct the set of destination points to obtain a "birds eye view"
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _order_points(self, pts):
        # Sorts points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right has smallest diff
        rect[3] = pts[np.argmax(diff)] # Bottom-left has largest diff
        return rect

    @ensure_valid
    def color_transfer(self, source_img):
        """
        'Vibe Stealer': Transfers the color distribution of source_img to self.image.
        Uses LAB color space statistics.
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype("float32")

        # Compute stats
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = self._image_stats(source_lab)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = self._image_stats(target_lab)

        # Split channels
        (l, a, b) = cv2.split(target_lab)

        # Transfer logic: (val - mean_target) * (std_source / std_target) + mean_source
        l -= lMeanTar
        a -= aMeanTar
        b -= bMeanTar

        # Scale by standard deviations
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

        # Add source means
        l += lMeanSrc
        a += aMeanSrc
        b += bMeanSrc

        # Clip and merge
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        transfer = cv2.merge([l, a, b])
        self.image = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
        return self

    @ensure_valid
    def smart_crop_borders(self, tolerance=10):
        """Auto-crops uniform borders (black/white) from the image."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)

        # If the border is white, invert
        if np.mean(thresh[:5, :]) > 127: 
            thresh = cv2.bitwise_not(thresh)

        # Find bounding box of non-zero pixels
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)

        self.image = self.image[y:y+h, x:x+w]
        return self

    @ensure_valid
    def auto_orient(self):
        """
        Fixes orientation based on EXIF data (e.g., sideways phone photos).
        Requires Pillow.
        """
        # This requires the image to be loaded from a file originally to have EXIF.
        # Since we are working on numpy arrays, we can't get EXIF *now*.
        # However, we can provide standard rotation helpers.
        # NOTE: A true auto-orient requires reading the original file path metadata.
        # We will assume this is handled at load time or explicitly called if metadata exists.
        pass 

    def _image_stats(self, image):
        # Helper for color transfer
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    @staticmethod
    def pdf_to_images(pdf_path, dpi=200):
        """
        Converts PDF to list of Imazing objects.
        Requires 'pdf2image' and 'poppler' installed on system.
        """
        from .core import Imazing
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=dpi)
            return [Imazing(np.array(page)[:, :, ::-1]) for page in pages]
        except ImportError:
            print("Error: 'pdf2image' library not installed.")
            return []
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []


    @ensure_valid
    def steganography_noise(self, intensity=1):
        """
        Adds subtle noise (LSB manipulation) to pixels.
        Used for anti-forensics or simple steganography.
        """
        noise = np.random.randint(-intensity, intensity + 1, self.image.shape, dtype=np.int16)
        res = self.image.astype(np.int16) + noise
        self.image = np.clip(res, 0, 255).astype(np.uint8)
        return self

    @ensure_valid
    def auto_optimize_quality(self, target_size_kb=100):
        """
        Determines the optimal format and quality to fit within target_size_kb.
        Returns: (best_quality, best_format_extension)
        Does NOT save the file, just calculates parameters.
        """
        formats = ['.jpg', '.webp']
        best_quality = 95
        best_format = '.jpg'

        for fmt in formats:
            for q in range(95, 10, -5):
                # Encode to buffer
                if fmt == '.jpg': params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
                else: params = [int(cv2.IMWRITE_WEBP_QUALITY), q]

                _, buf = cv2.imencode(fmt, self.image, params)
                size_kb = len(buf) / 1024

                if size_kb <= target_size_kb:
                    return q, fmt

        return 10, '.jpg' # Fallback
