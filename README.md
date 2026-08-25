# imazing

A unified computer-vision toolkit for Python. One consistent, chainable API
for the image and video operations you end up rebuilding on every project:
transforms, filters, detection, OCR, drawing, analysis, and video streaming.


## Install

```bash
pip install imazing
```

Optional extras, installed as needed:

```bash
pip install "imazing[ocr]"      # text extraction (pytesseract)
pip install "imazing[qr]"       # QR code / barcode decoding (pyzbar)
pip install "imazing[desktop]"  # screen capture + clipboard image support
pip install "imazing[full]"     # everything above
```

## Quick start (Python)

```python
from imazing import Imazing

(
    Imazing("photo.jpg")
    .resize(width=800)
    .blur(method="gaussian", ksize=5)
    .adjust_brightness_contrast(alpha=1.1, beta=10)
    .save("photo_edited.jpg")
)
```

Every operation returns `self`, so calls chain naturally. Load from a file
path, a URL, raw bytes, a base64 data URI, or a NumPy array — `Imazing()`
detects the source type automatically.

## Quick start (CLI)

```bash
imazing resize photo.jpg out.jpg --width 800
imazing convert photo.jpg gray.jpg --mode gray
imazing info photo.jpg
imazing --help
```

## License

MIT — see [LICENSE](https://github.com/MeetRVyas/imazing/blob/main/LICENSE).