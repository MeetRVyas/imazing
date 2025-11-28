
from setuptools import setup, find_packages

setup(
    name="imazing",
    version="1.0.0",
    description="Image & Video Processing",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "opencv-contrib-python",
        "numpy",
        "requests",
        "pyautogui",
        "pyperclip",
        "pillow",
        "pytesseract",
        "pyzbar",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
