"""Setup script for Math & STEM Content Processor"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="math-stem-processor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive library for extracting and processing mathematical content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/math-stem-processor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "lxml>=4.6.0",
        "pyyaml>=5.4.0",
        "regex>=2021.8.0",
    ],
    extras_require={
        "pdf": ["PyPDF2>=2.0.0", "pdfplumber>=0.5.0"],
        "ocr": ["pytesseract>=0.3.0", "opencv-python>=4.5.0"],
        "rendering": ["matplotlib>=3.3.0", "cairosvg>=2.5.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.10.0", "black>=21.0", "flake8>=3.9.0"],
    },
)
