#!/usr/bin/env python3
"""Setup script for NightSight."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ]

setup(
    name="nightsight",
    version="0.1.0",
    author="NightSight Team",
    description="Advanced night vision enhancement from standard cameras using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nightsight/nightsight",
    packages=find_packages(exclude=["tests", "scripts", "examples"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "full": [
            "rawpy>=0.18.0",
            "pywavelets>=1.3.0",
            "scipy>=1.7.0",
            "scikit-image>=0.19.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nightsight=scripts.inference:main",
            "nightsight-train=scripts.train:main",
            "nightsight-demo=scripts.demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="night-vision low-light enhancement deep-learning computer-vision",
)
