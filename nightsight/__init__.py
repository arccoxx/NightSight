"""
NightSight: Advanced Night Vision Enhancement Package

A comprehensive Python package for low-light and night vision image enhancement
combining state-of-the-art deep learning techniques with traditional image processing.

Key Features:
- Multiple deep learning architectures (Zero-DCE, Retinexformer, Diffusion Models)
- Traditional image processing (Retinex, CLAHE, Bilateral Filtering)
- Multi-frame temporal fusion for video enhancement
- Physics-guided enhancement with noise modeling
- Real-time inference support
- Modular and extensible architecture
"""

__version__ = "0.1.0"
__author__ = "NightSight Team"

from nightsight.core.base import BaseEnhancer
from nightsight.core.registry import ModelRegistry

# Convenient imports for common usage
from nightsight.pipelines.single_image import SingleImagePipeline
from nightsight.pipelines.video import VideoPipeline

__all__ = [
    "BaseEnhancer",
    "ModelRegistry",
    "SingleImagePipeline",
    "VideoPipeline",
    "__version__",
]
