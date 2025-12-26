"""Traditional image processing techniques for low-light enhancement."""

from nightsight.traditional.retinex import (
    single_scale_retinex,
    multi_scale_retinex,
    msrcr,
    RetinexEnhancer,
)
from nightsight.traditional.histogram import (
    histogram_equalization,
    clahe,
    adaptive_gamma,
    CLAHEEnhancer,
)
from nightsight.traditional.filters import (
    bilateral_filter,
    guided_filter,
    nlm_denoise,
    unsharp_mask,
)
from nightsight.traditional.frequency import (
    fft_filter,
    wavelet_denoise,
    homomorphic_filter,
)
from nightsight.traditional.edge import (
    detect_edges,
    enhance_edges,
    structure_tensor,
)
from nightsight.traditional.motion import (
    optical_flow,
    motion_detection,
    frame_difference,
)

__all__ = [
    # Retinex
    "single_scale_retinex",
    "multi_scale_retinex",
    "msrcr",
    "RetinexEnhancer",
    # Histogram
    "histogram_equalization",
    "clahe",
    "adaptive_gamma",
    "CLAHEEnhancer",
    # Filters
    "bilateral_filter",
    "guided_filter",
    "nlm_denoise",
    "unsharp_mask",
    # Frequency
    "fft_filter",
    "wavelet_denoise",
    "homomorphic_filter",
    # Edge
    "detect_edges",
    "enhance_edges",
    "structure_tensor",
    # Motion
    "optical_flow",
    "motion_detection",
    "frame_difference",
]
