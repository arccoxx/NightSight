"""
NightSight v2: Military Night Vision-Inspired Enhancement

Advanced low-light enhancement system inspired by military night vision technology.
Features depth-based object differentiation, bright outline superimposition,
and a comprehensive suite of AI-powered enhancements.

Key Features:
- Depth estimation for object differentiation
- Bright outline overlay (military night vision style)
- Zero-DCE++ low-light enhancement
- Edge detection with glowing outlines
- YOLOv8n object detection
- Super-resolution and denoising
- Color restoration
- Multi-object tracking with trajectories
- Scene-adaptive processing
"""

from nightsight.v2.models.nightsight_v2 import NightSightV2
from nightsight.v2.pipeline import NightSightV2Pipeline
from nightsight.v2.modules.depth_estimator import DepthEstimator
from nightsight.v2.modules.edge_outliner import EdgeOutliner
from nightsight.v2.modules.object_detector import ObjectDetector
from nightsight.v2.modules.tracker import MultiObjectTracker

__all__ = [
    'NightSightV2',
    'NightSightV2Pipeline',
    'DepthEstimator',
    'EdgeOutliner',
    'ObjectDetector',
    'MultiObjectTracker'
]

__version__ = '2.0.0'
