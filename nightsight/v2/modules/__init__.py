"""
NightSight v2 modules for various enhancement components.
"""

from nightsight.v2.modules.depth_estimator import DepthEstimator
from nightsight.v2.modules.edge_outliner import EdgeOutliner
from nightsight.v2.modules.object_detector import ObjectDetector
from nightsight.v2.modules.tracker import MultiObjectTracker
from nightsight.v2.modules.scene_classifier import SceneClassifier
from nightsight.v2.modules.super_resolution import SuperResolution
from nightsight.v2.modules.zerodce_plus import ZeroDCEPlusPlus

__all__ = [
    'DepthEstimator',
    'EdgeOutliner',
    'ObjectDetector',
    'MultiObjectTracker',
    'SceneClassifier',
    'SuperResolution',
    'ZeroDCEPlusPlus'
]
