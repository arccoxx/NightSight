"""
NightSight v2: Military Night Vision-Inspired Enhancement Model

Complete model integrating all v2 features:
- Depth estimation
- Zero-DCE++ enhancement
- Edge detection and glowing outlines
- Object detection and tracking
- Super-resolution
- Scene-adaptive processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import cv2

from nightsight.core.base import BaseModel
from nightsight.core.registry import ModelRegistry
from nightsight.v2.modules.depth_estimator import DepthEstimator
from nightsight.v2.modules.zerodce_plus import ZeroDCEPlusPlus
from nightsight.v2.modules.edge_outliner import EdgeOutliner
from nightsight.v2.modules.object_detector import ObjectDetector
from nightsight.v2.modules.tracker import MultiObjectTracker
from nightsight.v2.modules.super_resolution import SuperResolution
from nightsight.v2.modules.scene_classifier import SceneClassifier


@ModelRegistry.register("nightsight_v2")
class NightSightV2(BaseModel):
    """
    NightSight v2: Military night vision-inspired enhancement.

    Advanced low-light enhancement with depth-based object differentiation,
    bright outline overlays, and comprehensive AI-powered enhancements.
    """

    def __init__(
        self,
        device: str = "auto",
        use_depth: bool = True,
        use_zerodce: bool = True,
        use_edges: bool = True,
        use_detection: bool = True,
        use_tracking: bool = True,
        use_superres: bool = False,
        use_adaptive: bool = True,
        depth_model_path: Optional[str] = None,
        zerodce_model_path: Optional[str] = None,
        edge_model_path: Optional[str] = None,
        detector_model_path: Optional[str] = None,
        sr_model_path: Optional[str] = None,
        scene_model_path: Optional[str] = None
    ):
        """
        Initialize NightSight v2.

        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
            use_depth: Enable depth estimation
            use_zerodce: Enable Zero-DCE++ enhancement
            use_edges: Enable edge detection and outlines
            use_detection: Enable object detection
            use_tracking: Enable object tracking
            use_superres: Enable super-resolution
            use_adaptive: Enable scene-adaptive processing
            depth_model_path: Path to depth model weights
            zerodce_model_path: Path to Zero-DCE++ weights
            edge_model_path: Path to edge detector weights
            detector_model_path: Path to YOLO weights
            sr_model_path: Path to super-resolution weights
            scene_model_path: Path to scene classifier weights
        """
        super().__init__()

        self.device = self._get_device(device)

        # Module flags
        self.use_depth = use_depth
        self.use_zerodce = use_zerodce
        self.use_edges = use_edges
        self.use_detection = use_detection
        self.use_tracking = use_tracking
        self.use_superres = use_superres
        self.use_adaptive = use_adaptive

        # Initialize modules
        if use_depth:
            self.depth_estimator = DepthEstimator(
                model_path=depth_model_path,
                device=device
            )

        if use_zerodce:
            self.zerodce = ZeroDCEPlusPlus(
                model_path=zerodce_model_path,
                device=device
            )

        if use_edges:
            self.edge_outliner = EdgeOutliner(
                model_path=edge_model_path,
                device=device,
                use_deep_edges=True,
                use_traditional=True
            )

        if use_detection:
            self.object_detector = ObjectDetector(
                model_path=detector_model_path,
                device=device
            )

        if use_tracking:
            self.multi_tracker = MultiObjectTracker()

        if use_superres:
            self.super_resolution = SuperResolution(
                model_path=sr_model_path,
                device=device,
                scale=2
            )

        if use_adaptive:
            self.scene_classifier = SceneClassifier(
                model_path=scene_model_path,
                device=device,
                use_ml=False  # Use statistical for now
            )

    def _get_device(self, device: str) -> str:
        """Get device string."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def forward(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_components: bool = False,
        config: Optional[Dict] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Process image through NightSight v2 pipeline.

        Args:
            image: Input image (H, W, 3) in [0, 1] or [0, 255]
            return_components: Return intermediate results
            config: Optional configuration override

        Returns:
            Enhanced image, optionally with component dictionary
        """
        # Ensure numpy and normalize
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            image = image.transpose(1, 2, 0)

        if image.max() > 1.0:
            image = image / 255.0

        components = {}

        # Get adaptive configuration
        if self.use_adaptive and config is None:
            config = self.scene_classifier.get_adaptive_config(image)
            components['scene_config'] = config

        # Default config if not provided
        if config is None:
            config = self._get_default_config()

        # Step 1: Low-light enhancement with Zero-DCE++
        enhanced = image.copy()
        if self.use_zerodce and config.get('modules', {}).get('zerodce', {}).get('enabled', True):
            enhanced = self.zerodce.enhance(enhanced, return_numpy=True)
            components['zerodce_enhanced'] = enhanced.copy()

        # Step 1.5: Adaptive brightness boost for very dark scenes
        if 'scene_config' in components:
            scene_class = components['scene_config']['scene']['class_name']
            scene_brightness = components['scene_config']['scene']['brightness']

            # Apply aggressive gamma correction for very dark scenes
            if scene_class == 'very_dark' or scene_brightness < 0.15:
                # Calculate adaptive gamma (darker = stronger correction)
                adaptive_gamma = 0.3 + (scene_brightness * 0.5)  # Range: 0.3-0.375 for very dark
                enhanced = np.power(enhanced, adaptive_gamma)

                # Additional brightness boost
                brightness_boost = 1.2 + (0.15 - scene_brightness) * 2.0  # Boost more for darker scenes
                enhanced = np.clip(enhanced * brightness_boost, 0, 1)

                # Apply CLAHE for local contrast enhancement
                enhanced_uint8 = (enhanced * 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

                # Apply CLAHE to each channel
                for i in range(3):
                    enhanced_uint8[:, :, i] = clahe.apply(enhanced_uint8[:, :, i])

                enhanced = enhanced_uint8.astype(np.float32) / 255.0
                components['adaptive_boosted'] = enhanced.copy()
            elif scene_class == 'dark' or scene_brightness < 0.25:
                # Moderate boost for dark scenes
                adaptive_gamma = 0.45
                enhanced = np.power(enhanced, adaptive_gamma)
                enhanced = np.clip(enhanced * 1.15, 0, 1)

                # Lighter CLAHE for dark (not very dark) scenes
                enhanced_uint8 = (enhanced * 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                for i in range(3):
                    enhanced_uint8[:, :, i] = clahe.apply(enhanced_uint8[:, :, i])
                enhanced = enhanced_uint8.astype(np.float32) / 255.0
                components['adaptive_boosted'] = enhanced.copy()

        # Step 2: Super-resolution (if enabled)
        if self.use_superres and config.get('modules', {}).get('super_resolution', {}).get('enabled', False):
            enhanced = self.super_resolution.enhance(enhanced, return_numpy=True)
            components['super_resolved'] = enhanced.copy()

        # Step 3: Depth estimation
        depth_map = None
        if self.use_depth and config.get('modules', {}).get('depth_estimator', {}).get('enabled', True):
            depth_map = self.depth_estimator.estimate(enhanced, return_numpy=True)
            components['depth_map'] = depth_map.copy()

        # Step 4: Edge detection
        edges = None
        if self.use_edges and config.get('modules', {}).get('edge_outliner', {}).get('enabled', True):
            edges = self.edge_outliner.detect_edges_combined(enhanced)
            components['edges'] = edges.copy()

        # Step 5: Apply glowing outlines
        outlined = enhanced.copy()
        if self.use_edges and edges is not None:
            edge_config = config.get('modules', {}).get('edge_outliner', {})

            if depth_map is not None and self.use_depth:
                # Depth-aware colored outlines
                outlined = self.edge_outliner.create_depth_aware_outlines(
                    enhanced,
                    depth_map,
                    edges=edges,
                    thickness=edge_config.get('thickness', 2),
                    blur_radius=5
                )
            else:
                # Standard glowing outlines
                outlined = self.edge_outliner.apply_outline_to_image(
                    enhanced,
                    edges=edges,
                    glow_color=edge_config.get('color', (0, 255, 0)),
                    thickness=edge_config.get('thickness', 2),
                    intensity=edge_config.get('intensity', 0.8)
                )

            components['with_outlines'] = outlined.copy()

        # Step 6: Object detection
        detections = []
        if self.use_detection and config.get('modules', {}).get('object_detector', {}).get('enabled', True):
            detections = self.object_detector.detect(outlined)
            components['detections'] = detections

        # Step 7: Draw detection boxes
        result = outlined.copy()
        if self.use_detection and len(detections) > 0:
            result = self.object_detector.draw_glowing_boxes(
                result,
                detections,
                glow_radius=10,
                intensity=0.8
            )
            components['with_detections'] = result.copy()

        # Ensure result is in proper range
        result = np.clip(result, 0, 1)

        if return_components:
            return result, components

        return result

    def process_video_frame(
        self,
        frame: np.ndarray,
        config: Optional[Dict] = None,
        update_tracking: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single video frame with tracking.

        Args:
            frame: Input frame (H, W, 3)
            config: Optional configuration
            update_tracking: Update object tracker

        Returns:
            Enhanced frame and tracking info
        """
        # Process frame
        enhanced, components = self.forward(frame, return_components=True, config=config)

        # Update tracker if enabled
        tracks = []
        if self.use_tracking and update_tracking and 'detections' in components:
            tracks = self.multi_tracker.update(components['detections'])
            components['tracks'] = tracks

            # Draw tracks
            if len(tracks) > 0:
                enhanced = self.multi_tracker.draw_glowing_tracks(
                    enhanced,
                    tracks,
                    glow_radius=5
                )

        return enhanced, components

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'modules': {
                'zerodce': {'enabled': self.use_zerodce},
                'depth_estimator': {'enabled': self.use_depth},
                'edge_outliner': {
                    'enabled': self.use_edges,
                    'color': (0, 255, 0),
                    'thickness': 2,
                    'intensity': 0.8
                },
                'super_resolution': {'enabled': self.use_superres},
                'object_detector': {'enabled': self.use_detection},
                'tracker': {'enabled': self.use_tracking}
            }
        }

    def reset_tracker(self):
        """Reset object tracker state."""
        if self.use_tracking:
            self.multi_tracker = MultiObjectTracker()

    def get_config(self) -> Dict:
        """Get current configuration."""
        return {
            'device': self.device,
            'use_depth': self.use_depth,
            'use_zerodce': self.use_zerodce,
            'use_edges': self.use_edges,
            'use_detection': self.use_detection,
            'use_tracking': self.use_tracking,
            'use_superres': self.use_superres,
            'use_adaptive': self.use_adaptive
        }

    def set_module_enabled(self, module_name: str, enabled: bool):
        """
        Enable or disable a specific module.

        Args:
            module_name: Name of module ('depth', 'zerodce', 'edges', 'detection', 'tracking', 'superres', 'adaptive')
            enabled: Whether to enable the module
        """
        if module_name == 'depth':
            self.use_depth = enabled
        elif module_name == 'zerodce':
            self.use_zerodce = enabled
        elif module_name == 'edges':
            self.use_edges = enabled
        elif module_name == 'detection':
            self.use_detection = enabled
        elif module_name == 'tracking':
            self.use_tracking = enabled
        elif module_name == 'superres':
            self.use_superres = enabled
        elif module_name == 'adaptive':
            self.use_adaptive = enabled
        else:
            raise ValueError(f"Unknown module: {module_name}")
