"""
Object detection module using YOLOv8n for real-time detection.

Detects and highlights objects with bright bounding boxes for
enhanced visibility in low-light night vision scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union


class ObjectDetector:
    """
    Object detector using YOLOv8n for real-time detection.

    Provides object detection with highlighted bounding boxes
    and support for filtering specific classes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_ultralytics: bool = True
    ):
        """
        Initialize object detector.

        Args:
            model_path: Path to YOLO model weights or 'yolov8n.pt' for pretrained
            device: Device to run on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_ultralytics: Use ultralytics YOLO (if available)
        """
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_ultralytics = use_ultralytics

        # Try to use ultralytics YOLO
        if use_ultralytics:
            try:
                from ultralytics import YOLO
                model_path = model_path or 'yolov8n.pt'
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.backend = 'ultralytics'
            except ImportError:
                print("Warning: ultralytics not installed. Using custom lightweight detector.")
                self.backend = 'custom'
                self.model = self._create_custom_detector()
        else:
            self.backend = 'custom'
            self.model = self._create_custom_detector()

        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _get_device(self, device: str) -> str:
        """Get device string."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _create_custom_detector(self):
        """Create custom lightweight detector as fallback."""
        print("Note: Using placeholder detector. Install ultralytics for full YOLO support.")
        return None

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Detect objects in image.

        Args:
            image: Input image (H, W, 3)
            classes: List of class names to detect (None for all)

        Returns:
            List of detections, each dict containing:
                - bbox: [x1, y1, x2, y2]
                - conf: confidence score
                - class: class name
                - class_id: class ID
        """
        if self.backend == 'ultralytics':
            return self._detect_ultralytics(image, classes)
        else:
            return self._detect_custom(image, classes)

    def _detect_ultralytics(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Detect using ultralytics YOLO."""
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        # Parse results
        detections = []
        boxes = results.boxes

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'

            # Filter by class if specified
            if classes is not None and class_name not in classes:
                continue

            detections.append({
                'bbox': bbox.tolist(),
                'conf': float(conf),
                'class': class_name,
                'class_id': class_id
            })

        return detections

    def _detect_custom(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Custom detection (placeholder - returns empty)."""
        # This is a placeholder. For full functionality, install ultralytics.
        # Could implement a simple sliding window detector or other lightweight method here.
        return []

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        glow_effect: bool = True,
        show_labels: bool = True,
        show_conf: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (H, W, 3)
            detections: List of detection dicts
            color: RGB color for boxes
            thickness: Box line thickness
            glow_effect: Add glowing effect to boxes
            show_labels: Show class labels
            show_conf: Show confidence scores

        Returns:
            Image with drawn boxes (H, W, 3)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]

            # Draw bounding box
            if glow_effect:
                # Draw thicker semi-transparent outer glow
                overlay = img_vis.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness + 4)
                cv2.addWeighted(overlay, 0.4, img_vis, 0.6, 0, img_vis)

            # Draw main box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            if show_labels:
                label = det['class']
                if show_conf:
                    label += f" {det['conf']:.2f}"

                # Get label size
                (label_w, label_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )

                # Draw label background
                cv2.rectangle(
                    img_vis,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w + 5, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    img_vis,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        return img_vis

    def draw_glowing_boxes(
        self,
        image: np.ndarray,
        detections: List[Dict],
        base_color: Tuple[int, int, int] = (0, 255, 0),
        glow_radius: int = 10,
        intensity: float = 0.8
    ) -> np.ndarray:
        """
        Draw boxes with strong glowing effect (military night vision style).

        Args:
            image: Input image (H, W, 3)
            detections: List of detection dicts
            base_color: Base RGB color
            glow_radius: Radius of glow effect
            intensity: Glow intensity

        Returns:
            Image with glowing boxes (H, W, 3)
        """
        # Ensure image is uint8
        input_is_float = image.dtype != np.uint8
        if input_is_float:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        # Create separate layer for glowing boxes
        glow_layer = np.zeros_like(img_vis)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]

            # Draw box on glow layer
            cv2.rectangle(glow_layer, (x1, y1), (x2, y2), base_color, 3)

            # Draw corners for extra emphasis
            corner_len = min(30, (x2 - x1) // 4, (y2 - y1) // 4)

            # Top-left
            cv2.line(glow_layer, (x1, y1), (x1 + corner_len, y1), base_color, 4)
            cv2.line(glow_layer, (x1, y1), (x1, y1 + corner_len), base_color, 4)

            # Top-right
            cv2.line(glow_layer, (x2, y1), (x2 - corner_len, y1), base_color, 4)
            cv2.line(glow_layer, (x2, y1), (x2, y1 + corner_len), base_color, 4)

            # Bottom-left
            cv2.line(glow_layer, (x1, y2), (x1 + corner_len, y2), base_color, 4)
            cv2.line(glow_layer, (x1, y2), (x1, y2 - corner_len), base_color, 4)

            # Bottom-right
            cv2.line(glow_layer, (x2, y2), (x2 - corner_len, y2), base_color, 4)
            cv2.line(glow_layer, (x2, y2), (x2, y2 - corner_len), base_color, 4)

        # Apply glow effect
        glow_blurred = cv2.GaussianBlur(
            glow_layer,
            (glow_radius * 2 + 1, glow_radius * 2 + 1),
            0
        )

        # Combine with original image
        result = cv2.addWeighted(img_vis, 1.0, glow_layer, intensity, 0)
        result = cv2.addWeighted(result, 1.0, glow_blurred, intensity * 0.5, 0)

        # Convert back to float32 [0, 1] if input was float
        if input_is_float:
            result = result.astype(np.float32) / 255.0

        return result

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get distinct color for each class.

        Args:
            class_name: Name of the class

        Returns:
            RGB color tuple
        """
        # Predefined colors for common classes
        color_map = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),          # Red
            'truck': (255, 100, 0),      # Orange
            'bicycle': (0, 255, 255),    # Cyan
            'motorcycle': (255, 0, 255), # Magenta
            'chair': (255, 255, 0),      # Yellow
            'couch': (128, 0, 255),      # Purple
            'dog': (0, 128, 255),        # Light blue
            'cat': (255, 128, 0),        # Orange
        }

        return color_map.get(class_name, (0, 255, 0))

    def draw_class_colored_boxes(
        self,
        image: np.ndarray,
        detections: List[Dict],
        glow_effect: bool = True
    ) -> np.ndarray:
        """
        Draw boxes with different colors for each class.

        Args:
            image: Input image (H, W, 3)
            detections: List of detection dicts
            glow_effect: Add glow effect

        Returns:
            Image with colored boxes (H, W, 3)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        # Draw each detection with class-specific color
        for det in detections:
            color = self.get_class_color(det['class'])

            # Create temporary image with single detection
            temp_dets = [det]

            if glow_effect:
                img_vis = self.draw_glowing_boxes(
                    img_vis,
                    temp_dets,
                    base_color=color
                )
            else:
                img_vis = self.draw_detections(
                    img_vis,
                    temp_dets,
                    color=color,
                    glow_effect=False
                )

        return img_vis
