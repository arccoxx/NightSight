"""
Multi-object tracking with trajectory prediction.

Tracks detected objects across frames and predicts their motion
trajectories for improved awareness in night vision applications.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import deque


class KalmanTracker:
    """
    Kalman filter-based tracker for a single object.

    Tracks object position and velocity, predicts future positions.
    """

    def __init__(self, initial_bbox: np.ndarray, track_id: int):
        """
        Initialize Kalman tracker.

        Args:
            initial_bbox: Initial bounding box [x1, y1, x2, y2]
            track_id: Unique track ID
        """
        self.track_id = track_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        # State: [x_center, y_center, area, aspect_ratio, vx, vy, v_area, v_aspect]
        self.kf = cv2.KalmanFilter(8, 4)

        # State transition matrix
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        # Measurement matrix
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        # Process noise
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01

        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        # Initialize state from bbox
        x1, y1, x2, y2 = initial_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        aspect = (x2 - x1) / (y2 - y1 + 1e-8)

        self.kf.statePost = np.array([cx, cy, area, aspect, 0, 0, 0, 0], dtype=np.float32)

        # Track history for trajectory
        self.history = deque(maxlen=30)
        self.history.append((cx, cy))

    def predict(self) -> np.ndarray:
        """
        Predict next state.

        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        state = self.kf.predict()
        cx, cy, area, aspect = state[0], state[1], state[2], state[3]

        # Convert back to bbox
        w = np.sqrt(area * aspect)
        h = area / w

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return np.array([x1, y1, x2, y2])

    def update(self, bbox: np.ndarray):
        """
        Update tracker with new measurement.

        Args:
            bbox: Measured bounding box [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        aspect = (x2 - x1) / (y2 - y1 + 1e-8)

        measurement = np.array([cx, cy, area, aspect], dtype=np.float32)
        self.kf.correct(measurement)

        self.hits += 1
        self.time_since_update = 0

        # Update history
        self.history.append((cx, cy))

    def get_state(self) -> np.ndarray:
        """Get current state as bounding box."""
        state = self.kf.statePost
        cx, cy, area, aspect = state[0], state[1], state[2], state[3]

        w = np.sqrt(area * aspect)
        h = area / w

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return np.array([x1, y1, x2, y2])

    def predict_trajectory(self, num_steps: int = 10) -> List[Tuple[float, float]]:
        """
        Predict future trajectory.

        Args:
            num_steps: Number of future steps to predict

        Returns:
            List of (x, y) center positions
        """
        # Save current state
        saved_state = self.kf.statePost.copy()
        saved_cov = self.kf.errorCovPost.copy()

        trajectory = []

        for _ in range(num_steps):
            state = self.kf.predict()
            trajectory.append((state[0], state[1]))

        # Restore state
        self.kf.statePost = saved_state
        self.kf.errorCovPost = saved_cov

        return trajectory


class MultiObjectTracker:
    """
    Multi-object tracker with trajectory prediction.

    Tracks multiple objects across frames using Kalman filters
    and Hungarian algorithm for data association.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize multi-object tracker.

        Args:
            max_age: Maximum frames to keep track alive without updates
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.next_id = 0

    def update(
        self,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts with 'bbox' key

        Returns:
            List of tracked objects with track IDs and predictions
        """
        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()
            tracker.time_since_update += 1
            tracker.age += 1

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections(
            detections,
            self.trackers
        )

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(np.array(detections[det_idx]['bbox']))

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_tracker = KalmanTracker(
                np.array(detections[det_idx]['bbox']),
                self.next_id
            )
            self.next_id += 1
            self.trackers.append(new_tracker)

        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # Return confirmed tracks
        results = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or tracker.age <= self.min_hits:
                bbox = tracker.get_state()
                results.append({
                    'track_id': tracker.track_id,
                    'bbox': bbox.tolist(),
                    'age': tracker.age,
                    'hits': tracker.hits,
                    'history': list(tracker.history),
                    'predicted_trajectory': tracker.predict_trajectory()
                })

        return results

    def _associate_detections(
        self,
        detections: List[Dict],
        trackers: List[KalmanTracker]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to trackers using IoU.

        Returns:
            matched pairs, unmatched detections, unmatched trackers
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(trackers)))

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))

        for d, det in enumerate(detections):
            for t, tracker in enumerate(trackers):
                iou_matrix[d, t] = self._compute_iou(
                    np.array(det['bbox']),
                    tracker.get_state()
                )

        # Use greedy matching (simple but fast)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

        # Match in order of IoU
        while len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            # Find best match
            max_iou = 0
            best_det = -1
            best_trk = -1

            for d in unmatched_dets:
                for t in unmatched_trks:
                    if iou_matrix[d, t] > max_iou:
                        max_iou = iou_matrix[d, t]
                        best_det = d
                        best_trk = t

            # If best match is above threshold, add it
            if max_iou > self.iou_threshold:
                matched.append((best_det, best_trk))
                unmatched_dets.remove(best_det)
                unmatched_trks.remove(best_trk)
            else:
                break

        return matched, unmatched_dets, unmatched_trks

    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-8)

    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Dict],
        show_trajectory: bool = True,
        show_prediction: bool = True,
        trajectory_color: Tuple[int, int, int] = (0, 255, 255),
        prediction_color: Tuple[int, int, int] = (255, 255, 0)
    ) -> np.ndarray:
        """
        Draw tracks with trajectories on image.

        Args:
            image: Input image (H, W, 3)
            tracks: List of track dicts
            show_trajectory: Show past trajectory
            show_prediction: Show predicted future trajectory
            trajectory_color: RGB color for past trajectory
            prediction_color: RGB color for future prediction

        Returns:
            Image with drawn tracks (H, W, 3)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        for track in tracks:
            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in track['bbox']]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID
            label = f"ID: {track['track_id']}"
            cv2.putText(
                img_vis,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Draw past trajectory
            if show_trajectory and len(track['history']) > 1:
                history = track['history']
                for i in range(len(history) - 1):
                    pt1 = (int(history[i][0]), int(history[i][1]))
                    pt2 = (int(history[i + 1][0]), int(history[i + 1][1]))

                    # Fade older points
                    alpha = (i + 1) / len(history)
                    color = tuple(int(c * alpha) for c in trajectory_color)

                    cv2.line(img_vis, pt1, pt2, color, 2)
                    cv2.circle(img_vis, pt2, 3, color, -1)

            # Draw predicted trajectory
            if show_prediction and 'predicted_trajectory' in track:
                traj = track['predicted_trajectory']
                if len(traj) > 0:
                    # Start from current position
                    history = track['history']
                    if len(history) > 0:
                        last_pt = (int(history[-1][0]), int(history[-1][1]))

                        for i, (x, y) in enumerate(traj):
                            pt = (int(x), int(y))

                            # Fade future points
                            alpha = 1.0 - (i / len(traj)) * 0.7
                            color = tuple(int(c * alpha) for c in prediction_color)

                            cv2.line(img_vis, last_pt, pt, color, 2, cv2.LINE_AA)
                            cv2.circle(img_vis, pt, 2, color, -1)

                            last_pt = pt

        return img_vis

    def draw_glowing_tracks(
        self,
        image: np.ndarray,
        tracks: List[Dict],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        trajectory_color: Tuple[int, int, int] = (0, 255, 255),
        glow_radius: int = 5
    ) -> np.ndarray:
        """
        Draw tracks with glowing effect (military night vision style).

        Args:
            image: Input image (H, W, 3)
            tracks: List of track dicts
            box_color: RGB color for boxes
            trajectory_color: RGB color for trajectories
            glow_radius: Glow blur radius

        Returns:
            Image with glowing tracks (H, W, 3)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_vis = image.copy()

        # Create glow layer
        glow_layer = np.zeros_like(img_vis)

        for track in tracks:
            # Draw box
            x1, y1, x2, y2 = [int(v) for v in track['bbox']]
            cv2.rectangle(glow_layer, (x1, y1), (x2, y2), box_color, 3)

            # Draw track ID
            label = f"{track['track_id']}"
            cv2.putText(
                glow_layer,
                label,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Draw trajectory
            if 'history' in track and len(track['history']) > 1:
                history = track['history']
                for i in range(len(history) - 1):
                    pt1 = (int(history[i][0]), int(history[i][1]))
                    pt2 = (int(history[i + 1][0]), int(history[i + 1][1]))
                    cv2.line(glow_layer, pt1, pt2, trajectory_color, 3)

        # Apply glow
        glow_blurred = cv2.GaussianBlur(
            glow_layer,
            (glow_radius * 2 + 1, glow_radius * 2 + 1),
            0
        )

        # Combine
        result = cv2.add(img_vis, glow_layer)
        result = cv2.addWeighted(result, 1.0, glow_blurred, 0.5, 0)

        return result
