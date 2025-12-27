"""
NightSight v2 Pipeline for single images and video processing.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Union, Dict
from tqdm import tqdm

from nightsight.v2.models.nightsight_v2 import NightSightV2
from nightsight.utils.io import load_image, save_image


class NightSightV2Pipeline:
    """
    Complete pipeline for NightSight v2 enhancement.

    Provides easy-to-use interface for processing images and videos.
    """

    def __init__(
        self,
        device: str = "auto",
        use_all_features: bool = True,
        **model_kwargs
    ):
        """
        Initialize NightSight v2 pipeline.

        Args:
            device: Device to run on
            use_all_features: Enable all features by default
            **model_kwargs: Additional arguments for NightSightV2 model
        """
        if use_all_features:
            model_kwargs.setdefault('use_depth', True)
            model_kwargs.setdefault('use_zerodce', True)
            model_kwargs.setdefault('use_edges', True)
            model_kwargs.setdefault('use_detection', True)
            model_kwargs.setdefault('use_tracking', True)
            model_kwargs.setdefault('use_superres', False)  # Disabled by default (slow)
            model_kwargs.setdefault('use_adaptive', True)

        self.model = NightSightV2(device=device, **model_kwargs)

    def enhance_image(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
        return_components: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Enhance a single image.

        Args:
            image: Input image path or array
            output_path: Optional output path
            config: Optional configuration override
            return_components: Return intermediate results

        Returns:
            Enhanced image, optionally with components dict
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = load_image(str(image), dtype="float32")

        # Ensure float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0

        # Process
        result = self.model.forward(image, return_components=return_components, config=config)

        # Save if requested
        if output_path:
            if return_components:
                enhanced, components = result
                save_image((enhanced * 255).astype(np.uint8), output_path)
            else:
                enhanced = result
                save_image((enhanced * 255).astype(np.uint8), output_path)

        return result

    def enhance_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        config: Optional[Dict] = None,
        show_progress: bool = True
    ):
        """
        Enhance a video file.

        Args:
            video_path: Input video path
            output_path: Output video path
            config: Optional configuration override
            show_progress: Show progress bar
        """
        # Reset tracker for new video
        self.model.reset_tracker()

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        pbar = tqdm(total=total_frames, desc="Enhancing video") if show_progress else None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb.astype(np.float32) / 255.0

            # Process frame
            enhanced, _ = self.model.process_video_frame(frame_rgb, config=config)

            # Convert back to BGR uint8
            enhanced_bgr = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(enhanced_bgr)

            if pbar:
                pbar.update(1)

        # Cleanup
        cap.release()
        out.release()

        if pbar:
            pbar.close()

    def process_webcam(
        self,
        camera_id: int = 0,
        config: Optional[Dict] = None,
        display_fps: bool = True,
        save_output: Optional[str] = None
    ):
        """
        Process webcam feed in real-time.

        Args:
            camera_id: Camera device ID
            config: Optional configuration override
            display_fps: Show FPS counter
            save_output: Optional path to save output video
        """
        import time

        # Reset tracker
        self.model.reset_tracker()

        # Open camera
        cap = cv2.VideoCapture(camera_id)

        # Give camera time to initialize
        time.sleep(1.0)

        # Try reading a test frame to ensure camera is working
        ret, _ = cap.read()
        if not ret:
            cap.release()
            # Try reopening
            time.sleep(0.5)
            cap = cv2.VideoCapture(camera_id)
            time.sleep(1.0)

        if not cap.isOpened():
            print(f"\nError: Could not open camera {camera_id}")
            print("Please check:")
            print("  1. Camera is connected and not in use by another application")
            print("  2. Camera permissions are enabled")
            print("  3. Try a different camera ID with --camera argument")
            return

        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer if saving
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_output, fourcc, 20.0, (width, height))

        print("Press 'q' or ESC to quit, 's' to save screenshot")
        print("Keys 1-7: Toggle modules (1=depth, 2=zerodce, 3=edges, 4=detection, 5=tracking, 6=sr, 7=adaptive)")
        print("You can also close the window by clicking the X button")
        print()

        fps_history = []

        try:
            while True:
                start_time = time.time()

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = frame_rgb.astype(np.float32) / 255.0

                # Process frame
                enhanced, components = self.model.process_video_frame(frame_rgb, config=config)

                # Convert back to BGR
                enhanced_bgr = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                # Display FPS
                if display_fps:
                    fps = 1.0 / (time.time() - start_time + 1e-6)
                    fps_history.append(fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = np.mean(fps_history)

                    cv2.putText(
                        enhanced_bgr,
                        f"FPS: {avg_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    # Display active modules
                    active_modules = []
                    if self.model.use_zerodce:
                        active_modules.append("ZeroDCE")
                    if self.model.use_depth:
                        active_modules.append("Depth")
                    if self.model.use_edges:
                        active_modules.append("Edges")
                    if self.model.use_detection:
                        active_modules.append("Detect")
                    if self.model.use_tracking:
                        active_modules.append("Track")

                    modules_str = " | ".join(active_modules)
                    cv2.putText(
                        enhanced_bgr,
                        modules_str,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1
                    )

                # Display
                cv2.imshow('NightSight v2', enhanced_bgr)

                # Check if window was closed by user (clicking X)
                if cv2.getWindowProperty('NightSight v2', cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Save if requested
                if out:
                    out.write(enhanced_bgr)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"nightsight_v2_{timestamp}.png"
                    cv2.imwrite(filename, enhanced_bgr)
                    print(f"Saved screenshot: {filename}")
                elif key == ord('1'):
                    self.model.set_module_enabled('depth', not self.model.use_depth)
                    print(f"Depth estimation: {'ON' if self.model.use_depth else 'OFF'}")
                elif key == ord('2'):
                    self.model.set_module_enabled('zerodce', not self.model.use_zerodce)
                    print(f"Zero-DCE++: {'ON' if self.model.use_zerodce else 'OFF'}")
                elif key == ord('3'):
                    self.model.set_module_enabled('edges', not self.model.use_edges)
                    print(f"Edge outlines: {'ON' if self.model.use_edges else 'OFF'}")
                elif key == ord('4'):
                    self.model.set_module_enabled('detection', not self.model.use_detection)
                    print(f"Object detection: {'ON' if self.model.use_detection else 'OFF'}")
                elif key == ord('5'):
                    self.model.set_module_enabled('tracking', not self.model.use_tracking)
                    print(f"Object tracking: {'ON' if self.model.use_tracking else 'OFF'}")
                elif key == ord('6'):
                    self.model.set_module_enabled('superres', not self.model.use_superres)
                    print(f"Super-resolution: {'ON' if self.model.use_superres else 'OFF'}")
                elif key == ord('7'):
                    self.model.set_module_enabled('adaptive', not self.model.use_adaptive)
                    print(f"Adaptive processing: {'ON' if self.model.use_adaptive else 'OFF'}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError during webcam processing: {e}")
        finally:
            # Cleanup
            print("\nCleaning up...")
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

    def compare_with_v1(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Compare v2 enhancement with v1.

        Args:
            image: Input image
            output_path: Optional path to save comparison

        Returns:
            Comparison image (side-by-side)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = load_image(str(image), dtype="float32")

        # Ensure float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Process with v1
        from nightsight.models.hybrid import NightSightNet
        v1_model = NightSightNet()
        v1_model.eval()

        import torch
        with torch.no_grad():
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            v1_enhanced = v1_model(image_tensor).squeeze(0).numpy().transpose(1, 2, 0)

        # Process with v2
        v2_enhanced = self.model.forward(image)

        # Create comparison
        comparison = np.hstack([
            image,
            v1_enhanced,
            v2_enhanced
        ])

        # Add labels
        h, w = image.shape[:2]
        comparison_vis = (comparison * 255).astype(np.uint8)

        cv2.putText(comparison_vis, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_vis, "NightSight v1", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_vis, "NightSight v2", (w * 2 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save if requested
        if output_path:
            save_image(comparison_vis, output_path)

        return comparison_vis.astype(np.float32) / 255.0
