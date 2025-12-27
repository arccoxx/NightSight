"""
Comprehensive test suite for v2 components.
Tests all changed components to ensure functionality before publishing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import cv2
from nightsight.v2 import NightSightV2Pipeline
from nightsight.v2.modules.edge_outliner import EdgeOutliner
from nightsight.v2.modules.object_detector import ObjectDetector
from nightsight.v2.modules.depth_estimator import DepthEstimator
from nightsight.v2.models.nightsight_v2 import NightSightV2


def test_edge_outliner_dtype():
    """Test that edge outliner preserves data types correctly."""
    print("\n" + "=" * 60)
    print("Testing EdgeOutliner Data Type Handling")
    print("=" * 60)

    try:
        outliner = EdgeOutliner(device='cpu')

        # Test with float32 input
        image_float = np.random.rand(100, 100, 3).astype(np.float32)
        edges = outliner.detect_edges_combined(image_float)

        # Test standard outlines
        result = outliner.apply_outline_to_image(image_float, edges=edges)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        assert result.max() <= 1.0, "Float32 output exceeds 1.0"
        print("[OK] Standard outlines preserve float32")

        # Test depth-aware outlines
        depth = np.random.rand(100, 100).astype(np.float32)
        result_depth = outliner.create_depth_aware_outlines(
            image_float, depth, edges=edges
        )
        assert result_depth.dtype == np.float32, f"Expected float32, got {result_depth.dtype}"
        assert result_depth.max() <= 1.0, "Float32 output exceeds 1.0"
        print("[OK] Depth-aware outlines preserve float32")

        return True

    except Exception as e:
        print(f"[FAIL] EdgeOutliner test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_object_detector_dtype():
    """Test that object detector preserves data types correctly."""
    print("\n" + "=" * 60)
    print("Testing ObjectDetector Data Type Handling")
    print("=" * 60)

    try:
        detector = ObjectDetector(device='cpu', use_ultralytics=False)

        # Test with float32 input
        image_float = np.random.rand(100, 100, 3).astype(np.float32)

        # Create dummy detections
        detections = [
            {
                'bbox': [10, 10, 50, 50],
                'conf': 0.9,
                'class': 'person',
                'class_id': 0
            }
        ]

        result = detector.draw_glowing_boxes(image_float, detections)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        assert result.max() <= 1.0, "Float32 output exceeds 1.0"
        print("[OK] Glowing boxes preserve float32")

        return True

    except Exception as e:
        print(f"[FAIL] ObjectDetector test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_brightness():
    """Test adaptive brightness boosting for dark scenes."""
    print("\n" + "=" * 60)
    print("Testing Adaptive Brightness Boosting")
    print("=" * 60)

    try:
        model = NightSightV2(
            device='cpu',
            use_depth=True,
            use_zerodce=True,
            use_edges=False,
            use_detection=False,
            use_tracking=False,
            use_superres=False,
            use_adaptive=True
        )

        # Create very dark image
        very_dark_image = np.random.rand(100, 100, 3).astype(np.float32) * 0.1

        enhanced, components = model.forward(very_dark_image, return_components=True)

        # Check that scene was classified as very dark
        if 'scene_config' in components:
            scene_class = components['scene_config']['scene']['class_name']
            brightness = components['scene_config']['scene']['brightness']
            print(f"  Scene class: {scene_class}")
            print(f"  Scene brightness: {brightness:.3f}")

            # Check that adaptive boosting was applied
            if 'adaptive_boosted' in components:
                print("[OK] Adaptive boosting applied for very dark scene")
            else:
                print("[WARN] Adaptive boosting not applied")

        # Check that output is brighter than input
        input_mean = very_dark_image.mean()
        output_mean = enhanced.mean()

        print(f"  Input mean brightness: {input_mean:.3f}")
        print(f"  Output mean brightness: {output_mean:.3f}")
        print(f"  Brightness increase: {(output_mean / input_mean):.2f}x")

        assert output_mean > input_mean, "Output should be brighter than input"
        print("[OK] Adaptive brightness boosting working")

        return True

    except Exception as e:
        print(f"[FAIL] Adaptive brightness test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test full pipeline integration."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline Integration")
    print("=" * 60)

    try:
        pipeline = NightSightV2Pipeline(
            device='cpu',
            use_all_features=True
        )

        # Test with float32 image
        test_image = np.random.rand(100, 100, 3).astype(np.float32)

        enhanced, components = pipeline.enhance_image(
            test_image,
            return_components=True
        )

        # Verify output
        assert enhanced.dtype == np.float32, f"Expected float32, got {enhanced.dtype}"
        assert enhanced.shape == test_image.shape, "Shape mismatch"
        assert enhanced.max() <= 1.0, "Output exceeds 1.0"
        assert enhanced.min() >= 0.0, "Output below 0.0"

        print(f"  Input shape: {test_image.shape}")
        print(f"  Output shape: {enhanced.shape}")
        print(f"  Output dtype: {enhanced.dtype}")
        print(f"  Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
        print("[OK] Pipeline integration test passed")

        return True

    except Exception as e:
        print(f"[FAIL] Pipeline integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_frame_processing():
    """Test video frame processing."""
    print("\n" + "=" * 60)
    print("Testing Video Frame Processing")
    print("=" * 60)

    try:
        pipeline = NightSightV2Pipeline(
            device='cpu',
            use_all_features=False,
            use_tracking=True,
            use_detection=True
        )

        # Process multiple frames
        for i in range(3):
            frame = np.random.rand(100, 100, 3).astype(np.float32)
            enhanced, components = pipeline.model.process_video_frame(frame)

            assert enhanced.dtype == np.float32, f"Frame {i}: Wrong dtype"
            assert enhanced.shape == frame.shape, f"Frame {i}: Shape mismatch"

        print(f"  Processed 3 frames successfully")
        print("[OK] Video frame processing test passed")

        return True

    except Exception as e:
        print(f"[FAIL] Video frame processing test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("NightSight v2 Comprehensive Test Suite")
    print("=" * 60)

    results = {}

    # Run tests
    results['edge_outliner_dtype'] = test_edge_outliner_dtype()
    results['object_detector_dtype'] = test_object_detector_dtype()
    results['adaptive_brightness'] = test_adaptive_brightness()
    results['pipeline_integration'] = test_pipeline_integration()
    results['video_frame_processing'] = test_video_frame_processing()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    print()
    if all_passed:
        print("All comprehensive tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
