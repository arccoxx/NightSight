"""
Repository validation script.
Validates that both v1 and v2 are intact and functional.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

def test_v1_imports():
    """Test that all v1 components can be imported."""
    print("\n" + "=" * 60)
    print("Testing V1 Imports")
    print("=" * 60)

    try:
        from nightsight.models.zerodce import ZeroDCE
        print("[OK] ZeroDCE")
    except Exception as e:
        print(f"[FAIL] ZeroDCE: {e}")
        return False

    try:
        from nightsight.models.hybrid import NightSightNet
        print("[OK] NightSightNet (Hybrid)")
    except Exception as e:
        print(f"[FAIL] NightSightNet: {e}")
        return False

    try:
        from nightsight.models.retinexformer import Retinexformer
        print("[OK] Retinexformer")
    except Exception as e:
        print(f"[FAIL] Retinexformer: {e}")
        return False

    try:
        from nightsight.pipelines.single_image import SingleImagePipeline
        print("[OK] SingleImagePipeline")
    except Exception as e:
        print(f"[FAIL] SingleImagePipeline: {e}")
        return False

    return True


def test_v1_functionality():
    """Test that v1 models can process images."""
    print("\n" + "=" * 60)
    print("Testing V1 Functionality")
    print("=" * 60)

    try:
        from nightsight.models.hybrid import NightSightNet

        model = NightSightNet()
        model.eval()

        # Create test input
        test_input = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            output = model(test_input)

        assert output.shape == test_input.shape, "Output shape mismatch"
        print("[OK] NightSightNet forward pass")
        return True

    except Exception as e:
        print(f"[FAIL] V1 functionality test: {e}")
        return False


def test_v2_imports():
    """Test that all v2 components can be imported."""
    print("\n" + "=" * 60)
    print("Testing V2 Imports")
    print("=" * 60)

    try:
        from nightsight.v2 import NightSightV2Pipeline
        print("[OK] NightSightV2Pipeline")
    except Exception as e:
        print(f"[FAIL] NightSightV2Pipeline: {e}")
        return False

    try:
        from nightsight.v2.models.nightsight_v2 import NightSightV2
        print("[OK] NightSightV2")
    except Exception as e:
        print(f"[FAIL] NightSightV2: {e}")
        return False

    try:
        from nightsight.v2.modules.depth_estimator import DepthEstimator
        print("[OK] DepthEstimator")
    except Exception as e:
        print(f"[FAIL] DepthEstimator: {e}")
        return False

    try:
        from nightsight.v2.modules.zerodce_plus import ZeroDCEPlusPlus
        print("[OK] ZeroDCEPlusPlus")
    except Exception as e:
        print(f"[FAIL] ZeroDCEPlusPlus: {e}")
        return False

    try:
        from nightsight.v2.modules.edge_outliner import EdgeOutliner
        print("[OK] EdgeOutliner")
    except Exception as e:
        print(f"[FAIL] EdgeOutliner: {e}")
        return False

    try:
        from nightsight.v2.modules.object_detector import ObjectDetector
        print("[OK] ObjectDetector")
    except Exception as e:
        print(f"[FAIL] ObjectDetector: {e}")
        return False

    try:
        from nightsight.v2.modules.tracker import MultiObjectTracker
        print("[OK] MultiObjectTracker")
    except Exception as e:
        print(f"[FAIL] MultiObjectTracker: {e}")
        return False

    try:
        from nightsight.v2.modules.scene_classifier import SceneClassifier
        print("[OK] SceneClassifier")
    except Exception as e:
        print(f"[FAIL] SceneClassifier: {e}")
        return False

    return True


def test_v2_functionality():
    """Test that v2 can process images."""
    print("\n" + "=" * 60)
    print("Testing V2 Functionality")
    print("=" * 60)

    try:
        from nightsight.v2 import NightSightV2Pipeline

        pipeline = NightSightV2Pipeline(device='cpu', use_all_features=False)

        # Create test input
        test_image = np.random.rand(100, 100, 3).astype(np.float32)

        enhanced = pipeline.enhance_image(test_image)

        assert enhanced.shape == test_image.shape, "Output shape mismatch"
        print("[OK] NightSightV2 image enhancement")
        return True

    except Exception as e:
        print(f"[FAIL] V2 functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("NightSight Repository Validation")
    print("=" * 60)

    results = {}

    # Test v1
    results['v1_imports'] = test_v1_imports()
    results['v1_functionality'] = test_v1_functionality()

    # Test v2
    results['v2_imports'] = test_v2_imports()
    results['v2_functionality'] = test_v2_functionality()

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    print()
    if all_passed:
        print("All validation tests passed!")
        return 0
    else:
        print("Some validation tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
