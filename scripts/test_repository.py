#!/usr/bin/env python3
"""
Comprehensive Repository Test Suite

Tests all components of the NightSight repository to ensure everything works.

Usage:
    python scripts/test_repository.py
"""

import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class RepositoryTester:
    """Comprehensive repository testing."""

    def __init__(self):
        self.root = Path(__file__).parent.parent
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []

    def log(self, message, level="INFO"):
        """Log test message."""
        prefix = {
            "INFO": "[i]",
            "PASS": "[+]",
            "FAIL": "[-]",
            "WARN": "[!]"
        }
        print(f"{prefix.get(level, '')} {message}")

    def test_imports(self):
        """Test 1: Verify all critical imports work."""
        self.log("\n" + "="*70)
        self.log("TEST 1: Import Validation", "INFO")
        self.log("="*70)

        imports_to_test = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("yaml", "PyYAML"),
            ("tqdm", "tqdm"),
            ("matplotlib", "Matplotlib"),
        ]

        for module, name in imports_to_test:
            try:
                __import__(module)
                self.log(f"{name:20} ... OK", "PASS")
                self.tests_passed += 1
            except ImportError as e:
                self.log(f"{name:20} ... FAILED: {e}", "FAIL")
                self.tests_failed += 1
                self.errors.append(f"Import {name}: {e}")

        # Test NightSight modules
        nightsight_modules = [
            "nightsight.models.hybrid",
            "nightsight.models.zerodce",
            "nightsight.traditional.histogram",
            "nightsight.traditional.retinex",
            "nightsight.utils.checkpoint",
            "nightsight.data.datasets",
            "nightsight.losses.color",
            "nightsight.metrics",
        ]

        self.log("\nNightSight Modules:", "INFO")
        for module in nightsight_modules:
            try:
                __import__(module)
                self.log(f"{module:40} ... OK", "PASS")
                self.tests_passed += 1
            except Exception as e:
                self.log(f"{module:40} ... FAILED: {e}", "FAIL")
                self.tests_failed += 1
                self.errors.append(f"Module {module}: {e}")

    def test_file_structure(self):
        """Test 2: Verify repository structure."""
        self.log("\n" + "="*70)
        self.log("TEST 2: Repository Structure", "INFO")
        self.log("="*70)

        required_paths = [
            ("models/nightsight_best.pth", "Best model checkpoint"),
            ("models/nightsight_best.json", "Model metadata"),
            ("models/README.md", "Model documentation"),
            ("outputs/samples/index.html", "Sample gallery"),
            ("outputs/samples/SUMMARY.md", "Sample summary"),
            ("outputs/method_comparison/README.md", "Comparison documentation"),
            ("scripts/train.py", "Training script"),
            ("scripts/realtime_demo.py", "Real-time demo"),
            ("scripts/test_all_methods.py", "Method testing"),
            ("nightsight/__init__.py", "Package init"),
            ("README.md", "Main README"),
            (".gitignore", "Git ignore file"),
        ]

        for path, description in required_paths:
            full_path = self.root / path
            if full_path.exists():
                size = full_path.stat().st_size if full_path.is_file() else "DIR"
                size_str = f"{size/1024/1024:.1f}MB" if isinstance(size, int) and size > 1024*1024 else (f"{size/1024:.1f}KB" if isinstance(size, int) else "DIR")
                self.log(f"{description:40} ({size_str:>8}) ... OK", "PASS")
                self.tests_passed += 1
            else:
                self.log(f"{description:40} ... MISSING", "FAIL")
                self.tests_failed += 1
                self.errors.append(f"Missing file: {path}")

    def test_model_loading(self):
        """Test 3: Verify model checkpoint loads correctly."""
        self.log("\n" + "="*70)
        self.log("TEST 3: Model Loading", "INFO")
        self.log("="*70)

        try:
            import torch
            from nightsight.models.hybrid import NightSightNet
            from nightsight.utils.checkpoint import load_checkpoint

            # Test model creation
            self.log("Creating NightSightNet model...", "INFO")
            model = NightSightNet()
            param_count = sum(p.numel() for p in model.parameters())
            self.log(f"Model created: {param_count:,} parameters", "PASS")
            self.tests_passed += 1

            # Test checkpoint loading
            checkpoint_path = self.root / "models/nightsight_best.pth"
            if checkpoint_path.exists():
                self.log(f"Loading checkpoint: {checkpoint_path}", "INFO")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, info = load_checkpoint(str(checkpoint_path), model, device=device)

                self.log(f"Checkpoint loaded successfully", "PASS")
                self.log(f"  Epoch: {info.get('epoch', 'N/A')}", "INFO")
                self.log(f"  PSNR: {info.get('metrics', {}).get('psnr', 'N/A'):.2f} dB", "INFO")
                self.log(f"  SSIM: {info.get('metrics', {}).get('ssim', 'N/A'):.4f}", "INFO")
                self.tests_passed += 1
            else:
                self.log("Checkpoint file not found", "FAIL")
                self.tests_failed += 1
                self.errors.append("Model checkpoint missing")

        except Exception as e:
            self.log(f"Model loading failed: {e}", "FAIL")
            self.tests_failed += 1
            self.errors.append(f"Model loading: {e}")

    def test_inference(self):
        """Test 4: Run sample inference."""
        self.log("\n" + "="*70)
        self.log("TEST 4: Sample Inference", "INFO")
        self.log("="*70)

        try:
            import torch
            import numpy as np
            from nightsight.models.hybrid import NightSightNet
            from nightsight.utils.checkpoint import load_checkpoint

            # Create dummy input
            self.log("Creating test input (640x480)...", "INFO")
            dummy_input = np.random.rand(480, 640, 3).astype(np.float32)

            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = NightSightNet()
            checkpoint_path = self.root / "models/nightsight_best.pth"

            if checkpoint_path.exists():
                model, _ = load_checkpoint(str(checkpoint_path), model, device=device)
                model = model.to(device)
                model.eval()

                # Run inference
                tensor = torch.from_numpy(dummy_input.transpose(2, 0, 1)).unsqueeze(0).to(device)

                start_time = time.time()
                with torch.no_grad():
                    output = model(tensor)
                inference_time = (time.time() - start_time) * 1000

                # Validate output
                output_np = output.squeeze(0).cpu().numpy()

                self.log(f"Inference successful", "PASS")
                self.log(f"  Input shape: {dummy_input.shape}", "INFO")
                self.log(f"  Output shape: {output_np.shape}", "INFO")
                self.log(f"  Inference time: {inference_time:.1f}ms", "INFO")
                self.log(f"  Device: {device}", "INFO")
                self.tests_passed += 1
            else:
                self.log("Checkpoint not found, skipping inference test", "WARN")

        except Exception as e:
            self.log(f"Inference failed: {e}", "FAIL")
            self.tests_failed += 1
            self.errors.append(f"Inference: {e}")

    def test_traditional_methods(self):
        """Test 5: Verify traditional enhancement methods."""
        self.log("\n" + "="*70)
        self.log("TEST 5: Traditional Enhancement Methods", "INFO")
        self.log("="*70)

        try:
            import numpy as np
            import cv2
            from nightsight.traditional.histogram import histogram_equalization, CLAHEEnhancer
            from nightsight.traditional.retinex import RetinexEnhancer

            # Create test image
            test_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)

            methods = [
                ("Histogram Equalization", lambda img: histogram_equalization(img)),
                ("CLAHE", lambda img: CLAHEEnhancer().enhance(img)),
                ("Retinex", lambda img: RetinexEnhancer().enhance(img.astype(np.float32) / 255.0)),
            ]

            for method_name, method_func in methods:
                try:
                    result = method_func(test_img)
                    self.log(f"{method_name:30} ... OK", "PASS")
                    self.tests_passed += 1
                except Exception as e:
                    self.log(f"{method_name:30} ... FAILED: {e}", "FAIL")
                    self.tests_failed += 1
                    self.errors.append(f"{method_name}: {e}")

        except Exception as e:
            self.log(f"Traditional methods test failed: {e}", "FAIL")
            self.tests_failed += 1
            self.errors.append(f"Traditional methods: {e}")

    def test_sample_outputs(self):
        """Test 6: Verify sample outputs exist and are valid."""
        self.log("\n" + "="*70)
        self.log("TEST 6: Sample Outputs Validation", "INFO")
        self.log("="*70)

        samples_dir = self.root / "outputs/samples"

        expected_methods = [
            "bilateral", "clahe", "gamma_mean", "gamma_median",
            "hist_eq", "retinex", "nightsight_trained", "zerodce_untrained"
        ]

        for method in expected_methods:
            method_dir = samples_dir / method
            if method_dir.exists():
                samples = list(method_dir.glob("sample_*.png"))
                if len(samples) >= 3:
                    self.log(f"{method:30} ({len(samples)} samples) ... OK", "PASS")
                    self.tests_passed += 1
                else:
                    self.log(f"{method:30} ... INCOMPLETE ({len(samples)}/3)", "WARN")
            else:
                self.log(f"{method:30} ... MISSING", "FAIL")
                self.tests_failed += 1
                self.errors.append(f"Missing method directory: {method}")

        # Check comparison panels
        panels_dir = samples_dir / "comparison_panels"
        if panels_dir.exists():
            panels = list(panels_dir.glob("*.png"))
            self.log(f"Comparison panels ({len(panels)}) ... OK", "PASS")
            self.tests_passed += 1
        else:
            self.log("Comparison panels ... MISSING", "FAIL")
            self.tests_failed += 1

    def test_scripts(self):
        """Test 7: Verify scripts are syntactically correct."""
        self.log("\n" + "="*70)
        self.log("TEST 7: Script Validation", "INFO")
        self.log("="*70)

        scripts_dir = self.root / "scripts"

        for script_path in sorted(scripts_dir.glob("*.py")):
            try:
                # Check if script can be compiled
                with open(script_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, str(script_path), 'exec')
                self.log(f"{script_path.name:40} ... OK", "PASS")
                self.tests_passed += 1
            except SyntaxError as e:
                self.log(f"{script_path.name:40} ... SYNTAX ERROR", "FAIL")
                self.tests_failed += 1
                self.errors.append(f"Syntax error in {script_path.name}: {e}")
            except Exception as e:
                self.log(f"{script_path.name:40} ... ERROR: {e}", "FAIL")
                self.tests_failed += 1

    def test_documentation(self):
        """Test 8: Verify documentation completeness."""
        self.log("\n" + "="*70)
        self.log("TEST 8: Documentation Validation", "INFO")
        self.log("="*70)

        docs_to_check = [
            (self.root / "README.md", "Main README", ["NightSight", "Installation", "Usage"]),
            (self.root / "models/README.md", "Model README", ["Performance", "Usage", "PSNR"]),
            (self.root / "outputs/samples/SUMMARY.md", "Samples Summary", ["Methods", "Results"]),
        ]

        for doc_path, name, keywords in docs_to_check:
            if doc_path.exists():
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                missing = [kw for kw in keywords if kw.lower() not in content.lower()]
                if not missing:
                    self.log(f"{name:40} ... OK", "PASS")
                    self.tests_passed += 1
                else:
                    self.log(f"{name:40} ... INCOMPLETE (missing: {', '.join(missing)})", "WARN")
            else:
                self.log(f"{name:40} ... MISSING", "FAIL")
                self.tests_failed += 1
                self.errors.append(f"Missing documentation: {name}")

    def test_dependencies(self):
        """Test 9: Check all dependencies are installed."""
        self.log("\n" + "="*70)
        self.log("TEST 9: Dependency Check", "INFO")
        self.log("="*70)

        requirements_path = self.root / "requirements.txt"
        if requirements_path.exists():
            with open(requirements_path, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            for req in requirements:
                # Extract package name (before >=, ==, etc.)
                pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()

                # Skip optional packages in comments
                if pkg_name.startswith('#'):
                    continue

                try:
                    __import__(pkg_name.replace('-', '_'))
                    self.log(f"{pkg_name:30} ... OK", "PASS")
                    self.tests_passed += 1
                except ImportError:
                    self.log(f"{pkg_name:30} ... NOT INSTALLED", "WARN")
        else:
            self.log("requirements.txt not found", "WARN")

    def run_all_tests(self):
        """Run all tests."""
        start_time = time.time()

        self.log("\n" + "="*70)
        self.log("NIGHTSIGHT REPOSITORY - COMPREHENSIVE TEST SUITE", "INFO")
        self.log("="*70)

        # Run all tests
        self.test_imports()
        self.test_file_structure()
        self.test_model_loading()
        self.test_inference()
        self.test_traditional_methods()
        self.test_sample_outputs()
        self.test_scripts()
        self.test_documentation()
        self.test_dependencies()

        # Summary
        total_time = time.time() - start_time

        self.log("\n" + "="*70)
        self.log("TEST SUMMARY", "INFO")
        self.log("="*70)
        self.log(f"Tests Passed: {self.tests_passed}", "PASS")
        if self.tests_failed > 0:
            self.log(f"Tests Failed: {self.tests_failed}", "FAIL")
        self.log(f"Total Time: {total_time:.2f}s", "INFO")

        if self.errors:
            self.log("\n" + "="*70)
            self.log("ERRORS ENCOUNTERED:", "FAIL")
            self.log("="*70)
            for i, error in enumerate(self.errors, 1):
                self.log(f"{i}. {error}", "FAIL")

        self.log("\n" + "="*70)
        if self.tests_failed == 0:
            self.log("SUCCESS: ALL TESTS PASSED - Repository is ready for publication!", "PASS")
        else:
            self.log(f"WARNING: {self.tests_failed} tests failed - Please review errors above", "WARN")
        self.log("="*70)

        return self.tests_failed == 0


def main():
    tester = RepositoryTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
