"""
Repository cleanup script.
Removes temporary files, __pycache__ directories, and other cruft.
"""

import os
import shutil
from pathlib import Path

def cleanup_repo():
    """Clean up repository."""
    root = Path(__file__).parent.parent

    print("=" * 60)
    print("Repository Cleanup")
    print("=" * 60)
    print()

    # Remove __pycache__ directories
    print("Removing __pycache__ directories...")
    pycache_dirs = list(root.rglob("__pycache__"))
    for dir_path in pycache_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"  Removed: {dir_path.relative_to(root)}")
        except Exception as e:
            print(f"  Failed to remove {dir_path}: {e}")

    # Remove .pyc files
    print("\nRemoving .pyc files...")
    pyc_files = list(root.rglob("*.pyc"))
    for file_path in pyc_files:
        try:
            file_path.unlink()
            print(f"  Removed: {file_path.relative_to(root)}")
        except Exception as e:
            print(f"  Failed to remove {file_path}: {e}")

    # Remove .DS_Store files (Mac)
    print("\nRemoving .DS_Store files...")
    ds_store_files = list(root.rglob(".DS_Store"))
    for file_path in ds_store_files:
        try:
            file_path.unlink()
            print(f"  Removed: {file_path.relative_to(root)}")
        except Exception as e:
            print(f"  Failed to remove {file_path}: {e}")

    # Remove test_camera.py (utility script, not needed for release)
    test_camera = root / "scripts" / "test_camera.py"
    if test_camera.exists():
        print("\nRemoving utility scripts...")
        try:
            test_camera.unlink()
            print(f"  Removed: test_camera.py")
        except Exception as e:
            print(f"  Failed to remove test_camera.py: {e}")

    print("\n" + "=" * 60)
    print("Cleanup complete!")
    print("=" * 60)

if __name__ == '__main__':
    cleanup_repo()
