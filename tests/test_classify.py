"""
SatQuery AI — Phase 3 Smoke Test: Scene Classification
========================================================
Verifies that classify_scene returns a valid result for various input
types and that the return dict matches the same shape as ask_vqa.

Run from project root:
    python tests/test_classify.py
"""

import sys
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.classify.classify_tool import classify_scene, EUROSAT_CLASSES
from tools.vqa.vqa_tool import ask_vqa
from tools.geotiff_utils import load_geotiff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"

SAMPLE_RGB = PROJECT_ROOT / "data" / "sample_rgb.tif"


# ---------------------------------------------------------------------------
# Synthetic test images
# ---------------------------------------------------------------------------

def _make_forest(size=256):
    """Dense green blobs → should classify as Forest."""
    img = Image.new("RGB", (size, size), (30, 100, 30))
    draw = ImageDraw.Draw(img)
    np.random.seed(42)
    for _ in range(50):
        cx, cy = np.random.randint(10, 246, 2)
        r = np.random.randint(8, 20)
        g = np.random.randint(60, 130)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, g, 15))
    return img


def _make_residential(size=256):
    """Grid of rectangles → should classify as Residential."""
    img = Image.new("RGB", (size, size), (180, 180, 170))
    draw = ImageDraw.Draw(img)
    # Roads
    for y in range(0, size, 64):
        draw.rectangle([0, y, size, y + 4], fill=(100, 100, 100))
    for x in range(0, size, 64):
        draw.rectangle([x, 0, x + 4, size], fill=(100, 100, 100))
    # Buildings inside the grid cells
    np.random.seed(7)
    for bx in range(8, size, 64):
        for by in range(8, size, 64):
            c = tuple(np.random.randint(140, 220, 3))
            draw.rectangle([bx, by, bx + 50, by + 50], fill=c, outline=(80, 80, 80))
    return img


def _make_water(size=256):
    """Blue field → should classify as SeaLake."""
    img = Image.new("RGB", (size, size), (30, 60, 160))
    draw = ImageDraw.Draw(img)
    # subtle wave lines
    for y in range(0, size, 12):
        draw.line([(0, y), (size, y + 3)], fill=(40, 70, 170), width=2)
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    results = []

    # ------------------------------------------------------------------
    # Test 1: Return shape matches ask_vqa
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: Return dict shape matches VQA tool")
    print("=" * 60)
    try:
        forest = _make_forest()
        res = classify_scene(forest)

        # ask_vqa keys
        vqa_keys = {"answer", "raw_answer", "model_used", "elapsed_s",
                     "status", "error"}
        classify_keys = set(res.keys())

        missing = vqa_keys - classify_keys
        assert not missing, f"Missing keys vs VQA: {missing}"
        assert "confidence" in classify_keys, "Missing 'confidence' field"
        assert "backend" in classify_keys, "Missing 'backend' field"

        print(f"  {PASS}  Return shape is VQA-compatible + extras (confidence, backend, all_classes)")
        results.append(("Return shape", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Return shape", False))

    # ------------------------------------------------------------------
    # Test 2: Classify from PIL Image
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: Classify from PIL Image (forest scene)")
    print("=" * 60)
    try:
        forest = _make_forest()
        res = classify_scene(forest)
        assert res["status"] == "ok", f"Status is {res['status']}: {res['error']}"
        assert res["answer"], "Empty answer"
        assert res["backend"] in ("cnn", "vlm"), f"Unknown backend: {res['backend']}"

        print(f"  {PASS}  Classified as: {res['answer']}  (conf={res['confidence']:.3f})")
        print(f"         Backend: {res['backend']}  Model: {res['model_used']}  Time: {res['elapsed_s']}s")
        results.append(("PIL Image classify", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("PIL Image classify", False))

    # ------------------------------------------------------------------
    # Test 3: Classify from file path (GeoTIFF)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: Classify from GeoTIFF file path")
    print("=" * 60)
    try:
        if not SAMPLE_RGB.exists():
            from tests.generate_sample import create_synthetic_geotiff
            create_synthetic_geotiff(str(SAMPLE_RGB))

        res = classify_scene(str(SAMPLE_RGB))
        assert res["status"] == "ok", f"Status is {res['status']}: {res['error']}"
        assert res["answer"], "Empty answer"

        print(f"  {PASS}  Classified as: {res['answer']}  (conf={res['confidence']:.3f})")
        print(f"         Backend: {res['backend']}  Time: {res['elapsed_s']}s")
        results.append(("GeoTIFF classify", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("GeoTIFF classify", False))

    # ------------------------------------------------------------------
    # Test 4: Classify from LoadedImage
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: Classify from LoadedImage object")
    print("=" * 60)
    try:
        if not SAMPLE_RGB.exists():
            from tests.generate_sample import create_synthetic_geotiff
            create_synthetic_geotiff(str(SAMPLE_RGB))

        loaded = load_geotiff(SAMPLE_RGB)
        res = classify_scene(loaded)
        assert res["status"] == "ok", f"Status is {res['status']}: {res['error']}"
        assert res["answer"], "Empty answer"

        print(f"  {PASS}  Classified as: {res['answer']}  (conf={res['confidence']:.3f})")
        results.append(("LoadedImage classify", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("LoadedImage classify", False))

    # ------------------------------------------------------------------
    # Test 5: Multiple scene types (batch)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 5: Classify multiple scene types")
    print("=" * 60)
    scenes = [
        ("Forest",      _make_forest()),
        ("Residential", _make_residential()),
        ("Water",       _make_water()),
    ]
    batch_ok = True
    for scene_name, scene_img in scenes:
        try:
            res = classify_scene(scene_img)
            assert res["status"] == "ok", f"{scene_name}: {res['error']}"
            print(f"  {scene_name:15s} -> {res['answer']:25s} "
                  f"(conf={res['confidence']:.3f}, {res['backend']})")
        except Exception as e:
            print(f"  {scene_name:15s} -> {FAIL} {e}")
            batch_ok = False

    if batch_ok:
        print(f"\n  {PASS}  All scenes classified without errors")
    else:
        print(f"\n  {FAIL}  Some scenes failed")
    results.append(("Multi-scene batch", batch_ok))

    # ------------------------------------------------------------------
    # Test 6: Error handling — bad input
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 6: Error handling for bad inputs")
    print("=" * 60)
    try:
        # Non-existent file
        res = classify_scene("does_not_exist.tif")
        assert res["status"] == "error", "Should have returned error for missing file"
        assert "not found" in res["error"].lower() or "not found" in res["error"], \
            f"Unexpected error message: {res['error']}"

        # Unsupported type
        res = classify_scene(12345)
        assert res["status"] == "error", "Should have returned error for bad type"

        print(f"  {PASS}  Bad inputs handled gracefully (no crash)")
        results.append(("Error handling", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Error handling", False))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n  {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n>>> Phase 3 complete — classification tool is working!")
    else:
        print("\n[!] Some tests failed — review output above.")

    return results


if __name__ == "__main__":
    run_tests()
