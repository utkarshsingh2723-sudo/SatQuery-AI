"""
SatQuery AI -- Phase 4 Smoke Test: Change Detection
=====================================================
Generates synthetic bi-temporal image pairs with known changes,
runs detect_changes, and verifies the pipeline produces a valid
change map + narration without crashing.

Run from project root:
    python tests/test_change.py
"""

import sys
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.change.change_tool import detect_changes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"
OUTPUT_DIR = PROJECT_ROOT / "data" / "change_test_output"


# ---------------------------------------------------------------------------
# Synthetic bi-temporal pairs
# ---------------------------------------------------------------------------

def _make_pair_construction(size=256):
    """Before: empty field.  After: field + new buildings."""
    # Before -- green field
    before = Image.new("RGB", (size, size), (100, 160, 80))
    draw_b = ImageDraw.Draw(before)
    draw_b.rectangle([0, 120, size, 136], fill=(120, 110, 90))  # dirt road

    # After -- same field but with buildings added
    after = before.copy()
    draw_a = ImageDraw.Draw(after)
    # Add buildings
    draw_a.rectangle([40, 40, 90, 80], fill=(180, 170, 160), outline=(100, 100, 100))
    draw_a.rectangle([140, 50, 200, 100], fill=(190, 180, 170), outline=(100, 100, 100))
    draw_a.rectangle([60, 160, 120, 210], fill=(170, 165, 155), outline=(100, 100, 100))

    return before, after, "construction"


def _make_pair_deforestation(size=256):
    """Before: dense forest.  After: partially cleared."""
    np.random.seed(99)
    # Before -- dense forest
    before = Image.new("RGB", (size, size), (30, 100, 30))
    draw_b = ImageDraw.Draw(before)
    for _ in range(60):
        cx, cy = np.random.randint(10, 246, 2)
        r = np.random.randint(8, 18)
        g = np.random.randint(60, 130)
        draw_b.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, g, 15))

    # After -- same but right half is cleared (brown)
    after = before.copy()
    draw_a = ImageDraw.Draw(after)
    draw_a.rectangle([128, 0, size, size], fill=(170, 150, 110))  # cleared area
    # Some remaining stumps
    for _ in range(8):
        cx = np.random.randint(135, 250)
        cy = np.random.randint(10, 246)
        draw_a.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(80, 60, 40))

    return before, after, "deforestation"


def _make_pair_identical(size=256):
    """Two identical images -- should report ~0% change."""
    img = Image.new("RGB", (size, size), (120, 140, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 100, 100], fill=(80, 80, 80))
    draw.ellipse([150, 150, 220, 220], fill=(60, 100, 160))
    return img.copy(), img.copy(), "identical"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Test 1: Return shape matches VQA interface
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: Return dict shape (VQA-compatible + change extras)")
    print("=" * 60)
    try:
        before, after, _ = _make_pair_construction()
        res = detect_changes(before, after, narrate=False)

        vqa_keys = {"answer", "raw_answer", "model_used", "elapsed_s",
                     "status", "error"}
        change_extras = {"change_map", "diff_image", "change_mask", "stats"}
        all_required = vqa_keys | change_extras

        missing = all_required - set(res.keys())
        assert not missing, f"Missing keys: {missing}"

        print(f"  {PASS}  All required keys present: {sorted(res.keys())}")
        results.append(("Return shape", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Return shape", False))

    # ------------------------------------------------------------------
    # Test 2: Construction change (CV-only, no VLM)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: Construction change detection (CV-only)")
    print("=" * 60)
    try:
        before, after, label = _make_pair_construction()
        res = detect_changes(before, after, narrate=False)

        assert res["status"] == "ok", f"Status: {res['status']}, Error: {res['error']}"
        assert res["change_map"] is not None, "Missing change_map"
        assert res["diff_image"] is not None, "Missing diff_image"
        assert res["change_mask"] is not None, "Missing change_mask"
        assert res["stats"]["change_pct"] > 0, "Expected some change"
        assert isinstance(res["change_map"], Image.Image), "change_map not PIL Image"

        # Save outputs
        res["change_map"].save(OUTPUT_DIR / "construction_overlay.jpg", quality=90)
        res["diff_image"].save(OUTPUT_DIR / "construction_diff.jpg", quality=90)
        before.save(OUTPUT_DIR / "construction_before.jpg", quality=90)
        after.save(OUTPUT_DIR / "construction_after.jpg", quality=90)

        print(f"  {PASS}  Changed: {res['stats']['change_pct']}%  "
              f"({res['stats']['changed_pixels']} px)")
        print(f"         Threshold: {res['stats']['threshold']}")
        print(f"         Time: {res['elapsed_s']}s")
        print(f"         Outputs saved to {OUTPUT_DIR}")
        results.append(("Construction change", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Construction change", False))

    # ------------------------------------------------------------------
    # Test 3: Deforestation change
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: Deforestation change detection (CV-only)")
    print("=" * 60)
    try:
        before, after, label = _make_pair_deforestation()
        res = detect_changes(before, after, narrate=False)

        assert res["status"] == "ok"
        assert res["stats"]["change_pct"] > 10, \
            f"Expected significant change, got {res['stats']['change_pct']}%"

        res["change_map"].save(OUTPUT_DIR / "deforestation_overlay.jpg", quality=90)
        before.save(OUTPUT_DIR / "deforestation_before.jpg", quality=90)
        after.save(OUTPUT_DIR / "deforestation_after.jpg", quality=90)

        print(f"  {PASS}  Changed: {res['stats']['change_pct']}%  "
              f"(expected >10% for half-cleared forest)")
        print(f"         Time: {res['elapsed_s']}s")
        results.append(("Deforestation change", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Deforestation change", False))

    # ------------------------------------------------------------------
    # Test 4: Identical images (should report ~0% change)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: Identical images (expect ~0% change)")
    print("=" * 60)
    try:
        before, after, label = _make_pair_identical()
        res = detect_changes(before, after, narrate=False)

        assert res["status"] == "ok"
        assert res["stats"]["change_pct"] < 5, \
            f"Expected <5% change for identical images, got {res['stats']['change_pct']}%"

        print(f"  {PASS}  Changed: {res['stats']['change_pct']}%  (expected ~0%)")
        results.append(("Identical images", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Identical images", False))

    # ------------------------------------------------------------------
    # Test 5: Different-sized inputs (co-registration should handle)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 5: Different-sized inputs (resize co-registration)")
    print("=" * 60)
    try:
        before = Image.new("RGB", (256, 256), (100, 160, 80))
        after = Image.new("RGB", (512, 512), (90, 150, 70))
        # Add a big change to the after image
        draw = ImageDraw.Draw(after)
        draw.rectangle([100, 100, 400, 400], fill=(200, 200, 200))

        res = detect_changes(before, after, narrate=False)
        assert res["status"] == "ok"
        assert res["change_map"].size == (256, 256), \
            f"Output size should match before image, got {res['change_map'].size}"

        print(f"  {PASS}  Handled 256x256 vs 512x512 input")
        print(f"         Output size: {res['change_map'].size}")
        print(f"         Changed: {res['stats']['change_pct']}%")
        results.append(("Different sizes", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Different sizes", False))

    # ------------------------------------------------------------------
    # Test 6: GeoTIFF file inputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 6: GeoTIFF file path inputs")
    print("=" * 60)
    try:
        sample_rgb = PROJECT_ROOT / "data" / "sample_rgb.tif"
        if not sample_rgb.exists():
            from tests.generate_sample import create_synthetic_geotiff
            create_synthetic_geotiff(str(sample_rgb))

        # Use the same file as both before/after (should detect ~0% change)
        res = detect_changes(str(sample_rgb), str(sample_rgb), narrate=False)
        assert res["status"] == "ok"

        print(f"  {PASS}  GeoTIFF inputs accepted")
        print(f"         Changed: {res['stats']['change_pct']}%")
        results.append(("GeoTIFF inputs", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("GeoTIFF inputs", False))

    # ------------------------------------------------------------------
    # Test 7: Error handling
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 7: Error handling for bad inputs")
    print("=" * 60)
    try:
        # Missing file
        res = detect_changes("nonexistent.tif", "also_missing.tif", narrate=False)
        assert res["status"] == "error"
        assert "not found" in res["error"].lower()

        # Bad type
        res = detect_changes(12345, "test.tif", narrate=False)
        assert res["status"] == "error"

        print(f"  {PASS}  Bad inputs handled gracefully (no crash)")
        results.append(("Error handling", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Error handling", False))

    # ------------------------------------------------------------------
    # Test 8: VLM narration (will fail gracefully if Ollama not running)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 8: VLM narration attempt")
    print("=" * 60)
    try:
        before, after, _ = _make_pair_construction()
        res = detect_changes(before, after, narrate=True, timeout=30, max_retries=1)

        if res["model_used"] != "cv-only":
            print(f"  {PASS}  VLM narration succeeded")
            print(f"         Model: {res['model_used']}")
            print(f"         Answer: {res['answer'][:150]}...")
        else:
            print(f"  [SKIP] VLM unavailable, fell back to CV-only (expected if Ollama is off)")
            print(f"         Answer: {res['answer'][:150]}")

        # Either way, the result should be valid
        assert res["status"] == "ok"
        assert res["change_map"] is not None
        results.append(("VLM narration", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("VLM narration", False))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4 RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n  {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n>>> Phase 4 complete -- change detection tool is working!")
    elif passed_count >= 6:
        print("\n[!] Core change detection works. Check failing tests above.")
    else:
        print("\n[X] Issues to resolve before proceeding.")

    return results


if __name__ == "__main__":
    run_tests()
