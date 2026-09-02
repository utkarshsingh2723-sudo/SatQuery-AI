"""
SatQuery AI — Phase 1 End-to-End Smoke Test
=============================================
Acceptance criteria: load a GeoTIFF → send to qwen2.5vl → get text answer.

Run from the project root:
    python tests/test_phase1_e2e.py

This script:
  1. Generates a synthetic GeoTIFF if none exists.
  2. Loads it via geotiff_utils and verifies metadata.
  3. Sends it to the VLM via ollama_client with a hardcoded question.
  4. Prints the answer.

If steps 1-2 pass but step 3 fails, Ollama is probably not running or
the model isn't pulled.  That's a setup issue, not a code issue.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.geotiff_utils import load_geotiff, save_rgb_preview
from tools.ollama_client import query_vlm
from tests.generate_sample import create_synthetic_geotiff, create_synthetic_sar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLE_RGB = PROJECT_ROOT / "data" / "sample_rgb.tif"
SAMPLE_SAR = PROJECT_ROOT / "data" / "sample_sar.tif"

PASS = "[PASS]" 
FAIL = "[FAIL]"


def run_tests():
    results = []

    # ------------------------------------------------------------------
    # Test 1: Generate synthetic data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: Generate synthetic GeoTIFFs")
    print("=" * 60)
    try:
        create_synthetic_geotiff(str(SAMPLE_RGB))
        create_synthetic_sar(str(SAMPLE_SAR))
        assert SAMPLE_RGB.exists(), "RGB file not created"
        assert SAMPLE_SAR.exists(), "SAR file not created"
        print(f"  {PASS}  Synthetic data generated")
        results.append(("Generate samples", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Generate samples", False))
        print(f"\n[!] Cannot continue without sample data. Is rasterio installed?")
        return results

    # ------------------------------------------------------------------
    # Test 2: Load RGB GeoTIFF
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: Load RGB GeoTIFF via geotiff_utils")
    print("=" * 60)
    try:
        loaded = load_geotiff(SAMPLE_RGB)
        assert loaded.rgb_array.shape == (256, 256, 3), \
            f"Unexpected shape: {loaded.rgb_array.shape}"
        assert loaded.rgb_array.dtype == "uint8"
        assert loaded.metadata.has_georef, "Missing georeferencing"
        assert loaded.metadata.crs is not None
        assert loaded.metadata.band_count == 3

        preview = save_rgb_preview(loaded, SAMPLE_RGB.with_suffix(".preview.jpg"))
        print(f"  {PASS}  RGB loaded - {loaded.metadata.width}x{loaded.metadata.height}, "
              f"CRS={loaded.metadata.crs}")
        print(f"         Preview: {preview}")
        results.append(("Load RGB GeoTIFF", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Load RGB GeoTIFF", False))

    # ------------------------------------------------------------------
    # Test 3: Load SAR GeoTIFF (single-band float32)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: Load SAR GeoTIFF via geotiff_utils")
    print("=" * 60)
    try:
        loaded_sar = load_geotiff(SAMPLE_SAR)
        assert loaded_sar.rgb_array.shape == (256, 256, 3), \
            f"Unexpected shape: {loaded_sar.rgb_array.shape}"
        assert loaded_sar.rgb_array.dtype == "uint8"
        assert loaded_sar.metadata.band_count == 1

        preview = save_rgb_preview(loaded_sar, SAMPLE_SAR.with_suffix(".preview.jpg"))
        print(f"  {PASS}  SAR loaded - {loaded_sar.metadata.width}x{loaded_sar.metadata.height}, "
              f"dtype={loaded_sar.metadata.dtype}")
        print(f"         Preview: {preview}")
        results.append(("Load SAR GeoTIFF", True))
    except Exception as e:
        print(f"  {FAIL}  {e}")
        results.append(("Load SAR GeoTIFF", False))

    # ------------------------------------------------------------------
    # Test 4: VLM query (end-to-end)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: End-to-end VLM query (GeoTIFF -> Ollama -> answer)")
    print("=" * 60)
    try:
        loaded = load_geotiff(SAMPLE_RGB)
        question = (
            "This is a remote sensing satellite image. "
            "Describe what you see, including any patterns, land cover, "
            "or notable features."
        )
        print(f"  Sending to VLM: \"{question[:60]}...\"")
        print(f"  (This may take 30-120 seconds on first call...)\n")

        result = query_vlm(
            prompt=question,
            image=loaded.pil_image,
            system_prompt=(
                "You are an expert remote sensing analyst. "
                "Answer concisely based only on what you observe in the image."
            ),
        )

        assert result["answer"], "Empty answer"
        assert result["model_used"], "No model name"
        assert result["elapsed_s"] > 0, "Invalid timing"

        print(f"  {PASS}  VLM responded successfully")
        print(f"         Model : {result['model_used']}")
        print(f"         Time  : {result['elapsed_s']}s")
        print(f"         Answer: {result['answer'][:200]}...")
        results.append(("VLM E2E query", True))

    except Exception as e:
        print(f"  {FAIL}  {e}")
        print(f"\n  [!] Possible causes:")
        print(f"    - Ollama not running (start with: ollama serve)")
        print(f"    - Model not pulled (run: ollama pull qwen2.5vl:7b)")
        print(f"    - GPU out of memory")
        results.append(("VLM E2E query", False))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n  {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n>>> Phase 1 complete -- foundation is working!")
    elif passed_count >= 3:
        print("\n[!] GeoTIFF loading works. Fix the VLM connection to complete Phase 1.")
    else:
        print("\n[X] Core issues to resolve before proceeding.")

    return results


if __name__ == "__main__":
    run_tests()
