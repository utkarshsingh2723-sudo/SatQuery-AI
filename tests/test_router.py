"""
Quick integration test for the router.
Tests:
  1. Heuristic routing logic (no VLM, no tool dispatch)
  2. Tool dispatch for classify (uses CNN, no VLM)
  3. Tool dispatch for change + SAR (CV-only mode, no VLM)

VLM narration is NOT tested here — individual tools handle that.
"""

import sys
import logging
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from router.router import (
    route_query, _heuristic_classify,
    IMAGE_MODE_SINGLE, IMAGE_MODE_BITEMPORAL, IMAGE_MODE_SAR_OPTICAL,
    TASK_VQA, TASK_CLASSIFY, TASK_CHANGE, TASK_SAR,
)

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE_RGB = str(DATA / "sample_rgb.tif")
SAMPLE_SAR = str(DATA / "sample_sar.tif")


# ---------------------------------------------------------------------------
# Test 1: Heuristic routing (fast, no VLM or tool calls)
# ---------------------------------------------------------------------------

def test_heuristic_routing():
    print("\n=== Test 1: Heuristic Routing ===")
    cases = [
        ("How many buildings are in this image?", IMAGE_MODE_SINGLE, TASK_VQA),
        ("What type of land is shown?", IMAGE_MODE_SINGLE, TASK_CLASSIFY),
        ("Classify this scene", IMAGE_MODE_SINGLE, TASK_CLASSIFY),
        ("What changed between these images?", IMAGE_MODE_BITEMPORAL, TASK_CHANGE),
        ("Compare the before and after images", IMAGE_MODE_BITEMPORAL, TASK_CHANGE),
        ("Analyze the optical and SAR pair", IMAGE_MODE_SAR_OPTICAL, TASK_SAR),
        ("Has deforestation occurred?", IMAGE_MODE_BITEMPORAL, TASK_CHANGE),
        ("What features appear in both radar and optical?", IMAGE_MODE_SAR_OPTICAL, TASK_SAR),
        ("Is there a river visible?", IMAGE_MODE_SINGLE, TASK_VQA),
        ("Describe what you see", IMAGE_MODE_SINGLE, TASK_VQA),
    ]

    passed = 0
    for query, mode, expected in cases:
        task, score, reason = _heuristic_classify(query, mode)
        status = "OK" if task == expected else "FAIL"
        if task == expected:
            passed += 1
        print(f"  [{status:4s}] {task:12s} (score={score:2d})  expected={expected:12s}  q='{query}'")

    print(f"\n  {passed}/{len(cases)} passed")
    assert passed == len(cases), f"Some heuristic routing tests failed"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 2: Classify dispatch (uses CNN, no VLM needed)
# ---------------------------------------------------------------------------

def test_classify_dispatch():
    print("\n=== Test 2: Classify dispatch (CNN) ===")
    result = route_query(
        "Classify this scene",
        [SAMPLE_RGB],
        image_mode=IMAGE_MODE_SINGLE,
    )
    print(f"  Task:     {result['task']}")
    print(f"  Method:   {result['routing_method']}")
    print(f"  Status:   {result.get('status', '')}")
    print(f"  Answer:   {result.get('answer', '')}")
    print(f"  Backend:  {result.get('backend', '')}")
    assert result["task"] == TASK_CLASSIFY
    assert result.get("status") == "ok"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 3: Change detection dispatch (CV-only, no VLM)
# ---------------------------------------------------------------------------

def test_change_dispatch():
    print("\n=== Test 3: Change detection dispatch (CV-only) ===")
    # Import detect_changes directly to use narrate=False
    from tools.change.change_tool import detect_changes
    result = detect_changes(SAMPLE_RGB, SAMPLE_RGB, narrate=False)
    print(f"  Status:   {result.get('status', '')}")
    print(f"  Model:    {result.get('model_used', '')}")
    print(f"  Changed:  {result.get('stats', {}).get('change_pct', 'N/A')}%")
    assert result.get("status") == "ok"
    assert result.get("model_used") == "cv-only"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 4: SAR fusion dispatch (CV-only, no VLM)
# ---------------------------------------------------------------------------

def test_sar_dispatch():
    print("\n=== Test 4: SAR fusion dispatch (CV-only) ===")
    from tools.sar_fusion.sar_tool import analyze_sar_optical
    result = analyze_sar_optical(SAMPLE_RGB, SAMPLE_SAR, narrate=False)
    print(f"  Status:      {result.get('status', '')}")
    print(f"  Model:       {result.get('model_used', '')}")
    print(f"  Correlation: {result.get('stats', {}).get('correlation', 'N/A')}")
    print(f"  Edge agree:  {result.get('stats', {}).get('edge_agreement_pct', 'N/A')}%")
    assert result.get("status") == "ok"
    assert result.get("model_used") == "cv-only"
    print("  [PASS]")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_heuristic_routing()
    test_classify_dispatch()
    test_change_dispatch()
    test_sar_dispatch()

    print("\n=== All router integration tests passed ===")
