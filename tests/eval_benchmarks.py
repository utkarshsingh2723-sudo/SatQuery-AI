"""
SatQuery AI -- Combined Benchmark Evaluation Script (Phase 7)
===============================================================
Runs ALL four tool types against sample test data and produces a
consolidated report.

Sections:
  1. VQA evaluation (20 RSVQA-LR-style Q/A pairs)
  2. Scene classification evaluation (5 scenes via CNN)
  3. Change detection smoke test (CV-only, no VLM)
  4. SAR fusion smoke test (CV-only, no VLM)
  5. Router dispatch test (heuristic classification)

VLM-dependent tests (VQA, narration) require Ollama to be running.
If Ollama is unavailable, those sections are marked as SKIPPED rather
than failing the entire benchmark.

Run:  python tests/eval_benchmarks.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "benchmark_results"
SAMPLE_DIR = DATA_DIR / "vqa_eval_samples"
SAMPLE_RGB = DATA_DIR / "sample_rgb.tif"
SAMPLE_SAR = DATA_DIR / "sample_sar.tif"


# ── Synthetic image generators (reused from eval_vqa.py) ────────────────────

def _make_scene(name, size=256):
    """Generate a named synthetic scene image."""
    generators = {
        "urban": _make_urban,
        "rural": _make_rural,
        "water": _make_water,
        "forest": _make_forest,
        "barren": _make_barren,
    }
    return generators[name](size)


def _make_urban(size=256):
    img = Image.new("RGB", (size, size), (180, 190, 170))
    draw = ImageDraw.Draw(img)
    np.random.seed(10)
    draw.rectangle([0, 120, 256, 136], fill=(100, 100, 100))
    draw.rectangle([120, 0, 136, 256], fill=(100, 100, 100))
    buildings = [
        (20, 20, 60, 55), (70, 30, 105, 60), (150, 20, 200, 70),
        (20, 150, 70, 200), (150, 150, 210, 210), (210, 160, 245, 195),
    ]
    for b in buildings:
        c = tuple(np.random.randint(140, 220, 3))
        draw.rectangle(b, fill=c, outline=(80, 80, 80))
    return img


def _make_rural(size=256):
    img = Image.new("RGB", (size, size), (100, 160, 80))
    draw = ImageDraw.Draw(img)
    fields = [
        ((0, 0, 128, 128), (80, 140, 60)),
        ((128, 0, 256, 128), (120, 180, 70)),
        ((0, 128, 128, 256), (90, 170, 50)),
        ((128, 128, 256, 256), (60, 130, 40)),
    ]
    for rect, color in fields:
        draw.rectangle(rect, fill=color)
    draw.line([(0, 80), (60, 90), (130, 70), (200, 100), (256, 95)],
              fill=(60, 100, 180), width=8)
    return img


def _make_water(size=256):
    img = Image.new("RGB", (size, size), (40, 80, 150))
    draw = ImageDraw.Draw(img)
    draw.polygon([(0, 0), (256, 0), (256, 60), (200, 80), (100, 50), (0, 70)],
                 fill=(160, 150, 120))
    draw.ellipse([120, 120, 140, 130], fill=(200, 200, 180))
    return img


def _make_forest(size=256):
    img = Image.new("RGB", (size, size), (30, 100, 30))
    draw = ImageDraw.Draw(img)
    np.random.seed(42)
    for _ in range(40):
        cx = np.random.randint(10, 246)
        cy = np.random.randint(10, 246)
        r = np.random.randint(8, 20)
        g = np.random.randint(60, 130)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, g, 15))
    return img


def _make_barren(size=256):
    img = Image.new("RGB", (size, size), (200, 180, 140))
    draw = ImageDraw.Draw(img)
    draw.polygon([(0, 200), (80, 160), (160, 190), (256, 170), (256, 256), (0, 256)],
                 fill=(190, 170, 120))
    draw.polygon([(0, 180), (50, 150), (120, 170), (180, 140), (256, 160), (256, 200), (0, 200)],
                 fill=(210, 190, 150))
    return img


def _ensure_sample_images():
    """Generate sample images if they don't exist."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["urban", "rural", "water", "forest", "barren"]:
        path = SAMPLE_DIR / f"{name}.jpg"
        if not path.exists():
            _make_scene(name).save(path, quality=95)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _check_ollama():
    """Quick check if Ollama is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def _normalize_answer(text):
    text = text.lower().strip().rstrip('.')
    synonyms = {
        "agricultural": "vegetation", "farmland": "vegetation",
        "cropland": "vegetation", "crops": "vegetation",
        "fields": "vegetation", "grass": "vegetation",
        "grassland": "vegetation", "green": "vegetation",
        "plants": "vegetation", "residential": "urban",
        "city": "urban", "town": "urban", "buildings": "urban",
        "trees": "forest", "woodland": "forest", "wooded": "forest",
        "arid": "desert", "barren": "desert", "sand": "desert",
        "sandy": "desert", "dry": "desert", "ocean": "water",
        "sea": "water", "lake": "water", "river": "water",
    }
    return synonyms.get(text, text)


def _is_correct(pred, gt):
    pred_n = _normalize_answer(pred)
    gt_n = _normalize_answer(gt)
    return pred_n == gt_n or gt_n in pred_n or pred_n in gt_n


# ── Section 1: VQA Benchmark ───────────────────────────────────────────────

VQA_SAMPLES = [
    ("urban", "Is there a road in the image?", "yes", "presence"),
    ("urban", "Are there any buildings in this image?", "yes", "presence"),
    ("urban", "Is this an urban or rural area?", "urban", "rural_urban"),
    ("urban", "How many buildings can you see?", "6", "count"),
    ("urban", "Is there a river in this image?", "no", "presence"),
    ("urban", "What type of land use is shown?", "urban", "rural_urban"),
    ("rural", "Is there agricultural land in this image?", "yes", "presence"),
    ("rural", "Is this an urban or rural area?", "rural", "rural_urban"),
    ("rural", "Is there a water body in this image?", "yes", "presence"),
    ("rural", "Are there any buildings?", "no", "presence"),
    ("rural", "What is the dominant land cover?", "vegetation", "rural_urban"),
    ("water", "Is there water in this image?", "yes", "presence"),
    ("water", "What is the dominant feature?", "water", "rural_urban"),
    ("water", "Is there a forest in this image?", "no", "presence"),
    ("water", "Is there land visible?", "yes", "presence"),
    ("forest", "Is there vegetation in this image?", "yes", "presence"),
    ("forest", "What type of land cover is dominant?", "forest", "rural_urban"),
    ("forest", "Are there buildings in this image?", "no", "presence"),
    ("barren", "Is there vegetation in this image?", "no", "presence"),
    ("barren", "What type of terrain is shown?", "desert", "rural_urban"),
]


def run_vqa_benchmark(ollama_available):
    """Run VQA evaluation. Returns section results dict."""
    print("\n" + "=" * 70)
    print("  SECTION 1: VQA Evaluation (20 RSVQA-LR-style questions)")
    print("=" * 70)

    if not ollama_available:
        print("  [SKIPPED] Ollama not running -- VQA requires VLM")
        return {"status": "skipped", "reason": "Ollama unavailable",
                "total": len(VQA_SAMPLES), "answered": 0, "correct": 0,
                "accuracy_pct": 0, "results": []}

    from tools.vqa.vqa_tool import ask_vqa

    _ensure_sample_images()
    results = []
    correct = 0
    errors = 0

    for idx, (scene, question, gt, category) in enumerate(VQA_SAMPLES, 1):
        img_path = str(SAMPLE_DIR / f"{scene}.jpg")
        print(f"\n  [{idx:02d}/{len(VQA_SAMPLES)}] {scene}/{category}: {question}")

        t0 = time.perf_counter()
        try:
            res = ask_vqa(img_path, question, timeout=60, max_retries=1)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            elapsed = time.perf_counter() - t0
            res = {"answer": "", "raw_answer": "", "model_used": "", "status": "error",
                   "error": str(e), "elapsed_s": round(elapsed, 2)}

        pred = res.get("answer", "")
        status = res.get("status", "error")
        match = _is_correct(pred, gt) if status == "ok" else False

        if status == "error":
            errors += 1
            print(f"    ERROR: {res.get('error', '')[:80]}")
        else:
            if match:
                correct += 1
            print(f"    GT='{gt}' Pred='{pred}' {'[MATCH]' if match else '[MISS]'} ({res.get('elapsed_s', 0):.1f}s)")

        results.append({
            "index": idx, "scene": scene, "category": category,
            "question": question, "ground_truth": gt, "predicted": pred,
            "match": match, "status": status, "model": res.get("model_used", ""),
            "elapsed_s": round(elapsed, 2), "error": res.get("error"),
        })

    total = len(VQA_SAMPLES)
    answered = total - errors
    accuracy = correct / total * 100 if total else 0

    print(f"\n  VQA Summary: {correct}/{answered} answered correctly "
          f"({accuracy:.1f}% overall, {errors} errors)")

    # Per-category breakdown
    cats = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, {"total": 0, "correct": 0})
        cats[cat]["total"] += 1
        if r["match"]:
            cats[cat]["correct"] += 1

    for cat, s in sorted(cats.items()):
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"    {cat:15s}: {s['correct']}/{s['total']} ({pct:.0f}%)")

    return {"status": "ok", "total": total, "answered": answered,
            "correct": correct, "accuracy_pct": round(accuracy, 1),
            "errors": errors, "by_category": cats, "results": results}


# ── Section 2: Scene Classification Benchmark ──────────────────────────────

CLASSIFY_SAMPLES = [
    ("forest", "Forest"),
    ("urban", "Residential"),
    ("water", "SeaLake"),
    ("rural", "AnnualCrop"),
    ("barren", "Pasture"),   # barren is tricky — may not match perfectly
]


def run_classify_benchmark():
    """Run classification evaluation using CNN backend."""
    print("\n" + "=" * 70)
    print("  SECTION 2: Scene Classification (EuroSAT CNN)")
    print("=" * 70)

    from tools.classify.classify_tool import classify_scene, EUROSAT_CLASSES

    _ensure_sample_images()
    results = []
    correct = 0
    valid = 0

    for scene, expected_class in CLASSIFY_SAMPLES:
        img_path = str(SAMPLE_DIR / f"{scene}.jpg")
        t0 = time.perf_counter()
        res = classify_scene(img_path)
        elapsed = time.perf_counter() - t0

        status = res.get("status", "error")
        predicted = res.get("raw_answer", "")
        confidence = res.get("confidence", 0)
        backend = res.get("backend", "")

        # For classification, we check if the predicted class is valid EuroSAT
        is_valid = predicted in EUROSAT_CLASSES
        is_match = predicted == expected_class

        if is_valid:
            valid += 1
        if is_match:
            correct += 1

        match_str = "[MATCH]" if is_match else "[OK]" if is_valid else "[INVALID]"
        print(f"  {scene:10s} -> {predicted:25s} (conf={confidence:.3f}) "
              f"expected={expected_class:15s} {match_str}  [{backend}]")

        results.append({
            "scene": scene, "expected": expected_class, "predicted": predicted,
            "confidence": confidence, "match": is_match, "valid_class": is_valid,
            "backend": backend, "elapsed_s": round(elapsed, 2),
        })

    total = len(CLASSIFY_SAMPLES)
    print(f"\n  Classification Summary: {correct}/{total} exact matches, "
          f"{valid}/{total} valid EuroSAT classes")

    return {"status": "ok", "total": total, "exact_matches": correct,
            "valid_classes": valid, "results": results}


# ── Section 3: Change Detection Smoke Test ──────────────────────────────────

def run_change_benchmark():
    """Smoke test change detection with CV-only (no VLM)."""
    print("\n" + "=" * 70)
    print("  SECTION 3: Change Detection Smoke Test (CV-only)")
    print("=" * 70)

    from tools.change.change_tool import detect_changes

    results = []

    # Test 1: Same image → 0% change
    print("\n  Test 3a: Same image (should detect ~0% change)")
    t0 = time.perf_counter()
    res = detect_changes(str(SAMPLE_RGB), str(SAMPLE_RGB), narrate=False)
    elapsed = time.perf_counter() - t0

    status = res.get("status", "error")
    stats = res.get("stats", {})
    change_pct = stats.get("change_pct", -1)

    ok = status == "ok" and change_pct < 1.0
    print(f"    Status: {status}  Changed: {change_pct}%  Time: {elapsed:.1f}s  "
          f"{'[PASS]' if ok else '[FAIL]'}")
    results.append({"test": "same_image", "pass": ok, "change_pct": change_pct, "elapsed_s": round(elapsed, 2)})

    # Test 2: Different images → should detect change
    print("\n  Test 3b: Different scenes (should detect change)")
    _ensure_sample_images()
    urban = str(SAMPLE_DIR / "urban.jpg")
    forest = str(SAMPLE_DIR / "forest.jpg")

    t0 = time.perf_counter()
    res = detect_changes(urban, forest, narrate=False)
    elapsed = time.perf_counter() - t0

    status = res.get("status", "error")
    stats = res.get("stats", {})
    change_pct = stats.get("change_pct", -1)
    has_map = res.get("change_map") is not None

    ok = status == "ok" and change_pct > 5.0 and has_map
    print(f"    Status: {status}  Changed: {change_pct}%  Has map: {has_map}  Time: {elapsed:.1f}s  "
          f"{'[PASS]' if ok else '[FAIL]'}")
    results.append({"test": "different_scenes", "pass": ok, "change_pct": change_pct,
                     "has_map": has_map, "elapsed_s": round(elapsed, 2)})

    # Test 3: Error handling — missing file
    print("\n  Test 3c: Missing file (should return error gracefully)")
    res = detect_changes("nonexistent.tif", str(SAMPLE_RGB), narrate=False)
    ok = res.get("status") == "error" and res.get("error")
    print(f"    Status: {res.get('status')}  Error: {res.get('error', '')[:60]}  "
          f"{'[PASS]' if ok else '[FAIL]'}")
    results.append({"test": "error_handling", "pass": ok})

    passed = sum(1 for r in results if r["pass"])
    print(f"\n  Change Detection Summary: {passed}/{len(results)} tests passed")
    return {"status": "ok", "passed": passed, "total": len(results), "results": results}


# ── Section 4: SAR Fusion Smoke Test ────────────────────────────────────────

def run_sar_benchmark():
    """Smoke test SAR fusion with CV-only (no VLM)."""
    print("\n" + "=" * 70)
    print("  SECTION 4: SAR Fusion Smoke Test (CV-only)")
    print("=" * 70)

    from tools.sar_fusion.sar_tool import analyze_sar_optical

    results = []

    # Test 1: Optical + SAR pair
    print("\n  Test 4a: Optical + SAR analysis")
    if not SAMPLE_RGB.exists() or not SAMPLE_SAR.exists():
        print("    [SKIPPED] Sample files not found")
        return {"status": "skipped", "reason": "Sample files missing"}

    t0 = time.perf_counter()
    res = analyze_sar_optical(str(SAMPLE_RGB), str(SAMPLE_SAR), narrate=False)
    elapsed = time.perf_counter() - t0

    status = res.get("status", "error")
    stats = res.get("stats", {})
    has_composite = res.get("composite") is not None
    has_edges = res.get("edge_comparison") is not None
    corr = stats.get("correlation", None)

    ok = status == "ok" and has_composite and has_edges and corr is not None
    print(f"    Status: {status}  Correlation: {corr}  Composite: {has_composite}  "
          f"Edges: {has_edges}  Time: {elapsed:.1f}s  {'[PASS]' if ok else '[FAIL]'}")
    results.append({"test": "optical_sar_pair", "pass": ok, "stats": stats,
                     "elapsed_s": round(elapsed, 2)})

    # Test 2: Same image as both optical and SAR (should still work)
    print("\n  Test 4b: Same image as both (self-comparison)")
    t0 = time.perf_counter()
    res = analyze_sar_optical(str(SAMPLE_RGB), str(SAMPLE_RGB), narrate=False)
    elapsed = time.perf_counter() - t0

    status = res.get("status", "error")
    corr = res.get("stats", {}).get("correlation", None)

    ok = status == "ok" and corr is not None and corr > 0.9
    print(f"    Status: {status}  Correlation: {corr}  Time: {elapsed:.1f}s  "
          f"{'[PASS]' if ok else '[FAIL]'}")
    results.append({"test": "self_comparison", "pass": ok, "correlation": corr,
                     "elapsed_s": round(elapsed, 2)})

    passed = sum(1 for r in results if r["pass"])
    print(f"\n  SAR Fusion Summary: {passed}/{len(results)} tests passed")
    return {"status": "ok", "passed": passed, "total": len(results), "results": results}


# ── Section 5: Router Heuristic Test ────────────────────────────────────────

ROUTER_TEST_CASES = [
    ("How many buildings are in this image?", "single", "vqa"),
    ("What type of land is shown?", "single", "classify"),
    ("Classify this scene", "single", "classify"),
    ("What changed between these images?", "bitemporal", "change"),
    ("Compare the before and after images", "bitemporal", "change"),
    ("Analyze the optical and SAR pair", "sar_optical", "sar_fusion"),
    ("Is there a river visible?", "single", "vqa"),
    ("Describe what you see", "single", "vqa"),
    ("Has deforestation occurred?", "bitemporal", "change"),
    ("What features appear in both radar and optical?", "sar_optical", "sar_fusion"),
    ("How many roads are visible?", "single", "vqa"),
    ("What is the land cover classification?", "single", "classify"),
    ("Show me what's different in the two images", "bitemporal", "change"),
    ("Compare SAR with optical imagery", "sar_optical", "sar_fusion"),
    ("Is this area forested?", "single", "vqa"),
]


def run_router_benchmark():
    """Test the router heuristic classification."""
    print("\n" + "=" * 70)
    print("  SECTION 5: Router Heuristic Routing (15 queries)")
    print("=" * 70)

    from router.router import _heuristic_classify

    results = []
    correct = 0

    for query, mode, expected in ROUTER_TEST_CASES:
        task, score, reason = _heuristic_classify(query, mode)
        match = task == expected
        if match:
            correct += 1
        status = "OK" if match else "FAIL"
        print(f"  [{status:4s}] {task:12s} (score={score:2d})  expected={expected:12s}  q='{query}'")
        results.append({"query": query, "mode": mode, "expected": expected,
                        "predicted": task, "score": score, "match": match})

    total = len(ROUTER_TEST_CASES)
    print(f"\n  Router Summary: {correct}/{total} correct")
    return {"status": "ok", "correct": correct, "total": total, "results": results}


# ── Main benchmark runner ──────────────────────────────────────────────────

def run_all():
    """Run the complete benchmark suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  SatQuery AI -- Complete Benchmark Evaluation")
    print(f"#  Timestamp: {timestamp}")
    print("#" * 70)

    ollama_available = _check_ollama()
    print(f"\n  Ollama status: {'AVAILABLE' if ollama_available else 'NOT RUNNING'}")

    t_total = time.perf_counter()

    # Run all sections
    report = {
        "timestamp": timestamp,
        "ollama_available": ollama_available,
    }

    report["vqa"] = run_vqa_benchmark(ollama_available)
    report["classification"] = run_classify_benchmark()
    report["change_detection"] = run_change_benchmark()
    report["sar_fusion"] = run_sar_benchmark()
    report["router"] = run_router_benchmark()

    total_time = time.perf_counter() - t_total

    # ── Final summary ──
    print("\n" + "#" * 70)
    print("#  OVERALL BENCHMARK SUMMARY")
    print("#" * 70)

    sections = [
        ("VQA", report["vqa"]),
        ("Classification", report["classification"]),
        ("Change Detection", report["change_detection"]),
        ("SAR Fusion", report["sar_fusion"]),
        ("Router", report["router"]),
    ]

    for name, data in sections:
        st = data.get("status", "error")
        if st == "skipped":
            detail = f"SKIPPED ({data.get('reason', '')})"
        elif "accuracy_pct" in data:
            detail = f"{data.get('correct', 0)}/{data.get('total', 0)} ({data['accuracy_pct']}%)"
        elif "exact_matches" in data:
            detail = f"{data['exact_matches']}/{data['total']} exact, {data['valid_classes']}/{data['total']} valid"
        elif "passed" in data:
            detail = f"{data['passed']}/{data['total']} passed"
        elif "correct" in data:
            detail = f"{data['correct']}/{data['total']} correct"
        else:
            detail = st

        print(f"  {name:20s}: {detail}")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Save report ──
    report["total_time_s"] = round(total_time, 2)

    # Make JSON-serializable (remove PIL images from nested results)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if not isinstance(v, Image.Image)}
        elif isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    report_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=2, ensure_ascii=False)
    print(f"\n  Full report: {report_file}")

    # Human-readable summary
    summary_file = RESULTS_DIR / f"benchmark_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"SatQuery AI Benchmark Report -- {timestamp}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Ollama: {'Available' if ollama_available else 'Not running'}\n\n")
        for name, data in sections:
            st = data.get("status", "error")
            f.write(f"{name}:\n")
            if st == "skipped":
                f.write(f"  SKIPPED: {data.get('reason', '')}\n")
            elif "accuracy_pct" in data:
                f.write(f"  Accuracy: {data.get('correct', 0)}/{data.get('total', 0)} "
                        f"({data['accuracy_pct']}%)\n")
                f.write(f"  Errors: {data.get('errors', 0)}\n")
            elif "exact_matches" in data:
                f.write(f"  Exact: {data['exact_matches']}/{data['total']}\n")
                f.write(f"  Valid: {data['valid_classes']}/{data['total']}\n")
            elif "passed" in data:
                f.write(f"  Passed: {data['passed']}/{data['total']}\n")
            elif "correct" in data:
                f.write(f"  Correct: {data['correct']}/{data['total']}\n")
            f.write("\n")
        f.write(f"Total time: {total_time:.1f}s\n")
    print(f"  Summary:     {summary_file}")

    print()
    return report


if __name__ == "__main__":
    run_all()
