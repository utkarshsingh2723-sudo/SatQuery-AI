"""
SatQuery AI -- VQA Evaluation Script
======================================
Runs the VQA tool against RSVQA-LR-style question/image pairs and logs
predicted vs ground-truth answers.

Data strategy:
  - Ships with 20 hardcoded RSVQA-LR-style Q/A pairs using synthetic
    remote-sensing-like images generated at runtime.
  - If real RSVQA-LR data is placed in data/rsvqa_lr_sample/, it will
    use those instead.

Run:  python tests/eval_vqa.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.vqa.vqa_tool import ask_vqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLE_DIR = PROJECT_ROOT / "data" / "vqa_eval_samples"
RESULTS_DIR = PROJECT_ROOT / "data" / "vqa_eval_results"

# ---------------------------------------------------------------------------
# Synthetic sample generation
# ---------------------------------------------------------------------------

def _make_scene_urban(size=256):
    """Generate a synthetic 'urban' scene with rectangular buildings."""
    img = Image.new("RGB", (size, size), (180, 190, 170))  # greenish bg
    draw = ImageDraw.Draw(img)
    np.random.seed(10)
    # Roads
    draw.rectangle([0, 120, 256, 136], fill=(100, 100, 100))
    draw.rectangle([120, 0, 136, 256], fill=(100, 100, 100))
    # Buildings (rectangles)
    buildings = [
        (20, 20, 60, 55), (70, 30, 105, 60), (150, 20, 200, 70),
        (20, 150, 70, 200), (150, 150, 210, 210), (210, 160, 245, 195),
    ]
    for b in buildings:
        c = tuple(np.random.randint(140, 220, 3))
        draw.rectangle(b, fill=c, outline=(80, 80, 80))
    return img


def _make_scene_rural(size=256):
    """Generate a synthetic 'rural/agricultural' scene with fields."""
    img = Image.new("RGB", (size, size), (100, 160, 80))  # green base
    draw = ImageDraw.Draw(img)
    # Crop fields of different green shades
    fields = [
        ((0, 0, 128, 128), (80, 140, 60)),
        ((128, 0, 256, 128), (120, 180, 70)),
        ((0, 128, 128, 256), (90, 170, 50)),
        ((128, 128, 256, 256), (60, 130, 40)),
    ]
    for rect, color in fields:
        draw.rectangle(rect, fill=color)
    # A river
    draw.line([(0, 80), (60, 90), (130, 70), (200, 100), (256, 95)],
              fill=(60, 100, 180), width=8)
    return img


def _make_scene_water(size=256):
    """Generate a synthetic 'water body' scene."""
    img = Image.new("RGB", (size, size), (40, 80, 150))  # blue water
    draw = ImageDraw.Draw(img)
    # Shore on one side
    draw.polygon([(0, 0), (256, 0), (256, 60), (200, 80), (100, 50), (0, 70)],
                 fill=(160, 150, 120))
    # A small boat-like shape
    draw.ellipse([120, 120, 140, 130], fill=(200, 200, 180))
    return img


def _make_scene_forest(size=256):
    """Generate a synthetic 'forest' scene with dense green."""
    img = Image.new("RGB", (size, size), (30, 100, 30))
    draw = ImageDraw.Draw(img)
    np.random.seed(42)
    # Tree canopy blobs
    for _ in range(40):
        cx = np.random.randint(10, 246)
        cy = np.random.randint(10, 246)
        r = np.random.randint(8, 20)
        g = np.random.randint(60, 130)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(20, g, 15))
    return img


def _make_scene_barren(size=256):
    """Generate a synthetic 'barren/desert' scene."""
    img = Image.new("RGB", (size, size), (200, 180, 140))
    draw = ImageDraw.Draw(img)
    # Some dune-like shapes
    draw.polygon([(0, 200), (80, 160), (160, 190), (256, 170), (256, 256), (0, 256)],
                 fill=(190, 170, 120))
    draw.polygon([(0, 180), (50, 150), (120, 170), (180, 140), (256, 160), (256, 200), (0, 200)],
                 fill=(210, 190, 150))
    return img


# ---------------------------------------------------------------------------
# Sample dataset definition
# ---------------------------------------------------------------------------

# Each entry: (scene_generator, scene_name, question, ground_truth_answer, category)
# Categories mirror RSVQA-LR: presence, comparison, count, rural_urban
EVAL_SAMPLES = [
    # ---- Urban scene ----
    (_make_scene_urban, "urban", "Is there a road in the image?", "yes", "presence"),
    (_make_scene_urban, "urban", "Are there any buildings in this image?", "yes", "presence"),
    (_make_scene_urban, "urban", "Is this an urban or rural area?", "urban", "rural_urban"),
    (_make_scene_urban, "urban", "How many buildings can you see?", "6", "count"),
    (_make_scene_urban, "urban", "Is there a river in this image?", "no", "presence"),
    (_make_scene_urban, "urban", "What type of land use is shown?", "urban", "rural_urban"),

    # ---- Rural scene ----
    (_make_scene_rural, "rural", "Is there agricultural land in this image?", "yes", "presence"),
    (_make_scene_rural, "rural", "Is this an urban or rural area?", "rural", "rural_urban"),
    (_make_scene_rural, "rural", "Is there a water body in this image?", "yes", "presence"),
    (_make_scene_rural, "rural", "Are there any buildings?", "no", "presence"),
    (_make_scene_rural, "rural", "What is the dominant land cover?", "vegetation", "rural_urban"),

    # ---- Water scene ----
    (_make_scene_water, "water", "Is there water in this image?", "yes", "presence"),
    (_make_scene_water, "water", "What is the dominant feature?", "water", "rural_urban"),
    (_make_scene_water, "water", "Is there a forest in this image?", "no", "presence"),
    (_make_scene_water, "water", "Is there land visible?", "yes", "presence"),

    # ---- Forest scene ----
    (_make_scene_forest, "forest", "Is there vegetation in this image?", "yes", "presence"),
    (_make_scene_forest, "forest", "What type of land cover is dominant?", "forest", "rural_urban"),
    (_make_scene_forest, "forest", "Are there buildings in this image?", "no", "presence"),

    # ---- Barren scene ----
    (_make_scene_barren, "barren", "Is there vegetation in this image?", "no", "presence"),
    (_make_scene_barren, "barren", "What type of terrain is shown?", "desert", "rural_urban"),
]


def _generate_sample_images():
    """Generate and save synthetic sample images."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    generators = {
        "urban": _make_scene_urban,
        "rural": _make_scene_rural,
        "water": _make_scene_water,
        "forest": _make_scene_forest,
        "barren": _make_scene_barren,
    }
    paths = {}
    for name, gen in generators.items():
        path = SAMPLE_DIR / f"{name}.jpg"
        if not path.exists():
            img = gen()
            img.save(path, quality=95)
            logger.info("Generated sample: %s", path.name)
        paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Normalize an answer for comparison (lowercase, strip, common synonyms)."""
    text = text.lower().strip().rstrip('.')

    # Common synonyms / equivalences
    synonyms = {
        "agricultural": "vegetation",
        "farmland": "vegetation",
        "cropland": "vegetation",
        "crops": "vegetation",
        "fields": "vegetation",
        "grass": "vegetation",
        "grassland": "vegetation",
        "green": "vegetation",
        "greenery": "vegetation",
        "plants": "vegetation",
        "residential": "urban",
        "city": "urban",
        "town": "urban",
        "buildings": "urban",
        "built-up": "urban",
        "a city street": "urban",
        "trees": "forest",
        "woodland": "forest",
        "wooded": "forest",
        "arid": "desert",
        "barren": "desert",
        "sand": "desert",
        "sandy": "desert",
        "dry": "desert",
        "dry land": "desert",
        "land": "desert",
        "ocean": "water",
        "sea": "water",
        "lake": "water",
        "river": "water",
    }
    return synonyms.get(text, text)


def _is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if the predicted answer matches ground truth (fuzzy)."""
    pred_norm = _normalize_answer(predicted)
    gt_norm = _normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # Check containment (e.g., "yes, there is a road" contains "yes")
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True

    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval():
    """Run VQA evaluation on all sample pairs."""
    print("\n" + "=" * 70)
    print("  SatQuery AI -- VQA Evaluation (RSVQA-LR style)")
    print("=" * 70)

    # Generate sample images
    image_paths = _generate_sample_images()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    correct = 0
    total = len(EVAL_SAMPLES)
    errors = 0

    for idx, (gen_fn, scene, question, gt_answer, category) in enumerate(EVAL_SAMPLES, 1):
        img_path = image_paths[scene]

        print(f"\n--- [{idx:02d}/{total}] {scene} | {category} ---")
        print(f"  Q: {question}")
        print(f"  GT: {gt_answer}")

        t0 = time.perf_counter()
        result = ask_vqa(str(img_path), question)
        elapsed = time.perf_counter() - t0

        pred = result["answer"]
        status = result["status"]
        model = result["model_used"]
        match = _is_correct(pred, gt_answer) if status == "ok" else False

        if status == "error":
            errors += 1
            print(f"  ERROR: {result['error']}")
        else:
            if match:
                correct += 1
            marker = "[MATCH]" if match else "[MISS]"
            print(f"  Pred: {pred}  {marker}")
            print(f"  Model: {model}  Time: {elapsed:.1f}s")

        results.append({
            "index": idx,
            "scene": scene,
            "category": category,
            "question": question,
            "ground_truth": gt_answer,
            "predicted": pred,
            "raw_answer": result.get("raw_answer", ""),
            "match": match,
            "status": status,
            "model": model,
            "elapsed_s": round(elapsed, 2),
            "error": result.get("error"),
        })

    # --- Summary ---
    accuracy = correct / total * 100 if total > 0 else 0
    answered = total - errors

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total questions : {total}")
    print(f"  Answered        : {answered}")
    print(f"  Errors          : {errors}")
    print(f"  Correct (fuzzy) : {correct}/{answered} ({correct/answered*100:.0f}%)" if answered else "  N/A")
    print(f"  Overall accuracy: {accuracy:.1f}%")

    # Breakdown by category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["match"]:
            categories[cat]["correct"] += 1

    print("\n  By category:")
    for cat, stats in sorted(categories.items()):
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        print(f"    {cat:15s}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"vqa_eval_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total": total,
            "answered": answered,
            "correct": correct,
            "accuracy_pct": round(accuracy, 1),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved: {results_file}")

    # Also save a human-readable log
    log_file = RESULTS_DIR / f"vqa_eval_{timestamp}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"SatQuery AI VQA Evaluation -- {timestamp}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"{'#':>3}  {'Scene':8}  {'Cat':12}  {'GT':12}  {'Pred':12}  {'Match':6}  {'Time':6}\n")
        f.write(f"{'-'*3}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*6}\n")
        for r in results:
            match_str = "OK" if r["match"] else "MISS"
            if r["status"] == "error":
                match_str = "ERR"
            f.write(
                f"{r['index']:3d}  {r['scene']:8s}  {r['category']:12s}  "
                f"{r['ground_truth']:12s}  {r['predicted'][:12]:12s}  "
                f"{match_str:6s}  {r['elapsed_s']:5.1f}s\n"
            )
        f.write(f"\nOverall: {correct}/{total} ({accuracy:.1f}%)\n")

    print(f"  Log saved   : {log_file}")
    print()

    return results


if __name__ == "__main__":
    run_eval()
