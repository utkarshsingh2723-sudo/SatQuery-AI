"""
SatQuery AI — Agentic Router / Dispatcher
============================================
Reads a user's text query (and metadata about the images provided) and
decides which specialist tool to call:

  1. VQA          — single-image question answering
  2. Classify     — scene / land-cover classification
  3. Change       — bi-temporal change detection
  4. SAR Fusion   — optical-SAR joint analysis

Strategy (kept simple for hackathon reliability):
  - First pass: keyword/heuristic rules (fast, no VLM call needed).
  - Fallback: if heuristics are ambiguous, ask the VLM to classify the
    query intent via a few-shot prompt.

The router then calls the selected tool, formats the output into a final
response dict, and returns it to the caller (GUI or API).
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

# Project imports
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.geotiff_utils import LoadedImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------

TASK_VQA = "vqa"
TASK_CLASSIFY = "classify"
TASK_CHANGE = "change"
TASK_SAR = "sar_fusion"

ALL_TASKS = [TASK_VQA, TASK_CLASSIFY, TASK_CHANGE, TASK_SAR]

TASK_DESCRIPTIONS = {
    TASK_VQA: "Visual Question Answering — answer a question about a single image",
    TASK_CLASSIFY: "Scene Classification — classify the land-cover type of an image",
    TASK_CHANGE: "Change Detection — compare two images of the same area at different times",
    TASK_SAR: "SAR Fusion — joint analysis of an optical + SAR image pair",
}

# ---------------------------------------------------------------------------
# Input context (what the caller tells the router about the images)
# ---------------------------------------------------------------------------

IMAGE_MODE_SINGLE = "single"          # one image uploaded
IMAGE_MODE_BITEMPORAL = "bitemporal"  # two images, before/after
IMAGE_MODE_SAR_OPTICAL = "sar_optical"  # optical + SAR pair


# ---------------------------------------------------------------------------
# Keyword-based heuristic classification
# ---------------------------------------------------------------------------

# Patterns checked in order; first match wins.
# Each entry: (compiled regex, task, min_score_boost)
_KEYWORD_RULES = [
    # --- SAR fusion (check first — most specific) ---
    (re.compile(r'\bsar\b', re.I), TASK_SAR, 10),
    (re.compile(r'\bradar\b', re.I), TASK_SAR, 8),
    (re.compile(r'\bfusion\b', re.I), TASK_SAR, 6),
    (re.compile(r'\boptical.{0,10}sar\b', re.I), TASK_SAR, 10),
    (re.compile(r'\bsar.{0,10}optical\b', re.I), TASK_SAR, 10),
    (re.compile(r'\bmultimodal\b', re.I), TASK_SAR, 5),

    # --- Change detection ---
    (re.compile(r'\bchang(e|ed|es|ing)\b', re.I), TASK_CHANGE, 8),
    (re.compile(r'\bbefore\b.*\bafter\b', re.I), TASK_CHANGE, 8),
    (re.compile(r'\bafter\b.*\bbefore\b', re.I), TASK_CHANGE, 8),
    (re.compile(r'\bdiffer(ence|ent|s)\b', re.I), TASK_CHANGE, 6),
    (re.compile(r'\btemporal\b', re.I), TASK_CHANGE, 6),
    (re.compile(r'\bcompare\b', re.I), TASK_CHANGE, 4),
    (re.compile(r'\bevol(ve|ution)\b', re.I), TASK_CHANGE, 6),
    (re.compile(r'\burbaniz\b', re.I), TASK_CHANGE, 5),
    (re.compile(r'\bdeforest\b', re.I), TASK_CHANGE, 5),
    (re.compile(r'\bflood(ed|ing)?\b', re.I), TASK_CHANGE, 4),
    (re.compile(r'\bconstruct(ion|ed)\b', re.I), TASK_CHANGE, 4),

    # --- Classification ---
    (re.compile(r'\bclassif(y|ication|ied)\b', re.I), TASK_CLASSIFY, 10),
    (re.compile(r'\bscene\s+(type|class|category)\b', re.I), TASK_CLASSIFY, 8),
    (re.compile(r'\bland.?cover\b', re.I), TASK_CLASSIFY, 8),
    (re.compile(r'\bland.?use\b', re.I), TASK_CLASSIFY, 7),
    (re.compile(r'\bwhat\s+type\s+of\s+(area|land|terrain|region)\b', re.I), TASK_CLASSIFY, 7),
    (re.compile(r'\bidentify\s+the\s+(type|class|scene)\b', re.I), TASK_CLASSIFY, 7),
    (re.compile(r'\bcategori(ze|se)\b', re.I), TASK_CLASSIFY, 6),
    (re.compile(r'\b(forest|residential|industrial|highway|pasture|crop|river|lake)\b', re.I), TASK_CLASSIFY, 3),

    # --- VQA (broadest — catches general questions) ---
    (re.compile(r'\bhow\s+many\b', re.I), TASK_VQA, 6),
    (re.compile(r'\bcount\b', re.I), TASK_VQA, 5),
    (re.compile(r'\b(what|where|which|who|when)\b', re.I), TASK_VQA, 3),
    (re.compile(r'\bis\s+there\b', re.I), TASK_VQA, 4),
    (re.compile(r'\bare\s+there\b', re.I), TASK_VQA, 4),
    (re.compile(r'\bdescribe\b', re.I), TASK_VQA, 4),
    (re.compile(r'\b(yes|no)\s+question\b', re.I), TASK_VQA, 5),
    (re.compile(r'\?$', re.I), TASK_VQA, 2),
]


def _heuristic_classify(query: str, image_mode: str) -> tuple:
    """Score query against keyword rules + image mode context.

    Returns (task, confidence_score, reasoning).
    """
    scores = {t: 0 for t in ALL_TASKS}

    # --- Image mode gives a strong prior ---
    if image_mode == IMAGE_MODE_BITEMPORAL:
        scores[TASK_CHANGE] += 15
    elif image_mode == IMAGE_MODE_SAR_OPTICAL:
        scores[TASK_SAR] += 15

    # --- Keyword scoring ---
    for pattern, task, boost in _KEYWORD_RULES:
        if pattern.search(query):
            scores[task] += boost

    # Find the winner
    best_task = max(scores, key=scores.get)
    best_score = scores[best_task]

    # Compute a simple confidence: how far ahead is the winner?
    sorted_scores = sorted(scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    # If no keywords matched at all, default based on image mode
    if best_score == 0:
        if image_mode == IMAGE_MODE_BITEMPORAL:
            return TASK_CHANGE, 5, "Defaulting to change detection (bi-temporal pair provided)"
        elif image_mode == IMAGE_MODE_SAR_OPTICAL:
            return TASK_SAR, 5, "Defaulting to SAR fusion (optical+SAR pair provided)"
        else:
            return TASK_VQA, 2, "Defaulting to VQA (general query, single image)"

    reasoning = f"Keyword scoring: {dict(scores)}, gap={gap}"
    return best_task, best_score, reasoning


# ---------------------------------------------------------------------------
# VLM-based intent classification (fallback for ambiguous queries)
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "You are a query classifier for a satellite image analysis system. "
    "Given a user's query, classify it into exactly ONE of these task types:\n"
    "- vqa: The user is asking a specific question about what's in the image\n"
    "- classify: The user wants to know the scene/land-cover type\n"
    "- change: The user is asking about changes between two time periods\n"
    "- sar_fusion: The user is asking about optical vs SAR comparison\n\n"
    "Respond with ONLY the task type name (vqa, classify, change, or sar_fusion). "
    "Do not add any explanation."
)

ROUTER_FEW_SHOT = (
    "Examples:\n"
    "Q: 'How many buildings are in this image?' → vqa\n"
    "Q: 'What type of land is this?' → classify\n"
    "Q: 'What changed between these two images?' → change\n"
    "Q: 'Compare the optical and SAR images' → sar_fusion\n"
    "Q: 'Is there a river in this image?' → vqa\n"
    "Q: 'Classify the scene' → classify\n"
    "Q: 'Has the forest area decreased?' → change\n"
    "Q: 'What features are visible in both optical and radar?' → sar_fusion\n\n"
    "Now classify this query: '{query}'"
)


def _vlm_classify(query: str, image_mode: str) -> tuple:
    """Use the VLM to classify the query intent. Returns (task, reasoning)."""
    from tools.ollama_client import query_vlm
    from PIL import Image as PILImage

    # Create a tiny dummy image (VLM needs an image, but we only need text classification)
    dummy_img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))

    prompt = ROUTER_FEW_SHOT.format(query=query)

    try:
        result = query_vlm(
            prompt=prompt,
            image=dummy_img,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            timeout=30,
            max_retries=1,
        )

        raw = result["answer"].strip().lower()

        # Try to extract a valid task name from the VLM response
        for task in ALL_TASKS:
            if task in raw:
                return task, f"VLM classified as '{task}' (raw: '{raw}')"

        # VLM gave something unexpected — use image mode as fallback
        logger.warning("VLM router returned unexpected: '%s'", raw)
        if image_mode == IMAGE_MODE_BITEMPORAL:
            return TASK_CHANGE, f"VLM unclear ('{raw}'), defaulting to change (bi-temporal)"
        elif image_mode == IMAGE_MODE_SAR_OPTICAL:
            return TASK_SAR, f"VLM unclear ('{raw}'), defaulting to SAR fusion"
        return TASK_VQA, f"VLM unclear ('{raw}'), defaulting to VQA"

    except Exception as exc:
        logger.warning("VLM router call failed: %s", exc)
        if image_mode == IMAGE_MODE_BITEMPORAL:
            return TASK_CHANGE, f"VLM failed ({exc}), defaulting to change"
        elif image_mode == IMAGE_MODE_SAR_OPTICAL:
            return TASK_SAR, f"VLM failed ({exc}), defaulting to SAR fusion"
        return TASK_VQA, f"VLM failed ({exc}), defaulting to VQA"


# ---------------------------------------------------------------------------
# Task dispatch (route → call the right tool)
# ---------------------------------------------------------------------------

def _dispatch(
    task: str,
    query: str,
    images: list,
    image_mode: str,
    model: Optional[str],
    timeout: int,
    max_retries: int,
) -> dict:
    """Call the appropriate specialist tool and return its result dict."""

    if task == TASK_VQA:
        from tools.vqa.vqa_tool import ask_vqa
        if not images:
            return {"answer": "", "status": "error", "error": "No image provided for VQA"}
        return ask_vqa(images[0], query, model=model, timeout=timeout, max_retries=max_retries)

    elif task == TASK_CLASSIFY:
        from tools.classify.classify_tool import classify_scene
        if not images:
            return {"answer": "", "status": "error", "error": "No image provided for classification"}
        return classify_scene(images[0], model=model, timeout=timeout, max_retries=max_retries)

    elif task == TASK_CHANGE:
        from tools.change.change_tool import detect_changes
        if len(images) < 2:
            return {"answer": "", "status": "error",
                    "error": "Change detection requires two images (before and after)"}
        return detect_changes(
            images[0], images[1],
            model=model, timeout=timeout, max_retries=max_retries,
        )

    elif task == TASK_SAR:
        from tools.sar_fusion.sar_tool import analyze_sar_optical
        if len(images) < 2:
            return {"answer": "", "status": "error",
                    "error": "SAR fusion requires two images (optical and SAR)"}
        return analyze_sar_optical(
            images[0], images[1],
            question=query, model=model, timeout=timeout, max_retries=max_retries,
        )

    else:
        return {"answer": "", "status": "error", "error": f"Unknown task: {task}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Confidence threshold: if heuristic score is below this, use VLM fallback
HEURISTIC_CONFIDENCE_THRESHOLD = 5


def route_query(
    query: str,
    images: list,
    *,
    image_mode: str = IMAGE_MODE_SINGLE,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
) -> dict:
    """Route a user query to the appropriate specialist tool.

    Parameters
    ----------
    query : str
        The user's natural-language question or instruction.
    images : list[str | Path | PIL.Image | LoadedImage]
        The uploaded image(s).
    image_mode : str
        One of "single", "bitemporal", "sar_optical".
    model : str, optional
        Force a specific Ollama model for all VLM calls.
    timeout : int
        Seconds per VLM attempt.
    max_retries : int
        Retries per model before fallback.

    Returns
    -------
    dict
        {
            "task": str,              # which tool was selected
            "task_description": str,  # human-readable task description
            "routing_method": str,    # "heuristic" or "vlm"
            "routing_reason": str,    # why this task was chosen
            "routing_elapsed_s": float,
            ... + all fields from the specialist tool's result
        }
    """
    if not query or not query.strip():
        return {
            "task": "",
            "task_description": "",
            "routing_method": "none",
            "routing_reason": "Empty query",
            "routing_elapsed_s": 0.0,
            "answer": "",
            "status": "error",
            "error": "Empty query provided",
        }

    query = query.strip()
    t0 = time.perf_counter()

    # --- Step 1: Heuristic classification ---
    task, score, reasoning = _heuristic_classify(query, image_mode)
    routing_method = "heuristic"

    logger.info(
        "Router heuristic: task=%s  score=%d  threshold=%d  reason=%s",
        task, score, HEURISTIC_CONFIDENCE_THRESHOLD, reasoning,
    )

    # --- Step 2: VLM fallback if heuristic is ambiguous ---
    if score < HEURISTIC_CONFIDENCE_THRESHOLD:
        logger.info("Heuristic confidence low (%d < %d), using VLM fallback",
                     score, HEURISTIC_CONFIDENCE_THRESHOLD)
        task, vlm_reasoning = _vlm_classify(query, image_mode)
        reasoning = vlm_reasoning
        routing_method = "vlm"

    routing_elapsed = time.perf_counter() - t0

    logger.info(
        "Router decision: task=%s  method=%s  routing_time=%.2fs",
        task, routing_method, routing_elapsed,
    )

    # --- Step 3: Dispatch to the specialist tool ---
    try:
        tool_result = _dispatch(
            task, query, images, image_mode,
            model, timeout, max_retries,
        )
    except Exception as exc:
        logger.error("Tool dispatch failed: %s", exc)
        tool_result = {
            "answer": "",
            "status": "error",
            "error": f"Tool execution failed: {exc}",
        }

    # --- Build final response ---
    response = {
        "task": task,
        "task_description": TASK_DESCRIPTIONS.get(task, ""),
        "routing_method": routing_method,
        "routing_reason": reasoning,
        "routing_elapsed_s": round(routing_elapsed, 2),
    }
    response.update(tool_result)

    return response


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_queries = [
        ("How many buildings are in this image?", IMAGE_MODE_SINGLE),
        ("What type of land is shown?", IMAGE_MODE_SINGLE),
        ("Classify this scene", IMAGE_MODE_SINGLE),
        ("What changed between these images?", IMAGE_MODE_BITEMPORAL),
        ("Compare the before and after images", IMAGE_MODE_BITEMPORAL),
        ("Analyze the optical and SAR pair", IMAGE_MODE_SAR_OPTICAL),
        ("Is there a river visible?", IMAGE_MODE_SINGLE),
        ("Describe what you see", IMAGE_MODE_SINGLE),
        ("Has deforestation occurred?", IMAGE_MODE_BITEMPORAL),
        ("What features appear in both radar and optical?", IMAGE_MODE_SAR_OPTICAL),
    ]

    print("\n=== Router Heuristic Tests ===\n")
    for q, mode in test_queries:
        task, score, reason = _heuristic_classify(q, mode)
        print(f"  [{task:12s}] (score={score:2d})  mode={mode:12s}  q='{q}'")

    print("\nDone. No VLM calls needed for heuristic-only routing test.")
