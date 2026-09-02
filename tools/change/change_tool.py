"""
SatQuery AI -- Change Detection Specialist Tool
=================================================
Takes a bi-temporal image pair (before / after), computes a change map
using classical CV, and optionally asks the VLM to narrate the changes
in plain language.

Pipeline:
  1. Resolve both images to RGB arrays (GeoTIFF-aware).
  2. Co-register: resize the "after" image to match the "before" image
     dimensions, then run ORB feature matching + affine warp for
     approximate alignment.  Falls back to simple resize if feature
     matching fails (acceptable for hackathon).
  3. Compute a per-pixel difference map (grayscale absolute diff),
     apply Otsu thresholding to produce a binary change mask, and
     create a coloured overlay highlighting changed regions.
  4. (Optional) Send before, after, and change-map images to the VLM
     and ask it to describe what changed.

All heavy pixel math is done with OpenCV/NumPy -- the VLM only writes
the natural-language summary.

Return dict mirrors ask_vqa shape for router compatibility.
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Project imports
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.geotiff_utils import load_geotiff, LoadedImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ORB feature-matching parameters for co-registration
ORB_MAX_FEATURES = 1000
MATCH_RATIO_THRESH = 0.75       # Lowe's ratio test
MIN_GOOD_MATCHES = 10           # need at least this many to compute affine

# Change-map colouring
CHANGE_COLOUR = (0, 0, 255)     # Red (BGR) for changed regions in the overlay
OVERLAY_ALPHA = 0.45            # Transparency of the change overlay

# Supported formats
SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Image resolution helpers
# ---------------------------------------------------------------------------

def _resolve_to_rgb(
    image: Union[str, Path, Image.Image, LoadedImage],
) -> np.ndarray:
    """Convert any supported image input to a (H, W, 3) uint8 BGR array.

    Returns BGR because OpenCV is the primary consumer.
    """
    if isinstance(image, LoadedImage):
        rgb = image.rgb_array
    elif isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        if path.suffix.lower() in {'.tif', '.tiff'}:
            loaded = load_geotiff(path)
            rgb = loaded.rgb_array
        else:
            rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resolve_to_pil(image) -> Image.Image:
    """Convert any supported input to a PIL Image (for VLM)."""
    if isinstance(image, LoadedImage):
        return image.pil_image
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() in {'.tif', '.tiff'}:
            return load_geotiff(path).pil_image
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


# ---------------------------------------------------------------------------
# Co-registration
# ---------------------------------------------------------------------------

def _coregister(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
) -> np.ndarray:
    """Attempt ORB feature-based alignment of *after* to *before*.

    Falls back to simple resize if not enough features match.
    Returns the aligned *after* image with the same shape as *before*.
    """
    h, w = before_bgr.shape[:2]

    # Resize after to same dimensions first (prerequisite for alignment)
    after_resized = cv2.resize(after_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale for feature detection
    gray_before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_resized, cv2.COLOR_BGR2GRAY)

    try:
        orb = cv2.ORB.create(nfeatures=ORB_MAX_FEATURES)
        kp1, des1 = orb.detectAndCompute(gray_before, None)
        kp2, des2 = orb.detectAndCompute(gray_after, None)

        if des1 is None or des2 is None:
            logger.info("No features detected, using resize-only alignment")
            return after_resized

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = bf.knnMatch(des2, des1, k=2)

        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < MATCH_RATIO_THRESH * n.distance:
                    good.append(m)

        logger.info("ORB matches: %d good / %d raw", len(good), len(raw_matches))

        if len(good) < MIN_GOOD_MATCHES:
            logger.info(
                "Not enough good matches (%d < %d), using resize-only alignment",
                len(good), MIN_GOOD_MATCHES,
            )
            return after_resized

        # Compute affine transform (after -> before)
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            logger.info("Affine estimation failed, using resize-only alignment")
            return after_resized

        aligned = cv2.warpAffine(after_resized, M, (w, h))
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        logger.info("Affine alignment applied (%d inliers)", inlier_count)
        return aligned

    except Exception as exc:
        logger.warning("Co-registration failed (%s), using resize-only", exc)
        return after_resized


# ---------------------------------------------------------------------------
# Change map computation
# ---------------------------------------------------------------------------

def _compute_change_map(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute a change map between two co-registered BGR images.

    Returns
    -------
    diff_gray : (H, W) uint8
        Absolute difference in grayscale.
    change_mask : (H, W) uint8
        Binary mask (0 or 255) of changed pixels (Otsu threshold).
    overlay : (H, W, 3) uint8
        The *before* image with changed regions highlighted in red.
    stats : dict
        {"total_pixels", "changed_pixels", "change_pct", "threshold"}
    """
    gray_before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)

    # Absolute difference
    diff_gray = cv2.absdiff(gray_before, gray_after)

    # Otsu thresholding for adaptive binarisation
    threshold_val, change_mask = cv2.threshold(
        diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Apply small morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

    # Build overlay: before image + red highlight on changed regions
    overlay = before_bgr.copy()
    red_layer = np.full_like(overlay, CHANGE_COLOUR, dtype=np.uint8)
    mask_bool = change_mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        overlay, 1.0 - OVERLAY_ALPHA, red_layer, OVERLAY_ALPHA, 0
    )[mask_bool]

    # Draw contours around changed regions for visibility
    contours, _ = cv2.findContours(
        change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    total = change_mask.size
    changed = int(np.count_nonzero(change_mask))
    stats = {
        "total_pixels": total,
        "changed_pixels": changed,
        "change_pct": round(changed / total * 100, 2) if total else 0.0,
        "threshold": float(threshold_val),
    }

    logger.info(
        "Change map: threshold=%.0f  changed=%.1f%% (%d/%d px)",
        threshold_val, stats["change_pct"], changed, total,
    )

    return diff_gray, change_mask, overlay, stats


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _gray_to_pil(gray: np.ndarray) -> Image.Image:
    """Convert a grayscale array to a PIL Image."""
    return Image.fromarray(gray, mode="L")


# ---------------------------------------------------------------------------
# VLM narration
# ---------------------------------------------------------------------------

CHANGE_SYSTEM_PROMPT = (
    "You are a remote sensing analyst specializing in change detection. "
    "You are given satellite images of the same area taken at two different "
    "times, along with a change map highlighting regions that changed. "
    "Describe what changed between the two time periods in clear, concise "
    "natural language. Focus on the type of change (construction, "
    "deforestation, flooding, vegetation growth, etc.) and its approximate "
    "extent. Keep your answer to 2-4 sentences."
)

CHANGE_PROMPT = (
    "I am showing you three images of the same area:\n"
    "1. The BEFORE image (earlier time)\n"
    "2. The AFTER image (later time)\n"
    "3. A CHANGE MAP where red regions indicate detected changes\n\n"
    "The change map shows that approximately {change_pct}% of the area "
    "has changed.\n\n"
    "Describe what changed between the two time periods. "
    "Be specific about the type of change and where it occurred."
)


def _narrate_changes(
    before_pil: Image.Image,
    after_pil: Image.Image,
    overlay_pil: Image.Image,
    stats: dict,
    model: Optional[str],
    timeout: int,
    max_retries: int,
) -> dict:
    """Ask the VLM to describe the changes. Returns query_vlm result dict."""
    from tools.ollama_client import query_vlm_multi_image

    prompt = CHANGE_PROMPT.format(change_pct=stats["change_pct"])

    return query_vlm_multi_image(
        prompt=prompt,
        images=[before_pil, after_pil, overlay_pil],
        system_prompt=CHANGE_SYSTEM_PROMPT,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_changes(
    before: Union[str, Path, Image.Image, LoadedImage],
    after: Union[str, Path, Image.Image, LoadedImage],
    *,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
    narrate: bool = True,
) -> dict:
    """Detect changes between a bi-temporal image pair.

    Parameters
    ----------
    before : str | Path | PIL.Image | LoadedImage
        The earlier (T1) image.
    after : str | Path | PIL.Image | LoadedImage
        The later (T2) image.
    model : str, optional
        Force a specific Ollama model for VLM narration.
    timeout : int
        Seconds per VLM attempt.
    max_retries : int
        Retries per model before fallback.
    narrate : bool
        If True (default), ask the VLM to describe the changes.
        Set to False for a pure-CV result (no VLM call needed).

    Returns
    -------
    dict
        {
            "answer": str,                # VLM narration (or stats summary if narrate=False)
            "raw_answer": str,            # unprocessed VLM output
            "model_used": str,            # Ollama model or "cv-only"
            "elapsed_s": float,           # total time
            "status": str,               # "ok" or "error"
            "error": str | None,          # error message if status != "ok"
            "change_map": PIL.Image,      # coloured overlay image
            "diff_image": PIL.Image,      # raw grayscale diff
            "change_mask": PIL.Image,     # binary change mask
            "stats": dict,               # {total_pixels, changed_pixels, change_pct, threshold}
        }
    """
    t0 = time.perf_counter()

    # --- Resolve inputs to BGR arrays ---
    try:
        before_bgr = _resolve_to_rgb(before)
        after_bgr = _resolve_to_rgb(after)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _error_result(str(exc))

    # --- Co-register ---
    after_aligned = _coregister(before_bgr, after_bgr)

    # --- Compute change map ---
    diff_gray, change_mask, overlay_bgr, stats = _compute_change_map(
        before_bgr, after_aligned
    )

    # Convert outputs to PIL
    overlay_pil = _bgr_to_pil(overlay_bgr)
    diff_pil = _gray_to_pil(diff_gray)
    mask_pil = _gray_to_pil(change_mask)

    # --- VLM narration (optional) ---
    if narrate:
        try:
            before_pil = _bgr_to_pil(before_bgr)
            after_pil = _bgr_to_pil(after_aligned)

            vlm_result = _narrate_changes(
                before_pil, after_pil, overlay_pil, stats,
                model, timeout, max_retries,
            )
            elapsed = time.perf_counter() - t0

            logger.info(
                "Change detection complete: %.1f%% changed, narration by %s in %.1fs",
                stats["change_pct"], vlm_result["model_used"], elapsed,
            )

            return {
                "answer": vlm_result["answer"],
                "raw_answer": vlm_result["answer"],
                "model_used": vlm_result["model_used"],
                "elapsed_s": round(elapsed, 2),
                "status": "ok",
                "error": None,
                "change_map": overlay_pil,
                "diff_image": diff_pil,
                "change_mask": mask_pil,
                "stats": stats,
            }
        except Exception as exc:
            logger.warning("VLM narration failed (%s), returning CV-only result", exc)
            # Fall through to CV-only result below

    # --- CV-only result (no VLM or VLM failed) ---
    elapsed = time.perf_counter() - t0

    summary = (
        f"Change detection found {stats['change_pct']}% of the area changed "
        f"({stats['changed_pixels']:,} of {stats['total_pixels']:,} pixels) "
        f"using an adaptive threshold of {stats['threshold']:.0f}."
    )

    logger.info("Change detection (CV-only): %.1f%% changed in %.1fs",
                stats["change_pct"], elapsed)

    return {
        "answer": summary,
        "raw_answer": summary,
        "model_used": "cv-only",
        "elapsed_s": round(elapsed, 2),
        "status": "ok",
        "error": None,
        "change_map": overlay_pil,
        "diff_image": diff_pil,
        "change_mask": mask_pil,
        "stats": stats,
    }


def _error_result(msg: str, elapsed: float = 0.0) -> dict:
    """Standardised error dict."""
    return {
        "answer": "",
        "raw_answer": "",
        "model_used": "",
        "elapsed_s": round(elapsed, 2),
        "status": "error",
        "error": msg,
        "change_map": None,
        "diff_image": None,
        "change_mask": None,
        "stats": {},
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(_sys.argv) < 3:
        print("Usage: python change_tool.py <before_image> <after_image> [--no-vlm]")
        print('Example: python change_tool.py before.tif after.tif')
        _sys.exit(1)

    before_path = _sys.argv[1]
    after_path = _sys.argv[2]
    use_vlm = "--no-vlm" not in _sys.argv

    print(f"\n> Before  : {before_path}")
    print(f"> After   : {after_path}")
    print(f"> Narrate : {'yes' if use_vlm else 'no (CV only)'}\n")

    res = detect_changes(before_path, after_path, narrate=use_vlm)

    print(f"  Status     : {res['status']}")
    print(f"  Model      : {res['model_used']}")
    print(f"  Time       : {res['elapsed_s']}s")
    print(f"  Changed    : {res['stats'].get('change_pct', 'N/A')}%")
    print(f"  Answer     : {res['answer'][:200]}")
    if res['error']:
        print(f"  Error      : {res['error']}")

    # Save outputs
    if res['change_map']:
        out = Path(before_path).with_name("change_overlay.jpg")
        res['change_map'].save(out, quality=90)
        print(f"  Overlay    : {out}")
    if res['diff_image']:
        out = Path(before_path).with_name("change_diff.jpg")
        res['diff_image'].save(out, quality=90)
        print(f"  Diff       : {out}")
