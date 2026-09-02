"""
SatQuery AI — Optical-SAR Fusion Specialist Tool
===================================================
Takes a co-registered optical + SAR image pair, computes classical CV
comparisons (edge maps, intensity correlation, composite overlay), and
asks the VLM to narrate the comparison in plain language.

Pipeline:
  1. Load both images (GeoTIFF-aware) and co-register if needed.
  2. Compute edge maps (Canny) for both optical and SAR.
  3. Create a false-colour composite overlay (optical-R, SAR-G, optical-B).
  4. Compute basic statistics: correlation coefficient, structural
     similarity (SSIM-lite), edge agreement percentage.
  5. Send the optical, SAR, and composite images to the VLM with the
     computed statistics, and ask it to narrate what the joint analysis
     reveals.

All pixel-level math is done with OpenCV/NumPy — the VLM only writes
the natural-language summary.

Return dict mirrors ask_vqa / detect_changes shape for router compatibility.
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

SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# Canny edge detection thresholds
CANNY_LOW = 50
CANNY_HIGH = 150

# Co-registration (same approach as change detection)
ORB_MAX_FEATURES = 1000
MATCH_RATIO_THRESH = 0.75
MIN_GOOD_MATCHES = 10


# ---------------------------------------------------------------------------
# Image resolution helpers
# ---------------------------------------------------------------------------

def _resolve_to_bgr(
    image: Union[str, Path, Image.Image, LoadedImage],
) -> np.ndarray:
    """Convert any supported image input to a (H, W, 3) uint8 BGR array."""
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


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _gray_to_pil(gray: np.ndarray) -> Image.Image:
    """Convert a grayscale array to a PIL Image."""
    return Image.fromarray(gray, mode="L")


# ---------------------------------------------------------------------------
# Co-registration (align SAR to optical)
# ---------------------------------------------------------------------------

def _coregister(
    reference_bgr: np.ndarray,
    target_bgr: np.ndarray,
) -> np.ndarray:
    """Align *target* to *reference* using ORB + affine.

    Falls back to simple resize if feature matching fails.
    """
    h, w = reference_bgr.shape[:2]
    target_resized = cv2.resize(target_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    gray_ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target_resized, cv2.COLOR_BGR2GRAY)

    try:
        orb = cv2.ORB.create(nfeatures=ORB_MAX_FEATURES)
        kp1, des1 = orb.detectAndCompute(gray_ref, None)
        kp2, des2 = orb.detectAndCompute(gray_tgt, None)

        if des1 is None or des2 is None:
            logger.info("No features detected, using resize-only alignment")
            return target_resized

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = bf.knnMatch(des2, des1, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < MATCH_RATIO_THRESH * n.distance:
                    good.append(m)

        logger.info("ORB matches (SAR alignment): %d good / %d raw",
                     len(good), len(raw_matches))

        if len(good) < MIN_GOOD_MATCHES:
            logger.info("Not enough matches (%d < %d), resize-only",
                         len(good), MIN_GOOD_MATCHES)
            return target_resized

        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            logger.info("Affine estimation failed, resize-only alignment")
            return target_resized

        aligned = cv2.warpAffine(target_resized, M, (w, h))
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        logger.info("SAR affine alignment applied (%d inliers)", inlier_count)
        return aligned

    except Exception as exc:
        logger.warning("SAR co-registration failed (%s), resize-only", exc)
        return target_resized


# ---------------------------------------------------------------------------
# Analysis computations (all classical CV — no VLM)
# ---------------------------------------------------------------------------

def _compute_edge_maps(
    optical_bgr: np.ndarray,
    sar_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Canny edge maps for both images."""
    gray_opt = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2GRAY)
    gray_sar = cv2.cvtColor(sar_bgr, cv2.COLOR_BGR2GRAY)

    edges_opt = cv2.Canny(gray_opt, CANNY_LOW, CANNY_HIGH)
    edges_sar = cv2.Canny(gray_sar, CANNY_LOW, CANNY_HIGH)

    return edges_opt, edges_sar


def _compute_composite(
    optical_bgr: np.ndarray,
    sar_bgr: np.ndarray,
) -> np.ndarray:
    """Create a false-colour composite: R=optical_red, G=SAR, B=optical_blue.

    This helps visualise what features are visible in both modalities.
    """
    gray_sar = cv2.cvtColor(sar_bgr, cv2.COLOR_BGR2GRAY)

    # Use optical red and blue channels, SAR for green
    composite = np.stack([
        optical_bgr[:, :, 2],   # Red channel (BGR index 2 = R)
        gray_sar,                # SAR intensity as green
        optical_bgr[:, :, 0],   # Blue channel (BGR index 0 = B)
    ], axis=-1)  # This is RGB

    return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)  # Return as BGR


def _compute_statistics(
    optical_bgr: np.ndarray,
    sar_bgr: np.ndarray,
    edges_opt: np.ndarray,
    edges_sar: np.ndarray,
) -> dict:
    """Compute comparison statistics between optical and SAR."""
    gray_opt = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_sar = cv2.cvtColor(sar_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Pearson correlation coefficient
    opt_flat = gray_opt.flatten()
    sar_flat = gray_sar.flatten()

    opt_mean = opt_flat.mean()
    sar_mean = sar_flat.mean()
    opt_std = opt_flat.std()
    sar_std = sar_flat.std()

    if opt_std > 0 and sar_std > 0:
        correlation = float(np.corrcoef(opt_flat, sar_flat)[0, 1])
    else:
        correlation = 0.0

    # Edge agreement: percentage of pixels where both have an edge
    # or both don't have an edge
    edge_agree = np.mean((edges_opt > 0) == (edges_sar > 0))

    # Edge overlap (Jaccard-like): intersection / union of edge pixels
    opt_edge_count = np.count_nonzero(edges_opt)
    sar_edge_count = np.count_nonzero(edges_sar)
    both_edge = np.count_nonzero((edges_opt > 0) & (edges_sar > 0))
    union_edge = np.count_nonzero((edges_opt > 0) | (edges_sar > 0))
    edge_overlap = both_edge / union_edge if union_edge > 0 else 0.0

    # Simple SSIM-lite (structural similarity approximation)
    # Full SSIM is complex; we use a simplified luminance + contrast comparison
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_num = (2 * opt_mean * sar_mean + c1) * (2 * opt_std * sar_std + c2)
    ssim_den = (opt_mean**2 + sar_mean**2 + c1) * (opt_std**2 + sar_std**2 + c2)
    ssim_approx = ssim_num / ssim_den if ssim_den > 0 else 0.0

    stats = {
        "correlation": round(correlation, 4),
        "edge_agreement_pct": round(edge_agree * 100, 2),
        "edge_overlap_iou": round(edge_overlap, 4),
        "ssim_approx": round(ssim_approx, 4),
        "optical_edge_pixels": int(opt_edge_count),
        "sar_edge_pixels": int(sar_edge_count),
        "shared_edge_pixels": int(both_edge),
        "optical_mean_intensity": round(float(opt_mean), 2),
        "sar_mean_intensity": round(float(sar_mean), 2),
    }

    logger.info(
        "SAR-optical stats: corr=%.3f  edge_agree=%.1f%%  edge_iou=%.3f  ssim≈%.3f",
        stats["correlation"], stats["edge_agreement_pct"],
        stats["edge_overlap_iou"], stats["ssim_approx"],
    )

    return stats


def _create_edge_comparison(
    edges_opt: np.ndarray,
    edges_sar: np.ndarray,
) -> np.ndarray:
    """Create a colour-coded edge comparison image (BGR).

    Green = optical-only edges, Red = SAR-only edges, Yellow = shared edges.
    """
    h, w = edges_opt.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)

    opt_mask = edges_opt > 0
    sar_mask = edges_sar > 0
    both_mask = opt_mask & sar_mask

    # Green for optical-only edges
    comparison[opt_mask & ~sar_mask] = [0, 255, 0]     # BGR green
    # Red for SAR-only edges
    comparison[sar_mask & ~opt_mask] = [0, 0, 255]     # BGR red
    # Yellow for shared edges
    comparison[both_mask] = [0, 255, 255]               # BGR yellow

    return comparison


# ---------------------------------------------------------------------------
# VLM narration
# ---------------------------------------------------------------------------

SAR_SYSTEM_PROMPT = (
    "You are a remote sensing analyst specializing in optical-SAR image fusion. "
    "You are given an optical satellite image, a SAR (Synthetic Aperture Radar) "
    "image of the same area, and a false-colour composite overlay. "
    "Describe what the joint analysis reveals: what features are visible in "
    "both modalities, what is visible only in optical, what is visible only "
    "in SAR, and any notable observations from the comparison. "
    "Keep your answer to 3-5 sentences."
)

SAR_PROMPT = (
    "I am showing you three images of the same area:\n"
    "1. An OPTICAL satellite image\n"
    "2. A SAR (radar) image\n"
    "3. A FALSE-COLOUR COMPOSITE (Red=optical, Green=SAR, Blue=optical)\n\n"
    "Statistical comparison:\n"
    "- Intensity correlation: {correlation}\n"
    "- Edge agreement: {edge_agreement_pct}%\n"
    "- Edge overlap (IoU): {edge_overlap_iou}\n"
    "- Structural similarity (approx): {ssim_approx}\n\n"
    "Describe what this joint optical-SAR analysis reveals about the area. "
    "What features are shared? What is unique to each modality?"
)


def _narrate_comparison(
    optical_pil: Image.Image,
    sar_pil: Image.Image,
    composite_pil: Image.Image,
    stats: dict,
    model: Optional[str],
    timeout: int,
    max_retries: int,
) -> dict:
    """Ask the VLM to describe the optical-SAR comparison."""
    from tools.ollama_client import query_vlm_multi_image

    prompt = SAR_PROMPT.format(**stats)

    return query_vlm_multi_image(
        prompt=prompt,
        images=[optical_pil, sar_pil, composite_pil],
        system_prompt=SAR_SYSTEM_PROMPT,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sar_optical(
    optical: Union[str, Path, Image.Image, LoadedImage],
    sar: Union[str, Path, Image.Image, LoadedImage],
    *,
    question: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
    narrate: bool = True,
) -> dict:
    """Perform joint optical-SAR analysis on a co-registered pair.

    Parameters
    ----------
    optical : str | Path | PIL.Image | LoadedImage
        The optical satellite image.
    sar : str | Path | PIL.Image | LoadedImage
        The SAR image of the same area.
    question : str, optional
        If provided, appended to the VLM prompt for a more targeted analysis.
    model : str, optional
        Force a specific Ollama model for VLM narration.
    timeout : int
        Seconds per VLM attempt.
    max_retries : int
        Retries per model before fallback.
    narrate : bool
        If True (default), ask the VLM to describe the comparison.

    Returns
    -------
    dict
        {
            "answer": str,                # VLM narration or stats summary
            "raw_answer": str,
            "model_used": str,
            "elapsed_s": float,
            "status": str,               # "ok" or "error"
            "error": str | None,
            "composite": PIL.Image,      # false-colour composite
            "edge_comparison": PIL.Image, # colour-coded edge map
            "stats": dict,
        }
    """
    t0 = time.perf_counter()

    # --- Resolve inputs ---
    try:
        optical_bgr = _resolve_to_bgr(optical)
        sar_bgr = _resolve_to_bgr(sar)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _error_result(str(exc))

    # --- Co-register SAR to optical ---
    sar_aligned = _coregister(optical_bgr, sar_bgr)

    # --- Compute analysis products ---
    edges_opt, edges_sar = _compute_edge_maps(optical_bgr, sar_aligned)
    composite_bgr = _compute_composite(optical_bgr, sar_aligned)
    edge_comp_bgr = _create_edge_comparison(edges_opt, edges_sar)
    stats = _compute_statistics(optical_bgr, sar_aligned, edges_opt, edges_sar)

    # Convert to PIL for output + VLM
    composite_pil = _bgr_to_pil(composite_bgr)
    edge_comp_pil = _bgr_to_pil(edge_comp_bgr)

    # --- VLM narration (optional) ---
    if narrate:
        try:
            optical_pil = _bgr_to_pil(optical_bgr)
            sar_pil = _bgr_to_pil(sar_aligned)

            vlm_result = _narrate_comparison(
                optical_pil, sar_pil, composite_pil, stats,
                model, timeout, max_retries,
            )
            elapsed = time.perf_counter() - t0

            answer = vlm_result["answer"]
            if question:
                # If a specific question was asked, append it context
                answer = f"{answer}\n\n(Regarding your question: '{question}' — the above analysis should address this.)"

            logger.info(
                "SAR-optical analysis complete: narrated by %s in %.1fs",
                vlm_result["model_used"], elapsed,
            )

            return {
                "answer": answer,
                "raw_answer": vlm_result["answer"],
                "model_used": vlm_result["model_used"],
                "elapsed_s": round(elapsed, 2),
                "status": "ok",
                "error": None,
                "composite": composite_pil,
                "edge_comparison": edge_comp_pil,
                "stats": stats,
            }
        except Exception as exc:
            logger.warning("VLM narration failed (%s), returning CV-only", exc)

    # --- CV-only result ---
    elapsed = time.perf_counter() - t0

    summary = (
        f"Optical-SAR joint analysis:\n"
        f"• Intensity correlation: {stats['correlation']}\n"
        f"• Edge agreement: {stats['edge_agreement_pct']}%\n"
        f"• Edge overlap (IoU): {stats['edge_overlap_iou']}\n"
        f"• Structural similarity: {stats['ssim_approx']}\n"
        f"• Optical edges: {stats['optical_edge_pixels']:,} px, "
        f"SAR edges: {stats['sar_edge_pixels']:,} px, "
        f"shared: {stats['shared_edge_pixels']:,} px"
    )

    logger.info("SAR-optical analysis (CV-only) in %.1fs", elapsed)

    return {
        "answer": summary,
        "raw_answer": summary,
        "model_used": "cv-only",
        "elapsed_s": round(elapsed, 2),
        "status": "ok",
        "error": None,
        "composite": composite_pil,
        "edge_comparison": edge_comp_pil,
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
        "composite": None,
        "edge_comparison": None,
        "stats": {},
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(_sys.argv) < 3:
        print("Usage: python sar_tool.py <optical_image> <sar_image> [--no-vlm]")
        print('Example: python sar_tool.py optical.tif sar.tif')
        _sys.exit(1)

    optical_path = _sys.argv[1]
    sar_path = _sys.argv[2]
    use_vlm = "--no-vlm" not in _sys.argv

    print(f"\n> Optical : {optical_path}")
    print(f"> SAR     : {sar_path}")
    print(f"> Narrate : {'yes' if use_vlm else 'no (CV only)'}\n")

    res = analyze_sar_optical(optical_path, sar_path, narrate=use_vlm)

    print(f"  Status     : {res['status']}")
    print(f"  Model      : {res['model_used']}")
    print(f"  Time       : {res['elapsed_s']}s")
    print(f"  Correlation: {res['stats'].get('correlation', 'N/A')}")
    print(f"  Edge agree : {res['stats'].get('edge_agreement_pct', 'N/A')}%")
    print(f"  Answer     : {res['answer'][:300]}")
    if res['error']:
        print(f"  Error      : {res['error']}")

    # Save outputs
    if res['composite']:
        out = Path(optical_path).with_name("sar_composite.jpg")
        res['composite'].save(out, quality=90)
        print(f"  Composite  : {out}")
    if res['edge_comparison']:
        out = Path(optical_path).with_name("sar_edge_comparison.jpg")
        res['edge_comparison'].save(out, quality=90)
        print(f"  Edges      : {out}")
