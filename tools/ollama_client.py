"""
SatQuery AI — Ollama VLM Client
================================
Sends image + text queries to a local Ollama server.
Primary model:  qwen2.5vl:7b
Fallback chain: moondream -> llava:7b-v1.6-q4

Design decisions:
  - A threading Lock ensures only ONE VLM call runs at a time (VRAM OOM guard).
  - Retries up to 2 times on the primary model before falling to the next model.
  - Timeout is per-attempt, not cumulative.
"""

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Union

import ollama
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PRIMARY_MODEL = "qwen2.5vl:7b"
DEFAULT_MODEL_FALLBACKS = [
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
    "moondream",
]
PRIMARY_MODEL = DEFAULT_PRIMARY_MODEL
FALLBACK_MODELS = DEFAULT_MODEL_FALLBACKS

DEFAULT_TIMEOUT_S = 120          # seconds per attempt
DEFAULT_MAX_RETRIES = 2          # retries on the *same* model before fallback
MAX_IMAGE_DIM = 768              # resize longest edge — reduces VRAM usage significantly
DEFAULT_NUM_CTX = 2048           # context window — 2048 saves ~1GB VRAM vs 4096

# Global lock — prevents parallel VLM calls (VRAM OOM protection)
_vlm_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(image: Union[str, Path, Image.Image, bytes]) -> str:
    """Return a base64-encoded JPEG string suitable for the Ollama API.

    Accepts a file path, a PIL Image, or raw bytes.
    Resizes large images so the VLM doesn't choke on huge GeoTIFFs.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB (drop alpha / palette issues)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize if any dimension exceeds MAX_IMAGE_DIM
    w, h = pil_img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        pil_img = pil_img.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_model(
    model: str,
    prompt: str,
    images_b64: Union[str, list],
    timeout: int,
    system_prompt: Optional[str] = None,
) -> str:
    """Single attempt to call an Ollama model. Raises on failure/timeout.

    images_b64 can be a single base64 string or a list of them.
    """
    if isinstance(images_b64, str):
        images_b64 = [images_b64]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": prompt,
        "images": images_b64,
    })

    # ollama.chat is synchronous; we enforce an external timeout via threading
    result_container = {"response": None, "error": None}

    def _run():
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={"num_ctx": DEFAULT_NUM_CTX},
            )
            result_container["response"] = resp["message"]["content"]
        except Exception as exc:
            result_container["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        raise TimeoutError(
            f"Ollama call to '{model}' timed out after {timeout}s"
        )
    if result_container["error"]:
        raise result_container["error"]

    response = result_container["response"]
    # Handle None and whitespace-only responses
    if response is None or (isinstance(response, str) and not response.strip()):
        raise RuntimeError(f"Empty response from '{model}'")

    return response.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_vlm(
    prompt: str,
    image: Union[str, Path, Image.Image, bytes],
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Send an image + text query to the VLM and return the answer.

    Parameters
    ----------
    prompt : str
        The user's question or instruction.
    image : str | Path | PIL.Image | bytes
        The input image (path, PIL object, or raw bytes).
    system_prompt : str, optional
        An optional system message prepended to the conversation.
    model : str, optional
        Force a specific model (skips the fallback chain).
    timeout : int
        Seconds to wait per attempt before declaring a timeout.
    max_retries : int
        Retries on the *same* model before moving to the fallback.

    Returns
    -------
    dict
        {"answer": str, "model_used": str, "elapsed_s": float}
    """
    # Build the ordered model list
    if model:
        models_to_try = [model]
    else:
        models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS

    image_b64 = _encode_image(image)

    # Acquire the global lock — only one VLM call at a time
    with _vlm_lock:
        last_error = None
        for mdl in models_to_try:
            for attempt in range(1, max_retries + 1):
                logger.info(
                    "VLM call: model=%s  attempt=%d/%d",
                    mdl, attempt, max_retries,
                )
                t0 = time.perf_counter()
                try:
                    answer = _call_model(
                        mdl, prompt, image_b64, timeout, system_prompt
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "VLM success: model=%s  elapsed=%.1fs", mdl, elapsed
                    )
                    return {
                        "answer": answer.strip(),
                        "model_used": mdl,
                        "elapsed_s": round(elapsed, 2),
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_error = exc
                    logger.warning(
                        "VLM attempt failed: model=%s  attempt=%d  "
                        "elapsed=%.1fs  error=%s",
                        mdl, attempt, elapsed, exc,
                    )

            logger.warning(
                "All %d attempts exhausted for model '%s', trying next fallback.",
                max_retries, mdl,
            )

        # All models failed
        raise RuntimeError(
            f"All VLM models failed. Last error: {last_error}"
        ) from last_error


def query_vlm_multi_image(
    prompt: str,
    images: list,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Send multiple images + a text query to the VLM.

    Same contract as query_vlm but accepts a list of images.
    Used by change detection (before/after/diff) and SAR fusion tools.

    Parameters
    ----------
    prompt : str
        The user's question or instruction.
    images : list[str | Path | PIL.Image | bytes]
        List of images to include in the query.
    system_prompt, model, timeout, max_retries : same as query_vlm.

    Returns
    -------
    dict
        {"answer": str, "model_used": str, "elapsed_s": float}
    """
    if not images:
        raise ValueError("At least one image is required")

    # Build ordered model list
    if model:
        models_to_try = [model]
    else:
        models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS

    images_b64 = [_encode_image(img) for img in images]

    with _vlm_lock:
        last_error = None
        for mdl in models_to_try:
            for attempt in range(1, max_retries + 1):
                logger.info(
                    "VLM multi-image call: model=%s  images=%d  attempt=%d/%d",
                    mdl, len(images_b64), attempt, max_retries,
                )
                t0 = time.perf_counter()
                try:
                    answer = _call_model(
                        mdl, prompt, images_b64, timeout, system_prompt
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "VLM success: model=%s  elapsed=%.1fs", mdl, elapsed
                    )
                    return {
                        "answer": answer.strip(),
                        "model_used": mdl,
                        "elapsed_s": round(elapsed, 2),
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_error = exc
                    logger.warning(
                        "VLM attempt failed: model=%s  attempt=%d  "
                        "elapsed=%.1fs  error=%s",
                        mdl, attempt, elapsed, exc,
                    )

            logger.warning(
                "All %d attempts exhausted for model '%s', trying next fallback.",
                max_retries, mdl,
            )

        raise RuntimeError(
            f"All VLM models failed. Last error: {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python ollama_client.py <image_path> [question]")
        print('Example: python ollama_client.py sample.jpg "What do you see?"')
        sys.exit(1)

    img_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."

    print(f"\n> Image : {img_path}")
    print(f"> Query : {question}\n")

    result = query_vlm(prompt=question, image=img_path)

    print(f"[OK] Model : {result['model_used']}")
    print(f"[OK] Time  : {result['elapsed_s']}s")
    print(f"[OK] Answer:\n{result['answer']}")
