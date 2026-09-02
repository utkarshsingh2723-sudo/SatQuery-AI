"""
SatQuery AI -- VQA Specialist Tool
====================================
Takes a remote-sensing image + natural-language question, returns a
concise text answer by calling the Ollama VLM with RS-tuned prompts.

This is a *specialist tool* invoked by the router, not a general chatbot.
It enforces short, factual answers appropriate for VQA benchmarks
(RSVQA-LR, VRSBench).
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

from PIL import Image

# Project imports
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ollama_client import query_vlm
from tools.geotiff_utils import load_geotiff, LoadedImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VQA_SYSTEM_PROMPT = (
    "You are a remote sensing image analyst. "
    "Answer questions about satellite and aerial images. "
    "Keep answers very short: one word, a number, or a brief phrase. "
    "For yes/no questions, answer only 'yes' or 'no'. "
    "For counting questions, answer with just the number."
)

# Simpler prompt that works across model sizes (moondream struggles with
# the structured "Question: ...\nAnswer:" format)
VQA_PROMPT_TEMPLATE = (
    "Look at this satellite image and answer briefly. {question}"
)

# ---------------------------------------------------------------------------
# Answer post-processing
# ---------------------------------------------------------------------------

def _clean_answer(raw: str) -> str:
    """Clean up VLM output to extract a concise VQA answer.

    Handles common VLM quirks: repeating the question, adding
    explanations after the answer, markdown formatting, etc.
    """
    text = raw.strip()

    # Remove markdown bold/italic
    text = re.sub(r'[*_]{1,3}', '', text)

    # If the model repeated the question, take only what comes after "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # If there are multiple lines, take the first non-empty one
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if lines:
        text = lines[0]

    # Remove trailing periods and common filler
    text = text.rstrip('.')
    text = re.sub(r'^(The answer is|It is|I see|This is)\s+', '', text, flags=re.IGNORECASE)

    # Remove leading bullet / numbering
    text = re.sub(r'^[-\d.)\s]+', '', text).strip()

    # Collapse whitespace
    text = ' '.join(text.split())

    return text if text else "unknown"


def _is_garbled(answer: str) -> bool:
    """Detect if the model output is garbled / nonsensical."""
    if not answer or answer == "unknown":
        return False  # empty is handled, not garbled

    # Too long for a VQA answer
    if len(answer) > 200:
        return True

    # Mostly non-alphanumeric characters
    alnum = sum(1 for c in answer if c.isalnum() or c.isspace())
    if len(answer) > 5 and alnum / len(answer) < 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.webp'}

def _validate_image_path(path: Path) -> None:
    """Raise ValueError if the image format is not supported."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_vqa(
    image: Union[str, Path, Image.Image, LoadedImage],
    question: str,
    *,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
) -> dict:
    """Ask a visual question about a remote sensing image.

    Parameters
    ----------
    image : str | Path | PIL.Image | LoadedImage
        The input image. Can be a file path (GeoTIFF or standard image),
        a PIL Image object, or a LoadedImage from geotiff_utils.
    question : str
        The natural-language question to ask about the image.
    model : str, optional
        Force a specific Ollama model (skips fallback chain).
    timeout : int
        Seconds per VLM attempt.
    max_retries : int
        Retries per model before fallback.

    Returns
    -------
    dict
        {
            "answer": str,        # cleaned concise answer
            "raw_answer": str,    # unprocessed VLM output
            "model_used": str,    # which model actually responded
            "elapsed_s": float,   # time taken
            "status": str,        # "ok", "garbled", "error"
            "error": str | None,  # error message if status != "ok"
        }
    """
    # --- Validate question ---
    question = (question or "").strip()
    if not question:
        return {
            "answer": "",
            "raw_answer": "",
            "model_used": "",
            "elapsed_s": 0.0,
            "status": "error",
            "error": "Empty question provided",
        }

    # --- Resolve image to PIL ---
    pil_image = None
    try:
        if isinstance(image, LoadedImage):
            pil_image = image.pil_image
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, (str, Path)):
            path = Path(image)
            _validate_image_path(path)

            if path.suffix.lower() in {'.tif', '.tiff'}:
                loaded = load_geotiff(path)
                pil_image = loaded.pil_image
            else:
                pil_image = Image.open(path).convert("RGB")
        else:
            return {
                "answer": "",
                "raw_answer": "",
                "model_used": "",
                "elapsed_s": 0.0,
                "status": "error",
                "error": f"Unsupported image type: {type(image).__name__}",
            }
    except (FileNotFoundError, ValueError) as exc:
        return {
            "answer": "",
            "raw_answer": "",
            "model_used": "",
            "elapsed_s": 0.0,
            "status": "error",
            "error": str(exc),
        }

    # --- Call VLM ---
    prompt = VQA_PROMPT_TEMPLATE.format(question=question)

    try:
        result = query_vlm(
            prompt=prompt,
            image=pil_image,
            system_prompt=VQA_SYSTEM_PROMPT,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as exc:
        logger.error("VQA call failed: %s", exc)
        return {
            "answer": "",
            "raw_answer": "",
            "model_used": "",
            "elapsed_s": 0.0,
            "status": "error",
            "error": f"VLM call failed: {exc}",
        }

    raw_answer = result["answer"]
    clean = _clean_answer(raw_answer)

    # --- Check for garbled output ---
    status = "ok"
    error = None
    if _is_garbled(clean):
        status = "garbled"
        error = f"Model output appears garbled (length={len(clean)})"
        clean = "unknown"
        logger.warning("Garbled VQA output: '%s...'", raw_answer[:100])

    logger.info(
        "VQA: q='%s'  a='%s'  model=%s  time=%.1fs  status=%s",
        question[:50], clean, result["model_used"], result["elapsed_s"], status,
    )

    return {
        "answer": clean,
        "raw_answer": raw_answer,
        "model_used": result["model_used"],
        "elapsed_s": result["elapsed_s"],
        "status": status,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(_sys.argv) < 3:
        print("Usage: python vqa_tool.py <image_path> <question>")
        print('Example: python vqa_tool.py sample.tif "How many buildings are visible?"')
        _sys.exit(1)

    img_path = _sys.argv[1]
    q = _sys.argv[2]

    print(f"\n> Image   : {img_path}")
    print(f"> Question: {q}\n")

    res = ask_vqa(img_path, q)

    print(f"  Status : {res['status']}")
    print(f"  Model  : {res['model_used']}")
    print(f"  Time   : {res['elapsed_s']}s")
    print(f"  Answer : {res['answer']}")
    if res['raw_answer'] != res['answer']:
        print(f"  Raw    : {res['raw_answer'][:150]}")
    if res['error']:
        print(f"  Error  : {res['error']}")
