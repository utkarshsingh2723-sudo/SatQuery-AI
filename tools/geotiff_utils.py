"""
SatQuery AI — GeoTIFF / TIFF Utilities
========================================
Loads GeoTIFF and plain TIFF files, converts them to RGB arrays / PIL
Images that the VLM can consume, and preserves georeferencing metadata
for downstream tools (change detection, SAR fusion, overlay rendering).

Design decisions:
  - Uses rasterio for GeoTIFF (CRS, transform, nodata, etc.).
  - Falls back to PIL/tifffile for plain TIFFs without geo metadata.
  - Multispectral images (>3 bands): selects bands 1-2-3 as RGB by
    default, but callers can override.  Single-band images are
    converted to a pseudocolor (grayscale) RGB for VLM consumption.
  - SAR images (typically single-band float32): applies log-scaling +
    percentile stretch so the VLM gets a visually meaningful image.
  - Normalises pixel values to uint8 [0-255] for all outputs.
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeoMetadata:
    """Lightweight container for georeferencing info extracted from a GeoTIFF."""
    crs: Optional[str] = None              # e.g. "EPSG:4326"
    transform: Optional[tuple] = None      # affine transform (6 floats)
    bounds: Optional[tuple] = None         # (left, bottom, right, top)
    width: int = 0
    height: int = 0
    band_count: int = 0
    dtype: str = ""
    nodata: Optional[float] = None
    filepath: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def has_georef(self) -> bool:
        return self.crs is not None and self.transform is not None


@dataclass
class LoadedImage:
    """Bundle returned by load_geotiff: RGB array, PIL image, raw bands, metadata."""
    rgb_array: np.ndarray          # (H, W, 3) uint8
    pil_image: Image.Image         # RGB PIL Image, ready for the VLM
    raw_bands: np.ndarray          # (bands, H, W) original dtype
    metadata: GeoMetadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentile_stretch(
    arr: np.ndarray, lo: float = 2.0, hi: float = 98.0
) -> np.ndarray:
    """Stretch a float/int array to uint8 [0-255] using percentile clipping."""
    vmin = np.nanpercentile(arr, lo)
    vmax = np.nanpercentile(arr, hi)
    if vmax - vmin < 1e-6:
        # Constant image — return mid-gray
        return np.full(arr.shape, 128, dtype=np.uint8)
    stretched = (arr.astype(np.float64) - vmin) / (vmax - vmin)
    stretched = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    return stretched


def _is_sar_likely(band_data: np.ndarray, meta: GeoMetadata) -> bool:
    """Heuristic: if single-band float32 with very skewed values, likely SAR."""
    if meta.band_count != 1:
        return False
    if band_data.dtype not in (np.float32, np.float64):
        return False
    # SAR backscatter typically has a high dynamic range with many near-zero values
    finite = band_data[np.isfinite(band_data)]
    if finite.size == 0:
        return False
    skew = np.mean(finite) / (np.median(finite) + 1e-12)
    return skew > 3.0


def _sar_to_rgb(band: np.ndarray) -> np.ndarray:
    """Convert a single SAR band to a viewable RGB via dB log-scaling + stretch."""
    # Avoid log(0)
    eps = np.finfo(np.float32).eps
    safe = np.where(np.isfinite(band) & (band > 0), band, eps)
    db = 10.0 * np.log10(safe)
    stretched = _percentile_stretch(db, lo=1, hi=99)
    return np.stack([stretched, stretched, stretched], axis=-1)


def _bands_to_rgb(
    raw: np.ndarray,
    band_indices: Tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """Convert selected bands from (bands, H, W) to (H, W, 3) uint8."""
    if raw.shape[0] == 1:
        # Single band -> grayscale -> triplicate
        band = raw[0]
        stretched = _percentile_stretch(band)
        return np.stack([stretched, stretched, stretched], axis=-1)

    selected = []
    for idx in band_indices:
        if idx < raw.shape[0]:
            selected.append(_percentile_stretch(raw[idx]))
        else:
            # Pad with zeros if band index is out of range
            selected.append(np.zeros((raw.shape[1], raw.shape[2]), dtype=np.uint8))
    return np.stack(selected, axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_geotiff(
    filepath: Union[str, Path],
    rgb_bands: Tuple[int, ...] = (0, 1, 2),
    force_sar: bool = False,
) -> LoadedImage:
    """Load a GeoTIFF / TIFF file and return an RGB-ready image + metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to the .tif / .tiff file.
    rgb_bands : tuple of int
        0-indexed band indices to map to R, G, B. Default (0, 1, 2).
    force_sar : bool
        If True, apply SAR dB log-scaling regardless of heuristic detection.

    Returns
    -------
    LoadedImage
        Contains .rgb_array (H,W,3 uint8), .pil_image, .raw_bands, .metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")

    meta = GeoMetadata(filepath=str(filepath))

    # --- Try rasterio first (handles GeoTIFF properly) ---
    try:
        import rasterio

        with rasterio.open(filepath) as src:
            raw = src.read()  # (bands, H, W)
            meta.crs = str(src.crs) if src.crs else None
            meta.transform = tuple(src.transform)[:6] if src.transform else None
            meta.bounds = tuple(src.bounds) if src.bounds else None
            meta.width = src.width
            meta.height = src.height
            meta.band_count = src.count
            meta.dtype = str(src.dtypes[0])
            meta.nodata = src.nodata
            meta.tags = dict(src.tags())

        logger.info(
            "Loaded via rasterio: %s  shape=%s  crs=%s  dtype=%s",
            filepath.name, raw.shape, meta.crs, meta.dtype,
        )

    except Exception as rio_err:
        # Fallback to PIL for plain TIFFs
        logger.warning(
            "rasterio failed (%s), falling back to PIL for %s",
            rio_err, filepath.name,
        )
        pil_img = Image.open(filepath)
        arr = np.array(pil_img)

        if arr.ndim == 2:
            raw = arr[np.newaxis, :, :]   # (1, H, W)
        elif arr.ndim == 3:
            raw = arr.transpose(2, 0, 1)  # (C, H, W)
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

        meta.width = raw.shape[2]
        meta.height = raw.shape[1]
        meta.band_count = raw.shape[0]
        meta.dtype = str(raw.dtype)

    # --- Handle nodata -> NaN for float types ---
    if meta.nodata is not None and np.issubdtype(raw.dtype, np.floating):
        raw = np.where(raw == meta.nodata, np.nan, raw)

    # --- Convert to RGB ---
    is_sar = force_sar or _is_sar_likely(raw, meta)
    if is_sar:
        logger.info("SAR image detected — applying dB log-scaling")
        rgb = _sar_to_rgb(raw[0])
    else:
        rgb = _bands_to_rgb(raw, rgb_bands)

    pil_image = Image.fromarray(rgb, mode="RGB")

    return LoadedImage(
        rgb_array=rgb,
        pil_image=pil_image,
        raw_bands=raw,
        metadata=meta,
    )


def save_rgb_preview(
    loaded: LoadedImage,
    output_path: Union[str, Path],
    quality: int = 90,
) -> Path:
    """Save the RGB preview as a JPEG for quick inspection."""
    output_path = Path(output_path)
    loaded.pil_image.save(output_path, format="JPEG", quality=quality)
    logger.info("Saved RGB preview: %s", output_path)
    return output_path


def get_image_bytes(loaded: LoadedImage, fmt: str = "JPEG") -> bytes:
    """Return the RGB image as in-memory bytes (useful for API calls)."""
    buf = io.BytesIO()
    loaded.pil_image.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python geotiff_utils.py <path_to_tiff>")
        sys.exit(1)

    path = sys.argv[1]
    result = load_geotiff(path)

    print(f"\n[OK] Loaded       : {result.metadata.filepath}")
    print(f"  Dimensions   : {result.metadata.width} x {result.metadata.height}")
    print(f"  Bands        : {result.metadata.band_count}")
    print(f"  CRS          : {result.metadata.crs or 'None (plain TIFF)'}")
    print(f"  Bounds       : {result.metadata.bounds}")
    print(f"  Dtype        : {result.metadata.dtype}")
    print(f"  RGB shape    : {result.rgb_array.shape}")
    print(f"  PIL size     : {result.pil_image.size}")

    preview_path = Path(path).with_suffix(".preview.jpg")
    save_rgb_preview(result, preview_path)
    print(f"  Preview saved: {preview_path}")
