"""
Generate a small synthetic GeoTIFF for testing.
This avoids needing real satellite data to verify the pipeline works.
Creates a 256×256 3-band image with a simple pattern and fake EPSG:4326 georef.
"""

import numpy as np
from pathlib import Path


def create_synthetic_geotiff(
    output_path: str = "data/sample_rgb.tif",
    width: int = 256,
    height: int = 256,
    bands: int = 3,
) -> str:
    """Create a synthetic GeoTIFF with geometric patterns and georeferencing."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate a visually interesting pattern (not just noise)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    # Band 1 (Red): radial gradient from center
    cx, cy = width / 2, height / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    band1 = (255 * (1 - r / r.max())).astype(np.uint8)

    # Band 2 (Green): diagonal stripes
    band2 = ((128 + 127 * np.sin((x + y) * 0.1))).astype(np.uint8)

    # Band 3 (Blue): checkerboard
    band3 = (((x // 32 + y // 32) % 2) * 200 + 55).astype(np.uint8)

    data = np.stack([band1, band2, band3])  # (3, H, W)

    # Fake georeferencing: somewhere over central India (for ISRO context)
    transform = from_bounds(
        77.5, 12.9, 77.7, 13.1, width, height,
    )

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.update_tags(
            DESCRIPTION="Synthetic test GeoTIFF for SatQuery AI",
            SOURCE="generate_sample.py",
        )

    print(f"[OK] Created synthetic GeoTIFF: {output_path}")
    print(f"  Size: {width}x{height}, {bands} bands, EPSG:4326")
    return str(output_path)


def create_synthetic_sar(
    output_path: str = "data/sample_sar.tif",
    width: int = 256,
    height: int = 256,
) -> str:
    """Create a synthetic single-band SAR-like GeoTIFF (float32, skewed values)."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Simulate SAR backscatter: exponential-ish distribution
    np.random.seed(42)
    base = np.random.exponential(scale=0.05, size=(height, width)).astype(np.float32)

    # Add some structure (simulated buildings / bright targets)
    base[80:120, 80:120] += 0.8   # bright square
    base[160:180, 50:70] += 0.5   # another feature

    data = base[np.newaxis, :, :]  # (1, H, W)

    transform = from_bounds(
        77.5, 12.9, 77.7, 13.1, width, height,
    )

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)

    print(f"[OK] Created synthetic SAR GeoTIFF: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    create_synthetic_geotiff()
    create_synthetic_sar()
    print("\nSample data ready in data/")
