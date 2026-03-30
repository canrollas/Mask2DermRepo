"""
Optics-inspired dermoscopy standardization pipeline.

Implements the physics-based simulation described in Mask2Derm §2.1:
  - Circular field-of-view with feathered aperture mask M(d)
  - Radial vignetting falloff V(d)
  - Barrel lens distortion
  - Final: I'(x,y) = I(x_d, y_d) · M(d) · V(d)
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core optical effect functions
# ---------------------------------------------------------------------------

def _distance_map(h: int, w: int) -> tuple[np.ndarray, float, float]:
    """Return per-pixel radial distance from image centre and the centre coords."""
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return d, cx, cy


def apply_circular_mask(
    img: np.ndarray,
    R: float | None = None,
    delta: float | None = None,
) -> np.ndarray:
    """Apply a circular aperture with smooth feathered edges.

    M(d) = 1          if d <= R
           smooth      if R < d <= R + delta   (cosine blend to 0)
           0           if d > R + delta

    Args:
        img:   H×W×3 uint8 image.
        R:     Aperture radius in pixels. Defaults to 0.47 × min(H, W).
        delta: Feathering width in pixels. Defaults to 0.03 × min(H, W).

    Returns:
        Image with corners zeroed out (black outside the circular aperture).
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if R is None:
        R = 0.47 * min_dim
    if delta is None:
        delta = 0.03 * min_dim

    d, _, _ = _distance_map(h, w)

    mask = np.ones((h, w), dtype=np.float32)
    # Feathered transition zone
    in_transition = (d > R) & (d <= R + delta)
    t = (d[in_transition] - R) / delta          # 0 → 1 across transition
    mask[in_transition] = 0.5 * (1.0 + np.cos(math.pi * t))
    mask[d > R + delta] = 0.0

    result = img.astype(np.float32)
    result *= mask[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignetting(
    img: np.ndarray,
    R: float | None = None,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> np.ndarray:
    """Apply radial vignetting falloff.

    V(d) = (1 - alpha) + alpha * (1 - (d/R)^2)^gamma

    Args:
        img:   H×W×3 uint8 image.
        R:     Reference radius (pixels). Defaults to 0.47 × min(H, W).
        alpha: Vignetting strength (0 = no effect, 1 = full).
        gamma: Falloff rate (higher → sharper edge darkening).

    Returns:
        Vignette-corrected image.
    """
    h, w = img.shape[:2]
    if R is None:
        R = 0.47 * min(h, w)

    d, _, _ = _distance_map(h, w)
    ratio = np.clip(d / R, 0.0, 1.0)
    V = (1.0 - alpha) + alpha * (1.0 - ratio ** 2) ** gamma
    V = np.clip(V, 0.0, 1.0).astype(np.float32)

    result = img.astype(np.float32) * V[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_barrel_distortion(
    img: np.ndarray,
    k1: float = -0.3,
) -> np.ndarray:
    """Apply radial barrel distortion.

    Computes distorted coordinates:
        X = (x - cx) / fx,  Y = (y - cy) / fy,  r² = X² + Y²
        x_d = fx * X * (1 + k1 * r²) + cx
        y_d = fy * Y * (1 + k1 * r²) + cy

    with fx = fy = max(H, W) as a normalisation focal length.

    Args:
        img: H×W×3 uint8 image.
        k1:  Radial distortion coefficient. Negative → barrel effect.

    Returns:
        Barrel-distorted image (same size, remapped with bilinear interpolation).
    """
    h, w = img.shape[:2]
    f = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    # Build source coordinate maps
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    X = (xs - cx) / f
    Y = (ys - cy) / f
    r2 = X ** 2 + Y ** 2

    factor = 1.0 + k1 * r2
    x_src = (f * X * factor + cx).astype(np.float32)
    y_src = (f * Y * factor + cy).astype(np.float32)

    return cv2.remap(img, x_src, y_src, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# Full standardization pipeline
# ---------------------------------------------------------------------------

def standardize_image(
    img: np.ndarray,
    size: int = 256,
    apply_barrel: bool = True,
    apply_vignette: bool = True,
    apply_aperture: bool = True,
    k1: float = -0.3,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> np.ndarray:
    """Full optics-inspired standardization pipeline.

    Steps:
        1. Resize to (size × size)
        2. Barrel distortion       (if apply_barrel)
        3. Vignetting              (if apply_vignette)
        4. Circular aperture mask  (if apply_aperture)

    Args:
        img:             H×W×3 uint8 BGR or RGB image (cv2 or PIL-sourced).
        size:            Target square resolution.
        apply_barrel:    Apply lens barrel distortion.
        apply_vignette:  Apply radial vignetting.
        apply_aperture:  Apply circular FOV crop.
        k1:              Barrel distortion coefficient.
        alpha:           Vignetting strength.
        gamma:           Vignetting falloff exponent.

    Returns:
        Standardized H×W×3 uint8 image.
    """
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if apply_barrel:
        out = apply_barrel_distortion(out, k1=k1)
    if apply_vignette:
        out = apply_vignetting(out, alpha=alpha, gamma=gamma)
    if apply_aperture:
        out = apply_circular_mask(out)
    return out


def standardize_pil(pil_img: Image.Image, size: int = 256, **kwargs) -> Image.Image:
    """Convenience wrapper: PIL Image → standardize → PIL Image (RGB)."""
    arr = np.array(pil_img.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    out_bgr = standardize_image(arr_bgr, size=size, **kwargs)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo(input_path: str, output_path: str, size: int = 256) -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {input_path}")

    stages = {
        "original": cv2.resize(img, (size, size)),
        "barrel": apply_barrel_distortion(cv2.resize(img, (size, size))),
        "vignette": apply_vignetting(cv2.resize(img, (size, size))),
        "full": standardize_image(img, size=size),
    }

    row = np.hstack(list(stages.values()))
    labels = list(stages.keys())

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        cv2.putText(row, label, (i * size + 5, 20), font, 0.6, (255, 255, 255), 1)

    cv2.imwrite(output_path, row)
    print(f"Demo saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optics pipeline demo")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="demo_optics.png", help="Output comparison image")
    parser.add_argument("--size", type=int, default=256, help="Target resolution")
    args = parser.parse_args()
    _demo(args.input, args.output, args.size)
