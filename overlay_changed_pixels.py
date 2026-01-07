from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def delta_lsb_mask(cover: np.ndarray, stego: np.ndarray, mode: str = "any") -> np.ndarray:
    """
    Returns boolean mask (H,W) where LSB differs.
    mode:
      - "r"|"g"|"b" : mask for that channel only
      - "any"       : mask where any of R/G/B differs
      - "all"       : mask where all of R/G/B differ (rare)
    """
    c = cover & 1
    s = stego & 1
    d = (c != s)  # (H,W,3) bool

    mode = mode.lower()
    if mode == "r":
        return d[:, :, 0]
    if mode == "g":
        return d[:, :, 1]
    if mode == "b":
        return d[:, :, 2]
    if mode == "all":
        return d[:, :, 0] & d[:, :, 1] & d[:, :, 2]
    # default "any"
    return d[:, :, 0] | d[:, :, 1] | d[:, :, 2]

def apply_overlay(base_rgb: np.ndarray,
                  mask: np.ndarray,
                  color=(255, 0, 0),
                  alpha=0.6) -> np.ndarray:
    """
    base_rgb: (H,W,3) uint8
    mask: (H,W) bool
    color: overlay RGB
    alpha: blending factor for masked pixels
    """
    out = base_rgb.astype(np.float32).copy()
    overlay = np.array(color, dtype=np.float32)

    # Blend only masked pixels
    out[mask] = (1 - alpha) * out[mask] + alpha * overlay
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(description="Overlay changed LSB pixels on top of original/stego image.")
    ap.add_argument("--cover", required=True, help="Original cover image (PNG)")
    ap.add_argument("--stego", required=True, help="Stego image (PNG)")
    ap.add_argument("--base", default="cover", choices=["cover", "stego"],
                    help="Which image to use as background.")
    ap.add_argument("--mode", default="any", choices=["any", "r", "g", "b", "all"],
                    help="Which LSB changes to highlight.")
    ap.add_argument("--alpha", type=float, default=0.7, help="Overlay alpha [0..1].")
    ap.add_argument("--color", default="255,0,0", help="Overlay color as R,G,B (e.g. 255,0,0).")
    ap.add_argument("--save", default="overlay.png", help="Output image path.")
    ap.add_argument("--show", action="store_true", help="Show window.")
    args = ap.parse_args()

    cover = load_rgb(args.cover)
    stego = load_rgb(args.stego)

    if cover.shape != stego.shape:
        raise SystemExit(f"Size mismatch: {cover.shape} vs {stego.shape}")

    mask = delta_lsb_mask(cover, stego, args.mode)

    base = cover if args.base == "cover" else stego

    color = tuple(int(x) for x in args.color.split(","))
    out = apply_overlay(base, mask, color=color, alpha=args.alpha)

    Image.fromarray(out).save(args.save)

    # stats
    ratio = mask.mean()
    print(f"Changed LSB pixels (mode={args.mode}): {ratio:.6f} ({ratio*100:.4f}%)")
    print(f"Saved overlay: {args.save}")

    if args.show:
        plt.figure(figsize=(10, 6))
        plt.title(f"Overlay: changed LSB pixels (mode={args.mode}, {ratio*100:.3f}%)")
        plt.imshow(out)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
