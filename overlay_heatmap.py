from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def delta_any_mask(cover: np.ndarray, stego: np.ndarray) -> np.ndarray:
    c = cover & 1
    s = stego & 1
    d = (c != s)
    return d[:, :, 0] | d[:, :, 1] | d[:, :, 2]

def block_density(mask: np.ndarray, block: int) -> np.ndarray:
    H, W = mask.shape
    H2 = (H // block) * block
    W2 = (W // block) * block
    m = mask[:H2, :W2]
    # reshape to blocks and average
    m = m.reshape(H2 // block, block, W2 // block, block)
    dens = m.mean(axis=(1, 3))  # (Hb, Wb)
    return dens, H2, W2

def main():
    ap = argparse.ArgumentParser(description="Overlay block density heatmap of changed LSB pixels.")
    ap.add_argument("--cover", required=True)
    ap.add_argument("--stego", required=True)
    ap.add_argument("--base", default="cover", choices=["cover", "stego"])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--save", default="overlay_heatmap.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cover = load_rgb(args.cover)
    stego = load_rgb(args.stego)
    if cover.shape != stego.shape:
        raise SystemExit("Size mismatch")

    base = cover if args.base == "cover" else stego
    mask = delta_any_mask(cover, stego)

    dens, H2, W2 = block_density(mask, args.block)
    base2 = base[:H2, :W2]

    ratio = mask.mean()
    print(f"Overall changed pixels (any channel): {ratio:.6f} ({ratio*100:.4f}%)")
    print(f"Block size: {args.block}x{args.block}, heatmap shape: {dens.shape}")

    plt.figure(figsize=(10, 6))
    plt.title(f"Changed LSB density heatmap (block={args.block})")
    plt.imshow(base2)
    plt.imshow(dens, alpha=0.55, interpolation="nearest", extent=(0, W2, H2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    if args.show:
        plt.show()
    print(f"Saved heatmap overlay: {args.save}")

if __name__ == "__main__":
    main()
