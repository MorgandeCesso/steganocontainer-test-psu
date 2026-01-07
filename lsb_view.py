from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def lsb_plane(rgb: np.ndarray, mode: str) -> np.ndarray:
    r = rgb[:, :, 0] & 1
    g = rgb[:, :, 1] & 1
    b = rgb[:, :, 2] & 1

    mode = mode.lower()
    if mode in ("r", "red"):
        plane = r
    elif mode in ("g", "green"):
        plane = g
    elif mode in ("b", "blue"):
        plane = b
    elif mode in ("all", "combined", "rgb"):
        plane = r ^ g ^ b
    else:
        raise ValueError("mode must be one of: r, g, b, combined")

    return (plane * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image (cover.png or stego.png)")
    ap.add_argument("--mode", default="combined", help="r|g|b|combined")
    ap.add_argument("--save", default="lsb_view.png", help="Output image path")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rgb = load_rgb(args.image)
    lsb = lsb_plane(rgb, args.mode)

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"LSB-plane visualization ({args.mode.upper()})")

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(lsb, cmap="gray")
    ax2.set_title(f"LSB plane: {args.mode.upper()}")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
