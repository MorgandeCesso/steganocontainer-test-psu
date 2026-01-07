from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_rgb(p): return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

def lsb(rgb, ch):
    idx = {"r":0, "g":1, "b":2}[ch]
    return (rgb[:, :, idx] & 1).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True)
    ap.add_argument("--stego", required=True)
    ap.add_argument("--ch", default="r", help="r|g|b")
    ap.add_argument("--save", default="delta_lsb_view.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    c = load_rgb(args.cover)
    s = load_rgb(args.stego)
    if c.shape != s.shape:
        raise SystemExit("Cover and stego sizes must match")

    delta = (lsb(c, args.ch) ^ lsb(s, args.ch)) * 255

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Delta LSB map (white = changed bit) [{args.ch.upper()}]")

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(s)
    ax1.set_title("Stego image (normal view)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(delta, cmap="gray")
    ax2.set_title(f"DELTA LSB: {args.ch.upper()}")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
