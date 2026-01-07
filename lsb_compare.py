from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def lsb_planes(rgb: np.ndarray):
    r = rgb[:, :, 0] & 1
    g = rgb[:, :, 1] & 1
    b = rgb[:, :, 2] & 1
    combined = r ^ g ^ b  # combined view (xor)
    return r, g, b, combined

def to_img01(plane: np.ndarray) -> np.ndarray:
    # 0/1 -> 0/255 for visibility
    return (plane.astype(np.uint8) * 255)

def save_gray(arr_0_255: np.ndarray, path: str):
    Image.fromarray(arr_0_255.astype(np.uint8), mode="L").save(path)

def changed_ratio(delta01: np.ndarray) -> float:
    # delta01 is 0/1
    return float(delta01.mean())

def main():
    ap = argparse.ArgumentParser(description="Compare LSB planes of cover and stego images, save delta maps and stats.")
    ap.add_argument("--cover", required=True, help="Original cover PNG")
    ap.add_argument("--stego", required=True, help="Stego PNG")
    ap.add_argument("--outdir", default="lsb_compare_out", help="Output directory")
    ap.add_argument("--show", action="store_true", help="Show visualization window")
    ap.add_argument("--amp", type=int, default=1, help="Amplify delta visualization (1, 8, 16, 32, 64...)")
    args = ap.parse_args()

    cover = load_rgb(args.cover)
    stego = load_rgb(args.stego)

    if cover.shape != stego.shape:
        raise SystemExit(f"Size mismatch: cover {cover.shape} vs stego {stego.shape}")

    cR, cG, cB, cC = lsb_planes(cover)
    sR, sG, sB, sC = lsb_planes(stego)

    # XOR difference: 1 where LSB differs
    dR = cR ^ sR
    dG = cG ^ sG
    dB = cB ^ sB
    dC = cC ^ sC

    os.makedirs(args.outdir, exist_ok=True)
    base_c = os.path.splitext(os.path.basename(args.cover))[0]
    base_s = os.path.splitext(os.path.basename(args.stego))[0]
    tag = f"{base_c}_VS_{base_s}"

    # Save LSB planes
    save_gray(to_img01(cR), os.path.join(args.outdir, f"{tag}_cover_LSB_R.png"))
    save_gray(to_img01(cG), os.path.join(args.outdir, f"{tag}_cover_LSB_G.png"))
    save_gray(to_img01(cB), os.path.join(args.outdir, f"{tag}_cover_LSB_B.png"))
    save_gray(to_img01(cC), os.path.join(args.outdir, f"{tag}_cover_LSB_COMBINED.png"))

    save_gray(to_img01(sR), os.path.join(args.outdir, f"{tag}_stego_LSB_R.png"))
    save_gray(to_img01(sG), os.path.join(args.outdir, f"{tag}_stego_LSB_G.png"))
    save_gray(to_img01(sB), os.path.join(args.outdir, f"{tag}_stego_LSB_B.png"))
    save_gray(to_img01(sC), os.path.join(args.outdir, f"{tag}_stego_LSB_COMBINED.png"))

    # Save delta maps (where bits changed)
    save_gray(to_img01(dR), os.path.join(args.outdir, f"{tag}_DELTA_LSB_R.png"))
    save_gray(to_img01(dG), os.path.join(args.outdir, f"{tag}_DELTA_LSB_G.png"))
    save_gray(to_img01(dB), os.path.join(args.outdir, f"{tag}_DELTA_LSB_B.png"))
    save_gray(to_img01(dC), os.path.join(args.outdir, f"{tag}_DELTA_LSB_COMBINED.png"))

    # Amplified delta (visual trick): show changes brighter (still 0/255 but we can “thicken” via simple scaling)
    # Note: scaling doesn't change 0/255; for a stronger visual, we can blur or dilate, but keep it simple & pure.
    # We'll provide a contrast trick: convert delta to 0/255 then apply a gamma-like mapping via power.
    # Here: map 0->0, 255->255 always; to "see more", best is to show delta itself + % stats (below).
    # So "amp" is used in the preview only: repeat pixels (nearest neighbor upscale).
    amp = max(1, int(args.amp))

    # Stats
    r_ratio = changed_ratio(dR)
    g_ratio = changed_ratio(dG)
    b_ratio = changed_ratio(dB)
    c_ratio = changed_ratio(dC)
    overall = (r_ratio + g_ratio + b_ratio) / 3.0

    print("LSB changed ratio (fraction of pixels with changed LSB):")
    print(f"  R: {r_ratio:.6f} ({r_ratio*100:.4f}%)")
    print(f"  G: {g_ratio:.6f} ({g_ratio*100:.4f}%)")
    print(f"  B: {b_ratio:.6f} ({b_ratio*100:.4f}%)")
    print(f"  Combined (R^G^B): {c_ratio:.6f} ({c_ratio*100:.4f}%)")
    print(f"  Overall avg (R,G,B): {overall:.6f} ({overall*100:.4f}%)")
    print(f"Saved images to: {args.outdir}")

    if args.show:
        def upscale(img_0_255: np.ndarray, factor: int) -> np.ndarray:
            if factor == 1:
                return img_0_255
            return np.repeat(np.repeat(img_0_255, factor, axis=0), factor, axis=1)

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Delta LSB maps (white = changed bit)", fontsize=12)

        axs = [
            fig.add_subplot(2, 2, 1),
            fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3),
            fig.add_subplot(2, 2, 4),
        ]
        imgs = [
            upscale(to_img01(dR), amp),
            upscale(to_img01(dG), amp),
            upscale(to_img01(dB), amp),
            upscale(to_img01(dC), amp),
        ]
        titles = [
            f"DELTA R (changed {r_ratio*100:.3f}%)",
            f"DELTA G (changed {g_ratio*100:.3f}%)",
            f"DELTA B (changed {b_ratio*100:.3f}%)",
            f"DELTA Combined (changed {c_ratio*100:.3f}%)",
        ]

        for ax, im, t in zip(axs, imgs, titles):
            ax.imshow(im, cmap="gray", interpolation="nearest")
            ax.set_title(t)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
