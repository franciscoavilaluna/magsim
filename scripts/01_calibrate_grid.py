import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HOUGH_DP = 1.2
HOUGH_MIN_DIST = 55
HOUGH_PARAM1 = 60
HOUGH_PARAM2 = 28
HOUGH_MIN_R = 28
HOUGH_MAX_R = 55
MAGNET_MASK_W = 280
MAGNET_MASK_H = 60
EXPECTED_COMPASSES = 80

def mask_magnet_region(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    masked = gray.copy()
    y0 = max(0, cy - MAGNET_MASK_H)
    y1 = min(h, cy + MAGNET_MASK_H)
    x0 = max(0, cx - MAGNET_MASK_W)
    x1 = min(w, cx + MAGNET_MASK_W)
    masked[y0:y1, x0:x1] = 0
    return masked


def detect_compasses(image_path: Path) -> list[dict]:
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"[ERROR] No se pudo leer: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_masked = mask_magnet_region(gray)
    blurred = cv2.GaussianBlur(gray_masked, (7, 7), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R,
        maxRadius=HOUGH_MAX_R,
    )

    if circles is None:
        sys.exit("[ERROR] No se detectaron círculos. Ajusta HOUGH_PARAM2 (bájalo).")

    circles = np.round(circles[0]).astype(int)
    print(f"[INFO] Detectados {len(circles)} círculos (esperados ~{EXPECTED_COMPASSES})")

    if abs(len(circles) - EXPECTED_COMPASSES) > 10:
        print("[WARN] La cantidad difiere bastante. Revisa los parámetros Hough.")

    compasses = []
    for i, (cx, cy, r) in enumerate(circles):
        compasses.append({"id": i, "cx": int(cx), "cy": int(cy), "r": int(r)})

    compasses.sort(key=lambda c: (c["cy"] // 60, c["cx"]))
    for i, c in enumerate(compasses):
        c["id"] = i

    return compasses, img


def visualize(img: np.ndarray, compasses: list[dict], out_path: Path):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(rgb)

    for c in compasses:
        circle = mpatches.Circle(
            (c["cx"], c["cy"]), c["r"],
            linewidth=1.5, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(circle)
        ax.text(
            c["cx"], c["cy"], str(c["id"]),
            color="yellow", fontsize=5, ha="center", va="center", fontweight="bold"
        )

    ax.set_title(
        f"Grid calibrado — {len(compasses)} brújulas detectadas\n"
        "Círculo verde = brújula detectada | número = ID",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"[OK] Imagen de validación guardada: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Calibrar grid de brújulas")
    parser.add_argument("--image", required=True, help="Ruta a una foto de ejemplo")
    parser.add_argument("--output-dir", default="../output", help="Directorio de salida")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Procesando: {image_path.name}")
    compasses, img = detect_compasses(image_path)

    grid_path = output_dir / "grid.json"
    meta = {
        "source_image": str(image_path),
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "n_compasses": len(compasses),
        "compasses": compasses
    }
    with open(grid_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Grid guardado: {grid_path}")

    vis_path = output_dir / "grid_validation.png"
    visualize(img, compasses, vis_path)


if __name__ == "__main__":
    main()
