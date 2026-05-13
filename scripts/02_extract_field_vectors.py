import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


RED_HSV_LOWER1 = np.array([0,   100, 60])
RED_HSV_UPPER1 = np.array([12,  255, 255])
RED_HSV_LOWER2 = np.array([163, 100, 60])
RED_HSV_UPPER2 = np.array([180, 255, 255])
COMPASS_ROI_MARGIN = 0.82

MAGNET_W_CM = 15.3
MAGNET_H_CM =  1.3
MAGNET_H_PX = 46
SCALE_TOL = 0.30
BAND_MARGIN_PX = 130
COMPASS_EXCL_SCALE = 1.25
BG_DIFF_THRESHOLD = 18
BG_DIFF_MIN = 10
MORPH_CONNECT_W = 25
MORPH_CONNECT_H = 5

def to_python(obj):
    if isinstance(obj, dict):    return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):    return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


def extract_red_mask(roi: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1)
    m2   = cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
    mask = cv2.bitwise_or(m1, m2)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def angle_from_mask(mask: np.ndarray, cx: int, cy: int):
    px = np.column_stack(np.where(mask > 0))
    if len(px) < 8:
        return None
    dy = float(px[:, 0].mean()) - cy
    dx = float(px[:, 1].mean()) - cx
    angle_deg = (math.degrees(math.atan2(dy, dx)) - 90.0) % 360.0
    return round(angle_deg, 2)


def extract_vectors(img: np.ndarray, grid: dict) -> list[dict]:
    h, w = img.shape[:2]
    sx, sy = w / grid["image_width"], h / grid["image_height"]
    out = []
    for c in grid["compasses"]:
        cx    = int(c["cx"] * sx)
        cy    = int(c["cy"] * sy)
        r     = int(c["r"]  * (sx + sy) / 2)
        roi_r = int(r * COMPASS_ROI_MARGIN)
        x0    = max(0, cx - roi_r);  y0 = max(0, cy - roi_r)
        x1    = min(w, cx + roi_r);  y1 = min(h, cy + roi_r)
        roi   = img[y0:y1, x0:x1]
        if roi.size == 0:
            out.append({"compass_id": int(c["id"]), "cx": cx, "cy": cy,
                        "angle_deg": None})
            continue
        mask  = extract_red_mask(roi)
        angle = angle_from_mask(mask, roi_r, roi_r)
        out.append({"compass_id": int(c["id"]), "cx": cx, "cy": cy,
                    "angle_deg": angle})
    return out


def _compass_excl_mask(grid: dict, img_h: int, img_w: int) -> np.ndarray:
    sx, sy = img_w / grid["image_width"], img_h / grid["image_height"]
    m = np.ones((img_h, img_w), np.uint8)
    for c in grid["compasses"]:
        cv2.circle(m,
            (int(c["cx"] * sx), int(c["cy"] * sy)),
            int(c["r"] * (sx + sy) / 2 * COMPASS_EXCL_SCALE), 0, -1)
    return m


def _compute_band(grid: dict, img_h: int, img_w: int) -> tuple[int, int]:
    sy  = img_h / grid["image_height"]
    ys  = sorted(c["cy"] * sy for c in grid["compasses"])
    top, bot, gap = img_h // 3, 2 * img_h // 3, 0
    for i in range(1, len(ys)):
        g = ys[i] - ys[i - 1]
        if g > gap:
            gap, top, bot = g, int(ys[i - 1]), int(ys[i])
    return max(0, top - BAND_MARGIN_PX), min(img_h, bot + BAND_MARGIN_PX)


def _estimate_background(gray_band: np.ndarray, excl_band: np.ndarray) -> float:
    px = gray_band[excl_band > 0]
    return float(np.median(px)) if len(px) > 0 else 128.0


def _best_rect(binary: np.ndarray, exp_w: float, exp_h: float,
               tol: float, y_offset: int, method: str) -> dict | None:
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score, best = -1.0, None
    for cnt in cnts:
        if cv2.contourArea(cnt) < 200:
            continue
        rect = cv2.minAreaRect(cnt)
        (rx, ry), (rw, rh), angle = rect
        if rh > rw:
            rw, rh = rh, rw
            angle  = (angle + 90) % 180
        w_ok = exp_w * (1 - tol) <= rw <= exp_w * (1 + tol)
        h_ok = exp_h * (1 - tol) <= rh <= exp_h * (1 + tol)
        if not (w_ok or h_ok):
            continue
        if not w_ok and not (exp_w*(1-tol*2) <= rw <= exp_w*(1+tol*2)):
            continue
        if not h_ok and not (exp_h*(1-tol*2) <= rh <= exp_h*(1+tol*2)):
            continue
        w_err = abs(rw - exp_w) / exp_w
        h_err = abs(rh - exp_h) / exp_h
        score = max(0.0, 1.0 - (w_err + h_err) / 2.0)
        if score > best_score:
            best_score = score
            box = np.int32(cv2.boxPoints(rect))
            box[:, 1] += y_offset
            ang_r  = math.radians(angle)
            hw     = rw / 2
            abs_cy = float(ry) + y_offset
            best   = {
                "cx": float(rx), "cy": abs_cy,
                "x1": int(rx - hw*math.cos(ang_r)),
                "y1": int(abs_cy - hw*math.sin(ang_r)),
                "x2": int(rx + hw*math.cos(ang_r)),
                "y2": int(abs_cy + hw*math.sin(ang_r)),
                "width_px":  float(rw), "height_px": float(rh),
                "angle_deg": float(angle), "score": float(score),
                "box":       box.tolist(),
                "w_err_pct": round(w_err * 100, 1),
                "h_err_pct": round(h_err * 100, 1),
                "method":    method,
            }
    return best


def detect_magnet(img: np.ndarray, grid: dict,
                  magnet_w_cm: float, magnet_h_cm: float,
                  magnet_h_px: float) -> dict | None:
    H, W      = img.shape[:2]
    px_per_cm = magnet_h_px / magnet_h_cm
    exp_w     = magnet_w_cm * px_per_cm
    exp_h     = float(magnet_h_px)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    excl  = _compass_excl_mask(grid, H, W)
    y0,y1 = _compute_band(grid, H, W)

    gray_b = gray[y0:y1]
    excl_b = excl[y0:y1]
    bg_val = _estimate_background(gray_b, excl_b)

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CONNECT_W, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, MORPH_CONNECT_H))

    best = None
    for thresh in [BG_DIFF_THRESHOLD, BG_DIFF_THRESHOLD - 5, BG_DIFF_MIN]:
        if thresh < 5:
            break
        diff   = cv2.absdiff(gray_b, np.full_like(gray_b, int(bg_val)))
        diff   = cv2.bitwise_and(diff, excl_b)
        _, bin_ = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        bin_   = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kh, iterations=4)
        bin_   = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kv, iterations=2)
        bin_   = cv2.morphologyEx(bin_, cv2.MORPH_OPEN,  kh, iterations=1)
        r      = _best_rect(bin_, exp_w, exp_h, SCALE_TOL, y0,
                             f"bg_sub_t{thresh}")
        if r and r["score"] > 0.3:
            best = r
            break

    if best is None:
        blur   = cv2.GaussianBlur(gray_b, (5, 5), 2)
        edges  = cv2.Canny(blur, 20, 80)
        edges  = cv2.bitwise_and(edges, excl_b)
        kh2    = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kh2, iterations=3)
        best   = _best_rect(closed, exp_w, exp_h, 0.50, y0, "canny_fallback")

    if best:
        best["px_per_cm"]  = round(px_per_cm, 2)
        best["expected_w"] = round(exp_w, 1)
        best["expected_h"] = round(exp_h, 1)
        best["bg_val"]     = round(bg_val, 1)
    return best


def process_photo(img_path: Path, grid: dict, debug_dir,
                  magnet_w_cm, magnet_h_cm, magnet_h_px) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {"photo": img_path.name, "error": "no se pudo leer"}

    H, W    = img.shape[:2]
    vectors = extract_vectors(img, grid)
    magnet  = detect_magnet(img, grid, magnet_w_cm, magnet_h_cm, magnet_h_px)

    if debug_dir is not None:
        dbg   = img.copy()
        y0,y1 = _compute_band(grid, H, W)
        ov = dbg.copy()
        cv2.rectangle(ov, (0, y0), (W, y1), (160, 110, 0), -1)
        cv2.addWeighted(ov, 0.15, dbg, 0.85, 0, dbg)
        cv2.rectangle(dbg, (0, y0), (W, y1), (160, 110, 0), 1)

        px_per_cm = magnet_h_px / magnet_h_cm
        bg_val    = magnet["bg_val"] if magnet else 0
        cv2.putText(dbg,
            f"bg={bg_val:.0f}  buscando: {magnet_w_cm*px_per_cm:.0f}x{magnet_h_px:.0f}px",
            (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 80), 1)

        n_ok = 0
        for v in vectors:
            if v["angle_deg"] is None:
                cv2.circle(dbg, (v["cx"], v["cy"]), 3, (0, 0, 180), -1)
                continue
            n_ok += 1
            a  = math.radians(v["angle_deg"])
            a_rad = math.radians(v["angle_deg"])
            ex = int(v["cx"] + 15 * math.sin(a_rad))
            ey = int(v["cy"] - 15 * math.cos(a_rad))
            cv2.arrowedLine(dbg, (v["cx"], v["cy"]), (ex, ey),
                            (0, 200, 0), 1, tipLength=0.35)

        if magnet:
            box = np.array(magnet["box"], np.int32)
            cv2.drawContours(dbg, [box], 0, (0, 220, 255), 3)
            mcx, mcy = int(magnet["cx"]), int(magnet["cy"])
            cv2.circle(dbg, (mcx, mcy), 9, (0, 0, 255), -1)
            cv2.circle(dbg, (mcx, mcy), 9, (255, 255, 255), 2)
            lbl = (f"{magnet['method']}  "
                   f"w={int(magnet['width_px'])}px  "
                   f"score={magnet['score']:.2f}")
            cv2.putText(dbg, lbl, (max(10, mcx - 150), mcy - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        else:
            cv2.putText(dbg, "NO MAGNET", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        pct = 100 * n_ok / max(len(vectors), 1)
        cv2.putText(dbg, f"vectors: {n_ok}/{len(vectors)} ({pct:.0f}%)",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        cv2.imwrite(str(debug_dir / f"debug_{img_path.stem}.jpg"),
                    dbg, [cv2.IMWRITE_JPEG_QUALITY, 75])

    return to_python({"photo": img_path.name,
                      "magnet_px": magnet, "vectors": vectors})


def main():
    ap = argparse.ArgumentParser(description="Extraccion de vectores v8-final")
    ap.add_argument("--photos-dir",  required=True)
    ap.add_argument("--grid",        required=True)
    ap.add_argument("--output-dir",  default="data")
    ap.add_argument("--debug",       action="store_true")
    ap.add_argument("--ext",         default="jpg")
    ap.add_argument("--single",      default=None)
    ap.add_argument("--magnet-w-cm", type=float, default=MAGNET_W_CM)
    ap.add_argument("--magnet-h-cm", type=float, default=MAGNET_H_CM)
    ap.add_argument("--magnet-h-px", type=float, default=MAGNET_H_PX)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.grid) as f:
        grid = json.load(f)

    photos = ([Path(args.single)] if args.single
              else sorted(Path(args.photos_dir).glob(f"*.{args.ext}")))
    if not photos:
        sys.exit(f"[ERROR] No se encontraron .{args.ext}")

    px_per_cm = args.magnet_h_px / args.magnet_h_cm
    print(f"[INFO] {len(photos)} fotos | {grid['n_compasses']} brujulas")
    print(f"[INFO] Escala {px_per_cm:.1f} px/cm | "
          f"buscando {args.magnet_w_cm*px_per_cm:.0f}x{args.magnet_h_px:.0f}px")

    debug_dir = None
    if args.debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    results, n_ok_mag, vecs, methods = [], 0, [], {}
    for p in tqdm(photos, unit="foto"):
        r = process_photo(p, grid, debug_dir,
                          args.magnet_w_cm, args.magnet_h_cm, args.magnet_h_px)
        results.append(r)
        if r.get("magnet_px"):
            n_ok_mag += 1
            m = r["magnet_px"].get("method", "?")
            methods[m] = methods.get(m, 0) + 1
        vecs.append(sum(1 for v in r.get("vectors", [])
                        if v.get("angle_deg") is not None))

    out = output_dir / "field_data.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    T = len(results); tv = sum(vecs); tp = T * grid["n_compasses"]
    print(f"\n{'─'*54}")
    print(f"  Iman detectado:   {n_ok_mag}/{T}  ({100*n_ok_mag/max(T,1):.0f}%)")
    print(f"  Vectores:         {tv}/{tp}  ({100*tv/max(tp,1):.1f}%)")
    if vecs:
        print(f"  Vecs/foto:        "
              f"min={min(vecs)} med={int(np.median(vecs))} max={max(vecs)}")
    if methods:
        print(f"  Metodos iman:")
        for m, c in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"    {m:30s}: {c}")
    print(f"  Salida: {out}")
    print(f"{'─'*54}")


if __name__ == "__main__":
    main()
