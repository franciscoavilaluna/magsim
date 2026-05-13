import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib no disponible — sin graficas")

def dipole_field(obs_x, obs_y, cx, cy, theta, half_len, q_m):
    nx = cx - half_len * math.cos(theta)
    ny = cy - half_len * math.sin(theta)
    sx = cx + half_len * math.cos(theta)
    sy = cy + half_len * math.sin(theta)

    def pole(px, py, c):
        dx, dy = obs_x - px, obs_y - py
        r3 = np.maximum(dx**2 + dy**2, 1.0) ** 1.5
        return c * dx / r3, c * dy / r3

    Bxn, Byn = pole(nx, ny, +q_m)
    Bxs, Bys = pole(sx, sy, -q_m)
    return Bxn + Bxs, Byn + Bys


def field_to_angle(Bx, By):
    return (np.degrees(np.arctan2(By, Bx)) - 90.0) % 360.0


def ang_diff(a, b):
    return ((a - b + 180) % 360) - 180


def build_field_map(field_data, px_per_cm, half_len_px,
                    min_dist_px=30.0):
    all_rx, all_ry, all_Bx, all_By = [], [], [], []
    n_used = 0

    for record in field_data:
        mag = record.get("magnet_px")
        if not mag:
            continue
        vectors = [v for v in record.get("vectors", [])
                   if v.get("angle_deg") is not None]
        if len(vectors) < 5:
            continue

        cx    = float(mag["cx"])
        cy    = float(mag["cy"])
        theta = math.radians(float(mag.get("angle_deg", 0)))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for v in vectors:
            dx   = v["cx"] - cx
            dy   = v["cy"] - cy
            dist = math.hypot(dx, dy)
            if dist < min_dist_px:
                continue

            rx =  dx * cos_t + dy * sin_t
            ry = -dx * sin_t + dy * cos_t

            a      = math.radians(v["angle_deg"])
            bx_img = math.sin(a)
            by_img = -math.cos(a)

            bx_rel =  bx_img * cos_t + by_img * sin_t
            by_rel = -bx_img * sin_t + by_img * cos_t

            all_rx.append(rx);  all_ry.append(ry)
            all_Bx.append(bx_rel); all_By.append(by_rel)

        n_used += 1

    return {
        "rx": all_rx, "ry": all_ry,
        "Bx": all_Bx, "By": all_By,
        "n_photos":   n_used,
        "n_vectors":  len(all_rx),
        "px_per_cm":  float(px_per_cm),
        "half_len_px": float(half_len_px),
    }


def build_rbf(field_map):
    pts   = np.column_stack([field_map["rx"], field_map["ry"]])
    Bx    = np.array(field_map["Bx"])
    By    = np.array(field_map["By"])
    rbfBx = RBFInterpolator(pts, Bx, kernel="thin_plate_spline", smoothing=1.0)
    rbfBy = RBFInterpolator(pts, By, kernel="thin_plate_spline", smoothing=1.0)
    return rbfBx, rbfBy


def map_rmse(field_map, rbfBx, rbfBy, frac=0.2):
    n   = len(field_map["rx"])
    idx = np.random.choice(n, max(20, int(n * frac)), replace=False)
    pts = np.column_stack([
        np.array(field_map["rx"])[idx],
        np.array(field_map["ry"])[idx]
    ])
    bxp = rbfBx(pts); byp = rbfBy(pts)
    bxo = np.array(field_map["Bx"])[idx]
    byo = np.array(field_map["By"])[idx]
    ap  = np.degrees(np.arctan2(byp, bxp)) % 360
    ao  = np.degrees(np.arctan2(byo, bxo)) % 360
    return float(np.sqrt(np.mean(ang_diff(ap, ao)**2)))


def plot_report(field_map, rbfBx, rbfBy, dipole_rmses, half_len, out):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    ax  = axes[0]
    hl  = half_len
    x_  = np.linspace(-hl * 2, hl * 2, 60)
    y_  = np.linspace(-hl * 1.5, hl * 1.5, 40)
    XX, YY = np.meshgrid(x_, y_)
    pts_g  = np.column_stack([XX.ravel(), YY.ravel()])
    BX = rbfBx(pts_g).reshape(XX.shape)
    BY = rbfBy(pts_g).reshape(XX.shape)
    mg = np.hypot(BX, BY)
    mg = np.clip(mg, 0, np.percentile(mg, 95))
    ax.contourf(XX, YY, mg, levels=20, cmap="plasma")
    ax.streamplot(XX, YY, BX, BY, color="white", linewidth=0.6,
                  density=1.5, arrowsize=0.8)
    ax.axvline(-hl, color="lime", linewidth=2, label="Polo N")
    ax.axvline(+hl, color="red",  linewidth=2, label="Polo S")
    ax.set_title("Campo interpolado (coords relativas al iman)")
    ax.set_xlabel("x relativo (px)"); ax.set_ylabel("y relativo (px)")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    ax  = axes[1]
    rx  = np.array(field_map["rx"])
    ry  = np.array(field_map["ry"])
    ang = np.degrees(np.arctan2(
        np.array(field_map["By"]), np.array(field_map["Bx"]))) % 360
    sc  = ax.scatter(rx, ry, c=ang, cmap="hsv", s=4,
                     alpha=0.5, vmin=0, vmax=360)
    plt.colorbar(sc, ax=ax, label="Angulo (°)")
    ax.axvline(-hl, color="lime", linewidth=2, label="Polo N")
    ax.axvline(+hl, color="red",  linewidth=2, label="Polo S")
    ax.set_title(f"{field_map['n_vectors']} mediciones, "
                 f"{field_map['n_photos']} fotos")
    ax.set_xlabel("x relativo (px)"); ax.set_ylabel("y relativo (px)")
    ax.set_aspect("equal"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(dipole_rmses, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(np.median(dipole_rmses), color="orange", linewidth=2,
               label=f"mediana={np.median(dipole_rmses):.1f}°")
    ax.set_xlabel("RMSE dipolo (°)"); ax.set_ylabel("Fotos")
    ax.set_title("Sanity check: modelo dipolo")
    ax.legend()

    plt.suptitle("Calibracion del campo magnetico", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafica: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-data",       required=True)
    ap.add_argument("--grid",             required=True)
    ap.add_argument("--output-dir",       default="data")
    ap.add_argument("--magnet-length-cm", type=float, default=15.3)
    ap.add_argument("--magnet-h-px",      type=float, default=46.0)
    ap.add_argument("--magnet-h-cm",      type=float, default=1.3)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.field_data) as f:
        field_data = json.load(f)

    px_per_cm = args.magnet_h_px / args.magnet_h_cm
    half_len  = (args.magnet_length_cm / 2.0) * px_per_cm

    print(f"[INFO] Escala: {px_per_cm:.2f} px/cm  |  "
          f"longitud iman: {2*half_len:.0f} px")
    print(f"[INFO] {len(field_data)} fotos")

    print("\n[PASO 1] Construyendo mapa de campo...")
    fm = build_field_map(field_data, px_per_cm, half_len)
    print(f"  {fm['n_vectors']} vectores de {fm['n_photos']} fotos")

    if fm["n_vectors"] < 50:
        print("[ERROR] Muy pocos vectores")
        return

    print("[INFO] Ajustando RBF...")
    rbfBx, rbfBy = build_rbf(fm)
    rmse_map = map_rmse(fm, rbfBx, rbfBy)
    print(f"  RMSE mapa interpolado: {rmse_map:.1f}°")

    n_save = min(2000, fm["n_vectors"])
    idx    = np.random.choice(fm["n_vectors"], n_save, replace=False)
    fm_save = {k: ([fm[k][i] for i in idx] if isinstance(fm[k], list) else fm[k])
               for k in fm}
    fm_save["map_rmse_deg"] = float(rmse_map)
    with open(output_dir / "field_map.json", "w") as f:
        json.dump(fm_save, f, indent=2)
    print(f"[OK] field_map.json guardado")

    print("\n[PASO 2] Sanity check dipolo...")
    valid = [r for r in field_data
             if r.get("magnet_px") and
             sum(1 for v in r.get("vectors",[]) if v.get("angle_deg") is not None) >= 10]

    dipole_rmses = []
    q_m_init = half_len ** 2
    q_ms     = []

    for record in valid[:30]:
        mag   = record["magnet_px"]
        vecs  = [v for v in record["vectors"] if v.get("angle_deg") is not None]
        ox    = np.array([v["cx"] for v in vecs], dtype=float)
        oy    = np.array([v["cy"] for v in vecs], dtype=float)
        oa    = np.array([v["angle_deg"] for v in vecs], dtype=float)
        cx, cy = float(mag["cx"]), float(mag["cy"])
        theta  = math.radians(float(mag.get("angle_deg", 0)))

        def obj(p):
            q, = p
            if q <= 0: return 1e10
            Bx, By = dipole_field(ox, oy, cx, cy, theta, half_len, q)
            return float(np.mean(ang_diff(field_to_angle(Bx, By), oa)**2))

        res  = minimize(obj, [q_m_init], method="Nelder-Mead",
                        options={"maxiter":3000,"xatol":1e3,"fatol":0.5})
        q_fit = abs(res.x[0])
        Bx, By = dipole_field(ox, oy, cx, cy, theta, half_len, q_fit)
        rmse   = float(np.sqrt(np.mean(ang_diff(field_to_angle(Bx,By), oa)**2)))
        dipole_rmses.append(rmse)
        q_ms.append(q_fit)
        if rmse < 20:
            q_m_init = 0.7*q_m_init + 0.3*q_fit

    q_m_consensus = float(np.median(q_ms))
    print(f"  RMSE dipolo mediana: {np.median(dipole_rmses):.1f}°")
    print(f"  q_m consenso:        {q_m_consensus:.4e}")

    params = {
        "magnet_length_cm":       args.magnet_length_cm,
        "magnet_h_cm":            args.magnet_h_cm,
        "magnet_h_px":            args.magnet_h_px,
        "px_per_cm":              float(px_per_cm),
        "half_len_px":            float(half_len),
        "q_m":                    q_m_consensus,
        "pole_N_side":            "left",
        "angle_convention":       "angle_from_mask: (atan2(dy,dx)-90)%360",
        "field_map_file":         "field_map.json",
        "map_rmse_deg":           float(rmse_map),
        "dipole_rmse_median_deg": float(np.median(dipole_rmses)),
        "n_photos":               fm["n_photos"],
        "n_vectors":              fm["n_vectors"],
    }
    with open(output_dir / "model_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"[OK] model_params.json guardado")

    if HAS_MPL:
        plot_report(fm, rbfBx, rbfBy, dipole_rmses,
                    half_len, output_dir / "calibration_report.png")

    quality = ("excelente" if rmse_map < 10 else
               "bueno"     if rmse_map < 20 else
               "aceptable" if rmse_map < 35 else "revisar")
    print(f"\n{'═'*50}")
    print(f"  Mapa interpolado RMSE: {rmse_map:.1f}°  [{quality}]")
    print(f"  Dipolo RMSE mediana:   {np.median(dipole_rmses):.1f}°")
    print(f"  La simulacion 3D usara field_map.json")
    print(f"{'═'*50}")
    print(f"\nSIGUIENTE PASO: construir la simulacion 3D")


if __name__ == "__main__":
    main()
