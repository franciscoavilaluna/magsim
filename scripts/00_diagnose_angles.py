import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def bar_magnet_field_2d(obs_x, obs_y, cx, cy, theta, half_len, q_m=1e8):
    nx = cx - half_len * math.cos(theta)
    ny = cy - half_len * math.sin(theta)
    sx = cx + half_len * math.cos(theta)
    sy = cy + half_len * math.sin(theta)

    def pole(px, py, c):
        dx, dy = obs_x - px, obs_y - py
        r3 = np.maximum(dx**2 + dy**2, 1.0) ** 1.5
        return c*dx/r3, c*dy/r3

    Bxn, Byn = pole(nx, ny, +q_m)
    Bxs, Bys = pole(sx, sy, -q_m)
    return Bxn+Bxs, Byn+Bys


def compass_angle_to_vec(angle_deg):
    rad = math.radians(angle_deg - 90)
    return math.cos(rad), math.sin(rad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-data",       required=True)
    ap.add_argument("--photos-dir",       required=True)
    ap.add_argument("--photo",            default=None)
    ap.add_argument("--output-dir",       default="data")
    ap.add_argument("--magnet-length-cm", type=float, default=15.3)
    ap.add_argument("--magnet-h-px",      type=float, default=46.0)
    ap.add_argument("--magnet-h-cm",      type=float, default=1.3)
    args = ap.parse_args()

    with open(args.field_data) as f:
        field_data = json.load(f)

    if args.photo:
        records = [r for r in field_data if r["photo"] == args.photo]
        if not records:
            print(f"[ERROR] No se encontro {args.photo}")
            return
        record = records[0]
    else:
        valid = [r for r in field_data
                 if r.get("magnet_px") and r.get("vectors")]

        def overlap_score(r):
            mag  = r["magnet_px"]
            mcx  = mag.get("cx", 0)
            vecs = [v for v in r["vectors"] if v.get("angle_deg") is not None]
            if not vecs:
                return 0
            v_xs  = [v["cx"] for v in vecs]
            v_cx  = (min(v_xs) + max(v_xs)) / 2
            dist  = abs(mcx - v_cx)
            n_ok  = sum(1 for v in vecs if v.get("angle_deg") is not None)
            return n_ok / max(dist + 1, 1)

        record = max(valid, key=overlap_score)
        print(f"[INFO] Usando foto mas centrada: {record['photo']}")

    mag = record["magnet_px"]
    vectors = [v for v in record["vectors"] if v.get("angle_deg") is not None]

    px_per_cm  = args.magnet_h_px / args.magnet_h_cm
    half_len   = (args.magnet_length_cm / 2.0) * px_per_cm
    cx, cy     = float(mag["cx"]), float(mag["cy"])
    theta      = math.radians(float(mag.get("angle_deg", 0)))

    img_path = Path(args.photos_dir) / record["photo"]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] No se pudo leer {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W    = img.shape[:2]

    obs_x = np.array([v["cx"] for v in vectors], dtype=float)
    obs_y = np.array([v["cy"] for v in vectors], dtype=float)
    obs_a = np.array([v["angle_deg"] for v in vectors], dtype=float)

    Bx,  By  = bar_magnet_field_2d(obs_x, obs_y, cx, cy, theta,            half_len)
    Bx2, By2 = bar_magnet_field_2d(obs_x, obs_y, cx, cy, theta + math.pi,  half_len)

    convenciones = {}
    for tag, bx_, by_ in [("theta_normal", Bx, By), ("theta+180", Bx2, By2)]:
        for conv, arr in [
            ("atan2(By,Bx)-90",  (np.degrees(np.arctan2( by_,  bx_)) - 90.0) % 360),
            ("atan2(By,Bx)+90",  (np.degrees(np.arctan2( by_,  bx_)) + 90.0) % 360),
            ("atan2(Bx,-By)-90", (np.degrees(np.arctan2( bx_, -by_)) - 90.0) % 360),
            ("atan2(-Bx,By)-90", (np.degrees(np.arctan2(-bx_,  by_)) - 90.0) % 360),
        ]:
            convenciones[f"{tag} | {conv}"] = arr

    print("\n[FOTOS CON IMAN MAS CENTRADO ENTRE BRUJULAS]")
    print(f"{'Foto':35s} {'cx_iman':>8} {'cx_bruj_centro':>15} {'distancia':>10}")
    print("─" * 72)
    scored = []
    for r in field_data:
        if not r.get("magnet_px"): continue
        vecs = [v for v in r.get("vectors",[]) if v.get("angle_deg") is not None]
        if not vecs: continue
        mcx  = r["magnet_px"].get("cx", 0)
        v_xs = [v["cx"] for v in vecs]
        vcx  = (min(v_xs) + max(v_xs)) / 2
        dist = abs(mcx - vcx)
        scored.append((dist, r["photo"], mcx, vcx))
    for dist, photo, mcx, vcx in sorted(scored)[:8]:
        print(f"  {photo:33s} {mcx:8.0f} {vcx:15.0f} {dist:10.0f}px")

    print(f"\n  → Usa la foto con menor distancia para el diagnostico:")
    print(f"    python scripts/00_diagnose_angles.py ... --photo {sorted(scored)[0][1]}")

    print(f"{'Convencion':20s}  RMSE vs medido")
    print("─" * 40)
    best_conv = None
    best_rmse = 999
    for name, pred in convenciones.items():
        diff = ((pred - obs_a + 180) % 360) - 180
        rmse = float(np.sqrt(np.mean(diff**2)))
        flag = " <-- MEJOR" if rmse < best_rmse else ""
        print(f"  {name:20s}  {rmse:6.1f}°{flag}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_conv = name
            best_pred = pred

    print(f"\n  Mejor convencion: {best_conv}  (RMSE={best_rmse:.1f}°)")

    if not HAS_MPL:
        print("[WARN] matplotlib no disponible")
        return

    uses_flipped = "theta+180" in best_conv
    theta_best   = theta + math.pi if uses_flipped else theta
    Bx_best, By_best = bar_magnet_field_2d(
        obs_x, obs_y, cx, cy, theta_best, half_len)

    nx_b  = cx + half_len * math.cos(theta_best)
    ny_b  = cy + half_len * math.sin(theta_best)
    sx_b  = cx - half_len * math.cos(theta_best)
    sy_b  = cy - half_len * math.sin(theta_best)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    ax.imshow(img_rgb)
    ax.set_title(f"Vectores MEDIDOS — {record['photo']}\n"
                 f"Verde = punta roja de la brujula", fontsize=10)

    L = 22
    for v in vectors:
        a   = math.radians(v["angle_deg"])
        vx  = math.sin(a)
        vy  = -math.cos(a)
        ax.annotate("", xy=(v["cx"]+L*vx, v["cy"]+L*vy),
            xytext=(v["cx"], v["cy"]),
            arrowprops=dict(arrowstyle="->", color="lime", lw=1.2))

    if mag.get("box"):
        ax.add_patch(plt.Polygon(np.array(mag["box"]),
                                 fill=False, edgecolor="cyan", lw=2))
    ax.plot(cx, cy, "wo", ms=6)
    nx0 = cx + half_len * math.cos(theta)
    ny0 = cy + half_len * math.sin(theta)
    sx0 = cx - half_len * math.cos(theta)
    sy0 = cy - half_len * math.sin(theta)
    ax.plot(nx0, ny0, "b^", ms=12, label=f"N modelo (angle={math.degrees(theta):.0f}°)")
    ax.plot(sx0, sy0, "rs", ms=12, label="S modelo")
    ax.legend(fontsize=8, loc="upper left")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(img_rgb, alpha=0.45)
    ax.set_title(f"Medido (verde) vs Predicho (naranja)\n"
                 f"{best_conv}\nRMSE={best_rmse:.1f}°", fontsize=9)

    for i, v in enumerate(vectors):
        a    = math.radians(v["angle_deg"])
        vx   = math.sin(a);  vy = -math.cos(a)
        ax.annotate("", xy=(v["cx"]+L*vx, v["cy"]+L*vy),
            xytext=(v["cx"], v["cy"]),
            arrowprops=dict(arrowstyle="->", color="lime", lw=1.2))

        ap   = math.radians(float(best_pred[i]))
        vxp  = math.sin(ap); vyp = -math.cos(ap)
        ax.annotate("", xy=(v["cx"]+L*vxp, v["cy"]+L*vyp),
            xytext=(v["cx"], v["cy"]),
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.2, alpha=0.75))

    if mag.get("box"):
        ax.add_patch(plt.Polygon(np.array(mag["box"]),
                                 fill=False, edgecolor="cyan", lw=2))
    ax.plot(nx_b, ny_b, "b^", ms=12, label="N (mejor conv)")
    ax.plot(sx_b, sy_b, "rs", ms=12, label="S (mejor conv)")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="lime",   label="Medido"),
        mpatches.Patch(color="orange", label="Predicho"),
        plt.Line2D([],[],marker="^",color="b",ls="",ms=8,label="Polo N"),
        plt.Line2D([],[],marker="s",color="r",ls="",ms=8,label="Polo S"),
    ], fontsize=8, loc="upper left")
    ax.axis("off")

    plt.tight_layout()
    out = output_dir / f"diagnose_{Path(record['photo']).stem}.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Diagnostico guardado: {out}")
    print(f"\n{'═'*55}")
    print(f"  Mejor convencion: {best_conv}")
    print(f"  RMSE:             {best_rmse:.1f}°")
    if uses_flipped:
        print(f"  NOTA: el polo N del iman en el modelo estaba al reves.")
        print(f"  En 03_calibrate_model.py cambia bar_magnet_field_2d para")
        print(f"  usar theta + math.pi al calcular las posiciones de los polos.")
    print(f"\n  Edita bxy_to_compass_angle() en 03_calibrate_model.py:")
    conv_code = best_conv.split("|")[-1].strip()
    print(f"  return ({conv_code}) % 360")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
