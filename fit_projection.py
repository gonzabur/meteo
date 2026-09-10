#!/usr/bin/env python3
"""
Prueba varias familias de proyeccion (LCC, Estereografica oblicua, Lambert
Azimutal Equal-Area) contra los puntos de control marcados con
SelectorPuntos.html, y elige la que minimiza el error en pixeles.

Para cualquier proyeccion, dados sus parametros geograficos (que solo
determinan la FORMA/rotacion, no la escala), px=(X-bx)/ax y py=(Y-by)/ay
se resuelven exactamente por minimos cuadrados (regresion lineal 1D). Asi
que solo optimizamos sobre 1-3 parametros geograficos por familia, no 7.

Uso:
  python3 fit_projection.py puntos.json
"""
import json
import sys
import numpy as np
from scipy.optimize import differential_evolution

R = 6378137.0  # constante de escala; se absorbe en ax/ay igualmente

def lcc_xy(lon_deg, lat_deg, lon0_deg, lat1_deg, lat2_deg):
    lon = np.radians(lon_deg); lat = np.radians(lat_deg)
    lon0 = np.radians(lon0_deg); lat1 = np.radians(lat1_deg); lat2 = np.radians(lat2_deg)
    lat0 = (lat1 + lat2) / 2.0
    if abs(lat1 - lat2) < 1e-10:
        n = np.sin(lat1)
    else:
        n = np.log(np.cos(lat1)/np.cos(lat2)) / np.log(np.tan(np.pi/4+lat2/2)/np.tan(np.pi/4+lat1/2))
    F = np.cos(lat1) * np.tan(np.pi/4+lat1/2)**n / n
    rho = R * F / np.tan(np.pi/4+lat/2)**n
    rho0 = R * F / np.tan(np.pi/4+lat0/2)**n
    theta = n * (lon - lon0)
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y

def _azimuthal_common(lon_deg, lat_deg, lon0_deg, lat0_deg):
    lon = np.radians(lon_deg); lat = np.radians(lat_deg)
    lon0 = np.radians(lon0_deg); lat0 = np.radians(lat0_deg)
    dlon = lon - lon0
    coslat, sinlat = np.cos(lat), np.sin(lat)
    coslat0, sinlat0 = np.cos(lat0), np.sin(lat0)
    cosc = sinlat0*sinlat + coslat0*coslat*np.cos(dlon)
    return dlon, coslat, sinlat, coslat0, sinlat0, cosc

def stereo_xy(lon_deg, lat_deg, lon0_deg, lat0_deg):
    dlon, coslat, sinlat, coslat0, sinlat0, cosc = _azimuthal_common(lon_deg, lat_deg, lon0_deg, lat0_deg)
    k = 2.0 * R / (1.0 + cosc)
    x = k * coslat * np.sin(dlon)
    y = k * (coslat0*sinlat - sinlat0*coslat*np.cos(dlon))
    return x, y

def laea_xy(lon_deg, lat_deg, lon0_deg, lat0_deg):
    dlon, coslat, sinlat, coslat0, sinlat0, cosc = _azimuthal_common(lon_deg, lat_deg, lon0_deg, lat0_deg)
    k = R * np.sqrt(2.0 / np.clip(1.0 + cosc, 1e-9, None))
    x = k * coslat * np.sin(dlon)
    y = k * (coslat0*sinlat - sinlat0*coslat*np.cos(dlon))
    return x, y

def rotated_xy(lon_deg, lat_deg, pole_lon_deg, pole_lat_deg):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    plon = np.radians(pole_lon_deg); plat = np.radians(pole_lat_deg)
    x = np.cos(lat)*np.cos(lon); y = np.cos(lat)*np.sin(lon); z = np.sin(lat)
    x1 = x*np.cos(plon) + y*np.sin(plon)
    y1 = -x*np.sin(plon) + y*np.cos(plon)
    z1 = z
    theta = np.pi/2 - plat
    x2 = x1*np.cos(theta) - z1*np.sin(theta)
    y2 = y1
    z2 = x1*np.sin(theta) + z1*np.cos(theta)
    lat2 = np.arcsin(np.clip(z2, -1, 1))
    lon2 = np.arctan2(y2, x2)
    return R*lon2, R*lat2

def fit_affine(px, X):
    # app usa: px = (X - b) / a  =>  X = a*px + b
    A = np.vstack([px, np.ones_like(px)]).T
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef  # [a, b]

def fit_xy(pxs, pys, X, Y):
    ax, bx = fit_affine(pxs, X)
    ay, by = fit_affine(pys, Y)
    pred_px = (X - bx) / ax
    pred_py = (Y - by) / ay
    err = np.sqrt((pred_px-pxs)**2 + (pred_py-pys)**2)
    return ax, bx, ay, by, err

MODELS = {
    "lcc": {
        "bounds": [(-20, 30), (20, 55), (40, 89.9)],  # lon0, lat1, lat2
        "xy": lambda p, lons, lats: lcc_xy(lons, lats, p[0], p[1], p[2]),
        "extra": lambda p: {"lon0": p[0], "lat0": (p[1]+p[2])/2.0, "lat1": p[1], "lat2": p[2]},
    },
    "stereo": {
        "bounds": [(-20, 30), (30, 89.9)],  # lon0, lat0
        "xy": lambda p, lons, lats: stereo_xy(lons, lats, p[0], p[1]),
        "extra": lambda p: {"lon0": p[0], "lat0": p[1]},
    },
    "laea": {
        "bounds": [(-20, 30), (30, 89.9)],  # lon0, lat0
        "xy": lambda p, lons, lats: laea_xy(lons, lats, p[0], p[1]),
        "extra": lambda p: {"lon0": p[0], "lat0": p[1]},
    },
    "rotated": {
        "bounds": [(-180, 180), (-90, 90)],  # pole_lon, pole_lat
        "xy": lambda p, lons, lats: rotated_xy(lons, lats, p[0], p[1]),
        "extra": lambda p: {"pole_lon": p[0], "pole_lat": p[1]},
    },
}

def objective(params, model, lons, lats, pxs, pys):
    X, Y = MODELS[model]["xy"](params, lons, lats)
    ax, bx, ay, by, err = fit_xy(pxs, pys, X, Y)
    return np.sqrt(np.mean(err**2))

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 fit_projection.py puntos.json"); sys.exit(1)
    with open(sys.argv[1]) as f:
        pts = json.load(f)
    pts = [p for p in pts if not p.get("skipped")]
    if len(pts) < 4:
        print(f"Solo {len(pts)} puntos validos; se recomiendan al menos 4-5."); sys.exit(1)

    lons = np.array([p["lon"] for p in pts])
    lats = np.array([p["lat"] for p in pts])
    pxs  = np.array([p["px"]  for p in pts])
    pys  = np.array([p["py"]  for p in pts])

    results = {}
    for name, spec in MODELS.items():
        result = differential_evolution(
            objective, spec["bounds"], args=(name, lons, lats, pxs, pys),
            maxiter=500, popsize=40, tol=1e-12, seed=0, polish=True
        )
        results[name] = result

    print(f"{'modelo':<10}{'RMSE(px)':>10}")
    for name, result in sorted(results.items(), key=lambda kv: kv[1].fun):
        print(f"{name:<10}{result.fun:>10.2f}")

    best_name = min(results, key=lambda n: results[n].fun)
    best = results[best_name]
    print(f"\n>>> Mejor modelo: {best_name}\n")

    X, Y = MODELS[best_name]["xy"](best.x, lons, lats)
    ax, bx, ay, by, errs = fit_xy(pxs, pys, X, Y)

    print(f"{'punto':<12}{'err(px)':>10}")
    for p, e in zip(pts, errs):
        print(f"{p['id']:<12}{e:>10.2f}")

    out = {"proj": best_name}
    out.update({k: round(float(v), 4) for k, v in MODELS[best_name]["extra"](best.x).items()})
    out.update({"ax": float(ax), "bx": float(bx), "ay": float(ay), "by": float(by)})
    print("\nParametros ajustados:")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
