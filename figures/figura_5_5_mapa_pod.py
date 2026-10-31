"""Figura 5.5. POD por evento de cada estación en prueba: (a) umbral en C13, (b) logística L2,
(c) CNN plana L2 (semilla 42). Círculos vacíos: estaciones con menos de 10 episodios evaluables.
Datos: modelo/salidas_21/por_estacion.csv, data_ingestion/ground_truth/anexo_A_estaciones.csv."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import pandas as pd, numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")
pe = pd.read_csv(RAIZ / "modelo/salidas_21/por_estacion.csv", dtype={"codigoestacion": str})
a = pd.read_csv(RAIZ / "data_ingestion/ground_truth/anexo_A_estaciones.csv", dtype=str)
a["codigoestacion"] = a["Código"].str.zfill(10)
a["lat"] = a["Latitud (°N)"].astype(float); a["lon"] = a["Longitud (°)"].astype(float)
pe["codigoestacion"] = pe.codigoestacion.str.zfill(10)
m = pe.merge(a[["codigoestacion", "lat", "lon"]], on="codigoestacion")
bog = gpd.read_file(RAIZ / "data/cartografia/perimetro_urbano_bogota.geojson").to_crs(4326)
K = ROL["linea_ref"]; EXT = (-74.36, -73.84, 4.45, 5.03)
PUE = {"Zipaquirá": (5.022, -74.004), "Cajicá": (4.918, -74.028), "Chía": (4.862, -74.058), "Funza": (4.716, -74.211),
       "Soacha": (4.579, -74.217), "La Calera": (4.721, -73.968), "Bogotá": (4.640, -74.105)}
cm = LinearSegmentedColormap.from_list("pod", ["#fff3c4", PALETA["wi-iceblue"], PALETA["wwu-blue"], "#0b2f45"])
MOD = [("umbral_c13", "(a) Umbral en C13"), ("logistica_L2", "(b) Logística L2"), ("cnn_plana_L2_s42", "(c) CNN plana L2")]

fig, axs = plt.subplots(1, 3, figsize=(13, 5.6), constrained_layout=True)
for ax, (k, t) in zip(axs, MOD):
    ax.set_facecolor("#f6f5f1"); bog.plot(ax=ax, color="#e3e1d9", edgecolor="#bdb9ad", lw=0.6, zorder=1)
    for n_, (la, lo) in PUE.items():
        ax.plot(lo, la, "s", ms=2.3, color="#8d8a80", zorder=2)
        ax.text(lo + 0.008, la + 0.006, n_, fontsize=7, color="#6d6a62", style="italic")
    s = m[m.modelo == k]; ok = s[~s.insuficiente]; ins = s[s.insuficiente]
    sc = ax.scatter(ok.lon, ok.lat, c=ok.pod_evento, cmap=cm, vmin=0, vmax=0.75, s=48, edgecolors=K, linewidths=0.5, zorder=4)
    ax.scatter(ins.lon, ins.lat, s=40, facecolors="none", edgecolors="#9e9e9e", linewidths=1, zorder=4)
    ax.text(0.03, 0.97, f"mediana: {ok.pod_evento.median():.2f}".replace(".", ","), transform=ax.transAxes, va="top", fontsize=8.5, color=K)
    ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:]); ax.set_aspect(1 / np.cos(np.radians(4.75)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values(): sp.set_color("#bbbbbb")
    ax.set_title(t, loc="left", fontsize=10, color=K)
km = 20 / (111.32 * np.cos(np.radians(4.75))); x0, y0 = EXT[0] + 0.025, EXT[2] + 0.02
axs[0].plot([x0, x0 + km], [y0, y0], color=K, lw=2); axs[0].text(x0 + km / 2, y0 + 0.012, "20 km", ha="center", fontsize=7.5, color=K)
cb = fig.colorbar(sc, ax=axs, shrink=0.62, pad=0.015); cb.set_label("POD por evento")
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
axs[1].legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec="#9e9e9e", ms=7, label="menos de 10 episodios evaluables")],
              loc="upper center", frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.savefig(SALIDA, dpi=230, bbox_inches="tight")
print("ok", SALIDA.name)
