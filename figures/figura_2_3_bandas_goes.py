"""Figura 2.3. Temperatura de brillo en las bandas 8, 10 y 13 del ABI, 6 de noviembre de 2024,
12:00 y 15:00 hora local (17:00 y 20:00 UTC).
Datos: data/cmip_cropped/CMIPC_20241106-{170000,200000}.nc, data/cache/estaciones.parquet,
data/cartografia/perimetro_urbano_bogota.geojson."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import ROL

import netCDF4, numpy as np, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")

ARCH = [RAIZ / "data/cmip_cropped/CMIPC_20241106-170000.nc", RAIZ / "data/cmip_cropped/CMIPC_20241106-200000.nc"]
FILAS = ["12:00 hora local\n(antes de la convección)", "15:00 hora local\n(convección desarrollada)"]
bog = gpd.read_file(RAIZ / "data/cartografia/perimetro_urbano_bogota.geojson").to_crs(4326)
est = pd.read_parquet(RAIZ / "data/cache/estaciones.parquet")
# Escala IR: colores bajo ~235 K, grises por encima (misma que las Figuras 5.6–5.8)
ir = LinearSegmentedColormap.from_list("ir", [(0, "#3b0f70"), (0.2, "#c43c75"), (0.32, "#fca50a"), (0.38, "#f0f0f0"), (1, "#202020")])
COLS = [("CMI_C08", "Banda 8 (6,2 µm)\nvapor de agua, troposfera alta", (200, 250), "Blues_r"),
        ("CMI_C10", "Banda 10 (7,3 µm)\nvapor de agua, troposfera media-baja", (200, 270), "Blues_r"),
        ("CMI_C13", "Banda 13 (10,3 µm)\nventana infrarroja", (200, 300), ir)]

ds = [netCDF4.Dataset(a) for a in ARCH]
lat = np.asarray(ds[0]["lat"][:]); lon = np.asarray(ds[0]["lon"][:])
ext = [lon.min(), lon.max(), lat.min(), lat.max()]
elat, elon = lat[est.fila.values], lon[est.col.values]

fig, axs = plt.subplots(2, 3, figsize=(10, 7.4), constrained_layout=True)
for j, (b, tit, (vmin, vmax), cm) in enumerate(COLS):
    for i, d in enumerate(ds):
        ax = axs[i, j]; z = np.ma.filled(d[b][:].astype(float), np.nan)
        im = ax.imshow(z, extent=ext, origin="upper" if lat[0] > lat[-1] else "lower", cmap=cm, vmin=vmin, vmax=vmax)
        bog.boundary.plot(ax=ax, color="white", lw=1.8); bog.boundary.plot(ax=ax, color=ROL["linea_ref"], lw=0.7)
        ax.plot(elon, elat, "o", ms=1.3, mfc="white", mec=ROL["linea_ref"], mew=0.3)
        if b == "CMI_C13" and i == 1:
            ax.contour(lon, lat, z, levels=[230], colors="white", linewidths=0.7, origin="upper")
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        if i == 0: ax.set_title(tit, fontsize=9.5)
        if j == 0: ax.set_ylabel(FILAS[i], fontsize=9.5)
    cb = fig.colorbar(im, ax=axs[:, j], orientation="horizontal", shrink=0.85, pad=0.02); cb.set_label("Temperatura de brillo (K)")
    if b == "CMI_C13":
        cb.ax.axvline(230, color="k", lw=1.2)
        cb.ax.text(230, 1.4, "230 K", ha="center", fontsize=8, transform=cb.ax.get_xaxis_transform())

km = 100 / (111.32 * np.cos(np.radians(4.6)))
x0, y0 = ext[0] + 0.15, ext[2] + 0.15
axs[1, 0].plot([x0, x0 + km], [y0, y0], color="white", lw=2.5)
axs[1, 0].text(x0 + km / 2, y0 + 0.08, "100 km", color="white", ha="center", fontsize=8)
fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
