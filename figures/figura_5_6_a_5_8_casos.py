"""Figuras 5.6–5.8. Casos de estudio (umbral en C13, logística L2, CNN plana L2 semilla 42).
Réplica de modelo/21-analisis_errores.ipynb (figura_caso): mismo instante t*, mismas estaciones (hasta 20)
y mismos umbrales de decisión. Izquierda: C13 en t* con el resultado de la CNN en cada estación;
derecha: pares positivos (franjas) y alertas (puntos) a lo largo del día pluviométrico.
Datos: modelo/salidas_s11/pred_test_*.parquet, data/cache/estaciones.parquet, data/cmip_cropped/*.nc."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import numpy as np, pandas as pd, netCDF4, geopandas as gpd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.8})
AQUI = Path(__file__).resolve().parent
S = RAIZ / "modelo/salidas_s11"
TZ = "America/Bogota"
MOD = [("umbral_c13", "Umbral en C13", 0.36, PALETA["wi-grey"]),
       ("logistica_L2", "Logística L2", 0.40, PALETA["wi-ocher"]),
       ("cnn_plana_L2_s42", "CNN plana L2", 0.58, PALETA["wwu-blue"])]
P = {m: pd.read_parquet(S / f"pred_test_{m}.parquet") for m, *_ in MOD}
base = P["umbral_c13"][["codigoestacion", "timestamp", "etiqueta"]]
est = pd.read_parquet(RAIZ / "data/cache/estaciones.parquet")
est["cod"] = est.codigoestacion.astype(str).str.zfill(10); est = est.set_index("cod")
bog = gpd.read_file(RAIZ / "data/cartografia/perimetro_urbano_bogota.geojson").to_crs(4326)
K = ROL["linea_ref"]
RES = {"acierto": PALETA["wwu-green"], "omisión": ROL["positivo"], "falsa alarma": PALETA["wi-ocher"], "correcto negativo": "#bdbdbd"}
ir = LinearSegmentedColormap.from_list("ir", [(0, "#3b0f70"), (0.2, "#c43c75"), (0.32, "#fca50a"), (0.38, "#f0f0f0"), (1, "#202020")])

def caso(nombre, dia, archivo_nc, salida, falsa=False):
    ini = (pd.Timestamp(dia).tz_localize(TZ) + pd.Timedelta(hours=7)).tz_convert("UTC"); fin = ini + pd.Timedelta(days=1)
    msk = ((base.timestamp >= ini) & (base.timestamp < fin)).values
    d = base[msk].reset_index(drop=True); y = d.etiqueta.values == 1
    A = {m: P[m].p.values[msk] >= u for m, _, u, _ in MOD}
    a = A["cnn_plana_L2_s42"]; clave = (a & ~y) if falsa else y
    ts = d[clave].groupby("timestamp").size().idxmax()
    act = d.assign(pos=y, al=np.column_stack(list(A.values())).any(1)).groupby("codigoestacion")[["pos", "al"]].sum()
    E = act[(act.pos + act.al) > 0].sort_values(["pos", "al"], ascending=False).index[:20].tolist()

    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.2], hspace=0.38, wspace=0.12)
    # mapa de C13 en t*
    ax0 = fig.add_subplot(gs[:, 0])
    n = netCDF4.Dataset(RAIZ / "data/cmip_cropped" / archivo_nc); lat = np.asarray(n["lat"][:]); lon = np.asarray(n["lon"][:])
    f0, f1, c0, c1 = 61, 121, 70, 120
    z = np.ma.filled(n["CMI_C13"][f0:f1, c0:c1].astype(float), np.nan)
    ext = [lon[c0], lon[c1 - 1], lat[f1 - 1], lat[f0]]
    im = ax0.imshow(z, extent=ext, cmap=ir, vmin=200, vmax=300, origin="upper")
    bog.boundary.plot(ax=ax0, color="white", lw=1.3); bog.boundary.plot(ax=ax0, color=K, lw=0.5)
    et = (d.timestamp == ts).values
    rr = np.select([a[et] & y[et], ~a[et] & y[et], a[et] & ~y[et]], ["acierto", "omisión", "falsa alarma"], "correcto negativo")
    for cod, r in zip(d.codigoestacion[et], rr):
        e = est.loc[cod]; ax0.plot(lon[int(e.col)], lat[int(e.fila)], "o", ms=7, color=RES[r], mec="white", mew=0.8, zorder=4)
    ax0.set_xlim(ext[0], ext[1]); ax0.set_ylim(ext[2], ext[3]); ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_xlabel(""); ax0.set_ylabel("")
    hl = ts.tz_convert(TZ).strftime("%H:%M")
    ax0.set_title(f"C13 a las {hl}, hora local,\ncon el resultado de la CNN plana L2", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax0, orientation="horizontal", fraction=0.045, pad=0.02); cb.set_label("Temperatura de brillo (K)")
    ax0.legend(handles=[Line2D([], [], marker="o", ls="", color=c, mec="white", ms=7, label=k) for k, c in RES.items()],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)
    # líneas de tiempo
    hora = lambda s: (s.dt.tz_convert(TZ) - ini.tz_convert(TZ)) / pd.Timedelta(hours=1)
    py = {c: i for i, c in enumerate(E)}; sel = d.codigoestacion.isin(E).values; sub = d[sel]
    h, yy = hora(sub.timestamp).values, sub.codigoestacion.map(py).values; pos = y[sel]
    xt = hora(pd.Series([ts])).iloc[0]
    for k, (m, lab, u, col) in enumerate(MOD):
        ax = fig.add_subplot(gs[k, 1])
        ax.scatter(h[pos], yy[pos], marker="s", s=16, color=ROL["positivo"], alpha=0.25, lw=0)
        al = A[m][sel]; ax.scatter(h[al], yy[al], s=10, color=col, zorder=3)
        ax.axvline(xt, color=K, lw=0.8, ls="--")
        ax.set_yticks(range(len(E))); ax.set_yticklabels([c[-4:] for c in E], fontsize=6)
        ax.set_ylim(len(E) - 0.4, -0.6)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xticklabels([f"{(7+x)%24:02d}:00" for x in range(0, 25, 3)])
        ax.set_title(lab, loc="left", fontsize=9.5, color=col if m != "umbral_c13" else "#555", fontweight="bold")
        for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
        ax.xaxis.grid(True, ls=":", alpha=0.4)
        if k == 0:
            ax.legend(handles=[Line2D([], [], marker="s", ls="", color=ROL["positivo"], alpha=0.35, ms=7, label="par positivo"),
                               Line2D([], [], marker="o", ls="", color=K, ms=5, label="alerta del modelo"),
                               Line2D([], [], ls="--", color=K, lw=0.8, label=f"instante del mapa ({hl})")],
                      loc="upper right", bbox_to_anchor=(1, 1.38), ncol=3, frameon=False, fontsize=8)
    ax.set_xlabel("Hora local (día pluviométrico, 07:00 a 07:00)")
    fig.savefig(AQUI / salida, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("ok", salida, nombre, hl, len(E))

caso("diurno", "2025-04-25", "CMIPC_20250425-182000.nc", "figura_5_6_caso_diurno.png")
caso("nocturna", "2025-03-11", "CMIPC_20250312-000000.nc", "figura_5_7_caso_omision_nocturna.png")
caso("falsa", "2025-10-25", "CMIPC_20251025-172000.nc", "figura_5_8_caso_falsa_alarma.png", falsa=True)
