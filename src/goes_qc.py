"""
Controles de calidad de los archivos GOES recortados (formato de download_cmip.py).

Reúne en un módulo los criterios establecidos en 07-goes_qc.ipynb, para aplicarlos sin
cambios a archivos que se incorporan al registro después de ese diagnóstico, como los
instantes reconstruidos desde L1b en 08-goes-inventory.ipynb. Los umbrales y la grilla de
referencia son los fijados en la notebook 07 (Secciones 3, 5, 6 y 8): si cambian allí,
deben cambiar aquí.

Todos los timestamps se tratan en UTC.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

BANDAS_NC = ["CMI_C08", "CMI_C09", "CMI_C10", "CMI_C13", "CMI_C14"]
BANDAS = [b[-3:] for b in BANDAS_NC]
PASO = pd.Timedelta("10min")

UMBRAL_NAN_PARCIAL = 0.05                                     # Sección 3
LIMITES_K = {"C08": (160, 300), "C09": (160, 310), "C10": (160, 320),
             "C13": (160, 340), "C14": (160, 340)}           # Sección 5
Q_SALTO = 0.999                                               # Sección 6
GRILLA_REF = {"ny": 198, "nx": 199, "lat0": 6.3914, "lon0": -75.8797}   # Sección 8


def _utc(serie):
    """Timestamps a UTC con zona; los que vienen sin zona se interpretan como UTC."""
    return pd.to_datetime(serie, utc=True)


def estadisticas_archivos(df):
    """Pasada única por archivo, idéntica a la Sección 1 de la notebook 07.
    df: columnas timestamp, satelite y filepath, ordenado por timestamp. El cambio respecto
    de la imagen anterior (rmsd_prev) se calcula solo entre filas consecutivas de df que
    distan exactamente 10 minutos."""
    filas, previo = [], {}
    df = df.assign(timestamp=_utc(df["timestamp"])).sort_values("timestamp")
    for ts, sat, fp in tqdm(df[["timestamp", "satelite", "filepath"]].itertuples(index=False),
                            total=len(df), desc="QC satelital"):
        fila = {"timestamp": ts, "satelite": sat, "filepath": fp, "error": None}
        fila["mtime"] = Path(fp).stat().st_mtime
        try:
            with xr.open_dataset(fp) as ds:
                fila["ny"], fila["nx"] = ds.sizes["y"], ds.sizes["x"]
                fila["lat0"], fila["lon0"] = float(ds["lat"][0]), float(ds["lon"][0])
                consecutivo = ("ts" in previo) and (ts - previo["ts"] == PASO)
                actual = {}
                for b in BANDAS_NC:
                    k = b[-3:]
                    if b not in ds.variables:
                        fila[f"{k}_presente"] = False
                        continue
                    v = ds[b].values.astype("float64")
                    actual[k] = v
                    fila[f"{k}_presente"] = True
                    fila[f"{k}_hash"] = hashlib.md5(v.tobytes()).hexdigest()
                    fila[f"{k}_frac_nan"] = float(np.isnan(v).mean())
                    if fila[f"{k}_frac_nan"] < 1:
                        fila[f"{k}_media"] = float(np.nanmean(v))
                        fila[f"{k}_std"] = float(np.nanstd(v))
                        fila[f"{k}_min"] = float(np.nanmin(v))
                        fila[f"{k}_max"] = float(np.nanmax(v))
                        if consecutivo and k in previo and previo[k].shape == v.shape:
                            fila[f"{k}_rmsd_prev"] = float(np.sqrt(np.nanmean((v - previo[k]) ** 2)))
                previo = {"ts": ts, **actual}
        except Exception as e:
            fila["error"] = str(e)[:80]
            previo = {}
        filas.append(fila)
    salida = pd.DataFrame(filas)
    for k in BANDAS:
        for sufijo in ["presente", "hash", "frac_nan", "media", "std", "min", "max", "rmsd_prev"]:
            if f"{k}_{sufijo}" not in salida.columns:
                salida[f"{k}_{sufijo}"] = np.nan
    return salida


def controles_internos(stats, qc_registro):
    """Controles de la Sección 10.1 de la notebook 07 sobre archivos nuevos: completitud,
    integridad (ninguna banda idéntica a la de otro archivo del registro), rango físico y
    geometría. qc_registro: estadísticas del registro (qc_archivos_goes.parquet)."""
    r = stats.copy()
    for k in BANDAS:
        r[f"{k}_ok"] = (r[f"{k}_presente"].fillna(False).astype(bool)
                        & (r[f"{k}_frac_nan"] < UMBRAL_NAN_PARCIAL))
    r["completo"] = r[[f"{k}_ok" for k in BANDAS]].all(axis=1)

    otros = qc_registro[~_utc(qc_registro["timestamp"]).isin(_utc(r["timestamp"]))]
    r["duplicado"] = False
    for k in BANDAS:
        r.loc[r[f"{k}_hash"].isin(otros[f"{k}_hash"].dropna()), "duplicado"] = True

    r["fuera_rango"] = False
    for k, (lo, hi) in LIMITES_K.items():
        r.loc[(r[f"{k}_min"] < lo) | (r[f"{k}_max"] > hi), "fuera_rango"] = True

    r["geometria_ok"] = ((r["ny"] == GRILLA_REF["ny"]) & (r["nx"] == GRILLA_REF["nx"])
                         & (r["lat0"].round(4) == GRILLA_REF["lat0"])
                         & (r["lon0"].round(4) == GRILLA_REF["lon0"]))
    r["supera_controles"] = (r["completo"] & ~r["duplicado"] & ~r["fuera_rango"]
                             & r["geometria_ok"] & r["error"].isna())
    return r


def continuidad(stats, nuevos, qc_registro):
    """Cambio entre imágenes consecutivas en las transiciones que involucran un archivo
    nuevo, como fracción del umbral p99.9 del registro (Sección 6 de la notebook 07).
    stats: estadísticas de la secuencia (archivos nuevos y sus vecinos del registro).
    nuevos: máscara booleana, alineada con stats, que marca los archivos nuevos."""
    s = stats.assign(nuevo=np.asarray(nuevos, dtype=bool)).sort_values("timestamp")
    s["previo_nuevo"] = s["nuevo"].shift(1, fill_value=False)
    trans = s[s["nuevo"] | s["previo_nuevo"]].copy()
    trans["transicion"] = np.select([trans["nuevo"] & trans["previo_nuevo"], trans["nuevo"]],
                                    ["nuevo → nuevo", "registro → nuevo"], "nuevo → registro")
    filas = []
    for k in BANDAS:
        umbral = qc_registro.groupby("satelite")[f"{k}_rmsd_prev"].quantile(Q_SALTO)
        trans[f"{k}_razon"] = trans[f"{k}_rmsd_prev"] / trans["satelite"].map(umbral)
        for tipo, g in trans.groupby("transicion")[f"{k}_razon"]:
            filas.append({"banda": k, "transicion": tipo, "n": int(g.notna().sum()),
                          "razon_max": float(g.max()), "sobre_umbral": int((g > 1).sum())})
    return pd.DataFrame(filas), trans
