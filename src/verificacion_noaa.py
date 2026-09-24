"""
Verificación de archivos GOES recortados contra el producto original de NOAA.

Se usa desde 07-goes_qc.ipynb (Sección 7). Vive en un módulo aparte, y no en la
notebook, porque se ejecuta en procesos paralelos: en Windows cada proceso importa
las funciones desde un archivo .py, y cada uno carga su propia copia de HDF5, que
no admite ser usada por varios hilos a la vez.

Todos los timestamps están en UTC.
"""
import re

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

PRODUCTO_NOAA = "ABI-L2-CMIPF"
PASO = pd.Timedelta("10min")
BANDAS = ["C08", "C09", "C10", "C13", "C14"]

_PATRON_INICIO = re.compile(r"_s(\d{13})")   # sAAAADDDHHMMSS en el nombre de NOAA
_fs = None                                  # un cliente S3 por proceso
_cache_ls = {}                              # listados de carpetas horarias, por proceso


def _s3():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


def _bucket(t, transicion):
    return "noaa-goes16" if t < transicion else "noaa-goes19"


def _inicio_escaneo(nombre):
    m = _PATRON_INICIO.search(nombre)
    return pd.to_datetime(m.group(1), format="%Y%j%H%M%S").tz_localize("UTC") if m else None


def _listar_hora(bucket, t):
    clave = (bucket, t.strftime("%Y%j%H"))
    if clave not in _cache_ls:
        try:
            _cache_ls[clave] = _s3().ls(f"{bucket}/{PRODUCTO_NOAA}/{t:%Y}/{t:%j}/{t:%H}/")
        except FileNotFoundError:
            _cache_ls[clave] = []
    return _cache_ls[clave]


def _leer_noaa(t, banda, bucket, recorte):
    """Recortes del dominio desde NOAA para todos los escaneos que inician dentro de
    [t, t + 10 min), el mismo criterio de búsqueda que download_cmip.py. En escaneos
    anómalos NOAA puede tener más de un archivo por banda y ventana (un escaneo fallido
    y su repetición), por eso se devuelven todos. Lista vacía si no existe ninguno."""
    (y0, y1), (x0, x1), (dy, dx) = recorte
    cand = [f for f in _listar_hora(bucket, t)
            if re.search(rf"-M\d{banda}_", f)
            and (ini := _inicio_escaneo(f)) is not None and t <= ini < t + PASO]
    recortes = []
    for f in cand:
        with _s3().open(f, "rb") as fh, xr.open_dataset(fh, engine="h5netcdf") as ds:
            recortes.append(ds["CMI"].isel(y=slice(y0 + dy, y1 + dy), x=slice(x0 + dx, x1 + dx))
                            .values.astype("float64"))
    return recortes


def verificar_archivo(t, fp, bloque, recortes, transicion):
    """Compara las cinco bandas de un archivo local con NOAA. Devuelve una fila por banda.

    recortes: {bucket: ((y0, y1), (x0, x1), (dy, dx))}, calibrado en la notebook.
    """
    bucket = _bucket(t, transicion)
    filas = []
    try:
        with xr.open_dataset(fp) as ds_local:
            locales = {b: (ds_local[f"CMI_{b}"].values.astype("float64")
                           if f"CMI_{b}" in ds_local.variables else None)
                       for b in BANDAS}
    except Exception as e:
        return [{"timestamp": t, "filepath": fp, "banda": b, "bloque": bloque,
                 "estado": "error", "n_candidatos": 0, "max_dif": np.nan, "frac_nan_local": np.nan,
                 "frac_nan_noaa": np.nan, "detalle": f"local: {str(e)[:70]}"} for b in BANDAS]

    for banda in BANDAS:
        fila = {"timestamp": t, "filepath": fp, "banda": banda, "bloque": bloque, "estado": None,
                "n_candidatos": 0, "max_dif": np.nan, "frac_nan_local": np.nan,
                "frac_nan_noaa": np.nan, "detalle": None}
        try:
            loc = locales[banda]
            noaa = _leer_noaa(t, banda, bucket, recortes[bucket])
            fila["n_candidatos"] = len(noaa)
            if loc is not None:
                fila["frac_nan_local"] = float(np.isnan(loc).mean())
            if noaa:
                # Fracción sin dato del candidato más completo: indica si el dato es recuperable
                fila["frac_nan_noaa"] = min(float(np.isnan(n).mean()) for n in noaa)
            if not noaa and loc is None:
                fila["estado"] = "falta_ambos"
            elif not noaa:
                fila["estado"] = "falta_noaa"
            elif loc is None:
                fila["estado"] = "falta_local"
            elif all(n.shape != loc.shape for n in noaa):
                fila["estado"] = "forma_distinta"
            elif any(n.shape == loc.shape and np.array_equal(loc, n, equal_nan=True) for n in noaa):
                fila["estado"] = "igual"
            else:
                fila["estado"] = "distinto"
                difs = [np.abs(loc - n) for n in noaa if n.shape == loc.shape]
                finitas = [np.nanmax(d) for d in difs if np.isfinite(d).any()]
                fila["max_dif"] = float(min(finitas)) if finitas else np.nan
        except Exception as e:
            fila["estado"], fila["detalle"] = "error", str(e)[:80]
        filas.append(fila)
    return filas