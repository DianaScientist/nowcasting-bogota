"""
Reconstrucción de la temperatura de brillo de las bandas IR del ABI a partir de la
radiancia calibrada de nivel L1b (ABI-L1b-RadF), para instantes en que NOAA no generó
el producto CMIP (ABI-L2-CMIPF) o lo generó incompleto.

Se usa desde 08-goes-inventory.ipynb. Vive en un módulo aparte, y no en la notebook,
porque se ejecuta en procesos paralelos: en Windows cada proceso importa las funciones
desde un archivo .py, y cada uno carga su propia copia de HDF5, que no admite ser usada
por varios hilos a la vez.

Criterios compartidos con download_cmip.py (mantener sincronizados):
- Dominio BOUND y forma del recorte (198 x 199).
- Cálculo de los índices del recorte sobre la grilla fija del propio archivo de NOAA.
- Geometría lat/lon del archivo de salida.

Todos los timestamps están en UTC, sin zona.
"""
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from netCDF4 import Dataset
from pyproj import Proj

PRODUCTO_L1B = "ABI-L1b-RadF"
PRODUCTO_CMIP = "ABI-L2-CMIPF"
BANDAS = ["C08", "C09", "C10", "C13", "C14"]
BOUND = {"lon": [-75.877, -72.277], "lat": [2.796, 6.396]}   # idéntico a download_cmip.py
FORMA_ESPERADA = (198, 199)
TRANSICION = pd.Timestamp("2025-04-04 15:10")                 # GOES-16 → GOES-19, UTC
COEFICIENTES = ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")
PASO = pd.Timedelta("10min")
UMBRAL_NAN = 0.05                                             # idéntico a download_cmip.py
NIVEL_ORIGEN = "L1b-RadF convertido a temperatura de brillo"

_PATRON_ESCANEO = re.compile(r"_s(\d{14})_")   # sAAAAJJJHHMMSSt: identifica un escaneo
_fs = None                                      # un cliente S3 por proceso
_cache_ls = {}                                  # listados de carpetas horarias, por proceso


# ---------------------------------------------------------------- utilidades
def _s3():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


def bucket_de(t):
    return "noaa-goes16" if pd.Timestamp(t) < TRANSICION else "noaa-goes19"


def escaneo(nombre):
    """Identificador del escaneo (inicio con décimas de segundo) en un nombre de NOAA."""
    m = _PATRON_ESCANEO.search(nombre)
    return m.group(1) if m else None


def _listar_hora(bucket, producto, s):
    """Archivos de la carpeta horaria que corresponde al escaneo s."""
    anio, dia, hora = s[:4], s[4:7], s[7:9]
    clave = (bucket, producto, anio, dia, hora)
    if clave not in _cache_ls:
        try:
            _cache_ls[clave] = _s3().ls(f"{bucket}/{producto}/{anio}/{dia}/{hora}/")
        except FileNotFoundError:
            _cache_ls[clave] = []
    return _cache_ls[clave]


def buscar_l1b(bucket, banda, s):
    """Archivos L1b de la banda con exactamente el mismo escaneo s."""
    return [f for f in _listar_hora(bucket, PRODUCTO_L1B, s)
            if re.search(rf"-M\d{banda}_", f) and escaneo(f) == s]


def escaneos_cmip(bucket, banda, t):
    """Escaneos CMIP de la banda que inician dentro de [t, t + 10 min), el mismo criterio
    de búsqueda que download_cmip.py. Se usa cuando el archivo local no guarda su
    procedencia (archivos anteriores a la versión actual del script de descarga)."""
    t = pd.Timestamp(t)
    fin = t + pd.Timedelta("10min")
    salida = []
    for f in _listar_hora(bucket, PRODUCTO_CMIP, t.strftime("%Y%j%H")):
        s = escaneo(f)
        if s is None or not re.search(rf"-M\d{banda}_", f):
            continue
        inicio = pd.to_datetime(s[:13], format="%Y%j%H%M%S")
        if t <= inicio < fin:
            salida.append(s)
    return sorted(set(salida))


# ---------------------------------------------------------------- geometría y lectura
def _indices_recorte(ds):
    """Mismo cálculo que indices_recorte() de download_cmip.py, sobre un dataset xarray."""
    proj = ds["goes_imager_projection"].attrs
    h = float(proj["perspective_point_height"])
    p = Proj(proj="geos", h=h, lon_0=float(proj["longitude_of_projection_origin"]),
             sweep=proj["sweep_angle_axis"])
    xmin, ymin = np.array(p(BOUND["lon"][0], BOUND["lat"][0])) / h
    xmax, ymax = np.array(p(BOUND["lon"][1], BOUND["lat"][1])) / h
    x, y = ds["x"].values, ds["y"].values
    sx = np.where((x >= xmin) & (x <= xmax))[0]
    sy = np.where((y >= ymin) & (y <= ymax))[0]
    return slice(sy.min(), sy.max() + 1), slice(sx.min(), sx.max() + 1), x, y, p, h


def leer_l1b(ruta_s3):
    """Recorte del dominio desde un archivo L1b remoto: radiancia, DQF, coeficientes de
    Planck y geometría. Solo se transfieren los bloques del archivo que cubren el recorte."""
    with _s3().open(ruta_s3, "rb") as fh, xr.open_dataset(fh, engine="h5netcdf") as ds:
        sy, sx, x, y, p, h = _indices_recorte(ds)
        rad = ds["Rad"].isel(y=sy, x=sx).values.astype("float64")
        dqf = ds["DQF"].isel(y=sy, x=sx).values.astype("float64")
        coef = {k: float(ds[k].values) for k in COEFICIENTES}
        geom = {"x": np.asarray(x[sx]), "y": np.asarray(y[sy]), "p": p, "h": h}
    return rad, dqf, coef, geom


def latlon(geom):
    """Vectores lat/lon del recorte, con la misma construcción que download_cmip.py."""
    xx, yy = np.meshgrid(geom["x"] * geom["h"], geom["y"] * geom["h"])
    lon, lat = geom["p"](xx, yy, inverse=True)
    return lat[:, 0], lon[0, :]


def rad_a_bt(rad, coef):
    """Temperatura de brillo (K) desde radiancia con la inversa de Planck del producto L1b."""
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = (coef["planck_fk2"] / np.log(coef["planck_fk1"] / rad + 1.0)
              - coef["planck_bc1"]) / coef["planck_bc2"]
    bt[~np.isfinite(bt) | ~(rad > 0)] = np.nan
    return bt


# ---------------------------------------------------------------- validación
def validar_timestamp(t, fp):
    """Reconstruye las cinco bandas de un instante que sí tiene CMIP en disco y las compara
    con él. El L1b se empareja con el CMIP por el mismo escaneo: el que registra el
    atributo de procedencia del archivo local o, si no lo tiene, el único escaneo CMIP de
    NOAA en la ventana del instante. Devuelve una fila por banda."""
    t = pd.Timestamp(t)
    bucket = bucket_de(t)
    with xr.open_dataset(fp) as ds:
        lat_loc, lon_loc = ds["lat"].values, ds["lon"].values
        locales = {b: (ds[f"CMI_{b}"].values.astype("float64"),
                       float(ds[f"CMI_{b}"].encoding["scale_factor"]),
                       float(ds[f"CMI_{b}"].encoding["add_offset"]))
                   for b in BANDAS}
        fuentes = {b: str(ds.attrs.get(f"fuente_{b}", "")) for b in BANDAS}

    filas = []
    for b in BANDAS:
        fila = {"timestamp": t, "banda": b, "estado": None, "origen_escaneo": None,
                "n_cmip": np.nan, "n_l1b": 0, "geom_ok": None,
                "max_dif_K": np.nan, "frac_igual_emp": np.nan, "frac_dif_1": np.nan,
                "nan_coincide": None, "frac_nan_cmip": np.nan, "n_dqf_mayor_1": np.nan,
                "detalle": None}
        try:
            s = escaneo(fuentes[b])
            if s is not None:
                fila["origen_escaneo"] = "procedencia"
            else:
                # Sin procedencia: el escaneo se identifica en NOAA. Solo se valida si hay
                # un único escaneo CMIP en la ventana, para no emparejar el equivocado.
                cands_cmip = escaneos_cmip(bucket, b, t)
                fila["n_cmip"] = len(cands_cmip)
                if len(cands_cmip) != 1:
                    fila["estado"] = "sin_cmip" if not cands_cmip else "varios_cmip"
                    filas.append(fila)
                    continue
                s = cands_cmip[0]
                fila["origen_escaneo"] = "noaa"
            cands = buscar_l1b(bucket, b, s)
            fila["n_l1b"] = len(cands)
            if not cands:
                fila["estado"] = "sin_l1b"
                filas.append(fila)
                continue

            rad, dqf, coef, geom = leer_l1b(cands[0])
            if rad.shape != FORMA_ESPERADA:
                fila["estado"], fila["detalle"] = "forma_distinta", str(rad.shape)
                filas.append(fila)
                continue
            lat, lon = latlon(geom)
            fila["geom_ok"] = bool(np.allclose(lat, lat_loc, atol=1e-4)
                                   and np.allclose(lon, lon_loc, atol=1e-4))

            bt = rad_a_bt(rad, coef)
            cmip, sf, off = locales[b]
            ambos = np.isfinite(bt) & np.isfinite(cmip)
            fila["nan_coincide"] = bool(np.array_equal(np.isnan(bt), np.isnan(cmip)))
            fila["frac_nan_cmip"] = float(np.isnan(cmip).mean())
            fila["n_dqf_mayor_1"] = int(np.nansum(dqf > 1))
            if ambos.any():
                fila["max_dif_K"] = float(np.abs(bt - cmip)[ambos].max())
                # Comparación en el espacio de enteros del empaquetado de CMIP
                q_rec = np.round((bt[ambos] - off) / sf)
                q_cmip = np.round((cmip[ambos] - off) / sf)
                dif = np.abs(q_rec - q_cmip)
                fila["frac_igual_emp"] = float((dif == 0).mean())
                fila["frac_dif_1"] = float((dif == 1).mean())
            fila["estado"] = "ok"
        except Exception as e:
            fila["estado"], fila["detalle"] = "error", str(e)[:120]
        filas.append(fila)
    return filas


# ---------------------------------------------------------------- reconstrucción
class SinDatoL1b(Exception):
    """NOAA no tiene L1b para alguna de las bandas en la ventana del instante."""


class Incompleto(Exception):
    """El L1b existe pero no pasa los mismos controles que exige download_cmip.py."""


def nombre_archivo(t):
    return f"CMIPC_{pd.Timestamp(t):%Y%m%d-%H%M%S}.nc"


def escaneos_l1b(bucket, banda, t):
    """Archivos L1b de la banda cuyo escaneo inicia dentro de [t, t + 10 min)."""
    t = pd.Timestamp(t)
    salida = []
    for f in _listar_hora(bucket, PRODUCTO_L1B, t.strftime("%Y%j%H")):
        s = escaneo(f)
        if s is None or not re.search(rf"-M\d{banda}_", f):
            continue
        if t <= pd.to_datetime(s[:13], format="%Y%j%H%M%S") < t + PASO:
            salida.append(f)
    return sorted(salida)


def referencia_empaquetado(fp_ref):
    """Tipo de dato y atributos de empaquetado de cada banda, tomados de un archivo CMIP
    local del mismo satélite. Se copian tal cual al archivo reconstruido."""
    ref = {}
    with Dataset(fp_ref) as ds:
        for b in BANDAS:
            v = ds.variables[f"CMI_{b}"]
            v.set_auto_maskandscale(False)
            ref[b] = {"dtype": v.dtype, "attrs": {k: v.getncattr(k) for k in v.ncattrs()}}
    return ref


def empaquetar(bt, ref_banda):
    """Temperatura de brillo (K) a los enteros crudos del producto CMIP."""
    at, dtype = ref_banda["attrs"], ref_banda["dtype"]
    q = np.round((bt - float(at["add_offset"])) / float(at["scale_factor"]))
    sin_signo = str(at.get("_Unsigned", "false")).lower() == "true"
    info = np.iinfo(np.dtype(dtype.str.replace("i", "u")) if sin_signo else dtype)
    fuera = np.isfinite(q) & ((q < info.min) | (q > info.max))
    if fuera.any():
        raise Incompleto(f"{int(fuera.sum())} píxeles fuera del rango del empaquetado")
    crudo = np.nan_to_num(q, nan=0).astype(info.dtype)
    if sin_signo:
        crudo = crudo.view(dtype)
    crudo[np.isnan(bt)] = at["_FillValue"]
    return crudo


def reconstruir_timestamp(t, dir_salida, ref, nombre_ref):
    """Reconstruye un instante desde L1b y lo escribe con el formato de download_cmip.py.
    Si hay más de un escaneo por banda en la ventana, usa el más completo en el dominio,
    igual que el script de descarga. Devuelve una fila con el resultado."""
    t = pd.Timestamp(t)
    bucket = bucket_de(t)
    destino = Path(dir_salida) / nombre_archivo(t)
    parcial = Path(dir_salida) / f".{destino.name}.part"
    fila = {"timestamp": t, "estado": None, "detalle": None, "archivo": str(destino)}
    try:
        if destino.exists():
            raise FileExistsError(f"{destino.name} ya existe en la carpeta de salida")
        bandas, fuentes, geom0 = {}, {}, None
        for b in BANDAS:
            cands = escaneos_l1b(bucket, b, t)
            if not cands:
                raise SinDatoL1b(f"{b} no existe en L1b")
            mejor = None
            for f in cands:
                rad, _, coef, geom = leer_l1b(f)
                bt = rad_a_bt(rad, coef)
                frac_nan = float(np.isnan(bt).mean())
                if mejor is None or frac_nan < mejor[1]:
                    mejor = (bt, frac_nan, geom, f.split("/")[-1])
            bt, frac_nan, geom, fuente = mejor
            if bt.shape != FORMA_ESPERADA:
                raise Incompleto(f"{b} con forma {bt.shape}")
            if frac_nan > UMBRAL_NAN:
                raise Incompleto(f"{b} con {frac_nan:.1%} de píxeles sin dato")
            bandas[b] = empaquetar(bt, ref[b])
            fuentes[b] = fuente
            fila[f"frac_nan_{b}"] = frac_nan
            geom0 = geom0 or geom

        lat_v, lon_v = latlon(geom0)
        with Dataset(parcial, "w", format="NETCDF4") as out:
            out.createDimension("y", FORMA_ESPERADA[0])
            out.createDimension("x", FORMA_ESPERADA[1])
            lat = out.createVariable("lat", "f4", ("y",), zlib=True, complevel=9)
            lon = out.createVariable("lon", "f4", ("x",), zlib=True, complevel=9)
            lat[:], lon[:] = lat_v, lon_v
            lat.units, lon.units = "degrees_north", "degrees_east"
            for b, crudo in bandas.items():
                at = ref[b]["attrs"]
                v = out.createVariable(f"CMI_{b}", crudo.dtype, ("y", "x"),
                                       zlib=True, complevel=9, fill_value=at["_FillValue"])
                v.set_auto_maskandscale(False)
                v.setncatts({k: val for k, val in at.items() if k != "_FillValue"})
                v[:] = crudo
            out.setncatts({
                "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "satelite": bucket.replace("noaa-", "").upper(),
                "producto": PRODUCTO_L1B,
                "nivel_origen": NIVEL_ORIGEN,
                "referencia_empaquetado": nombre_ref,
                **{f"fuente_{b}": fuentes[b] for b in BANDAS},
                "fecha_descarga_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        os.replace(parcial, destino)
        fila["estado"] = "escrito"
    except SinDatoL1b as e:
        fila["estado"], fila["detalle"] = "sin_dato_l1b", str(e)
    except Incompleto as e:
        fila["estado"], fila["detalle"] = "incompleto", str(e)
    except Exception as e:
        fila["estado"], fila["detalle"] = "error", str(e)[:120]
    finally:
        parcial.unlink(missing_ok=True)
    return fila


def comparar_crudo(fp_a, fp_b):
    """Compara dos archivos con el formato de download_cmip.py a nivel de enteros crudos."""
    filas = []
    with Dataset(fp_a) as a, Dataset(fp_b) as b:
        geom_igual = (np.allclose(a["lat"][:], b["lat"][:], atol=1e-4)
                      and np.allclose(a["lon"][:], b["lon"][:], atol=1e-4))
        for banda in BANDAS:
            va, vb = a.variables[f"CMI_{banda}"], b.variables[f"CMI_{banda}"]
            va.set_auto_maskandscale(False)
            vb.set_auto_maskandscale(False)
            ca, cb = np.array(va[:]), np.array(vb[:])
            attrs_iguales = ({k: str(va.getncattr(k)) for k in va.ncattrs()}
                             == {k: str(vb.getncattr(k)) for k in vb.ncattrs()})
            filas.append({"banda": banda, "geom_igual": geom_igual,
                          "dtype_igual": ca.dtype == cb.dtype, "attrs_iguales": attrs_iguales,
                          "frac_igual": float((ca == cb).mean()),
                          "max_dif_crudo": int(np.abs(ca.astype("int64") - cb.astype("int64")).max())})
    return filas
