"""
Descarga de ABI-L2-CMIPF (Cloud and Moisture Imagery, disco completo) de GOES-16/GOES-19
para las bandas 08, 09, 10, 13 y 14, recortada al dominio de ~400x400 km centrado en
Bogotá (4.596°N, 74.077°O) y guardada como un NetCDF comprimido por timestamp.

TODOS LOS TIEMPOS SON UTC: los argumentos, los nombres de archivo, los logs y la fecha de
transición entre satélites. Al re-descargar timestamps vistos en hora local, convertirlos
a UTC antes (Bogotá = UTC-5).

Garantías de esta versión (ver 07-goes_qc.ipynb, Secciones 4 y 7):
- Cada timestamp se procesa en su propia carpeta temporal, que se borra al terminar,
  falle o no. No hay carpeta compartida entre timestamps, sesiones ni procesos.
- Solo se usan archivos de NOAA cuyo escaneo inicia dentro de [t, t + 10 min). Si hay
  más de un escaneo por banda en esa ventana, se usa el más completo en el dominio.
- Un archivo solo se escribe si están las cinco bandas, con la forma esperada y sin
  píxeles faltantes por encima del umbral. Si no, se registra como incidencia.
- La escritura es atómica: el archivo final aparece solo cuando está completo.
- Si NOAA no tiene el dato, no se reintenta; si falla la red, se reintenta.
- Cada archivo guarda su procedencia (archivos fuente de NOAA, satélite, timestamp).
- Un archivo de bloqueo impide correr dos descargas a la vez sobre la misma carpeta.

Uso:
    # Rango (UTC)
    python download_cmip.py --date_ini "2024-04-20 06:00" --date_fin "2024-04-20 10:00" \
        --dir_salida ../../data/cmip_cropped

    # Lista de timestamps (UTC), p. ej. qc_redescarga_goes.csv de la notebook 07.
    # En modo lista se re-descarga siempre, aunque el timestamp figure como completado;
    # el archivo que se reemplaza se guarda en <dir_salida>/../cmip_reemplazados/.
    python download_cmip.py --lista qc_redescarga_goes.csv --dir_salida ../../data/cmip_cropped
"""
import argparse
import csv
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
from netCDF4 import Dataset
from pyproj import Proj

PRODUCTO = "ABI-L2-CMIPF"
CANALES = ["08", "09", "10", "13", "14"]
PASO = pd.Timedelta("10min")
TRANSICION = pd.Timestamp("2025-04-04 15:10", tz="UTC")     # GOES-16 → GOES-19 como GOES-East
BOUND = {"lon": [-75.877, -72.277], "lat": [2.796, 6.396]}   # dominio ~400x400 km
FORMA_ESPERADA = (198, 199)
UMBRAL_NAN = 0.05          # fracción máxima de píxeles sin dato por banda
REINTENTOS = 3
PATRON_INICIO = re.compile(r"_s(\d{13})")                    # sAAAADDDHHMMSS


# ---------------------------------------------------------------- utilidades
def bucket_de(t):
    return "noaa-goes16" if t < TRANSICION else "noaa-goes19"


def clave_log(t):
    """Formato histórico del log: 'AAAA-MM-DD HH:MM:SS', en UTC, sin zona."""
    return t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")


def nombre_archivo(t):
    return f"CMIPC_{t:%Y%m%d-%H%M%S}.nc"


def inicio_escaneo(nombre):
    m = PATRON_INICIO.search(nombre)
    return pd.to_datetime(m.group(1), format="%Y%j%H%M%S").tz_localize("UTC") if m else None


def con_reintentos(funcion, *args):
    for intento in range(1, REINTENTOS + 1):
        try:
            return funcion(*args)
        except FileNotFoundError:
            raise
        except Exception:
            if intento == REINTENTOS:
                raise
            time.sleep(5 * intento)


class SinDatoNOAA(Exception):
    """NOAA no tiene alguna de las bandas para este timestamp. No se reintenta."""


class Incompleto(Exception):
    """NOAA tiene el dato, pero no pasa la validación. No se escribe el archivo."""


# ---------------------------------------------------------------- NOAA
def candidatos(fs, t, canal):
    """Archivos de NOAA de la banda cuyo escaneo inicia dentro de [t, t + 10 min)."""
    ruta = f"{bucket_de(t)}/{PRODUCTO}/{t:%Y}/{t:%j}/{t:%H}/"
    try:
        archivos = con_reintentos(fs.ls, ruta)
    except FileNotFoundError:
        return []
    return [f for f in archivos
            if re.search(rf"-M\dC{canal}_", f)
            and (ini := inicio_escaneo(f)) is not None and t <= ini < t + PASO]


def indices_recorte(ds):
    proj = ds.variables["goes_imager_projection"]
    h = proj.perspective_point_height
    p = Proj(proj="geos", h=h, lon_0=proj.longitude_of_projection_origin,
             sweep=proj.sweep_angle_axis)
    xmin, ymin = np.array(p(BOUND["lon"][0], BOUND["lat"][0])) / h
    xmax, ymax = np.array(p(BOUND["lon"][1], BOUND["lat"][1])) / h
    x, y = ds.variables["x"][:], ds.variables["y"][:]
    sx = np.where((x >= xmin) & (x <= xmax))[0]
    sy = np.where((y >= ymin) & (y <= ymax))[0]
    return sy, sx, x, y, p, h


def leer_recorte(ruta_local):
    """Devuelve el recorte crudo (valores empaquetados, sin escalar), sus atributos,
    la fracción de píxeles sin dato y la geometría del recorte."""
    with Dataset(ruta_local) as ds:
        sy, sx, x, y, p, h = indices_recorte(ds)
        var = ds.variables["CMI"]
        # Fracción sin dato, con los valores ya decodificados
        escalado = var[sy.min():sy.max() + 1, sx.min():sx.max() + 1]
        frac_nan = float(np.ma.getmaskarray(escalado).mean())
        # Valores crudos para copiarlos idénticos al archivo de salida
        var.set_auto_maskandscale(False)
        crudo = np.array(var[sy.min():sy.max() + 1, sx.min():sx.max() + 1])
        atributos = {k: var.getncattr(k) for k in var.ncattrs()}
        geom = {"x": np.array(x[sx]), "y": np.array(y[sy]), "p": p, "h": h}
    return crudo, atributos, frac_nan, geom


# ---------------------------------------------------------------- un timestamp
def procesar_timestamp(fs, t, dir_salida, dir_trabajo, dir_reemplazados=None):
    """Descarga, valida y escribe un timestamp. Devuelve la ruta del archivo escrito."""
    tmp = Path(tempfile.mkdtemp(prefix="cmip_", dir=dir_trabajo))
    try:
        bandas, fuentes = {}, {}
        for canal in CANALES:
            cands = candidatos(fs, t, canal)
            if not cands:
                raise SinDatoNOAA(f"C{canal} no existe en NOAA")
            mejor = None
            for f in cands:
                local = tmp / Path(f).name
                con_reintentos(fs.get, f, str(local))
                crudo, atributos, frac_nan, geom = leer_recorte(local)
                if mejor is None or frac_nan < mejor[2]:
                    mejor = (crudo, atributos, frac_nan, geom, Path(f).name)
            crudo, atributos, frac_nan, geom, fuente = mejor
            if crudo.shape != FORMA_ESPERADA:
                raise Incompleto(f"C{canal} con forma {crudo.shape}")
            if frac_nan > UMBRAL_NAN:
                raise Incompleto(f"C{canal} con {frac_nan:.1%} de píxeles sin dato")
            bandas[canal] = (crudo, atributos, geom)
            fuentes[canal] = fuente

        # Escritura en temporal y reemplazo atómico
        geom = bandas[CANALES[0]][2]
        x_, y_ = np.meshgrid(geom["x"] * geom["h"], geom["y"] * geom["h"])
        lon_grid, lat_grid = geom["p"](x_, y_, inverse=True)

        destino = Path(dir_salida) / nombre_archivo(t)
        parcial = Path(dir_salida) / (f".{destino.name}.part")
        with Dataset(parcial, "w", format="NETCDF4") as out:
            out.createDimension("y", FORMA_ESPERADA[0])
            out.createDimension("x", FORMA_ESPERADA[1])
            lat = out.createVariable("lat", "f4", ("y",), zlib=True, complevel=9)
            lon = out.createVariable("lon", "f4", ("x",), zlib=True, complevel=9)
            lat[:], lon[:] = lat_grid[:, 0], lon_grid[0, :]
            lat.units, lon.units = "degrees_north", "degrees_east"
            for canal, (crudo, atributos, _) in bandas.items():
                fill = atributos.get("_FillValue")
                v = out.createVariable(f"CMI_C{canal}", crudo.dtype, ("y", "x"),
                                       zlib=True, complevel=9, fill_value=fill)
                v.set_auto_maskandscale(False)
                v.setncatts({k: val for k, val in atributos.items() if k != "_FillValue"})
                v[:] = crudo
            out.setncatts({
                "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "satelite": bucket_de(t).replace("noaa-", "").upper(),
                "producto": PRODUCTO,
                **{f"fuente_C{c}": fuentes[c] for c in CANALES},
                "fecha_descarga_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

        if destino.exists() and dir_reemplazados is not None:
            Path(dir_reemplazados).mkdir(parents=True, exist_ok=True)
            shutil.move(str(destino), str(Path(dir_reemplazados) / destino.name))
        os.replace(parcial, destino)
        return destino
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for resto in Path(dir_salida).glob(f".{nombre_archivo(t)}.part"):
            resto.unlink(missing_ok=True)


# ---------------------------------------------------------------- ejecución
def descargar(timestamps, dir_salida, log_ok, log_incidencias, forzar=False):
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_trabajo = dir_salida.parent / "cmip_trabajo"
    dir_trabajo.mkdir(exist_ok=True)
    dir_reemplazados = dir_salida.parent / "cmip_reemplazados" if forzar else None

    # Bloqueo: una sola descarga a la vez sobre esta carpeta
    candado = dir_salida / ".descarga.lock"
    try:
        fd = os.open(candado, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid {os.getpid()} desde {datetime.now():%Y-%m-%d %H:%M}".encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"⚠️ Ya hay una descarga en curso sobre {dir_salida} ({candado}). "
                         "Si no es así, borrar ese archivo y volver a lanzar.")

    try:
        completados = set()
        if Path(log_ok).exists():
            completados = set(Path(log_ok).read_text().split("\n"))
        nuevo_log = not Path(log_incidencias).exists()

        fs = s3fs.S3FileSystem(anon=True)
        with open(log_incidencias, "a", newline="") as fi:
            w = csv.writer(fi)
            if nuevo_log:
                w.writerow(["timestamp_utc", "estado", "detalle", "registrado_utc"])
            for t in timestamps:
                if clave_log(t) in completados and not forzar:
                    continue
                ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                try:
                    destino = procesar_timestamp(fs, t, dir_salida, dir_trabajo, dir_reemplazados)
                    if clave_log(t) not in completados:
                        with open(log_ok, "a") as fo:
                            fo.write(clave_log(t) + "\n")
                    print(f"✓ {clave_log(t)} UTC → {destino.name}")
                except SinDatoNOAA as e:
                    w.writerow([clave_log(t), "sin_dato_noaa", str(e), ahora])
                    print(f"· {clave_log(t)} UTC: sin dato en NOAA ({e})")
                except Incompleto as e:
                    w.writerow([clave_log(t), "incompleto", str(e), ahora])
                    print(f"· {clave_log(t)} UTC: incompleto ({e})")
                except Exception as e:
                    w.writerow([clave_log(t), "error", str(e)[:200], ahora])
                    print(f"✗ {clave_log(t)} UTC: error ({str(e)[:120]})")
                fi.flush()
    finally:
        candado.unlink(missing_ok=True)


def timestamps_rango(date_ini, date_fin):
    ini = pd.Timestamp(date_ini, tz="UTC")
    fin = pd.Timestamp(date_fin, tz="UTC")
    return list(pd.date_range(ini, fin, freq=PASO, inclusive="left"))


def timestamps_lista(ruta_csv):
    ts = pd.to_datetime(pd.read_csv(ruta_csv)["timestamp"], utc=True)
    return sorted(ts.dt.floor(PASO).unique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date_ini", type=str, help="Inicio del rango, UTC: 'AAAA-MM-DD HH:MM'")
    parser.add_argument("--date_fin", type=str, help="Fin del rango (excluido), UTC")
    parser.add_argument("--lista", type=str,
                        help="CSV con columna 'timestamp' (UTC). Re-descarga siempre.")
    parser.add_argument("--dir_salida", type=str, default="cmip_cropped")
    parser.add_argument("--log", type=str, default="cmip_log.txt",
                        help="Log de timestamps completados")
    parser.add_argument("--log_incidencias", type=str, default="cmip_log_incidencias.csv")
    args = parser.parse_args()

    if args.lista:
        ts = timestamps_lista(args.lista)
        forzar = True
    elif args.date_ini and args.date_fin:
        ts = timestamps_rango(args.date_ini, args.date_fin)
        forzar = False
    else:
        parser.error("Indicar --lista, o bien --date_ini y --date_fin")

    print(f"Timestamps a procesar: {len(ts)} (UTC)  |  salida: {args.dir_salida}")
    descargar(ts, args.dir_salida, args.log, args.log_incidencias, forzar=forzar)
    print("\nDescarga completada.")