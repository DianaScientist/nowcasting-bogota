"""Caché de imágenes GOES para el modelado (S2).

Un único arreglo uint16 (instante × banda × fila × columna) con los enteros
empaquetados del CMIP, recortado a la región que contiene todos los parches
32×32 de las estaciones. Sin pérdida: los valores en K se recuperan con el
mismo scale_factor/add_offset del archivo original, que se verifica idéntico
en cada archivo durante la construcción.

Productos (en `dir_cache`):
    imagenes.npy      uint16 (N, 5, H, W), memmap
    instantes.parquet una fila por instante: fila, timestamp, satelite, origen, filepath
    estaciones.parquet codigoestacion, fila/col en la grilla, fila_c/col_c en el recorte
    meta.json         bandas, recorte, scale/offset por banda, fecha, conteos
    progreso.npy      bool (N,), filas ya escritas (permite reanudar)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

BANDAS = ["CMI_C08", "CMI_C09", "CMI_C10", "CMI_C13", "CMI_C14"]
HALF_SIZE = 16
FILL_U16 = np.uint16(65535)   # _FillValue = -1 en int16 con _Unsigned = true


# --- Recorte ---------------------------------------------------------------------
def calcular_recorte(estaciones: pd.DataFrame, half_size: int = HALF_SIZE) -> dict:
    """Límites [f0, f1) × [c0, c1) que contienen todos los parches."""
    return {
        "f0": int(estaciones["fila"].min() - half_size),
        "f1": int(estaciones["fila"].max() + half_size),
        "c0": int(estaciones["col"].min() - half_size),
        "c1": int(estaciones["col"].max() + half_size),
    }


# --- Lectura de un archivo -------------------------------------------------------
def _codificacion(var) -> tuple:
    return (float(var.scale_factor), float(var.add_offset),
            str(var.getncattr("_Unsigned")), int(var.getncattr("_FillValue")))


def leer_crudo(filepath, recorte: dict, bandas=BANDAS):
    """Enteros crudos uint16 del recorte (B, H, W), codificación por banda y grilla lat/lon."""
    f0, f1, c0, c1 = recorte["f0"], recorte["f1"], recorte["c0"], recorte["c1"]
    with netCDF4.Dataset(filepath) as d:
        d.set_auto_maskandscale(False)
        crudo = np.stack([np.asarray(d[b][f0:f1, c0:c1]) for b in bandas])
        cod = {b: _codificacion(d[b]) for b in bandas}
        lat = np.asarray(d["lat"][:])
        lon = np.asarray(d["lon"][:])
    assert crudo.dtype == np.int16, f"{filepath}: tipo {crudo.dtype}, se esperaba int16"
    return crudo.view(np.uint16), cod, lat, lon


# --- Decodificación (réplica exacta de xarray + extraer_parche de 10) --------------
def decodificar(u16: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """uint16 (..., B, H, W) → K en float32, con la misma aritmética que xarray:
    astype(float32); *= scale (float32); += offset (float32)."""
    x = u16.astype(np.float32)
    x *= scale.astype(np.float32)[:, None, None]
    x += offset.astype(np.float32)[:, None, None]
    return x


def normalizar(k: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    """Igual que extraer_parche: float32 − media float64 → float64."""
    return (k - media[:, None, None]) / desvio[:, None, None]


# --- Construcción ----------------------------------------------------------------
def construir(inventario: pd.DataFrame, estaciones: pd.DataFrame, dir_cache,
              bandas=BANDAS, cada=2000, rutas=None) -> dict:
    """Construye (o reanuda) el caché. `rutas` permite remapear filepaths (pruebas)."""
    dir_cache = Path(dir_cache)
    dir_cache.mkdir(parents=True, exist_ok=True)

    utiles = (inventario[~inventario["excluido_qc"]]
              .sort_values("timestamp").reset_index(drop=True))
    assert utiles["timestamp"].is_unique, "Instantes duplicados en el inventario"
    n = len(utiles)
    recorte = calcular_recorte(estaciones)
    H, W = recorte["f1"] - recorte["f0"], recorte["c1"] - recorte["c0"]

    ruta_img, ruta_prog = dir_cache / "imagenes.npy", dir_cache / "progreso.npy"
    ruta_meta = dir_cache / "meta.json"
    forma = (n, len(bandas), H, W)

    if ruta_img.exists():
        img = np.lib.format.open_memmap(ruta_img, mode="r+")
        assert img.shape == forma and img.dtype == np.uint16, \
            f"Caché existente con forma {img.shape}; se esperaba {forma}. Borrar la carpeta para reconstruir."
        prog = np.load(ruta_prog)
        meta = json.loads(ruta_meta.read_text())
    else:
        img = np.lib.format.open_memmap(ruta_img, mode="w+", dtype=np.uint16, shape=forma)
        prog = np.zeros(n, dtype=bool)
        meta = None

    # Tablas auxiliares (se reescriben siempre; son deterministas)
    instantes = utiles[["timestamp", "satelite", "origen", "filepath"]].copy()
    instantes.insert(0, "fila", np.arange(n, dtype=np.int64))
    instantes.to_parquet(dir_cache / "instantes.parquet", index=False)
    est = estaciones[["codigoestacion", "fila", "col"]].copy()
    est["fila_c"] = est["fila"] - recorte["f0"]
    est["col_c"] = est["col"] - recorte["c0"]
    est.to_parquet(dir_cache / "estaciones.parquet", index=False)

    ref_cod = None if meta is None else {b: tuple(v) for b, v in meta["codificacion"].items()}
    ref_lat = ref_lon = None
    pendientes = np.flatnonzero(~prog)
    t0 = time.time()
    print(f"Caché {forma} uint16 ({img.nbytes / 1e9:.2f} GB) | pendientes {len(pendientes):,} de {n:,}")

    for k, i in enumerate(pendientes):
        fp = utiles.at[i, "filepath"]
        fp = rutas(fp) if rutas else fp
        u16, cod, lat, lon = leer_crudo(fp, recorte, bandas)

        if ref_cod is None:
            ref_cod = cod
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
        assert cod == ref_cod, f"{fp}: codificación distinta {cod} ≠ {ref_cod}"
        assert lat.shape == ref_lat.shape and lon.shape == ref_lon.shape, f"{fp}: grilla de otro tamaño"
        assert np.abs(lat - ref_lat).max() < 1e-4 and np.abs(lon - ref_lon).max() < 1e-4, \
            f"{fp}: grilla desplazada"
        assert not (u16 == FILL_U16).any(), f"{fp}: píxeles sin dato en el recorte"

        img[i] = u16
        prog[i] = True
        if (k + 1) % cada == 0 or k + 1 == len(pendientes):
            img.flush()
            np.save(ruta_prog, prog)
            _guardar_meta(ruta_meta, bandas, recorte, forma, ref_cod, prog)
            v = (k + 1) / (time.time() - t0)
            print(f"  {k + 1:>7,} / {len(pendientes):,}  ({v:,.0f} archivos/s, "
                  f"faltan ~{(len(pendientes) - k - 1) / v / 60:.0f} min)")

    if len(pendientes) == 0:
        print("Nada pendiente: el caché ya estaba completo.")
    return json.loads(ruta_meta.read_text())


def _guardar_meta(ruta, bandas, recorte, forma, cod, prog):
    meta = {
        "bandas": bandas,
        "recorte": recorte,
        "forma": list(forma),
        "dtype": "uint16",
        "codificacion": {b: list(cod[b]) for b in bandas},   # scale, offset, _Unsigned, _FillValue
        "filas_escritas": int(prog.sum()),
        "completo": bool(prog.all()),
        "actualizado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ruta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# --- Lectura ---------------------------------------------------------------------
class Cache:
    """Acceso de solo lectura. `en_memoria=True` carga el arreglo completo (~3 GB)."""

    def __init__(self, dir_cache, zscore: pd.DataFrame, en_memoria=False):
        dir_cache = Path(dir_cache)
        self.meta = json.loads((dir_cache / "meta.json").read_text())
        assert self.meta["completo"], "El caché no está completo"
        self.bandas = self.meta["bandas"]
        self.img = np.load(dir_cache / "imagenes.npy", mmap_mode=None if en_memoria else "r")
        self.instantes = pd.read_parquet(dir_cache / "instantes.parquet")
        self.estaciones = pd.read_parquet(dir_cache / "estaciones.parquet")
        self.fila_de = pd.Series(self.instantes["fila"].values, index=self.instantes["timestamp"])
        cod = self.meta["codificacion"]
        self.scale = np.array([cod[b][0] for b in self.bandas], dtype=np.float32)
        self.offset = np.array([cod[b][1] for b in self.bandas], dtype=np.float32)
        z = zscore.set_index("banda").loc[self.bandas]
        self.media, self.desvio = z["mean"].values, z["std"].values
        self.pos = {int(r.codigoestacion): (int(r.fila_c), int(r.col_c))
                    for r in self.estaciones.itertuples()}

    def parche_crudo(self, fila: int, codigoestacion: int, half_size=HALF_SIZE) -> np.ndarray:
        fc, cc = self.pos[int(codigoestacion)]
        return self.img[fila, :, fc - half_size:fc + half_size, cc - half_size:cc + half_size]

    def parche(self, timestamp, codigoestacion: int, half_size=HALF_SIZE) -> np.ndarray:
        """Equivalente a extraer_parche de 10: (B, 32, 32) z-score, float64."""
        u16 = self.parche_crudo(int(self.fila_de[timestamp]), codigoestacion, half_size)
        return normalizar(decodificar(u16, self.scale, self.offset), self.media, self.desvio)
