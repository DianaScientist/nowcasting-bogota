"""
Peldaños 1 y 2 de la escalera de complejidad (S5 del plan de modelado).

    Peldaño 1  umbral sobre el p10 de C13: regresión logística con esa sola variable
    Peldaño 2  regresión logística sobre estadísticas de parche por banda e instante
               (p10, p50, p90 y desvío estándar de la temperatura de brillo), L = 2, 3, 4

Las estadísticas dependen solo de (imagen, estación), no del par, así que se calculan una vez
para todo el caché y se guardan en data/cache/estadisticas_parche.npy; cada par las toma de
las filas de su historia.

Entrenamiento: todos los pares de train, ponderados para reproducir la proporción 1:2:2
(positivo : difícil : aleatorio) con que se muestrean las épocas de las redes. Equivale a
muestrear infinitas épocas: misma escala de probabilidades que las redes, sin descartar
negativos. Decisiones de S5, 12-oct-2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HALF = 16                                  # parche 32 × 32, igual que el cargador
ESTADISTICAS = ("p10", "p50", "p90", "desvio")
PERCENTILES = (10, 50, 90)
GRILLA_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
ZONA_LOCAL = "America/Bogota"


# --- Estadísticas de parche por (imagen, estación) ----------------------------------
def calcular_estadisticas(cache, lote: int = 100, progreso=None) -> tuple[np.ndarray, list[str]]:
    """
    Arreglo (n_imagenes, n_estaciones, n_bandas, 4) float32 en K: p10, p50, p90 y desvío
    estándar de cada banda sobre el parche 32×32 de cada estación.

    Los percentiles se calculan sobre los enteros del caché y se convierten a K con
    scale/offset (afín con escala positiva: igual al percentil en K, como `percentil_banda`
    de src/negativos.py). Devuelve también los códigos de estación en el orden del eje 1.
    """
    cods = sorted(cache.pos)
    pos = [cache.pos[c] for c in cods]
    n, nb = cache.img.shape[0], cache.img.shape[1]
    sc = cache.scale.astype(np.float64).reshape(1, 1, nb)
    of = cache.offset.astype(np.float64).reshape(1, 1, nb)
    salida = np.empty((n, len(cods), nb, len(ESTADISTICAS)), dtype=np.float32)
    pasos = range(0, n, lote)
    for r0 in (progreso(pasos) if progreso else pasos):
        bloque = np.asarray(cache.img[r0:r0 + lote])                                 # (B, nb, H, W)
        parches = np.stack([bloque[:, :, fc - HALF:fc + HALF, cc - HALF:cc + HALF]
                            .reshape(len(bloque), nb, -1) for fc, cc in pos], axis=1)  # (B, E, nb, 1024)
        q = np.percentile(parches, PERCENTILES, axis=3)                              # (3, B, E, nb)
        salida[r0:r0 + len(bloque), :, :, :3] = np.moveaxis(q * sc + of, 0, -1)
        salida[r0:r0 + len(bloque), :, :, 3] = parches.astype(np.float32).std(axis=3) * sc
    return salida, [str(c).zfill(10) for c in cods]


def guardar_estadisticas(ruta: Path, est: np.ndarray, codigos: list[str], bandas: list[str]):
    ruta = Path(ruta)
    np.save(ruta.with_suffix(".npy"), est)
    ruta.with_suffix(".json").write_text(json.dumps(
        {"forma": list(est.shape), "estaciones": codigos, "bandas": bandas,
         "estadisticas": list(ESTADISTICAS), "unidad": "K", "parche": 2 * HALF}, indent=2))


def leer_estadisticas(ruta: Path) -> tuple[np.ndarray, dict]:
    ruta = Path(ruta)
    meta = json.loads(ruta.with_suffix(".json").read_text())
    return np.load(ruta.with_suffix(".npy")), meta


# --- Variables de cada par ------------------------------------------------------------
def nombres_variables(L: int, bandas: list[str]) -> list[str]:
    """Orden: instante de la historia (más antiguo → t), banda, estadística."""
    return [f"{b}_{s}_lag{L - 1 - j}" for j in range(L) for b in bandas for s in ESTADISTICAS]


def variables_pares(pares: pd.DataFrame, est: np.ndarray, codigos: list[str], L: int) -> np.ndarray:
    """(N, L · n_bandas · 4) a partir de las filas fila_0 … fila_{L−1} de `preparar_pares`."""
    col = pd.Series(np.arange(len(codigos)), index=codigos)
    e = col.reindex(pares["codigoestacion"].astype(str).str.zfill(10)).to_numpy()
    assert not np.isnan(e.astype(float)).any(), "Estaciones del par sin estadísticas"
    filas = pares[[f"fila_{j}" for j in range(L)]].to_numpy(np.int64)               # (N, L)
    x = est[filas, e[:, None]]                                                         # (N, L, nb, 4)
    return x.reshape(len(pares), -1)


def p10_c13(pares: pd.DataFrame, est: np.ndarray, codigos: list[str], bandas: list[str], L: int) -> np.ndarray:
    """p10 de C13 en t (variable del peldaño 1)."""
    x = variables_pares(pares, est, codigos, L).reshape(len(pares), L, len(bandas), len(ESTADISTICAS))
    return x[:, -1, bandas.index("CMI_C13"), ESTADISTICAS.index("p10")]


def variables_hora(timestamps) -> np.ndarray:
    """Seno y coseno de la hora local de t (variante de diagnóstico, fuera de la escalera)."""
    h = pd.DatetimeIndex(timestamps).tz_convert(ZONA_LOCAL)
    a = 2 * np.pi * (h.hour + h.minute / 60) / 24
    return np.column_stack([np.sin(a), np.cos(a)])


# --- Pesos de entrenamiento -----------------------------------------------------------
def pesos_muestreo(bolsa: pd.Series, n_dificiles: float = 2, n_aleatorios: float = 2) -> np.ndarray:
    """
    Peso de cada par para que las bolsas sumen positivos : difíciles : aleatorios =
    1 : n_dificiles : n_aleatorios, como en el muestreo por época de las redes.
    """
    b = bolsa.to_numpy()
    n = {k: int((b == k).sum()) for k in ("positivo", "dificil", "aleatorio")}
    assert min(n.values()) > 0, f"Bolsa vacía: {n}"
    P = n["positivo"]
    w = np.select([b == "positivo", b == "dificil"],
                  [1.0, n_dificiles * P / n["dificil"]], n_aleatorios * P / n["aleatorio"])
    return w.astype(np.float64)


# --- Modelos --------------------------------------------------------------------------
def ajustar(x: np.ndarray, y: np.ndarray, w: np.ndarray, C: float | None):
    """Estandarización ponderada + regresión logística L2 (C = None: sin regularización efectiva)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    lr = LogisticRegression(C=C if C is not None else 1e10, max_iter=5000)   # C enorme = sin regularización
    modelo = make_pipeline(StandardScaler(), lr)
    modelo.fit(x, y, standardscaler__sample_weight=w, logisticregression__sample_weight=w)
    return modelo


def predecir(modelo, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return modelo.predict_proba(x)[:, 1]
