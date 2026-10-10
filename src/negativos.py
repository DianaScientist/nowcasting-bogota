"""
Criterios de negativos difíciles para el muestreo del cargador (S3 del plan de modelado).

Usado por 14-negativos_dificiles.ipynb. No modifica las etiquetas de 05: calcula atributos
auxiliares de cada par (estación, t) que el cargador usa para decidir a qué bolsa de
negativos pertenece un par.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from etiquetado import LARGO, PASO_INI_EVAL, PASO_FIN_VENTANA, bins_estacion

COLUMNAS_INT = ["timestamp", "int_max_ventana"]


def intensidad_ventana_estacion(csv_path: Path) -> pd.DataFrame:
    """
    Intensidad horaria máxima en la ventana de la etiqueta, para cada t de la grilla de 10 min (UTC).

    int_max_ventana = max de la intensidad rolling 60 min en τ ∈ (t+2h, t+3h], es decir, sobre
    las horas completas contenidas en (t+1h, t+3h]. Usa la misma grilla (bins_estacion), los
    mismos bins (PASO_INI_EVAL..PASO_FIN_VENTANA) y el mismo largo que etiquetar_estacion, de
    modo que, para cada t:
        etiqueta == 1  ⇔  int_max_ventana ≥ umbral
    (los bins sin registros valen 0, igual que en etiquetar_estacion).
    """
    b = bins_estacion(csv_path)
    if len(b) < LARGO:
        return pd.DataFrame(columns=COLUMNAS_INT)

    im = sliding_window_view(b["int_max"].values, LARGO)        # columna k = bin t + k·10 min
    return pd.DataFrame({"timestamp": b.index[:len(im)],
                         "int_max_ventana": im[:, PASO_INI_EVAL:PASO_FIN_VENTANA + 1].max(axis=1)})


def percentil_banda(cache, banda: str = "CMI_C13", q: float = 10, lote: int = 500,
                    progreso=None) -> pd.DataFrame:
    """
    Percentil q (en K) de una banda sobre el parche 32×32 de cada estación, para cada instante del caché.

    Calcula el percentil sobre los enteros empaquetados y lo convierte a K con scale/offset de la
    banda. Como la conversión es afín con escala positiva, el resultado coincide con el percentil
    de los valores en K (interpolación lineal, igual que np.nanpercentile en 12-dataset_eda).
    Devuelve una fila por (instante, estación): timestamp, codigoestacion (texto de 10 dígitos), valor.
    """
    k = cache.bandas.index(banda)
    h = 16
    cods = list(cache.pos)
    pos = [cache.pos[c] for c in cods]
    n = cache.img.shape[0]
    salida = np.empty((n, len(cods)), dtype=np.float64)
    pasos = range(0, n, lote)
    for r0 in (progreso(pasos) if progreso else pasos):
        bloque = np.asarray(cache.img[r0:r0 + lote, k])                       # (B, H, W) uint16
        parches = np.stack([bloque[:, fc - h:fc + h, cc - h:cc + h].reshape(len(bloque), -1)
                            for fc, cc in pos], axis=1)                        # (B, E, 1024)
        salida[r0:r0 + len(bloque)] = np.percentile(parches, q, axis=2)
    salida = salida * float(cache.scale[k]) + float(cache.offset[k])
    ins = cache.instantes.sort_values("fila")
    assert (ins["fila"].to_numpy() == np.arange(n)).all()
    ts = pd.DatetimeIndex(ins["timestamp"])
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")    # el caché guarda UTC sin zona
    return pd.DataFrame({"timestamp": ts.repeat(len(cods)),
                         "codigoestacion": np.tile([str(c).zfill(10) for c in cods], n),
                         "valor": salida.ravel()})


BANDAS_INT = ["sin lluvia", "< 50 %", "50–80 %", "80–100 %", "≥ umbral"]


def banda_intensidad(razon: pd.Series) -> pd.Series:
    """
    Clasifica la razón int_max_ventana / umbral en las bandas de BANDAS_INT
    (sin lluvia = razón 0). Solo la banda 50–80 % se usa como criterio de negativo difícil;
    la de 80–100 % queda dentro de la incertidumbre del P95 y se reserva para el análisis
    de errores (S12).
    """
    r = razon.to_numpy()
    return pd.Categorical(np.select([r == 0, r < 0.5, r < 0.8, r < 1.0], BANDAS_INT[:4], BANDAS_INT[4]),
                          categories=BANDAS_INT, ordered=True)
