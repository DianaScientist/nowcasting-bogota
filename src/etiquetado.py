"""
Etiqueta de lluvia fuerte en ventana (t+1h, t+3h] a partir de los registros sub-horarios de IDEAM.

Usado por 05-ground_truth_labels.ipynb (construcción de las etiquetas) y por
11-dataset_training.ipynb (figuras de los ejemplos del cruce con el inventario satelital).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TZ = "America/Bogota"

# --- Parámetros de la etiqueta ---
PASO_MIN    = 10     # grilla de 10 min (cadencia GOES)
INICIO_H    = 1      # la ventana empieza en t+1h (exclusivo)
FIN_H       = 3      # la ventana termina en t+3h (inclusivo)
ACUM_MIN    = 60     # acumulación con que se definió el umbral (04)
BLOQUE_MIN  = 30     # bloques de cobertura dentro de la ventana
PISO_HUMEDO = 0.2    # mm/h, piso de hora húmeda (04)

# Pasos relativos a t (bin k = registros en (t+(k-1)·10, t+k·10])
PASO_INI_VENTANA   = INICIO_H * 60 // PASO_MIN + 1               # 7  → primer bin de (t+1h, t+3h]
PASO_FIN_VENTANA   = FIN_H * 60 // PASO_MIN                      # 18 → último bin
PASO_INI_EVAL      = (INICIO_H * 60 + ACUM_MIN) // PASO_MIN + 1  # 13 → τ ∈ (t+2h, t+3h]
PASO_FIN_INMINENTE = (INICIO_H * 60 + ACUM_MIN) // PASO_MIN      # 12 → τ ∈ (t, t+2h]
PASOS_BLOQUE       = BLOQUE_MIN // PASO_MIN                      # 3
LARGO              = PASO_FIN_VENTANA + 1                        # 19 bins por t (t incluido)
N_BLOQUES          = (PASO_FIN_VENTANA - PASO_INI_VENTANA + 1) // PASOS_BLOQUE
assert N_BLOQUES == 4 and PASO_FIN_VENTANA - PASO_INI_EVAL + 1 == 6

COLUMNAS_ETIQ = ["timestamp", "etiqueta", "en_curso", "inminente",
                 "cobertura", "sin_dato_previo", "hueco_primera_hora", "int_t"]


def serie_estacion(csv_path: Path) -> pd.DataFrame:
    """Registro sub-horario en UTC con la acumulación de los 60 min que terminan en cada registro."""
    df = (pd.read_csv(csv_path, usecols=["fechaobservacion", "valorobservado"],
                      parse_dates=["fechaobservacion"])
            .dropna(subset=["valorobservado"])
            .sort_values("fechaobservacion")
            .set_index("fechaobservacion"))
    df.index = df.index.tz_localize(TZ).tz_convert("UTC")
    df["intensidad_mm_h"] = df["valorobservado"].rolling(f"{ACUM_MIN}min").sum()   # (τ-60, τ]
    return df


def bins_estacion(csv_path: Path) -> pd.DataFrame:
    """Grilla de 10 min en UTC (bins cerrados a la derecha) con presencia de registros e intensidad horaria máxima."""
    df = serie_estacion(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["tiene_dato", "int_max"])
    r = dict(rule=f"{PASO_MIN}min", closed="right", label="right")
    conteo = df["valorobservado"].resample(**r).count()
    grilla = pd.date_range(conteo.index.min(), conteo.index.max(), freq=f"{PASO_MIN}min")
    return pd.DataFrame({"tiene_dato": conteo.reindex(grilla, fill_value=0).values > 0,
                         "int_max": df["intensidad_mm_h"].resample(**r).max()
                                      .reindex(grilla).fillna(0).values},
                        index=grilla)


def etiquetar_estacion(csv_path: Path, umbral_mm_h: float) -> pd.DataFrame:
    """
    Etiqueta cada t de la grilla de 10 min (UTC):
      etiqueta  = 1   si algún lapso de 60 min dentro de (t+1h, t+3h] acumula ≥ umbral
                      (intensidad(τ) ≥ umbral para algún τ ∈ (t+2h, t+3h]), aunque falten registros;
                  0   si ninguno lo hace y los cuatro bloques de 30 min de la ventana tienen registros;
                  NaN si ninguno lo hace y algún bloque carece de registros.
      en_curso  = True si la hora que termina en t ya acumula ≥ umbral (τ ∈ (t-10 min, t]).
      inminente = True si etiqueta = 0, no hay lluvia en curso y algún lapso de 60 min
                  que termina en (t, t+2h] acumula ≥ umbral.
    Diagnóstico: cobertura (cuatro bloques con registros), sin_dato_previo (sin registros en (t-30 min, t]),
    hueco_primera_hora (algún bloque de 30 min de (t, t+1h] sin registros), int_t (intensidad horaria en t).
    """
    b = bins_estacion(csv_path)
    if len(b) < LARGO:
        return pd.DataFrame(columns=COLUMNAS_ETIQ)

    td = b["tiene_dato"].values
    hd = sliding_window_view(td, LARGO)                          # columna k = bin t + k·10 min
    ev = sliding_window_view(b["int_max"].values >= umbral_mm_h, LARGO)
    n  = len(hd)

    cobertura = np.column_stack([hd[:, i:i + PASOS_BLOQUE].any(axis=1)
                                 for i in range(PASO_INI_VENTANA, PASO_FIN_VENTANA + 1, PASOS_BLOQUE)]).all(axis=1)
    evento    = ev[:, PASO_INI_EVAL:PASO_FIN_VENTANA + 1].any(axis=1)
    en_curso  = ev[:, 0]                                         # bin de t: (t-10 min, t]
    etiqueta  = np.where(evento, 1.0, np.where(cobertura, 0.0, np.nan))
    inminente = (etiqueta == 0) & ~en_curso & ev[:, 1:PASO_FIN_INMINENTE + 1].any(axis=1)

    previo       = pd.Series(td).rolling(PASOS_BLOQUE, min_periods=1).max().astype(bool).values[:n]
    primera_hora = (hd[:, 1:1 + PASOS_BLOQUE].any(axis=1) &
                    hd[:, 1 + PASOS_BLOQUE:1 + 2 * PASOS_BLOQUE].any(axis=1))

    return pd.DataFrame({"timestamp": b.index[:n], "etiqueta": etiqueta,
                         "en_curso": en_curso, "inminente": inminente, "cobertura": cobertura,
                         "sin_dato_previo": ~previo, "hueco_primera_hora": ~primera_hora,
                         "int_t": b["int_max"].values[:n]})
