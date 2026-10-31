"""
Análisis de sensibilidad posteriores a la apertura de prueba (revisión del documento, 30-oct-2026).

No reentrena ni vuelve a abrir prueba.

1. Estabilidad del umbral de decisión (diagnóstico de la opción B, decisión 4 de S4a): en
   validación, umbral que maximiza el CSI re-elegido en cada réplica del bootstrap por días
   pluviométricos (B = 2.000, semilla 42). Se reporta su mediana, su intervalo del 95 %, la
   fracción de réplicas a ±0,05 del umbral fijado y el CSI de validación en los extremos del
   intervalo. Las regresiones logísticas se omiten: las predicciones de validación guardadas en
   `salidas_17` son anteriores al cambio de C (S5) y no reproducen los umbrales de `umbrales.csv`.
2. Pares de tormenta inminente: cota del efecto de reincorporarlos a la prueba como negativos.
   Los modelos no tienen predicciones para esos pares (están fuera del índice), así que se
   acotan la FAR y el CSI suponiendo que el modelo alerta en ellos (a) con su POD por par y
   (b) en todos.

Uso, desde la raíz del repo:  python modelo/sensibilidad_umbral_inminentes.py
Salidas: modelo/salidas_21/estabilidad_umbral.csv y modelo/salidas_21/sensibilidad_inminentes.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import evaluacion as ev  # noqa: E402

M = RAIZ / "modelo"
S11, S21 = M / "salidas_s11", M / "salidas_21"
ETIQUETAS = RAIZ / "data_ingestion" / "ground_truth" / "etiquetas_ground_truth.parquet"
B = 2000
MODELOS_L2 = ["umbral_c13", "logistica_L2", "cnn_plana_L2_s42", "cnn_plana_L2_s43", "cnn_plana_L2_s44",
              "resnet_L2_s42", "resnet_L2_s43", "resnet_L2_s44"]


def ruta_val(m):
    if m == "umbral_c13":
        return M / "salidas_17" / "pred_val_peldano1_umbral_c13.parquet"
    return M / "salidas_s9" / f"pred_val_{m}.parquet"


def estabilidad_umbral(umbrales):
    filas = []
    for m in umbrales.index:
        if m.startswith("logistica"):
            continue
        d = pd.read_parquet(ruta_val(m))
        y, p = d["etiqueta"].to_numpy() == 1, d["p"].to_numpy()
        elegido = ev.elegir_umbral(y, p)
        assert abs(elegido["umbral"] - umbrales[m]) < 1e-9, f"{m}: no reproduce el umbral fijado"
        bloques, tablas = ev.tablas_por_bloque_y_umbral(d["timestamp"], y, p)
        u = ev.estabilidad_umbral(tablas, ev.pesos_bootstrap(len(bloques), B=B, semilla=42))
        inf, sup = np.nanpercentile(u, [2.5, 97.5])
        csi = elegido["curva"].set_index("umbral")["csi"]
        filas.append(dict(modelo=m, umbral=umbrales[m], mediana_replicas=np.nanmedian(u),
                          ic_inf=inf, ic_sup=sup, frac_a_005=np.mean(np.abs(u - umbrales[m]) <= 0.05 + 1e-9),
                          csi_val=elegido["csi_max"], csi_val_en_ic_inf=csi.loc[round(inf, 2)],
                          csi_val_en_ic_sup=csi.loc[round(sup, 2)]))
    return pd.DataFrame(filas)


def sensibilidad_inminentes():
    base = pd.read_parquet(S11 / "pred_test_umbral_c13.parquet")[["codigoestacion", "timestamp"]]
    base["codigoestacion"] = base["codigoestacion"].astype(str).str.zfill(10)
    e = pd.read_parquet(ETIQUETAS)
    e["codigoestacion"] = e["codigoestacion"].astype(str).str.zfill(10)
    inm = e[(e["timestamp"].dt.year == 2025) & e["codigoestacion"].isin(set(base["codigoestacion"]))
            & e["inminente"] & (e["etiqueta"] == 0) & e["timestamp"].isin(set(base["timestamp"]))]
    n = len(inm)
    r = pd.read_csv(S11 / "resultados_por_modelo.csv", index_col=0)
    filas = []
    for m in MODELOS_L2:
        a, b, c = r.loc[m, ["aciertos", "falsas_alarmas", "omisiones"]]
        for escenario, k in (("actual", 0), ("alerta_con_su_pod", a / (a + c) * n), ("alerta_en_todos", n)):
            filas.append(dict(modelo=m, escenario=escenario, pares_inminentes=n, falsas_alarmas_extra=round(k),
                              csi=a / (a + b + k + c), far=(b + k) / (a + b + k)))
    return pd.DataFrame(filas)


if __name__ == "__main__":
    umbrales = pd.read_csv(S11 / "umbrales.csv", index_col=0)["umbral"]
    eu = estabilidad_umbral(umbrales)
    eu.to_csv(S21 / "estabilidad_umbral.csv", index=False)
    si = sensibilidad_inminentes()
    si.to_csv(S21 / "sensibilidad_inminentes.csv", index=False)
    pd.set_option("display.width", 200)
    print(eu.round(3).to_string())
    print(si.round(4).to_string())
