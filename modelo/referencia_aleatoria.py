"""
Análisis de sensibilidad posterior a la apertura de prueba (revisión del documento, 30-oct-2026).

No reentrena ni vuelve a abrir prueba: usa las predicciones y umbrales de decisión ya fijados en
S11 (`modelo/salidas_s11/`) y los episodios de prueba de S12 (`modelo/salidas_21/`).

1. Referencia aleatoria por evento. Para cada modelo con L = 2 se simulan alertas aleatorias con
   el mismo número de alertas que el modelo, de tres formas: uniforme, con la misma tasa por hora
   local y con la misma tasa por estación y hora. Se compara la POD por evento y la anticipación
   mediana del modelo con la distribución de las simulaciones.
2. Tasa base de la condición de vecindad usada en el análisis de falsas alarmas (5.6.2).

Asignación de pares a episodios: cada par positivo va al primer episodio de su estación con
fin > t + 1 h e inicio <= t + 3 h. Verificación: reproduce los avisados de la Tabla D.5
(umbral en C13 141; logística L2 357 frente a 358).

Uso, desde la raíz del repo:  python modelo/referencia_aleatoria.py
Salidas: modelo/salidas_21/referencia_aleatoria.csv y modelo/salidas_21/tasa_base_vecindad.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
S11 = RAIZ / "modelo" / "salidas_s11"
S21 = RAIZ / "modelo" / "salidas_21"
MODELOS = ["umbral_c13", "logistica_L2",
           "cnn_plana_L2_s42", "cnn_plana_L2_s43", "cnn_plana_L2_s44",
           "resnet_L2_s42", "resnet_L2_s43", "resnet_L2_s44"]
N_SIM = 500
SEMILLA = 7
ZONA = "America/Bogota"
UTC_NS = "datetime64[ns, UTC]"
H = pd.Timedelta(hours=1)
GRANDE = np.iinfo(np.int64).max


def cargar():
    ep = pd.read_csv(S21 / "episodios_prueba_detalle.csv", dtype={"codigoestacion": str})
    ep["codigoestacion"] = ep["codigoestacion"].str.zfill(10)
    for c in ("inicio", "fin"):
        ep[c] = pd.to_datetime(ep[c], utc=True).astype(UTC_NS)
    n_ep = int(ep["evaluable"].sum())
    base = pd.read_parquet(S11 / f"pred_test_{MODELOS[0]}.parquet")[["codigoestacion", "timestamp", "etiqueta"]]
    base["codigoestacion"] = base["codigoestacion"].astype(str).str.zfill(10)
    base["timestamp"] = base["timestamp"].astype(UTC_NS)
    umbrales = pd.read_csv(S11 / "umbrales.csv", index_col=0)["umbral"]
    return ep, n_ep, base, umbrales


def asignar(base, ep):
    pos = base[base["etiqueta"] == 1].copy()
    pos["i"] = pos.index
    pos["_k"] = pos["timestamp"] + H
    m = pd.merge_asof(pos.sort_values("_k"), ep.sort_values("fin")[["codigoestacion", "inicio", "fin", "id_episodio"]],
                      left_on="_k", right_on="fin", by="codigoestacion",
                      direction="forward", allow_exact_matches=False)
    m = m[m["inicio"] <= m["timestamp"] + 3 * H]
    assert len(m) == len(pos), "hay pares positivos sin episodio"
    return m[m["timestamp"] < m["inicio"]]          # solo pares que pueden dar una alerta válida


def referencia_aleatoria(base, ep, n_ep, umbrales):
    pos = asignar(base, ep)
    hora = base["timestamp"].dt.tz_convert(ZONA).dt.hour.to_numpy()
    est = base["codigoestacion"].astype("category").cat.codes.to_numpy()
    idx = pos["i"].to_numpy()
    ids, inv = np.unique(pos["id_episodio"].to_numpy(), return_inverse=True)
    t_pos = pos["timestamp"].astype("int64").to_numpy()
    inicio = pos.groupby("id_episodio")["inicio"].first().loc[ids].astype("int64").to_numpy()

    def resumen(alerta):
        avisado = np.zeros(len(ids), bool)
        np.logical_or.at(avisado, inv, alerta)
        primera = np.full(len(ids), GRANDE)
        np.minimum.at(primera, inv, np.where(alerta, t_pos, GRANDE))
        antic = (inicio[avisado] - primera[avisado]) / 3.6e12
        return avisado.sum(), (np.median(antic) if avisado.any() else np.nan)

    rng = np.random.default_rng(SEMILLA)
    filas = []
    for m in MODELOS:
        alerta = pd.read_parquet(S11 / f"pred_test_{m}.parquet")["p"].to_numpy() >= umbrales[m]
        n, med = resumen(alerta[idx])
        tasa_h = pd.Series(alerta).groupby(hora).mean().reindex(range(24), fill_value=0).to_numpy()
        tasa_eh = pd.Series(alerta).groupby([est, hora]).mean()
        tasa_eh = tasa_eh.reindex(pd.MultiIndex.from_arrays([est[idx], hora[idx]])).fillna(0).to_numpy()
        fila = dict(modelo=m, tasa_alerta=alerta.mean(), avisados=n, pod_evento=n / n_ep, anticipacion_mediana_h=med)
        for nombre, prob in (("uniforme", np.full(len(idx), alerta.mean())),
                             ("misma_hora", tasa_h[hora[idx]]),
                             ("misma_hora_estacion", tasa_eh)):
            sim = np.array([resumen(rng.random(len(idx)) < prob) for _ in range(N_SIM)])
            fila |= {f"{nombre}_pod": sim[:, 0].mean() / n_ep,
                     f"{nombre}_pod_ic_inf": np.percentile(sim[:, 0], 2.5) / n_ep,
                     f"{nombre}_pod_ic_sup": np.percentile(sim[:, 0], 97.5) / n_ep,
                     f"{nombre}_frac_sim_ge_modelo": (sim[:, 0] >= n).mean(),
                     f"{nombre}_anticipacion_mediana_h": np.nanmedian(sim[:, 1])}
        filas.append(fila)
    return pd.DataFrame(filas)


def tasa_base_vecindad():
    fa = pd.read_csv(S21 / "falsas_alarmas_por_grupo.csv")
    v = fa[fa["criterio"] == "vecindad"].copy()
    v["frac_negativos"] = v["n_negativos"] / v.groupby("modelo")["n_negativos"].transform("sum")
    v["frac_falsas_alarmas"] = v["n_alertas"] / v.groupby("modelo")["n_alertas"].transform("sum")
    v["razon_vs_base"] = v["frac_falsas_alarmas"] / v["frac_negativos"]
    return v[["modelo", "grupo", "n_negativos", "n_alertas", "tasa_alerta",
              "frac_negativos", "frac_falsas_alarmas", "razon_vs_base"]]


if __name__ == "__main__":
    ep, n_ep, base, umbrales = cargar()
    ra = referencia_aleatoria(base, ep, n_ep, umbrales)
    ra.to_csv(S21 / "referencia_aleatoria.csv", index=False)
    tb = tasa_base_vecindad()
    tb.to_csv(S21 / "tasa_base_vecindad.csv", index=False)
    pd.set_option("display.width", 200)
    print(ra.round(3).T.to_string())
    print(tb.round(3).to_string())
