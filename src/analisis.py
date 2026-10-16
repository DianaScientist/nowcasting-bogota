"""
Análisis de errores y casos sobre las predicciones de prueba (S12 del plan de modelado).

Trabaja en local con lo que dejó S11 (`modelo/salidas_s11/`): las predicciones de prueba de los
22 modelos y sus umbrales de decisión fijados en validación. No reentrena nada ni vuelve a abrir
prueba: es análisis descriptivo de resultados ya obtenidos.

Alcance acordado (16-oct-2026):
    - Métricas por evento y estratos (noche, terciles de umbral, temporada, nube fría) para las
      10 configuraciones (umbral, 3 logísticas, 6 redes por semilla y media ± dispersión).
    - Por estación (mínimo 10 episodios) y casos de estudio solo para tres modelos, uno por
      peldaño: umbral C13, logística L2 y CNN plana L2 (semilla 42, la que reproduce la búsqueda).
    - Casos: acierto diurno del 25-abr-2025 (fijo); omisión nocturna y falsa alarma elegidas por
      regla escrita antes de mirar (ver `elegir_casos`).
    - Errores por banda de intensidad en la ventana (incluida la banda 80–100 % del umbral) y por
      vecindad de tormenta (near-miss).
    - Sin importancia por permutación; en su lugar, coeficientes estandarizados de la logística.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import evaluacion as ev

REDES = ("cnn_plana", "resnet")
LS = (2, 3, 4)
SEMILLAS = (42, 43, 44)
MODELOS_DETALLE = {"umbral_c13": "1 · umbral C13", "logistica_L2": "2 · logística L2",
                   "cnn_plana_L2_s42": "3 · CNN plana L2"}
METRICAS_EVENTO = ("pod_evento", "far_evento", "frac_anticipacion_1h")
METRICAS_PARES = ("pod", "far", "csi", "ets")


def configuraciones() -> dict:
    """Configuración -> modelos (las redes, sus tres semillas). Igual que en S11."""
    cfg = {"umbral_c13": ["umbral_c13"]}
    cfg |= {f"logistica_L{L}": [f"logistica_L{L}"] for L in LS}
    cfg |= {f"{m}_L{L}": [f"{m}_L{L}_s{s}" for s in SEMILLAS] for m in REDES for L in LS}
    return cfg


# --- Carga -----------------------------------------------------------------------
def cargar_predicciones(dir_s11: Path, modelos) -> tuple[pd.DataFrame, dict]:
    """Pares de prueba (estación, t, etiqueta) y probabilidades de cada modelo, verificando que
    todos los archivos tienen los mismos pares en el mismo orden."""
    base, p = None, {}
    for m in modelos:
        d = pd.read_parquet(Path(dir_s11) / f"pred_test_{m}.parquet")
        if base is None:
            base = d[["codigoestacion", "timestamp", "etiqueta"]].copy()
            base["codigoestacion"] = base["codigoestacion"].astype(str).str.zfill(10)
        else:
            assert len(d) == len(base) and (d["timestamp"].to_numpy() == base["timestamp"].to_numpy()).all() \
                and (d["etiqueta"].to_numpy() == base["etiqueta"].to_numpy()).all(), f"{m}: pares distintos"
        p[m] = d["p"].to_numpy()
    ts = pd.DatetimeIndex(base["timestamp"])
    assert ts.tz is not None, "timestamps sin zona horaria"
    return base.reset_index(drop=True), p


# --- Eventos y estratos ----------------------------------------------------------
def particiones(pares: pd.DataFrame, ep: pd.DataFrame, av: pd.DataFrame, grupo_umbral: pd.Series,
                p10_par: pd.Series) -> dict:
    """Estratos de S4b alineados con pares, episodios y avisos (mismo código que 17-referencias)."""
    clave = lambda e, t: pd.MultiIndex.from_arrays([e, t])
    return {
        "total": (np.full(len(pares), "total"), np.full(len(ep), "total"), np.full(len(av), "total")),
        "noche": (ev.es_noche(pares.timestamp), ev.es_noche(ep.inicio), ev.es_noche(av.t_primera)),
        "umbral": (pares.codigoestacion.map(grupo_umbral).to_numpy(), ep.codigoestacion.map(grupo_umbral).to_numpy(),
                   av.codigoestacion.map(grupo_umbral).to_numpy()),
        "temporada": (ev.temporada(pares.timestamp), ev.temporada(ep.inicio), ev.temporada(av.t_primera)),
        "nube_fria": (ev.nube_fria(pares.p10_c13),
                      ev.nube_fria(p10_par.reindex(clave(ep.codigoestacion, ep.t_primer_par)).to_numpy()),
                      ev.nube_fria(p10_par.reindex(clave(av.codigoestacion, av.t_primera)).to_numpy())),
    }


def evaluar_modelo(pares: pd.DataFrame, p: np.ndarray, umbral: float, episodios: pd.DataFrame,
                   grupo_umbral: pd.Series, p10_par: pd.Series, pesos: np.ndarray, bloques: np.ndarray):
    """Episodios, avisos, anticipación y tabla de estratos (con la partición 'total') de un modelo."""
    d = pares.assign(y=pares["etiqueta"].to_numpy(), p=p)
    ep, av = ev.eventos(d, episodios, umbral)
    args = dict(pares=d, ep=ep, avisos=av, umbral=umbral, pesos=pesos, bloques=bloques)
    filas = [ev.evaluar_por_estrato(**args, estrato_pares=sp, estrato_ep=se, estrato_avisos=sa).assign(particion=k)
             for k, (sp, se, sa) in particiones(d, ep, av, grupo_umbral, p10_par).items()]
    return ep, av, ev.resumen_anticipacion(ep), a_numerico(pd.concat(filas, ignore_index=True))


def a_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """Las métricas por evento salen como arreglos de dimensión 0: se pasan a float."""
    for c in df.columns:
        if df[c].dtype == object and c not in ("estrato", "particion", "modelo", "codigoestacion"):
            try:
                df[c] = df[c].astype(float)
            except (TypeError, ValueError):
                pass
    return df


def por_configuracion(estratos: pd.DataFrame, cfgs: dict, metricas=METRICAS_EVENTO + METRICAS_PARES) -> pd.DataFrame:
    """Media ± dispersión entre semillas por configuración, partición y estrato.
    `estratos` lleva una columna 'modelo'. Los conteos se informan del primer modelo de la
    configuración (son iguales entre semillas para tormentas y episodios)."""
    filas = []
    for cfg, mods in cfgs.items():
        sub = estratos[estratos.modelo.isin(mods)]
        for (part, est), g in sub.groupby(["particion", "estrato"], sort=False):
            fila = {"configuracion": cfg, "particion": part, "estrato": est, "n_modelos": len(g),
                    "n_tormentas": int(g.n_tormentas.iloc[0]), "n_episodios": int(g.n_episodios.iloc[0])}
            for k in metricas:
                v = g[k].to_numpy(dtype=float)
                fila[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                fila[f"{k}_de_semillas"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 and np.isfinite(v).sum() > 1 else np.nan
                if len(mods) == 1:
                    fila[f"{k}_ic_inf"], fila[f"{k}_ic_sup"] = g[f"{k}_ic_inf"].iloc[0], g[f"{k}_ic_sup"].iloc[0]
            filas.append(fila)
    return pd.DataFrame(filas)


# --- Errores por intensidad en la ventana y por vecindad -------------------------
def tasa_alerta_negativos(pares: pd.DataFrame, p: np.ndarray, umbral: float) -> pd.DataFrame:
    """Fracción de negativos que reciben alerta, por banda de intensidad en la ventana (razón
    int_max_ventana / umbral) y por vecindad de tormenta (otra estación positiva o activa en t)."""
    neg = pares["etiqueta"].to_numpy() == 0
    alerta = (p >= umbral)[neg]
    d = pares.loc[neg]
    import negativos as ng
    banda = pd.Series(ng.banda_intensidad(d["razon"]), index=d.index).astype(str)
    vecindad = np.where(d["n_est_pos"].to_numpy() >= 1, "otra estación positiva en t",
                        np.where(d["n_est_activas"].to_numpy() >= 1, "otra estación con lluvia en curso o inminente",
                                 "sin tormenta en la red"))
    filas = []
    for nombre, grupos in (("intensidad en la ventana", banda.to_numpy()), ("vecindad", vecindad)):
        for g in pd.unique(grupos):
            m = grupos == g
            filas.append({"criterio": nombre, "grupo": g, "n_negativos": int(m.sum()),
                          "n_alertas": int(alerta[m].sum()), "tasa_alerta": float(alerta[m].mean())})
    return pd.DataFrame(filas)


# --- Casos de estudio --------------------------------------------------------------
def dia_pluviometrico(timestamps) -> np.ndarray:
    """Fecha local (07:00 a 07:00) que identifica el día pluviométrico de cada instante."""
    local = pd.DatetimeIndex(timestamps).tz_convert(ev.ZONA_LOCAL) - pd.Timedelta(hours=ev.HORA_INICIO_BLOQUE)
    return np.asarray(local.normalize().tz_localize(None).date)


def elegir_casos(ep: pd.DataFrame, av: pd.DataFrame, fijo="2025-04-25") -> pd.DataFrame:
    """
    Regla fijada antes de mirar (16-oct-2026), con los episodios y avisos de la CNN plana L2 s42:
      - acierto:     el día pluviométrico fijo (25-abr-2025);
      - omisión:     el día con más episodios nocturnos (inicio 19:00–06:59 local) sin avisar;
      - falsa alarma: el día con más avisos falsos.
    Empates: el día más temprano. Un día ya elegido no se repite (se toma el siguiente).
    """
    fijo = pd.Timestamp(fijo).date()
    elegidos = [fijo]
    om = ep[~ep.avisado & ev.es_noche(ep.inicio)].assign(dia=lambda x: dia_pluviometrico(x.inicio))
    fa = av[av.tipo == "falso"].assign(dia=lambda x: dia_pluviometrico(x.t_primera))
    filas = [{"caso": "acierto diurno", "dia": fijo, "regla": "fijo",
              "n": int((dia_pluviometrico(ep.loc[ep.avisado, "inicio"]) == fijo).sum())}]
    for caso, df, regla in (("omisión nocturna", om, "más episodios nocturnos sin avisar"),
                            ("falsa alarma", fa, "más avisos falsos")):
        cuenta = df.groupby("dia").size().reset_index(name="n").sort_values(["n", "dia"], ascending=[False, True])
        cuenta = cuenta[~cuenta.dia.isin(elegidos)]
        dia = cuenta.dia.iloc[0]
        elegidos.append(dia)
        filas.append({"caso": caso, "dia": dia, "regla": regla, "n": int(cuenta.n.iloc[0])})
    return pd.DataFrame(filas)


def ventana_dia(dia) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inicio y fin (UTC) del día pluviométrico que empieza a las 07:00 local de `dia`."""
    ini = pd.Timestamp(dia).tz_localize(ev.ZONA_LOCAL) + pd.Timedelta(hours=ev.HORA_INICIO_BLOQUE)
    return ini.tz_convert("UTC"), (ini + pd.Timedelta(days=1)).tz_convert("UTC")


def resultado_par(y: np.ndarray, alerta: np.ndarray) -> np.ndarray:
    """acierto / omisión / falsa alarma / correcto negativo, por par."""
    return np.select([alerta & y, ~alerta & y, alerta & ~y], ["acierto", "omisión", "falsa alarma"], "correcto negativo")


# --- Coeficientes de la logística --------------------------------------------------
def coeficientes_logistica(modelo, bandas, L: int, estadisticas=("p10", "p50", "p90", "desvio")) -> pd.DataFrame:
    """Coeficientes estandarizados (sobre variables z) de la logística, con su instante, banda y
    estadística. Orden de las variables: instante (más antiguo → t), banda, estadística."""
    coef = modelo[-1].coef_.ravel()
    assert len(coef) == L * len(bandas) * len(estadisticas)
    idx = pd.MultiIndex.from_product([[f"t−{(L - 1 - j) * 10} min" if j < L - 1 else "t" for j in range(L)],
                                      bandas, estadisticas], names=["instante", "banda", "estadistica"])
    return pd.Series(coef, index=idx, name="coef").reset_index()
