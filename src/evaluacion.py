"""Evaluación de pronósticos binarios de lluvia fuerte.

Métricas de la tabla de contingencia (POD, FAR, CSI, ETS), elección del
umbral de decisión por CSI en validación y bootstrap por bloques de tiempo
local con comparaciones pareadas entre modelos.

Convenciones:

- Tabla de contingencia: a = aciertos, b = falsas alarmas, c = sorpresas
  (tormentas sin alerta), d = rechazos correctos.
- Una alerta se emite cuando la probabilidad es >= umbral.
- Denominador cero -> NaN, nunca 0: FAR sin alertas o POD sin tormentas no
  se pueden calcular, y un 0 diría algo falso del modelo.
- Etiquetas o probabilidades NaN detienen la evaluación con un error: las
  etiquetas NaN se quitan antes (índice de 11-dataset_training y cargador),
  así que un NaN aquí indica un error aguas arriba.
- Bloques del bootstrap en hora local (America/Bogota), asignados por el
  instante t de la imagen, empezando a las 07:00 (día pluviométrico del
  IDEAM; es también el corte que menos episodios regionales parte). El largo del bloque es un parámetro (24 h por
  defecto) para la prueba de sensibilidad 6/12/24/48 h.
- Bootstrap: B = 10,000 réplicas, IC percentil 95 %, mismos pesos de bloque
  para todos los modelos comparados (diferencias pareadas).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ZONA_LOCAL = "America/Bogota"
GRILLA_UMBRALES = np.round(np.arange(0.01, 1.00, 0.01), 2)  # 0.01 ... 0.99
B_DEFECTO = 10_000
NIVEL_IC = 0.95
UMBRAL_AVISO_NAN = 0.01
HORA_INICIO_BLOQUE = 7  # día pluviométrico del IDEAM: de 07:00 a 07:00 local  # fracción de réplicas indefinidas que vuelve poco confiable un IC
METRICAS = ("pod", "far", "csi", "ets")


# ---------------------------------------------------------------------------
# Validación de entradas
# ---------------------------------------------------------------------------

def _validar(y, p=None):
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        raise ValueError(
            f"Hay {int(np.isnan(y).sum())} etiquetas NaN. Las etiquetas NaN deben "
            "quitarse antes de evaluar; si llegan aquí, algo falló aguas arriba."
        )
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Las etiquetas deben ser 0 o 1.")
    if p is None:
        return y.astype(bool)
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"Formas distintas: etiquetas {y.shape}, probabilidades {p.shape}.")
    if np.isnan(p).any():
        raise ValueError(f"Hay {int(np.isnan(p).sum())} probabilidades NaN.")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("Las probabilidades deben estar en [0, 1].")
    return y.astype(bool), p


# ---------------------------------------------------------------------------
# Tabla de contingencia y métricas
# ---------------------------------------------------------------------------

def tabla_contingencia(y, p, umbral):
    """Devuelve (a, b, c, d) como enteros para un umbral."""
    y, p = _validar(y, p)
    alerta = p >= umbral
    a = int(np.sum(alerta & y))
    b = int(np.sum(alerta & ~y))
    c = int(np.sum(~alerta & y))
    d = int(np.sum(~alerta & ~y))
    return a, b, c, d


def _div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def metricas(a, b, c, d):
    """POD, FAR, CSI y ETS. Acepta escalares o arreglos (vectorizado).

    ETS (Gilbert): aciertos por azar a_r = (a + b)(a + c) / n;
    ETS = (a - a_r) / (a + b + c - a_r).
    """
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    n = a + b + c + d
    a_r = _div((a + b) * (a + c), n)
    salida = {
        "pod": _div(a, a + c),
        "far": _div(b, a + b),
        "csi": _div(a, a + b + c),
        "ets": _div(a - a_r, a + b + c - a_r),
    }
    if salida["pod"].ndim == 0:
        return {k: float(v) for k, v in salida.items()}
    return salida


# ---------------------------------------------------------------------------
# Umbral de decisión
# ---------------------------------------------------------------------------

def tablas_por_umbral(y, p, grilla=GRILLA_UMBRALES):
    """Tabla (a, b, c, d) para cada umbral de la grilla; forma (len(grilla), 4).

    Vectorizado: cuenta positivos y negativos con p >= u mediante búsqueda en
    las probabilidades ordenadas.
    """
    y, p = _validar(y, p)
    grilla = np.asarray(grilla, dtype=float)
    p_pos = np.sort(p[y])
    p_neg = np.sort(p[~y])
    a = len(p_pos) - np.searchsorted(p_pos, grilla, side="left")
    b = len(p_neg) - np.searchsorted(p_neg, grilla, side="left")
    c = len(p_pos) - a
    d = len(p_neg) - b
    return np.stack([a, b, c, d], axis=1)


def _punto_medio_meseta(csi, grilla, tol=1e-12):
    """Elige el umbral en el punto medio de la meseta de CSI máximo.

    Si hay varias mesetas separadas con el mismo máximo, se usa la más ancha
    (a igual ancho, la de umbral más bajo). Dentro de una meseta con un número
    par de puntos se toma el más bajo de los dos centrales (regla fija).
    """
    if np.all(np.isnan(csi)):
        return np.nan, (np.nan, np.nan)
    maximo = np.nanmax(csi)
    en_max = np.abs(csi - maximo) <= tol
    en_max &= ~np.isnan(csi)
    mejor = None
    i = 0
    while i < len(en_max):
        if en_max[i]:
            j = i
            while j + 1 < len(en_max) and en_max[j + 1]:
                j += 1
            if mejor is None or (j - i) > (mejor[1] - mejor[0]):
                mejor = (i, j)
            i = j + 1
        else:
            i += 1
    i, j = mejor
    medio = i + (j - i) // 2
    return float(grilla[medio]), (float(grilla[i]), float(grilla[j]))


def elegir_umbral(y, p, grilla=GRILLA_UMBRALES):
    """Umbral que maximiza el CSI (decisión 4), con desempate por punto medio.

    Devuelve un diccionario con el umbral, el CSI máximo, los extremos de la
    meseta y la curva completa CSI-umbral (para guardarla como artefacto).
    """
    tablas = tablas_por_umbral(y, p, grilla)
    m = metricas(*tablas.T)
    umbral, (desde, hasta) = _punto_medio_meseta(m["csi"], np.asarray(grilla))
    curva = pd.DataFrame({"umbral": grilla, "a": tablas[:, 0], "b": tablas[:, 1],
                          "c": tablas[:, 2], "d": tablas[:, 3], **m})
    return {
        "umbral": umbral,
        "csi_max": float(np.nanmax(m["csi"])),
        "meseta_desde": desde,
        "meseta_hasta": hasta,
        "curva": curva,
    }


# ---------------------------------------------------------------------------
# Bloques de tiempo local
# ---------------------------------------------------------------------------

def bloque_local(timestamps, horas_bloque=24, hora_inicio=HORA_INICIO_BLOQUE):
    """Número de bloque de cada par según la hora local del instante t.

    Los bloques se cuentan desde las `hora_inicio` locales del 1-ene-1970, de
    modo que con 24 h y hora_inicio = 7 el bloque es el día pluviométrico
    (07:00 a 07:00) y con 12 h se corta a las 07:00 y a las 19:00. Exige timestamps con zona horaria (falla si no la tienen,
    para no repetir el error silencioso de mezclar UTC con y sin zona).
    """
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        raise ValueError(
            "Los timestamps no tienen zona horaria. Localícelos (p. ej. "
            "tz_localize('UTC')) antes de asignar bloques."
        )
    local = ts.tz_convert(ZONA_LOCAL).tz_localize(None)
    origen = pd.Timestamp("1970-01-01") + pd.Timedelta(hours=hora_inicio)
    horas = (local - origen) / pd.Timedelta(hours=1)
    return np.floor(np.asarray(horas) / horas_bloque).astype(np.int64)


def tablas_por_bloque(timestamps, y, p, umbral, horas_bloque=24, bloques=None):
    """Tabla (a, b, c, d) de cada bloque.

    Devuelve (ids_bloque, tablas) con tablas de forma (n_bloques, 4). Si se
    pasa `bloques`, las filas salen en ese orden (bloques sin pares quedan en
    cero), lo que permite alinear dos modelos para la comparación pareada.
    """
    y, p = _validar(y, p)
    ids = bloque_local(timestamps, horas_bloque)
    alerta = p >= umbral
    df = pd.DataFrame({
        "bloque": ids,
        "a": alerta & y, "b": alerta & ~y, "c": ~alerta & y, "d": ~alerta & ~y,
    })
    t = df.groupby("bloque")[["a", "b", "c", "d"]].sum()
    if bloques is not None:
        t = t.reindex(bloques, fill_value=0)
    return t.index.to_numpy(), t.to_numpy(dtype=np.int64)


def tablas_por_bloque_y_umbral(timestamps, y, p, horas_bloque=24, grilla=GRILLA_UMBRALES):
    """Tablas por bloque para toda la grilla de umbrales: forma (n_bloques, n_umbrales, 4).

    Solo para el diagnóstico de estabilidad del umbral (opción B de la decisión 4).
    """
    y, p = _validar(y, p)
    ids = bloque_local(timestamps, horas_bloque)
    bloques, inv = np.unique(ids, return_inverse=True)
    salida = np.zeros((len(bloques), len(grilla), 4), dtype=np.int64)
    for k in range(len(bloques)):
        sel = inv == k
        salida[k] = tablas_por_umbral(y[sel], p[sel], grilla)
    return bloques, salida


# ---------------------------------------------------------------------------
# Bootstrap por bloques
# ---------------------------------------------------------------------------

def pesos_bootstrap(n_bloques, B=B_DEFECTO, semilla=42):
    """Cuántas veces sale cada bloque en cada réplica: forma (B, n_bloques).

    Sortear n bloques con reemplazo equivale a una multinomial con
    probabilidades iguales. Los mismos pesos se usan para todos los modelos
    que se comparan (diferencias pareadas, decisión 3).
    """
    rng = np.random.default_rng(semilla)
    return rng.multinomial(n_bloques, np.full(n_bloques, 1.0 / n_bloques), size=B).astype(np.int64)


def _ic(valores, nivel=NIVEL_IC):
    validos = valores[~np.isnan(valores)]
    if len(validos) == 0:
        return np.nan, np.nan
    alfa = (1 - nivel) / 2
    return float(np.quantile(validos, alfa)), float(np.quantile(validos, 1 - alfa))


@dataclass
class ResultadoBootstrap:
    replicas: dict          # métrica -> arreglo (B,)
    ic: dict                # métrica -> (inf, sup)
    n_nan: dict             # métrica -> réplicas indefinidas
    poco_confiable: dict = field(default_factory=dict)  # métrica -> bool


def bootstrap(tablas, pesos):
    """Métricas en cada réplica a partir de las tablas por bloque y los pesos."""
    sumas = pesos @ tablas            # (B, 4): cada réplica suma sus bloques
    m = metricas(*sumas.T)
    B = pesos.shape[0]
    n_nan = {k: int(np.isnan(v).sum()) for k, v in m.items()}
    return ResultadoBootstrap(
        replicas=m,
        ic={k: _ic(v) for k, v in m.items()},
        n_nan=n_nan,
        poco_confiable={k: n_nan[k] / B > UMBRAL_AVISO_NAN for k in m},
    )


def comparar_pareado(tablas_A, tablas_B, pesos, metrica="csi"):
    """Diferencia pareada A − B con los mismos bloques sorteados en cada réplica.

    Las tablas de ambos modelos deben estar alineadas por bloque (mismo orden;
    usar el argumento `bloques` de `tablas_por_bloque`).
    """
    if tablas_A.shape != tablas_B.shape:
        raise ValueError("Las tablas de los dos modelos no están alineadas por bloque.")
    if not np.array_equal(tablas_A[:, [0, 2]].sum(1), tablas_B[:, [0, 2]].sum(1)):
        raise ValueError("Los modelos no se evaluaron sobre los mismos pares (difieren las tormentas por bloque).")
    vA = metricas(*(pesos @ tablas_A).T)[metrica]
    vB = metricas(*(pesos @ tablas_B).T)[metrica]
    delta = vA - vB
    validas = ~np.isnan(delta)
    puntual_A = metricas(*tablas_A.sum(0))[metrica]
    puntual_B = metricas(*tablas_B.sum(0))[metrica]
    return {
        "delta": puntual_A - puntual_B,
        "ic_inf": _ic(delta)[0],
        "ic_sup": _ic(delta)[1],
        "frac_gana_A": float(np.mean(delta[validas] > 0)) if validas.any() else np.nan,
        "n_nan": int((~validas).sum()),
        "replicas": delta,
    }


def estabilidad_umbral(tablas_bu, pesos, grilla=GRILLA_UMBRALES):
    """Diagnóstico (opción B de la decisión 4): umbral óptimo re-elegido en cada réplica."""
    sumas = np.einsum("rk,kuj->ruj", pesos, tablas_bu)   # (B, n_umbrales, 4)
    csi = metricas(*np.moveaxis(sumas, -1, 0))["csi"]    # (B, n_umbrales)
    grilla = np.asarray(grilla)
    return np.array([_punto_medio_meseta(fila, grilla)[0] for fila in csi])


# ---------------------------------------------------------------------------
# Evaluación completa de un conjunto y salida para MLflow
# ---------------------------------------------------------------------------

def evaluar(timestamps, y, p, umbral, conjunto, B=B_DEFECTO, semilla=42, horas_bloque=24):
    """Evalúa un modelo sobre un conjunto con un umbral ya fijado.

    Devuelve (metricas_planas, artefactos). `metricas_planas` tiene nombres
    `conjunto_metrica[_ic_inf|_ic_sup]` listos para mlflow.log_metrics;
    `artefactos` contiene las tablas por bloque y las réplicas.
    """
    bloques, tablas = tablas_por_bloque(timestamps, y, p, umbral, horas_bloque)
    puntual = metricas(*tablas.sum(0))
    pesos = pesos_bootstrap(len(bloques), B, semilla)
    bs = bootstrap(tablas, pesos)

    a, b, c, d = (int(x) for x in tablas.sum(0))
    planas = {f"{conjunto}_umbral": float(umbral)}
    for k in METRICAS:
        planas[f"{conjunto}_{k}"] = puntual[k]
        planas[f"{conjunto}_{k}_ic_inf"], planas[f"{conjunto}_{k}_ic_sup"] = bs.ic[k]
        planas[f"{conjunto}_replicas_nan_{k}"] = bs.n_nan[k]
    planas.update({
        f"{conjunto}_n_pares": a + b + c + d,
        f"{conjunto}_n_tormentas": a + c,
        f"{conjunto}_n_alertas": a + b,
        f"{conjunto}_n_bloques": len(bloques),
        f"{conjunto}_n_bloques_con_tormenta": int((tablas[:, [0, 2]].sum(1) > 0).sum()),
        f"{conjunto}_horas_bloque": horas_bloque,
        f"{conjunto}_B": B,
    })
    avisos = [k for k, v in bs.poco_confiable.items() if v]
    artefactos = {
        "tablas_por_bloque": pd.DataFrame(tablas, columns=list("abcd"), index=pd.Index(bloques, name="bloque")),
        "replicas": pd.DataFrame(bs.replicas),
        "avisos_ic_poco_confiable": avisos,
    }
    return planas, artefactos


def registrar_mlflow(planas, artefactos, prefijo_archivos):
    """Registra métricas y artefactos en la corrida de MLflow activa."""
    import mlflow  # import diferido: el módulo funciona sin MLflow instalado

    mlflow.log_metrics({k: v for k, v in planas.items() if np.isfinite(v)})
    artefactos["tablas_por_bloque"].to_parquet(f"{prefijo_archivos}_tablas_por_bloque.parquet")
    artefactos["replicas"].to_parquet(f"{prefijo_archivos}_replicas.parquet")
    mlflow.log_artifact(f"{prefijo_archivos}_tablas_por_bloque.parquet")
    mlflow.log_artifact(f"{prefijo_archivos}_replicas.parquet")


# ---------------------------------------------------------------------------
# Duración de episodios (respaldo físico del largo de bloque, decisión 2)
# ---------------------------------------------------------------------------

def rachas_positivas(df, col_estacion="estacion", col_tiempo="timestamp", col_etiqueta="etiqueta",
                     paso=pd.Timedelta(minutes=10)):
    """Rachas de pares positivos consecutivos.

    Devuelve un DataFrame con una fila por racha: por estación (columna
    `estacion`) y regionales (`estacion` = "red", instantes con al menos una
    estación positiva). Una racha se corta cuando pasan más de `paso` entre
    dos positivos. La duración de una racha es el lapso en que los pares
    dependen de la misma tormenta, que es lo que el bloque debe contener.
    """
    pos = df.loc[df[col_etiqueta] == 1, [col_estacion, col_tiempo]].copy()

    def _rachas(t):
        # Se compara en enteros (ns): con timestamps con zona, np.sort/np.diff
        # devuelven objetos y la comparación falla según la versión de pandas.
        t = pd.DatetimeIndex(t).unique().sort_values()
        if len(t) == 0:
            return pd.DataFrame(columns=["inicio", "fin", "duracion_h"])
        ns = t.as_unit("ns").asi8
        corte = np.r_[True, np.diff(ns) > pd.Timedelta(paso).value]
        grupo = np.cumsum(corte)
        r = pd.DataFrame({"t": t, "g": grupo}).groupby("g")["t"].agg(["min", "max"])
        r.columns = ["inicio", "fin"]
        r["duracion_h"] = (r["fin"] - r["inicio"] + paso) / pd.Timedelta(hours=1)
        return r.reset_index(drop=True)

    partes = []
    for est, g in pos.groupby(col_estacion):
        r = _rachas(g[col_tiempo])
        r.insert(0, "estacion", est)
        partes.append(r)
    r = _rachas(pos[col_tiempo])
    r.insert(0, "estacion", "red")
    partes.append(r)
    return pd.concat(partes, ignore_index=True)


# ===========================================================================
# ---- Métricas por evento y estratos
# ===========================================================================
#
# Convenciones (decisiones de S4b, 12-oct-2026):
#
# - Episodio: por estación, instantes de 10 min con intensidad horaria
#   (`int_t`) >= umbral de la estación; dos tramos se unen si la intensidad
#   queda bajo el umbral por TOLERANCIA (30 min) o menos. Inicio = primer
#   instante del episodio (la hora que termina en él alcanza el umbral).
# - Cada par positivo pertenece al episodio cuyo primer instante fuerte cae
#   en su ventana τ ∈ (t + 2 h, t + 3 h] (la misma ventana de la etiqueta);
#   si la ventana toca dos episodios, al primero.
# - Episodio evaluable: con al menos un par positivo en el conjunto evaluado.
# - Alerta válida: alerta en un par positivo del episodio con t < inicio.
#   Episodio avisado: al menos una alerta válida.
# - Aviso: alertas de una estación con pausas de TOLERANCIA o menos.
#   correcto (contiene una alerta válida) / tardío (alertas en positivos, todas
#   posteriores al inicio de su episodio) / falso (ninguna alerta en positivo).
# - Anticipación = inicio − t de la primera alerta válida.

TOLERANCIA = pd.Timedelta(minutes=30)
PASO = pd.Timedelta(minutes=10)
VENTANA_DESDE = pd.Timedelta(hours=2)   # τ ∈ (t + 2 h, t + 3 h]
VENTANA_HASTA = pd.Timedelta(hours=3)
HORAS_NOCHE = (19, 7)                   # noche = [19:00, 07:00) local
UMBRAL_NUBE_FRIA_K = 228.2              # p25 del p10 de C13 de los positivos de train (S3a)
MIN_EPISODIOS_ESTACION = 10
TEMPORADAS = {12: "DEF", 1: "DEF", 2: "DEF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def _rachas_tiempo(t, separacion_max):
    """Etiqueta de grupo para tiempos ordenados: nuevo grupo si el salto supera separacion_max."""
    ns = pd.DatetimeIndex(t).as_unit("ns").asi8
    corte = np.r_[True, np.diff(ns) > pd.Timedelta(separacion_max).value]
    return np.cumsum(corte)


def episodios_lluvia(intensidad, umbrales, col_estacion="codigoestacion", col_tiempo="timestamp",
                     col_int="int_t", tolerancia=TOLERANCIA, paso=PASO):
    """Episodios de lluvia fuerte por estación.

    `intensidad`: una fila por (estación, instante de 10 min) con la intensidad
    horaria en ese instante. `umbrales`: Series estación -> umbral (mm/h).
    Devuelve (episodios, instantes_fuertes). Los instantes con intensidad NaN
    se tratan como no fuertes.

    Una pausa de `tolerancia` bajo el umbral entre dos instantes fuertes
    equivale a un salto de tolerancia + paso entre ellos.
    """
    df = intensidad[[col_estacion, col_tiempo, col_int]].copy()
    df["umbral"] = df[col_estacion].map(umbrales)
    if df["umbral"].isna().any():
        faltan = df.loc[df["umbral"].isna(), col_estacion].unique()[:5]
        raise ValueError(f"Estaciones sin umbral: {list(faltan)}")
    fuertes = df[df[col_int] >= df["umbral"]].sort_values([col_estacion, col_tiempo])
    fuertes = fuertes[[col_estacion, col_tiempo]].reset_index(drop=True)
    n_ep = np.zeros(len(fuertes), dtype=np.int64)
    for _, idx in fuertes.groupby(col_estacion, sort=False).indices.items():
        n_ep[idx] = _rachas_tiempo(fuertes[col_tiempo].iloc[idx], tolerancia + paso)
    fuertes["n_ep"] = n_ep
    fuertes["id_episodio"] = fuertes.groupby([col_estacion, "n_ep"], sort=True).ngroup()
    episodios = (fuertes.groupby("id_episodio").agg(**{col_estacion: (col_estacion, "first"),
                                                     "inicio": (col_tiempo, "min"),
                                                     "fin": (col_tiempo, "max")}).reset_index())
    return episodios, fuertes[[col_estacion, col_tiempo, "id_episodio"]]


def asignar_episodio(pares, instantes_fuertes, col_estacion="codigoestacion", col_tiempo="timestamp",
                     desde=VENTANA_DESDE, hasta=VENTANA_HASTA):
    """Episodio de cada par: el del primer instante fuerte en (t + desde, t + hasta].

    Devuelve un arreglo de id_episodio (−1 si la ventana no toca ningún
    episodio) y la cantidad de episodios distintos que toca cada ventana
    (para contar los pares ambiguos).
    """
    p = pares[[col_estacion, col_tiempo]].copy()
    p["_orden"] = np.arange(len(p))
    p["_desde"] = p[col_tiempo] + desde
    f = instantes_fuertes.rename(columns={col_tiempo: "_tf"}).sort_values("_tf")
    p = p.sort_values("_desde")
    m = pd.merge_asof(p, f, left_on="_desde", right_on="_tf", by=col_estacion,
                      direction="forward", allow_exact_matches=False)
    dentro = m["_tf"] <= m[col_tiempo] + hasta
    m["id_episodio"] = np.where(dentro, m["id_episodio"], -1)
    # episodios distintos dentro de la ventana: comparar con el último instante fuerte de la ventana
    p2 = p.copy()
    p2["_hasta"] = p2[col_tiempo] + hasta
    m2 = pd.merge_asof(p2.sort_values("_hasta"), f, left_on="_hasta", right_on="_tf", by=col_estacion,
                       direction="backward", allow_exact_matches=True)
    m2 = m2.set_index("_orden").reindex(m["_orden"])
    ultimo_dentro = (m2["_tf"] > m2[col_tiempo] + desde).to_numpy()
    n_dist = np.where(dentro.to_numpy(),
                      1 + (ultimo_dentro & (m2["id_episodio"].to_numpy() != m["id_episodio"].to_numpy())), 0)
    m["_n"] = n_dist
    m = m.sort_values("_orden")
    return m["id_episodio"].to_numpy().astype(np.int64), m["_n"].to_numpy()


def eventos(pares, episodios, umbral, col_estacion="codigoestacion", col_tiempo="timestamp",
            tolerancia=TOLERANCIA, paso=PASO):
    """Episodios evaluables y avisos de un modelo.

    `pares`: conjunto evaluado con columnas estación, tiempo, y, p e
    id_episodio (de `asignar_episodio`). Devuelve (ep, avisos):

    - ep: un episodio evaluable por fila, con t del primer par positivo,
      avisado (bool) y anticipacion_h.
    - avisos: un aviso por fila con su estación, t de la primera alerta y tipo
      ("correcto", "tardío" o "falso").
    """
    y, p = _validar(pares["y"], pares["p"])
    if ((pares["id_episodio"] >= 0) != y).any():
        raise ValueError("Hay pares positivos sin episodio o negativos con episodio: "
                         "revisar la ventana o los umbrales usados para los episodios.")
    df = pares[[col_estacion, col_tiempo, "id_episodio"]].copy()
    df["y"], df["alerta"] = y, p >= umbral
    df = df.merge(episodios[["id_episodio", "inicio"]], on="id_episodio", how="left")
    df["valida"] = df["alerta"] & df["y"] & (df[col_tiempo] < df["inicio"])

    pos = df[df["y"]]
    ep = pos.groupby("id_episodio").agg(t_primer_par=(col_tiempo, "min"), n_pares=(col_tiempo, "size"))
    validas = pos[pos["valida"]].groupby("id_episodio")[col_tiempo].min().rename("t_primera_alerta")
    ep = ep.join(validas).join(episodios.set_index("id_episodio")[[col_estacion, "inicio", "fin"]])
    ep["avisado"] = ep["t_primera_alerta"].notna()
    ep["anticipacion_h"] = (ep["inicio"] - ep["t_primera_alerta"]) / pd.Timedelta(hours=1)
    ep = ep.reset_index()

    al = df[df["alerta"]].sort_values([col_estacion, col_tiempo]).copy()
    if len(al):
        al["n_av"] = al.groupby(col_estacion, sort=False)[col_tiempo].transform(
            lambda s: _rachas_tiempo(s, tolerancia + paso))
        av = al.groupby([col_estacion, "n_av"]).agg(
            t_primera=(col_tiempo, "min"), t_ultima=(col_tiempo, "max"), n_alertas=(col_tiempo, "size"),
            hay_valida=("valida", "any"), hay_positivo=("y", "any")).reset_index().drop(columns="n_av")
        av["tipo"] = np.select([av["hay_valida"], av["hay_positivo"]], ["correcto", "tardío"], "falso")
        av = av.drop(columns=["hay_valida", "hay_positivo"])
    else:
        av = pd.DataFrame(columns=[col_estacion, "t_primera", "t_ultima", "n_alertas", "tipo"])
    return ep, av


def metricas_evento(n_ep, n_avisados, n_avisos, n_falsos, n_avisados_1h):
    """POD y FAR por evento y fracción de avisados con ≥ 1 h. Escalares o arreglos."""
    return {
        "pod_evento": _div(n_avisados, n_ep),
        "far_evento": _div(n_falsos, n_avisos),
        "frac_anticipacion_1h": _div(n_avisados_1h, n_avisados),
    }


# ---------------------------------------------------------------------------
# Estratos
# ---------------------------------------------------------------------------

def es_noche(timestamps):
    h = pd.DatetimeIndex(timestamps).tz_convert(ZONA_LOCAL).hour
    return np.asarray((h >= HORAS_NOCHE[0]) | (h < HORAS_NOCHE[1]))


def temporada(timestamps):
    m = pd.DatetimeIndex(timestamps).tz_convert(ZONA_LOCAL).month
    return np.asarray(pd.Series(m).map(TEMPORADAS))


def terciles_umbral(umbrales):
    """Grupo de umbral por estación (bajo / medio / alto), con cortes de los umbrales mismos."""
    cortes = np.quantile(np.asarray(umbrales, dtype=float), [1 / 3, 2 / 3])
    grupo = np.where(umbrales <= cortes[0], "bajo", np.where(umbrales <= cortes[1], "medio", "alto"))
    return pd.Series(grupo, index=umbrales.index), cortes


def nube_fria(p10_c13):
    return np.asarray(p10_c13, dtype=float) <= UMBRAL_NUBE_FRIA_K


def _conteos(df, col_bloque, col_estrato, columnas, bloques, estratos):
    """Suma de columnas por (estrato, bloque) → arreglo (n_estratos, n_bloques, n_columnas)."""
    salida = np.zeros((len(estratos), len(bloques), len(columnas)), dtype=np.int64)
    if len(df) == 0:
        return salida
    t = df.groupby([col_estrato, col_bloque])[columnas].sum()
    pos_b = {b: i for i, b in enumerate(bloques)}
    for i, e in enumerate(estratos):
        if e in t.index.get_level_values(0):
            sub = t.loc[e]
            idx = [pos_b[b] for b in sub.index]
            salida[i, idx] = sub.to_numpy()
    return salida


def evaluar_por_estrato(pares, ep, avisos, umbral, estrato_pares, estrato_ep, estrato_avisos,
                        pesos, bloques, horas_bloque=24, col_tiempo="timestamp",
                        min_episodios=0):
    """Métricas por pares y por evento en cada estrato, con IC por bootstrap de bloques.

    Los estratos se pasan como arreglos alineados con `pares`, `ep` y `avisos`.
    Se usan los mismos `pesos` y `bloques` que en la evaluación completa, de
    modo que la suma de los estratos reproduce el total en cada réplica.
    Con `min_episodios` > 0, los estratos con menos episodios evaluables quedan
    marcados como insuficientes y sin métricas (regla por estación).
    """
    y, p = _validar(pares["y"], pares["p"])
    alerta = p >= umbral
    dp = pd.DataFrame({"bloque": bloque_local(pares[col_tiempo], horas_bloque), "estrato": estrato_pares,
                       "a": alerta & y, "b": alerta & ~y, "c": ~alerta & y, "d": ~alerta & ~y})
    de = pd.DataFrame({"bloque": bloque_local(ep["t_primer_par"], horas_bloque), "estrato": estrato_ep,
                       "n_ep": 1, "n_avisados": ep["avisado"].astype(int),
                       "n_avisados_1h": (ep["anticipacion_h"] >= 1).astype(int)})
    da = pd.DataFrame({"bloque": bloque_local(avisos["t_primera"], horas_bloque) if len(avisos) else [],
                       "estrato": estrato_avisos, "n_avisos": 1,
                       "n_falsos": (avisos["tipo"] == "falso").astype(int),
                       "n_tardios": (avisos["tipo"] == "tardío").astype(int)})
    estratos = sorted(set(dp["estrato"]) | set(de["estrato"]) | set(da["estrato"]), key=str)
    cp = _conteos(dp, "bloque", "estrato", list("abcd"), bloques, estratos)
    ce = _conteos(de, "bloque", "estrato", ["n_ep", "n_avisados", "n_avisados_1h"], bloques, estratos)
    ca = _conteos(da, "bloque", "estrato", ["n_avisos", "n_falsos", "n_tardios"], bloques, estratos)

    filas = []
    B = pesos.shape[0]
    for i, e in enumerate(estratos):
        tp, te, ta = cp[i].sum(0), ce[i].sum(0), ca[i].sum(0)
        fila = {"estrato": e, "n_pares": int(tp.sum()), "n_tormentas": int(tp[0] + tp[2]),
                "n_alertas": int(tp[0] + tp[1]), "n_episodios": int(te[0]), "n_avisados": int(te[1]),
                "n_avisos": int(ta[0]), "n_falsos": int(ta[1]), "n_tardios": int(ta[2])}
        insuf = min_episodios > 0 and te[0] < min_episodios
        fila["insuficiente"] = bool(insuf)
        if not insuf:
            puntual = {**metricas(*tp), **metricas_evento(te[0], te[1], ta[0], ta[1], te[2])}
            rep = {**metricas(*(pesos @ cp[i]).T)}
            se, sa = pesos @ ce[i], pesos @ ca[i]
            rep.update(metricas_evento(se[:, 0], se[:, 1], sa[:, 0], sa[:, 1], se[:, 2]))
            for k, v in puntual.items():
                fila[k] = v
                fila[f"{k}_ic_inf"], fila[f"{k}_ic_sup"] = _ic(rep[k])
                n_nan = int(np.isnan(rep[k]).sum())
                fila[f"{k}_replicas_nan"] = n_nan
                fila[f"{k}_poco_confiable"] = n_nan / B > UMBRAL_AVISO_NAN
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_anticipacion(ep):
    """Distribución de la anticipación de los episodios avisados (horas)."""
    a = ep.loc[ep["avisado"], "anticipacion_h"]
    if len(a) == 0:
        return {"n_avisados": 0}
    return {"n_avisados": int(len(a)), "mediana_h": float(a.median()),
            "q1_h": float(a.quantile(0.25)), "q3_h": float(a.quantile(0.75)),
            "frac_1h": float((a >= 1).mean())}
