"""
Cargador de datos para el entrenamiento y la evaluación (S3b del plan de modelado).

Une tres productos:
    índice     datasets/indice_completo.parquet        (11)  pares (estación, t), etiqueta, período, L_max
    atributos  datasets/negativos_atributos.parquet    (14)  criterios de negativos difíciles, sin NG
    caché      data/cache/                             (13)  imágenes uint16 recortadas

y entrega lotes (B, 5·L, 32, 32) float32 normalizados con su etiqueta.

Orden de canales: historia de la más antigua a la más reciente y, dentro de cada instante,
las cinco bandas en el orden del caché (C08, C09, C10, C13, C14):
    canal = j · 5 + b,   j = 0 → t − (L−1)·10 min, …, j = L−1 → t.

Conjuntos (decisión de 11, Sección 2):
    train  pares con L_max ≥ L (cada modelo L usa los pares que tienen su historia)
    val/test  pares con L_max = 4 (conjunto común a los tres modelos, comparación pareada)
Nueva Generación (0021206600) no entra en ningún conjunto (decisión del 8-oct-2026).

Muestreo por época (solo train): todos los positivos y, por cada positivo, `n_dificiles`
negativos de la bolsa difícil y `n_aleatorios` del resto, sorteados sin reemplazo con
semilla = semilla_base + época. Val y test se recorren completos, sin muestrear.

Uso típico:
    cache = Cache(dir_cache, zscore, en_memoria=True)
    pares = preparar_pares(indice, atributos, cache, periodo="train", L=2)
    ds    = DatasetPares(pares, cache, L=2)
    mu    = MuestreadorEpoca(pares, n_dificiles=2, n_aleatorios=2, semilla=42)
    dl    = cargador(ds, mu, tam_lote=256)
    for epoca in range(n):
        mu.fijar_epoca(epoca)
        for x, y in dl: ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

NG = "0021206600"
PASO = pd.Timedelta(minutes=10)
L_COMUN = 4                       # historia del conjunto común de val y test
HALF = 16                         # parche 32 × 32
UMBRAL_NUBE_FRIA = 228.2          # K, p25 del p10 de C13 de los positivos de train (14, Sección 3)
CRITERIOS = ("red", "red_activa", "casi_positivo", "temporal", "nube_fria")
CRITERIOS_DEFECTO = ("red_activa", "casi_positivo")


def cod10(s: pd.Series) -> pd.Series:
    """Código de estación como texto de 10 dígitos (las tablas lo guardan como texto o entero)."""
    return s.astype("int64").astype(str).str.zfill(10)


# --- Bolsas de negativos ----------------------------------------------------------
def mascara_dificil(d: pd.DataFrame, criterios=CRITERIOS_DEFECTO,
                    umbral_nube_fria: float = UMBRAL_NUBE_FRIA) -> np.ndarray:
    """Negativo difícil = unión de los criterios activados (definiciones en 14-negativos_dificiles)."""
    desconocidos = set(criterios) - set(CRITERIOS)
    assert not desconocidos, f"Criterios desconocidos: {desconocidos}"
    c = {
        "red":           lambda: d["n_est_pos"] >= 1,
        "red_activa":    lambda: d["n_est_activas"] >= 1,
        "casi_positivo": lambda: (d["razon"] >= 0.5) & (d["razon"] < 0.8),
        "temporal":      lambda: d["horas_a_positivo"] <= 24,
        "nube_fria":     lambda: d["p10_c13"] <= umbral_nube_fria,
    }
    return np.logical_or.reduce([c[k]().to_numpy() for k in criterios])


# --- Preparación de los pares de un conjunto --------------------------------------
def preparar_pares(indice: pd.DataFrame, atributos: pd.DataFrame, cache, periodo: str, L: int,
                   criterios=CRITERIOS_DEFECTO, umbral_nube_fria: float = UMBRAL_NUBE_FRIA) -> pd.DataFrame:
    """
    Pares de un período listos para el cargador: filas del caché de cada imagen de la historia
    (fila_0 … fila_{L-1}, de la más antigua a t), posición de la estación en el recorte y bolsa
    ('positivo', 'dificil', 'aleatorio'; la bolsa solo se usa para muestrear train).
    """
    assert periodo in ("train", "val", "test") and 1 <= L <= L_COMUN
    d = indice[indice["periodo"] == periodo].copy()
    d["codigoestacion"] = cod10(d["codigoestacion"])
    d = d[d["codigoestacion"] != NG]
    d = d[d["L_max"] >= L] if periodo == "train" else d[d["L_max"] == L_COMUN]

    a = atributos.copy()
    a["codigoestacion"] = cod10(a["codigoestacion"])
    assert str(d["timestamp"].dt.tz) == str(a["timestamp"].dt.tz) == "UTC"
    d = d.merge(a, on=["codigoestacion", "timestamp"], how="left", validate="one_to_one")
    assert d["p10_c13"].notna().all(), "Pares sin atributos (¿14 desactualizada?)"

    # Filas del caché para cada instante de la historia (el caché guarda UTC sin zona)
    t = d["timestamp"].dt.tz_convert(None)
    for j in range(L):
        ts_j = t - (L - 1 - j) * PASO
        d[f"fila_{j}"] = ts_j.map(cache.fila_de)
        assert d[f"fila_{j}"].notna().all(), f"Imagen faltante en el caché para el lag {L - 1 - j}"
        d[f"fila_{j}"] = d[f"fila_{j}"].astype(np.int64)

    pos = d["codigoestacion"].astype("int64").map(cache.pos)
    assert pos.notna().all(), "Estación sin posición en el caché"
    d["fila_c"] = [p[0] for p in pos]
    d["col_c"] = [p[1] for p in pos]

    dificil = mascara_dificil(d, criterios, umbral_nube_fria)
    d["bolsa"] = np.select([d["etiqueta"] == 1, dificil], ["positivo", "dificil"], "aleatorio")
    return d.reset_index(drop=True)


# --- Dataset por lotes ------------------------------------------------------------
class DatasetPares(Dataset):
    """
    Recibe una lista de índices (un lote completo, vía BatchSampler) y devuelve
    x (B, 5·L, 32, 32) float32 normalizado e y (B,) float32. La extracción es vectorizada
    por lote: una indexación avanzada por instante de la historia.
    """

    def __init__(self, pares: pd.DataFrame, cache, L: int):
        self.L = L
        self.img = cache.img
        self.filas = pares[[f"fila_{j}" for j in range(L)]].to_numpy(np.int64)       # (N, L)
        self.fc = pares["fila_c"].to_numpy(np.int64)
        self.cc = pares["col_c"].to_numpy(np.int64)
        self.y = pares["etiqueta"].to_numpy(np.float32)
        nb = len(cache.bandas)
        # (x·scale + offset − media) / desvío  =  x·a + b
        self.a = (cache.scale / cache.desvio).astype(np.float32).reshape(1, 1, nb, 1, 1)
        self.b = ((cache.offset - cache.media) / cache.desvio).astype(np.float32).reshape(1, 1, nb, 1, 1)
        self.nb = nb
        self._off = np.arange(-HALF, HALF)

    def __len__(self):
        return len(self.y)

    def crudo(self, idx) -> np.ndarray:
        """Enteros del caché (B, L, 5, 32, 32) uint16, sin normalizar."""
        idx = np.asarray(idx, dtype=np.int64)
        r = (self.fc[idx, None] + self._off)[:, None, None, :, None]                  # (B,1,1,32,1)
        c = (self.cc[idx, None] + self._off)[:, None, None, None, :]                  # (B,1,1,1,32)
        f = self.filas[idx][:, :, None, None, None]                                   # (B,L,1,1,1)
        bnd = np.arange(self.nb)[None, None, :, None, None]                           # (1,1,5,1,1)
        return np.asarray(self.img[f, bnd, r, c])

    def __getitem__(self, idx):
        u = self.crudo(idx).astype(np.float32)
        x = u * self.a + self.b                                                       # (B, L, 5, 32, 32)
        x = x.reshape(len(u), self.L * self.nb, 2 * HALF, 2 * HALF)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.y[np.asarray(idx)])


# --- Muestreo por época -----------------------------------------------------------
class MuestreadorEpoca(Sampler):
    """
    Índices de una época de entrenamiento: todos los positivos + n_dificiles·P de la bolsa
    difícil + n_aleatorios·P del resto, sin reemplazo, mezclados. Determinista dado
    (semilla, época). Si una bolsa tiene menos pares que los pedidos, la toma completa y avisa.
    """

    def __init__(self, pares: pd.DataFrame, n_dificiles: float = 2, n_aleatorios: float = 2, semilla: int = 42):
        b = pares["bolsa"].to_numpy()
        self.idx = {k: np.flatnonzero(b == k) for k in ("positivo", "dificil", "aleatorio")}
        P = len(self.idx["positivo"])
        assert P > 0, "Sin positivos"
        self.pedidos = {"dificil": int(round(n_dificiles * P)), "aleatorio": int(round(n_aleatorios * P))}
        for k, n in self.pedidos.items():
            if n > len(self.idx[k]):
                print(f"Aviso: se piden {n:,} pares de la bolsa '{k}' y tiene {len(self.idx[k]):,}; se toma completa.")
                self.pedidos[k] = len(self.idx[k])
        self.semilla, self.epoca = semilla, 0

    def fijar_epoca(self, epoca: int):
        self.epoca = epoca

    def indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.semilla + self.epoca)
        partes = [self.idx["positivo"]] + [rng.choice(self.idx[k], n, replace=False)
                                           for k, n in self.pedidos.items()]
        return rng.permutation(np.concatenate(partes))

    def __iter__(self):
        return iter(self.indices().tolist())

    def __len__(self):
        return len(self.idx["positivo"]) + sum(self.pedidos.values())


def cargador(ds: DatasetPares, muestreador: Sampler | None = None, tam_lote: int = 256,
             num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    """
    DataLoader por lotes. Con muestreador (train) recorre la época muestreada; sin él
    (val/test) recorre todos los pares en orden. batch_size=None porque el Dataset ya
    devuelve lotes completos.

    pin_memory=True acelera la copia de los lotes a la GPU, pero exige una GPU disponible;
    se activa solo en el entrenamiento (desde S6), no al leer datos.
    """
    base = muestreador if muestreador is not None else range(len(ds))
    return DataLoader(ds, sampler=BatchSampler(base, tam_lote, drop_last=False), batch_size=None,
                      num_workers=num_workers, pin_memory=pin_memory)
