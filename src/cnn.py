"""
Peldaño 3 de la escalera de complejidad: CNN plana (S6 del plan de modelado).

    Entrada  (B, 5·L, 32, 32) del cargador (src/cargador.py), canales lag-mayor
    Salida   logit (B,); la probabilidad es sigmoid(logit)

"Plana" = bloques convolucionales apilados sin conexiones residuales (las agrega la
ResNet del peldaño 4). Cada bloque: conv 3×3 → BN → ReLU → conv 3×3 → BN → ReLU →
max-pool 2×2. Tras el último bloque, promedio global, dropout y una capa lineal.

Pérdidas:
    csi  1 − CSI suave del lote: TP = Σ p·y, FP = Σ p·(1−y), FN = Σ (1−p)·y, con p = sigmoid(logit).
         Es la pérdida principal del borrador (4.4). Con p binaria coincide con 1 − CSI.
    bce  entropía cruzada binaria, como referencia y diagnóstico.

Selección (decisión 5 de 17-referencias, 14-oct-2026): la época que se conserva es la de mayor
PR-AUC en validación; el umbral de decisión se elige después por CSI con src/evaluacion.py.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import evaluacion as ev
from cargador import L_COMUN


# --- Reproducibilidad -------------------------------------------------------------
def fijar_semilla(semilla: int = 42, determinista: bool = True):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    if determinista:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Modelo -----------------------------------------------------------------------
def _bloque(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNPlana(nn.Module):
    """CNN sin conexiones residuales. Con canales=(32, 64, 128) el parche pasa 32 → 16 → 8 → 4."""

    def __init__(self, L: int, n_bandas: int = 5, canales=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        capas, c = [], n_bandas * L
        for c_out in canales:
            capas.append(_bloque(c, c_out))
            c = c_out
        self.cuerpo = nn.Sequential(*capas)
        self.cabeza = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 1))
        self.L = L

    def forward(self, x):
        return self.cabeza(self.cuerpo(x)).squeeze(1)


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# --- Pérdidas ---------------------------------------------------------------------
def perdida_csi(logits: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 − CSI suave del lote. Si el lote no tiene positivos ni alertas suaves, vale ~1."""
    p = torch.sigmoid(logits)
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    return 1 - tp / (tp + fp + fn + eps)


def perdida_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


PERDIDAS = {"csi": perdida_csi, "bce": perdida_bce}


# --- Datos de validación con historia más corta -----------------------------------
def ultimos_instantes(pares: pd.DataFrame, L: int) -> pd.DataFrame:
    """
    Val y test se preparan con L_COMUN = 4 (fila_0 … fila_3, de la más antigua a t). Un modelo
    con historia L usa los últimos L instantes: fila_{4−L+j} pasa a ser fila_j.
    """
    d = pares.drop(columns=[f"fila_{j}" for j in range(L_COMUN)])
    for j in range(L):
        d[f"fila_{j}"] = pares[f"fila_{L_COMUN - L + j}"].to_numpy()
    return d


# --- Predicción -------------------------------------------------------------------
@torch.no_grad()
def predecir(modelo: nn.Module, dl, disp: torch.device, perdida=None):
    """Probabilidades en el orden del cargador (sin muestreador) y, si se pasa, la pérdida media."""
    modelo.eval()
    ps, total, n = [], 0.0, 0
    for x, y in dl:
        x, y = x.to(disp, non_blocking=True), y.to(disp, non_blocking=True)
        logits = modelo(x)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        if perdida is not None:
            total += float(perdida(logits, y)) * len(y)
            n += len(y)
    p = np.concatenate(ps)
    return (p, total / n) if perdida is not None else p


# --- Sobreajuste deliberado de un lote --------------------------------------------
def sobreajustar_lote(modelo: nn.Module, x: torch.Tensor, y: torch.Tensor, perdida, disp: torch.device,
                      pasos: int = 300, lr: float = 1e-3) -> pd.DataFrame:
    """
    Prueba de humo: con un lote fijo, sin dropout ni regularización, la pérdida debe caer cerca
    de 0 y el lote quedar clasificado sin errores. Si no, hay un error en el modelo, la pérdida
    o los datos.
    """
    modelo.train()
    for m in modelo.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    x, y = x.to(disp), y.to(disp)
    filas = []
    for paso in range(pasos):
        opt.zero_grad()
        logits = modelo(x)
        l = perdida(logits, y)
        l.backward()
        opt.step()
        if paso % 10 == 0 or paso == pasos - 1:
            pred = (torch.sigmoid(logits.detach()) >= 0.5).float()
            filas.append({"paso": paso, "perdida": l.item(), "errores": int((pred != y).sum())})
    return pd.DataFrame(filas)


# --- Entrenamiento con parada temprana --------------------------------------------
def entrenar(modelo: nn.Module, dl_train, muestreador, dl_val, y_val: np.ndarray, cfg: dict,
             disp: torch.device, ruta_mejor: Path, registrar=None) -> pd.DataFrame:
    """
    Entrena época a época (cada época vuelve a sortear los negativos con `muestreador`).
    Cada `cfg["validar_cada"]` épocas predice validación completa y calcula PR-AUC (criterio de
    selección), AUC, CSI máximo sobre la grilla y pérdida de validación. Guarda en `ruta_mejor`
    el estado con mayor PR-AUC; se detiene si no mejora en `cfg["paciencia"]` validaciones
    seguidas o al llegar a `cfg["max_epocas"]`.

    `registrar(métricas, paso)` se llama en cada validación (p. ej. mlflow.log_metrics).
    Devuelve el historial (una fila por época; columnas de validación en las épocas validadas).
    """
    perdida = PERDIDAS[cfg["perdida"]]
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    mejor, sin_mejora, hist = -np.inf, 0, []
    for epoca in range(1, cfg["max_epocas"] + 1):
        t0 = time.time()
        muestreador.fijar_epoca(epoca)
        modelo.train()
        total, n = 0.0, 0
        for x, y in dl_train:
            x, y = x.to(disp, non_blocking=True), y.to(disp, non_blocking=True)
            opt.zero_grad()
            l = perdida(modelo(x), y)
            l.backward()
            opt.step()
            total += float(l) * len(y)
            n += len(y)
        fila = {"epoca": epoca, "train_perdida": total / n, "seg_train": time.time() - t0}

        if epoca % cfg["validar_cada"] == 0 or epoca == cfg["max_epocas"]:
            t1 = time.time()
            p, l_val = predecir(modelo, dl_val, disp, perdida)
            auc, ap = ev.auc_pr(y_val, p)
            sel = ev.elegir_umbral(y_val, p)
            fila.update({"val_perdida": l_val, "val_pr_auc": ap, "val_auc": auc,
                         "val_csi_max": sel["csi_max"], "val_umbral": sel["umbral"],
                         "seg_val": time.time() - t1})
            if ap > mejor:
                mejor, sin_mejora = ap, 0
                torch.save({"estado": modelo.state_dict(), "epoca": epoca, "val_pr_auc": ap, "cfg": cfg}, ruta_mejor)
                fila["mejor"] = True
            else:
                sin_mejora += 1
            if registrar:
                registrar({k: v for k, v in fila.items() if k != "epoca" and not isinstance(v, bool)}, epoca)
            print(f"época {epoca:3d} | pérdida train {fila['train_perdida']:.4f} val {l_val:.4f} | "
                  f"PR-AUC {ap:.4f} AUC {auc:.3f} CSI máx {sel['csi_max']:.4f} | "
                  f"{fila['seg_train']:.0f}+{fila['seg_val']:.0f} s{'  ← mejor' if fila.get('mejor') else ''}")
            hist.append(fila)
            if sin_mejora >= cfg["paciencia"]:
                print(f"Parada temprana: {cfg['paciencia']} validaciones sin mejorar el PR-AUC.")
                break
        else:
            if registrar:
                registrar({"train_perdida": fila["train_perdida"]}, epoca)
            hist.append(fila)
    return pd.DataFrame(hist)


def cargar_mejor(modelo: nn.Module, ruta_mejor: Path, disp: torch.device) -> dict:
    ck = torch.load(ruta_mejor, map_location=disp, weights_only=False)
    modelo.load_state_dict(ck["estado"])
    return ck
