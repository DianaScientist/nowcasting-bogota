"""
Búsqueda de hiperparámetros de la CNN plana (S7 del plan de modelado).

Una búsqueda Optuna por cada L ∈ {2, 3, 4} (opción B, 15-oct-2026): muestreo bayesiano TPE
con semilla fija, poda por mediana y parada temprana. Criterio de selección: PR-AUC en
validación (decisión 5 de 17-referencias); el umbral de decisión se elige después por CSI.

Espacio de búsqueda (acordado el 15-oct-2026):
    lr              log-uniforme 1e-4 – 3e-3
    weight_decay    log-uniforme 1e-6 – 1e-3
    dropout         uniforme 0 – 0.5
    proporcion      1:2:2, 1:4:4, 1:8:8   (positivo : difíciles : aleatorios)
    bolsa           red_activa+casi_positivo (por defecto de S3a) | + nube_fria (variante de S3a)
    perdida         bce | bce+csi0.1 | bce+csi0.5   (BCE + peso · (1 − CSI suave))
Fijos: canales (32, 64, 128), lote 256, semilla 42, máximo 20 épocas, paciencia 5.

Con --modelo resnet (S8) se busca la ResNet de Fan et al. (2024) con el mismo procedimiento y,
además, los hiperparámetros de arquitectura que Fan buscó (Tabla A2):
    filtros         16 | 32 | 64           (filtros iniciales F; Fan: 10–100, óptimo 64)
    alpha           uniforme 0 – 0.4       (pendiente de la Leaky ReLU; Fan: 0–0.4, óptimo 0.138)
    dropout         uniforme 0 – 0.4       (Fan: 0–0.4, óptimo 0.074)

Optimizaciones frente a 18-cnn_plana (mismo modelo, mismos datos):
    - El caché completo vive en la GPU (int32, ~6.3 GB) y los parches se arman ahí por
      indexación; no hay DataLoader ni copias CPU→GPU por lote.
    - Entrenamiento con autocast bf16 (A10G); la validación se predice en float32 para que
      el PR-AUC y la grilla de umbrales no sufran empates por redondeo.
    - Poda: una configuración se corta si su mejor PR-AUC hasta la época k queda bajo la
      mediana de las anteriores en esa misma época (desde la época 4, tras 5 configuraciones).

Reanudable: el estudio vive en mlruns/optuna.db; si la VM se cae, volver a lanzar el mismo
comando continúa donde quedó (las configuraciones interrumpidas se marcan como fallidas y
no cuentan para las 30).

Uso en la VM (dentro de tmux o con nohup):
    python src/buscar.py --prueba                       # 2 configuraciones × 2 épocas, L=2
    python src/buscar.py --L 2 3 4 --apagar             # búsqueda completa y apagado
    python src/buscar.py --L 2 3 4 --horas-max 20 --apagar

Salidas en modelo/salidas_s7/L{L}/ (CNN plana) o modelo/salidas_s8/L{L}/ (ResNet): un checkpoint e historial por configuración, la tabla de
configuraciones (configuraciones.csv) y las predicciones de validación de la mejor
(pred_val_mejor.parquet). Cada configuración es además una corrida de MLflow.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import cnn                                                               # noqa: E402
import evaluacion as ev                                                  # noqa: E402
from cache_imagenes import Cache                                         # noqa: E402
from cargador import HALF, L_COMUN, MuestreadorEpoca, preparar_pares     # noqa: E402

SEMILLA = 42
TAM_LOTE = 256
LOTE_PRED = 8192
CANALES = (32, 64, 128)
PROPORCIONES = {"1:2:2": (2, 2), "1:4:4": (4, 4), "1:8:8": (8, 8)}
BOLSAS = {"red_activa+casi_positivo": ("red_activa", "casi_positivo"),
          "red_activa+casi_positivo+nube_fria": ("red_activa", "casi_positivo", "nube_fria")}  # S3a: + nube fría
PERDIDAS = {"bce": 0.0, "bce+csi0.1": 0.1, "bce+csi0.5": 0.5}


# --- Datos en la GPU --------------------------------------------------------------
class ImagenesGPU:
    """Caché completo en la GPU como int32 (torch no indexa uint16) y la normalización x·a + b."""

    def __init__(self, cache: Cache, disp: torch.device, filas_por_copia: int = 4000):
        n = cache.img.shape[0]
        self.img = torch.empty(cache.img.shape, dtype=torch.int32, device=disp)
        for i in range(0, n, filas_por_copia):
            bloque = np.asarray(cache.img[i:i + filas_por_copia]).astype(np.int32)
            self.img[i:i + len(bloque)] = torch.from_numpy(bloque).to(disp)
        nb = len(cache.bandas)
        # Misma aritmética que DatasetPares: (x·scale + offset − media)/desvío = x·a + b, en float32
        a = (cache.scale / cache.desvio).astype(np.float32).reshape(1, 1, nb, 1, 1)
        b = ((cache.offset - cache.media) / cache.desvio).astype(np.float32).reshape(1, 1, nb, 1, 1)
        self.a, self.b = torch.from_numpy(a).to(disp), torch.from_numpy(b).to(disp)
        self.nb, self.disp = nb, disp
        self.off = torch.arange(-HALF, HALF, device=disp)
        self.bnd = torch.arange(nb, device=disp)[None, None, :, None, None]


class ConjuntoGPU:
    """Pares de un conjunto (salida de preparar_pares) con sus índices ya en la GPU."""

    def __init__(self, pares: pd.DataFrame, imgs: ImagenesGPU, L: int):
        d = imgs.disp
        self.L, self.imgs = L, imgs
        self.filas = torch.from_numpy(pares[[f"fila_{j}" for j in range(L)]].to_numpy(np.int64)).to(d)
        self.fc = torch.from_numpy(pares["fila_c"].to_numpy(np.int64)).to(d)
        self.cc = torch.from_numpy(pares["col_c"].to_numpy(np.int64)).to(d)
        self.y = torch.from_numpy(pares["etiqueta"].to_numpy(np.float32)).to(d)

    def __len__(self):
        return len(self.y)

    def lote(self, idx: torch.Tensor):
        """x (B, 5·L, 32, 32) float32 normalizado e y (B,), idéntico a DatasetPares."""
        im = self.imgs
        r = (self.fc[idx, None] + im.off)[:, None, None, :, None]
        c = (self.cc[idx, None] + im.off)[:, None, None, None, :]
        f = self.filas[idx][:, :, None, None, None]
        x = im.img[f, im.bnd, r, c].float() * im.a + im.b                     # (B, L, 5, 32, 32)
        return x.reshape(len(idx), self.L * im.nb, 2 * HALF, 2 * HALF), self.y[idx]


@torch.no_grad()
def predecir_gpu(modelo, conj: ConjuntoGPU) -> np.ndarray:
    """Probabilidades en float32 (sin autocast), en el orden de los pares."""
    modelo.eval()
    n, ps = len(conj), []
    for i in range(0, n, LOTE_PRED):
        idx = torch.arange(i, min(i + LOTE_PRED, n), device=conj.y.device)
        x, _ = conj.lote(idx)
        ps.append(torch.sigmoid(modelo(x).float()))
    return torch.cat(ps).cpu().numpy()


# --- Pérdida ----------------------------------------------------------------------
def perdida_compuesta(peso_csi: float):
    def f(logits, y):
        logits = logits.float()
        l = cnn.perdida_bce(logits, y)
        return l + peso_csi * cnn.perdida_csi(logits, y) if peso_csi > 0 else l
    return f


# --- Modelos ----------------------------------------------------------------------
MODELOS = {"cnn_plana": {"sesion": "s7", "etiqueta": "cnn-plana"},
           "resnet": {"sesion": "s8", "etiqueta": "resnet"}}


def construir_modelo(cfg: dict) -> torch.nn.Module:
    if cfg.get("modelo", "cnn_plana") == "resnet":
        return cnn.ResNetFan(L=cfg["L"], filtros=cfg["filtros"], alpha=cfg["alpha"], dropout=cfg["dropout"])
    return cnn.CNNPlana(L=cfg["L"], canales=CANALES, dropout=cfg["dropout"])


# --- Una configuración ------------------------------------------------------------
def entrenar_config(cfg: dict, tr: ConjuntoGPU, pares_tr: pd.DataFrame, val: ConjuntoGPU, y_val: np.ndarray,
                    disp: torch.device, ruta_mejor: Path, informar=None) -> pd.DataFrame:
    """
    Entrena con parada temprana por PR-AUC en validación (validada en cada época).
    `informar(valor, época)` recibe el mejor PR-AUC hasta la época (para la poda de Optuna);
    puede lanzar optuna.TrialPruned.
    """
    cnn.fijar_semilla(cfg["semilla"])
    modelo = construir_modelo(cfg).to(disp)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    perdida = perdida_compuesta(cfg["peso_csi"])
    n_d, n_a = PROPORCIONES[cfg["proporcion"]]
    mu = MuestreadorEpoca(pares_tr, n_dificiles=n_d, n_aleatorios=n_a, semilla=cfg["semilla"])
    usar_bf16 = disp.type == "cuda" and torch.cuda.is_bf16_supported()

    mejor, sin_mejora, hist = -np.inf, 0, []
    for epoca in range(1, cfg["max_epocas"] + 1):
        t0 = time.time()
        mu.fijar_epoca(epoca)
        orden = torch.from_numpy(mu.indices()).to(disp)
        modelo.train()
        total = torch.zeros((), device=disp)
        for i in range(0, len(orden), TAM_LOTE):
            idx = orden[i:i + TAM_LOTE]
            x, y = tr.lote(idx)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(disp.type, dtype=torch.bfloat16, enabled=usar_bf16):
                logits = modelo(x)
            l = perdida(logits, y)
            l.backward()
            opt.step()
            total += l.detach() * len(idx)
        if disp.type == "cuda":
            torch.cuda.synchronize()
        fila = {"epoca": epoca, "train_perdida": float(total) / len(orden), "seg_train": time.time() - t0}

        t1 = time.time()
        p = predecir_gpu(modelo, val)
        auc, ap = ev.auc_pr(y_val, p)
        sel = ev.elegir_umbral(y_val, p)
        fila.update({"val_pr_auc": ap, "val_auc": auc, "val_csi_max": sel["csi_max"],
                     "val_umbral": sel["umbral"], "seg_val": time.time() - t1})
        if not math.isfinite(fila["train_perdida"]):
            hist.append(fila)
            raise FloatingPointError(f"Pérdida no finita en la época {epoca}")
        if ap > mejor:
            mejor, sin_mejora, fila["mejor"] = ap, 0, True
            torch.save({"estado": modelo.state_dict(), "epoca": epoca, "val_pr_auc": ap, "cfg": cfg}, ruta_mejor)
        else:
            sin_mejora, fila["mejor"] = sin_mejora + 1, False
        hist.append(fila)
        print(f"  época {epoca:2d} | pérdida {fila['train_perdida']:.4f} | PR-AUC {ap:.4f} AUC {auc:.3f} "
              f"CSI {sel['csi_max']:.4f} | {fila['seg_train']:.1f}+{fila['seg_val']:.1f} s"
              f"{'  ← mejor' if fila['mejor'] else ''}", flush=True)
        if informar:
            informar(mejor, epoca, pd.DataFrame(hist))
        if sin_mejora >= cfg["paciencia"]:
            break
    return pd.DataFrame(hist)


# --- Estudio por L ----------------------------------------------------------------
def configurar_mlflow(modelo: str = "cnn_plana"):
    try:
        import mlflow
    except ImportError:
        return None
    (RAIZ / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlruns' / 'mlflow.db').as_posix()}")
    m = MODELOS[modelo]
    nombre = f"{m['sesion']}-busqueda-{m['etiqueta']}"
    if mlflow.get_experiment_by_name(nombre) is None:
        mlflow.create_experiment(nombre, artifact_location=(RAIZ / "mlruns" / "artefactos").as_uri())
    mlflow.set_experiment(nombre)
    return mlflow


def buscar_L(L: int, args, cache: Cache, imgs: ImagenesGPU, indice, atributos, val_pares, y_val,
             disp: torch.device, mlflow, limite: float | None):
    import optuna
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    sufijo = "-prueba" if args.prueba else ""
    m = MODELOS[args.modelo]
    dir_sal = RAIZ / "modelo" / f"salidas_{m['sesion']}{sufijo}" / f"L{L}"
    dir_sal.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pares_tr = {b: preparar_pares(indice, atributos, cache, "train", L, criterios=c) for b, c in BOLSAS.items()}
    tr = {b: ConjuntoGPU(p, imgs, L) for b, p in pares_tr.items()}
    val = ConjuntoGPU(cnn.ultimos_instantes(val_pares, L), imgs, L)
    for b, p in pares_tr.items():
        print(f"L={L} train [{b}]: {len(p):,} pares, {int(p.etiqueta.sum()):,} positivos, "
              f"bolsas {p.bolsa.value_counts().to_dict()}")
    print(f"L={L} val: {len(val):,} pares ({time.time() - t0:.0f} s preparando)", flush=True)

    almacen = optuna.storages.RDBStorage(
        f"sqlite:///{(RAIZ / 'mlruns' / 'optuna.db').as_posix()}",
        heartbeat_interval=60, grace_period=300)
    nombre_estudio = f"{m['sesion']}-{m['etiqueta']}-L{L}{sufijo}"
    try:
        previas = len(optuna.load_study(study_name=nombre_estudio, storage=almacen).trials)
    except KeyError:
        previas = 0
    # Semilla del muestreador desplazada por las configuraciones previas: al reanudar no repite
    # las mismas propuestas iniciales (reproducible dado el historial del estudio).
    estudio = optuna.create_study(
        study_name=nombre_estudio, storage=almacen, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA + previas, multivariate=True, n_startup_trials=8),
        pruner=(optuna.pruners.NopPruner() if args.prueba else
                optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)))
    optuna.storages.fail_stale_trials(estudio)

    estados_cuentan = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    hechos = len(estudio.get_trials(deepcopy=False, states=estados_cuentan))
    faltan = args.n_configs - hechos
    print(f"L={L}: {hechos} configuraciones hechas, faltan {max(faltan, 0)}", flush=True)

    def objetivo(trial: "optuna.Trial") -> float:
        cfg = {
            "L": L,
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4 if args.modelo == "resnet" else 0.5),
            "proporcion": trial.suggest_categorical("proporcion", list(PROPORCIONES)),
            "bolsa": trial.suggest_categorical("bolsa", list(BOLSAS)),
            "perdida": trial.suggest_categorical("perdida", list(PERDIDAS)),
            "semilla": SEMILLA, "max_epocas": args.max_epocas, "paciencia": args.paciencia,
            "tam_lote": TAM_LOTE, "modelo": args.modelo,
        }
        if args.modelo == "resnet":
            cfg["filtros"] = trial.suggest_categorical("filtros", [16, 32, 64])
            cfg["alpha"] = trial.suggest_float("alpha", 0.0, 0.4)
        else:
            cfg["canales"] = CANALES
        cfg["peso_csi"] = PERDIDAS[cfg["perdida"]]
        nombre = f"L{L}_config{trial.number:03d}"
        print(f"\n[{nombre}] {json.dumps({k: cfg[k] for k in trial.params}, default=str)}", flush=True)
        ruta = dir_sal / f"{nombre}.pt"

        def informar(mejor, epoca, hist):
            if mlflow:
                f = hist.iloc[-1]
                mlflow.log_metrics({k: float(f[k]) for k in ("train_perdida", "val_pr_auc", "val_auc",
                                                              "val_csi_max", "val_umbral", "seg_train", "seg_val")},
                                   step=epoca)
            trial.report(mejor, epoca)
            if trial.should_prune():
                hist.to_csv(dir_sal / f"historial_{nombre}.csv", index=False)
                print(f"  podada en la época {epoca}", flush=True)
                raise optuna.TrialPruned()

        corrida = mlflow.start_run(run_name=nombre) if mlflow else None
        try:
            if mlflow:
                mlflow.log_params({k: str(v) for k, v in cfg.items()} | {"trial": trial.number})
                mlflow.set_tags({"busqueda": f"{m['sesion']}-L{L}{sufijo}", "modelo": args.modelo})
            t = time.time()
            hist = entrenar_config(cfg, tr[cfg["bolsa"]], pares_tr[cfg["bolsa"]], val, y_val, disp, ruta, informar)
            hist.to_csv(dir_sal / f"historial_{nombre}.csv", index=False)
            m = hist.loc[hist.val_pr_auc.idxmax()]
            for k, v in {"epoca_mejor": int(m.epoca), "val_auc": float(m.val_auc), "val_csi_max": float(m.val_csi_max),
                         "val_umbral": float(m.val_umbral), "epocas": int(hist.epoca.max()),
                         "minutos": (time.time() - t) / 60}.items():
                trial.set_user_attr(k, v)
            if mlflow:
                mlflow.log_metrics({"mejor_val_pr_auc": float(m.val_pr_auc), "epoca_mejor": int(m.epoca)})
            return float(m.val_pr_auc)
        except optuna.TrialPruned:
            if mlflow:
                mlflow.set_tag("estado", "podada")
            raise
        finally:
            if corrida:
                mlflow.end_run()

    if faltan > 0:
        estudio.optimize(
            objetivo, n_trials=None, timeout=None if limite is None else max(limite - time.time(), 0),
            callbacks=[optuna.study.MaxTrialsCallback(args.n_configs, states=estados_cuentan),
                       lambda s, t: torch.cuda.empty_cache() if disp.type == "cuda" else None],
            catch=(FloatingPointError, torch.cuda.OutOfMemoryError), gc_after_trial=True)

    # Tabla de configuraciones y predicciones de validación de la mejor
    tabla = estudio.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    tabla.to_csv(dir_sal / "configuraciones.csv", index=False)
    completas = estudio.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not completas:
        print(f"L={L}: ninguna configuración completa todavía")
        return None
    best = estudio.best_trial
    ck = torch.load(dir_sal / f"L{L}_config{best.number:03d}.pt", map_location=disp, weights_only=False)
    modelo = construir_modelo(ck["cfg"]).to(disp)
    modelo.load_state_dict(ck["estado"])
    p = predecir_gpu(modelo, val)
    ap = ev.auc_pr(y_val, p)[1]
    assert abs(ap - best.value) < 1e-6, f"Las predicciones no reproducen el PR-AUC de la mejor ({ap} ≠ {best.value})"
    pd.DataFrame({"codigoestacion": val_pares.codigoestacion, "timestamp": val_pares.timestamp,
                  "etiqueta": y_val, "p": p}).to_parquet(dir_sal / "pred_val_mejor.parquet")
    print(f"\nL={L}: mejor configuración {best.number} | PR-AUC {best.value:.4f} | {best.params}", flush=True)
    return best


# --- Programa ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=list(MODELOS), default="cnn_plana")
    ap.add_argument("--L", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--n-configs", type=int, default=30, help="configuraciones (completas + podadas) por L")
    ap.add_argument("--max-epocas", type=int, default=20)
    ap.add_argument("--paciencia", type=int, default=5)
    ap.add_argument("--horas-max", type=float, default=None, help="tope de tiempo total (protege el presupuesto)")
    ap.add_argument("--prueba", action="store_true", help="2 configuraciones × 2 épocas en L=2, estudio aparte")
    ap.add_argument("--apagar", action="store_true", help="apagar la VM al terminar (también si hay error)")
    args = ap.parse_args()
    if args.prueba:
        args.L, args.n_configs, args.max_epocas = args.L if args.L != [2, 3, 4] else [2], 2, 2

    limite = None if args.horas_max is None else time.time() + args.horas_max * 3600
    t_ini = time.time()
    try:
        disp = cnn.dispositivo()
        print("PyTorch", torch.__version__, "| dispositivo:", disp, flush=True)
        if disp.type == "cuda":
            print(torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        zs = pd.read_csv(RAIZ / "data_ingestion" / "satellite" / "normalizacion_zscore.csv")
        cache = Cache(RAIZ / "data" / "cache", zs, en_memoria=False)
        indice = pd.read_parquet(RAIZ / "datasets" / "indice_completo.parquet")
        atributos = pd.read_parquet(RAIZ / "datasets" / "negativos_atributos.parquet")
        t = time.time()
        imgs = ImagenesGPU(cache, disp)
        print(f"Caché {tuple(imgs.img.shape)} en {disp} ({time.time() - t:.0f} s)", flush=True)
        val_pares = preparar_pares(indice, atributos, cache, "val", L_COMUN)
        y_val = val_pares.etiqueta.to_numpy()

        mlflow = configurar_mlflow(args.modelo)
        for L in args.L:
            if limite is not None and time.time() >= limite:
                print(f"Tope de {args.horas_max} h alcanzado; L={L} queda para la próxima vez")
                break
            buscar_L(L, args, cache, imgs, indice, atributos, val_pares, y_val, disp, mlflow, limite)
        print(f"\nFin: {(time.time() - t_ini) / 3600:.2f} h", flush=True)
    finally:
        if args.apagar:
            print("Apagando la VM en 1 minuto…", flush=True)
            subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)


if __name__ == "__main__":
    main()
