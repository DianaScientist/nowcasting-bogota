"""
Congelamiento de las redes (S9 del plan de modelado).

Las seis configuraciones de redes (CNN plana y ResNet × L = 2, 3, 4) se entrenan con 3 semillas
cada una (18 corridas). Para cada corrida: parada temprana por PR-AUC en validación (mismo
procedimiento que la búsqueda), predicciones de validación de la mejor época y umbral de
decisión por CSI en validación. La prueba sigue cerrada.

Regla de selección, fijada el 16-oct-2026 antes de correr S8b (opción 1A):
    - CNN plana: la mejor configuración (PR-AUC en validación) del estudio s7-cnn-plana-L{L}.
    - ResNet: la de mayor PR-AUC en validación entre el mejor de s8-resnet-L{L} y el mejor de
      s8b-resnet-L{L} (S8b amplió lr hacia abajo y alpha hacia arriba).
La selección se escribe en salidas_s9/seleccion.csv antes de entrenar.

Semillas: 42, 43, 44. La semilla 42 repite la de la búsqueda; la diferencia entre su PR-AUC y el
de la búsqueda mide el no determinismo de la GPU y queda en el resumen (pr_auc_busqueda).
El PR-AUC de validación del ganador de la búsqueda no se reporta como resultado (sesgo del
ganador); lo que se reporta es la media ± dispersión de las 3 semillas.

Reanudable: las corridas ya registradas en resumen.csv se saltan.

Uso en la VM:
    python src/congelar.py --prueba        # L=2, semilla 42, 1 época, salidas aparte
    python src/congelar.py --apagar        # 18 corridas y apagado

Salidas en modelo/salidas_s9/: seleccion.csv, resumen.csv y, por corrida, el checkpoint,
historial, predicciones de validación y curva CSI–umbral. Cada corrida es una corrida de MLflow
(experimento s9-congelamiento, etiqueta congelado=si).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import buscar as bu                                                      # noqa: E402
import cnn                                                               # noqa: E402
import evaluacion as ev                                                  # noqa: E402
from cache_imagenes import Cache                                         # noqa: E402
from cargador import L_COMUN, preparar_pares                             # noqa: E402

SEMILLAS = (42, 43, 44)
ESTUDIOS = {"cnn_plana": ["s7-cnn-plana"], "resnet": ["s8-resnet", "s8b-resnet"]}
CLAVES = ["modelo", "L", "semilla"]


# --- Selección --------------------------------------------------------------------
def seleccionar(almacen, modelo: str, L: int, prueba: bool) -> dict:
    """Mejor configuración completa entre los estudios del modelo para ese L (regla 1A)."""
    import optuna
    candidatas = []
    for prefijo in ESTUDIOS[modelo]:
        nombre = f"{prefijo}-L{L}"
        try:
            estudio = optuna.load_study(study_name=nombre, storage=almacen)
        except KeyError:
            if prueba:
                print(f"  Aviso: no existe el estudio {nombre}; en prueba se omite", flush=True)
                continue
            raise RuntimeError(f"Falta el estudio {nombre}: la selección no puede aplicar la regla completa")
        completas = estudio.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not completas:
            raise RuntimeError(f"El estudio {nombre} no tiene configuraciones completas")
        b = max(completas, key=lambda t: t.value)
        candidatas.append({"modelo": modelo, "L": L, "estudio": nombre, "trial": b.number,
                           "pr_auc_busqueda": float(b.value), "epoca_busqueda": b.user_attrs.get("epoca_mejor"),
                           "params": b.params})
        print(f"  {nombre}: mejor trial {b.number}, PR-AUC {b.value:.4f}", flush=True)
    if not candidatas:
        raise RuntimeError(f"Sin candidatas para {modelo} L={L}")
    elegida = max(candidatas, key=lambda c: c["pr_auc_busqueda"])
    elegida["candidatas"] = "; ".join(f"{c['estudio']}#{c['trial']}={c['pr_auc_busqueda']:.4f}" for c in candidatas)
    return elegida


def cfg_desde(sel: dict, semilla: int, max_epocas: int, paciencia: int) -> dict:
    p = sel["params"]
    cfg = {"L": sel["L"], **p, "semilla": semilla, "max_epocas": max_epocas, "paciencia": paciencia,
           "tam_lote": bu.TAM_LOTE, "modelo": sel["modelo"], "peso_csi": bu.PERDIDAS[p["perdida"]]}
    if sel["modelo"] == "cnn_plana":
        cfg["canales"] = bu.CANALES
    return cfg


# --- MLflow -----------------------------------------------------------------------
def configurar_mlflow(prueba: bool):
    try:
        import mlflow
    except ImportError:
        return None
    (RAIZ / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlruns' / 'mlflow.db').as_posix()}")
    nombre = "s9-congelamiento" + ("-prueba" if prueba else "")
    if mlflow.get_experiment_by_name(nombre) is None:
        mlflow.create_experiment(nombre, artifact_location=(RAIZ / "mlruns" / "artefactos").as_uri())
    mlflow.set_experiment(nombre)
    return mlflow


# --- Programa ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", choices=list(ESTUDIOS), default=list(ESTUDIOS))
    ap.add_argument("--L", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--semillas", type=int, nargs="+", default=list(SEMILLAS))
    ap.add_argument("--max-epocas", type=int, default=20)
    ap.add_argument("--paciencia", type=int, default=5)
    ap.add_argument("--prueba", action="store_true", help="L=2, semilla 42, 1 época, salidas aparte")
    ap.add_argument("--apagar", action="store_true", help="apagar la VM al terminar (también si hay error)")
    args = ap.parse_args()
    if args.prueba:
        args.L, args.semillas, args.max_epocas = [2], [42], 1

    t_ini = time.time()
    try:
        import optuna
        disp = cnn.dispositivo()
        print("PyTorch", torch.__version__, "| dispositivo:", disp, flush=True)
        dir_sal = RAIZ / "modelo" / ("salidas_s9" + ("-prueba" if args.prueba else ""))
        dir_sal.mkdir(parents=True, exist_ok=True)

        # 1. Selección (antes de entrenar)
        almacen = optuna.storages.RDBStorage(f"sqlite:///{(RAIZ / 'mlruns' / 'optuna.db').as_posix()}")
        print("\nSelección (regla 1A):", flush=True)
        sel = {(m, L): seleccionar(almacen, m, L, args.prueba) for m in args.modelos for L in args.L}
        tabla_sel = pd.DataFrame([{k: (json.dumps(v) if k == "params" else v) for k, v in s.items()}
                                  for s in sel.values()])
        tabla_sel.to_csv(dir_sal / "seleccion.csv", index=False)
        print(tabla_sel[["modelo", "L", "estudio", "trial", "pr_auc_busqueda"]].to_string(index=False), flush=True)

        # 2. Datos
        zs = pd.read_csv(RAIZ / "data_ingestion" / "satellite" / "normalizacion_zscore.csv")
        cache = Cache(RAIZ / "data" / "cache", zs, en_memoria=False)
        indice = pd.read_parquet(RAIZ / "datasets" / "indice_completo.parquet")
        atributos = pd.read_parquet(RAIZ / "datasets" / "negativos_atributos.parquet")
        t = time.time()
        imgs = bu.ImagenesGPU(cache, disp)
        print(f"\nCaché {tuple(imgs.img.shape)} en {disp} ({time.time() - t:.0f} s)", flush=True)
        val_pares = preparar_pares(indice, atributos, cache, "val", L_COMUN)
        y_val = val_pares.etiqueta.to_numpy()
        mlflow = configurar_mlflow(args.prueba)

        ruta_resumen = dir_sal / "resumen.csv"
        resumen = pd.read_csv(ruta_resumen) if ruta_resumen.exists() else pd.DataFrame(columns=CLAVES)
        hechas = set(map(tuple, resumen[CLAVES].astype({"L": int, "semilla": int}).values.tolist()))

        # 3. Corridas
        for L in args.L:
            bolsas = {sel[(m, L)]["params"]["bolsa"] for m in args.modelos}
            pares_tr = {b: preparar_pares(indice, atributos, cache, "train", L, criterios=bu.BOLSAS[b]) for b in bolsas}
            tr = {b: bu.ConjuntoGPU(p, imgs, L) for b, p in pares_tr.items()}
            val = bu.ConjuntoGPU(cnn.ultimos_instantes(val_pares, L), imgs, L)
            for m in args.modelos:
                s = sel[(m, L)]
                for semilla in args.semillas:
                    if (m, L, semilla) in hechas:
                        print(f"\n{m} L={L} semilla {semilla}: ya está, se salta", flush=True)
                        continue
                    nombre = f"{m}_L{L}_s{semilla}"
                    cfg = cfg_desde(s, semilla, args.max_epocas, args.paciencia)
                    print(f"\n[{nombre}] de {s['estudio']}#{s['trial']} | {json.dumps(s['params'])}", flush=True)
                    ruta = dir_sal / f"{nombre}.pt"
                    corrida = mlflow.start_run(run_name=nombre) if mlflow else None
                    try:
                        if mlflow:
                            mlflow.log_params({k: str(v) for k, v in cfg.items()} | {"estudio": s["estudio"], "trial": s["trial"]})
                            mlflow.set_tags({"congelado": "si", "modelo": m, "L": str(L), "semilla": str(semilla)})
                        t = time.time()
                        hist = bu.entrenar_config(cfg, tr[cfg["bolsa"]], pares_tr[cfg["bolsa"]], val, y_val, disp, ruta)
                        hist.to_csv(dir_sal / f"historial_{nombre}.csv", index=False)

                        ck = torch.load(ruta, map_location=disp, weights_only=False)
                        modelo = bu.construir_modelo(ck["cfg"]).to(disp)
                        modelo.load_state_dict(ck["estado"])
                        p = bu.predecir_gpu(modelo, val)
                        auc, ap_ = ev.auc_pr(y_val, p)
                        assert abs(ap_ - ck["val_pr_auc"]) < 1e-6, f"El checkpoint no reproduce su PR-AUC ({ap_} ≠ {ck['val_pr_auc']})"
                        u = ev.elegir_umbral(y_val, p)
                        u["curva"].to_csv(dir_sal / f"curva_umbral_{nombre}.csv", index=False)
                        pd.DataFrame({"codigoestacion": val_pares.codigoestacion, "timestamp": val_pares.timestamp,
                                      "etiqueta": y_val, "p": p}).to_parquet(dir_sal / f"pred_val_{nombre}.parquet")

                        fila = {"modelo": m, "L": L, "semilla": semilla, "estudio": s["estudio"], "trial": s["trial"],
                                "epoca_mejor": int(ck["epoca"]), "epocas": int(hist.epoca.max()),
                                "val_pr_auc": ap_, "val_auc": auc, "umbral": u["umbral"], "val_csi": u["csi_max"],
                                "meseta_desde": u["meseta_desde"], "meseta_hasta": u["meseta_hasta"],
                                "pr_auc_busqueda": s["pr_auc_busqueda"], "minutos": (time.time() - t) / 60}
                        resumen = pd.concat([resumen, pd.DataFrame([fila])], ignore_index=True)
                        resumen.to_csv(ruta_resumen, index=False)
                        if mlflow:
                            mlflow.log_metrics({k: float(fila[k]) for k in ("val_pr_auc", "val_auc", "umbral", "val_csi", "epoca_mejor")})
                        print(f"  → PR-AUC {ap_:.4f} | AUC {auc:.3f} | umbral {u['umbral']:.2f} CSI {u['csi_max']:.4f}"
                              f" | época {ck['epoca']} | {fila['minutos']:.1f} min", flush=True)
                    finally:
                        if corrida:
                            mlflow.end_run()
                        if disp.type == "cuda":
                            torch.cuda.empty_cache()

        # 4. Media ± dispersión por configuración
        r = resumen[resumen.modelo.isin(args.modelos) & resumen.L.isin(args.L)]
        agg = r.groupby(["modelo", "L"]).agg(n=("semilla", "size"),
                                             pr_auc_media=("val_pr_auc", "mean"), pr_auc_de=("val_pr_auc", "std"),
                                             auc_media=("val_auc", "mean"), csi_media=("val_csi", "mean"),
                                             umbral_media=("umbral", "mean"), pr_auc_busqueda=("pr_auc_busqueda", "first"))
        agg.to_csv(dir_sal / "resumen_por_configuracion.csv")
        print("\nResumen por configuración (validación):\n" + agg.round(4).to_string(), flush=True)
        print(f"\nFin: {(time.time() - t_ini) / 3600:.2f} h", flush=True)
    finally:
        if args.apagar:
            print("Apagando la VM en 1 minuto…", flush=True)
            subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)


if __name__ == "__main__":
    main()
