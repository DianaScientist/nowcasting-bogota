"""
Apertura única del conjunto de prueba (S11 del plan de modelado).

Evalúa en prueba, con todo congelado en validación:
    peldaño 1   umbral sobre el p10 de C13 (logística de una variable, como en S5)
    peldaño 2   regresión logística sobre estadísticas de parche, L = 2, 3, 4 (C = 10, S5)
    peldaño 3   CNN plana, L = 2, 3, 4 × 3 semillas (S9)
    peldaño 4   ResNet,    L = 2, 3, 4 × 3 semillas (S9)
Cada modelo usa el umbral de decisión que se eligió por CSI en validación (las redes, el de
salidas_s9/resumen.csv; las referencias se reajustan, son deterministas, y lo re-eligen).

Métricas por pares (POD, FAR, CSI, ETS) y sin umbral (AUC, PR-AUC), con IC del bootstrap por
bloques de día pluviométrico (B = 10,000, semilla 42, los mismos pesos para todos los modelos).
Cada red se reporta por semilla (con su IC) y por configuración como media ± dispersión entre sus
3 semillas (decisión del 15-oct-2026). Si más del 1 % de las réplicas de una métrica queda
indefinida (NaN), su IC se marca como poco confiable (decisión 6 de S4a).

Comparaciones pareadas fijadas antes de abrir prueba (escalera de complejidad, plan de modelado):
    logística L − umbral (L = 2, 3, 4); CNN plana L − logística L; ResNet L − CNN plana L;
    L = 3 y L = 4 − L = 2 dentro de logística, CNN plana y ResNet;
    mejor modelo − umbral, donde el mejor modelo es la configuración (de las 9) con mayor PR-AUC
    medio en validación.
Con redes, cada comparación se hace semilla por semilla (decisión del 15-oct-2026): red − referencia
con cada una de las 3 semillas; red − red emparejando la misma semilla (42−42, 43−43, 44−44). Se
reportan las tres diferencias y si la conclusión (IC sobre 0, bajo 0 o que lo incluye) cambia con
la semilla. Todas se reportan, en CSI y en PR-AUC.

En el ensayo, además, la sensibilidad del largo de bloque (6, 12, 24 y 48 h) del IC del CSI con
las redes finales en validación (plan, decisión 2 de S4a: revisar el bloque con las redes finales
antes de abrir prueba).

Las métricas por evento, los estratos y el análisis por estación son de S12: aquí se guardan
las predicciones de prueba de cada modelo para hacerlos sin volver a abrir nada.

Uso en la VM:
    python src/abrir_prueba.py --ensayo        # todo sobre VALIDACIÓN con B = 200 (no toca prueba)
    python src/abrir_prueba.py --apagar        # apertura real y apagado

El bootstrap del AUC y del PR-AUC (~20 min por modelo) corre en paralelo (--procesos, 4 por
defecto, uno por vCPU de la g5.xlarge). Reanudable: las réplicas ya guardadas no se recalculan.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import buscar as bu                                                      # noqa: E402
import cnn                                                               # noqa: E402
import evaluacion as ev                                                  # noqa: E402
import referencias as rf                                                 # noqa: E402
from cache_imagenes import Cache                                         # noqa: E402
from cargador import L_COMUN, preparar_pares                             # noqa: E402

SEMILLA = 42
C_LOGISTICA = {2: 10.0, 3: 10.0, 4: 10.0}      # cierre de S5 (C elegido por PR-AUC en validación)
REDES = ("cnn_plana", "resnet")
LS = (2, 3, 4)
SEMILLAS = (42, 43, 44)
METRICAS_UMBRAL = ("pod", "far", "csi", "ets")


# --- Configuraciones y comparaciones ----------------------------------------------
def configuraciones() -> dict:
    """nombre de configuración -> lista de modelos (los de las redes son sus 3 semillas)."""
    cfg = {"umbral_c13": ["umbral_c13"]}
    cfg |= {f"logistica_L{L}": [f"logistica_L{L}"] for L in LS}
    cfg |= {f"{m}_L{L}": [f"{m}_L{L}_s{s}" for s in SEMILLAS] for m in REDES for L in LS}
    return cfg


def comparaciones(mejor_modelo: str) -> list[tuple[str, str]]:
    c = [(f"logistica_L{L}", "umbral_c13") for L in LS]
    c += [(f"cnn_plana_L{L}", f"logistica_L{L}") for L in LS]
    c += [(f"resnet_L{L}", f"cnn_plana_L{L}") for L in LS]
    c += [(f"{m}_L{L}", f"{m}_L2") for m in ("logistica", *REDES) for L in (3, 4)]
    if mejor_modelo != "umbral_c13":                 # si el umbral fuera el mejor, no hay nada que comparar
        c += [(mejor_modelo, "umbral_c13")]
    return c


# --- Bootstrap del AUC en paralelo ------------------------------------------------
_COMPARTIDO: dict = {}


def _auc_un_modelo(nombre: str):
    d = _COMPARTIDO
    rep = ev.bootstrap_auc_pr(d["ts"], d["y"], d["p"][nombre], d["pesos"], d["bloques"])
    np.savez(d["dir"] / f"replicas_auc_{nombre}.npz", auc=rep["auc"], pr_auc=rep["pr_auc"])
    return nombre


# --- Programa ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensayo", action="store_true", help="todo sobre validación con B = 200; no toca prueba")
    ap.add_argument("--B", type=int, default=None, help="réplicas (por defecto 10,000; 200 en ensayo)")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--apagar", action="store_true")
    args = ap.parse_args()
    conjunto = "val" if args.ensayo else "test"
    B = args.B or (200 if args.ensayo else ev.B_DEFECTO)

    t_ini = time.time()
    try:
        dir_s9 = RAIZ / "modelo" / "salidas_s9"
        dir_sal = RAIZ / "modelo" / ("salidas_s11" + ("-ensayo" if args.ensayo else ""))
        dir_sal.mkdir(parents=True, exist_ok=True)
        print(f"Conjunto evaluado: {conjunto.upper()} | B = {B:,} | salidas en {dir_sal}", flush=True)

        # 0. Insumos congelados (S9) y regla del mejor modelo, antes de mirar prueba
        res9 = pd.read_csv(dir_s9 / "resumen.csv")
        assert len(res9) == 18 and not res9.duplicated(["modelo", "L", "semilla"]).any(), "S9 incompleta"
        for _, f in res9.iterrows():
            assert (dir_s9 / f"{f.modelo}_L{f.L}_s{f.semilla}.pt").exists(), f"Falta el checkpoint de {f.modelo} L{f.L} s{f.semilla}"
        cfgs = configuraciones()

        # 1. Datos
        disp = cnn.dispositivo()
        zs = pd.read_csv(RAIZ / "data_ingestion" / "satellite" / "normalizacion_zscore.csv")
        cache = Cache(RAIZ / "data" / "cache", zs, en_memoria=False)
        indice = pd.read_parquet(RAIZ / "datasets" / "indice_completo.parquet")
        atributos = pd.read_parquet(RAIZ / "datasets" / "negativos_atributos.parquet")
        val = preparar_pares(indice, atributos, cache, "val", L_COMUN)
        evl = val if args.ensayo else preparar_pares(indice, atributos, cache, "test", L_COMUN)
        y_val, y = val.etiqueta.to_numpy(), evl.etiqueta.to_numpy()
        print(f"val {len(val):,} pares ({int(y_val.sum()):,} positivos) | {conjunto} {len(evl):,} pares "
              f"({int(y.sum()):,} positivos, {evl.codigoestacion.nunique()} estaciones)", flush=True)

        p, umbral = {}, {}

        # 2. Referencias (deterministas: se reajustan como en S5)
        ruta_est = RAIZ / "data" / "cache" / "estadisticas_parche"
        if not ruta_est.with_suffix(".npy").exists():
            print("Calculando estadísticas de parche (no estaban en la VM)…", flush=True)
            t = time.time()
            est, codigos = rf.calcular_estadisticas(cache, lote=100)
            rf.guardar_estadisticas(ruta_est, est, codigos, cache.bandas)
            print(f"  {(time.time() - t) / 60:.1f} min", flush=True)
        est, meta = rf.leer_estadisticas(ruta_est)
        codigos, bandas = meta["estaciones"], meta["bandas"]
        dir_comun = RAIZ / "modelo" / "salidas_s11-comun"        # el ensayo y la apertura comparten los ajustes
        dir_comun.mkdir(parents=True, exist_ok=True)
        ruta_ref = dir_comun / "referencias.pkl"
        if ruta_ref.exists():
            refs = pickle.loads(ruta_ref.read_bytes())
        else:
            refs, t = {}, time.time()
            tr2 = preparar_pares(indice, atributos, cache, "train", 2)
            refs["umbral_c13"] = rf.ajustar(rf.p10_c13(tr2, est, codigos, bandas, 2), tr2.etiqueta.to_numpy(),
                                            rf.pesos_muestreo(tr2.bolsa), C=None)
            for L in LS:
                tr = tr2 if L == 2 else preparar_pares(indice, atributos, cache, "train", L)
                refs[f"logistica_L{L}"] = rf.ajustar(rf.variables_pares(tr, est, codigos, L), tr.etiqueta.to_numpy(),
                                                     rf.pesos_muestreo(tr.bolsa), C=C_LOGISTICA[L])
            ruta_ref.write_bytes(pickle.dumps(refs))
            print(f"Referencias ajustadas en {(time.time() - t) / 60:.1f} min", flush=True)

        def x_ref(nombre, pares):
            if nombre == "umbral_c13":
                return rf.p10_c13(pares, est, codigos, bandas, L_COMUN)
            L = int(nombre[-1])
            return rf.variables_pares(pares, est, codigos, L_COMUN)[:, -L * len(bandas) * 4:]

        pr_auc_val = {}
        for nombre, mod in refs.items():
            pv = rf.predecir(mod, x_ref(nombre, val))
            umbral[nombre] = ev.elegir_umbral(y_val, pv)["umbral"]
            pr_auc_val[nombre] = ev.auc_pr(y_val, pv)[1]
            print(f"  {nombre}: PR-AUC val {pr_auc_val[nombre]:.4f} | umbral {umbral[nombre]:.2f}", flush=True)

        # Regla del mejor modelo, fijada con validación antes de mirar prueba
        for (m, L), v in res9.groupby(["modelo", "L"]).val_pr_auc.mean().items():
            pr_auc_val[f"{m}_L{L}"] = v
        candidatos = {c: pr_auc_val[c] for c in cfgs}
        mejor_modelo = max(candidatos, key=candidatos.get)
        print(f"Mejor modelo (mayor PR-AUC medio en validación de las 9 configuraciones): {mejor_modelo} "
              f"({candidatos[mejor_modelo]:.4f})", flush=True)
        comps = comparaciones(mejor_modelo)
        (dir_sal / "diseno.json").write_text(json.dumps(
            {"conjunto": conjunto, "B": B, "semilla": SEMILLA, "mejor_modelo": mejor_modelo,
             "pr_auc_val_por_configuracion": candidatos, "configuraciones": cfgs, "comparaciones": comps}, indent=2))

        # 3. Redes congeladas: primero se verifica en validación que cada checkpoint reproduce
        #    su PR-AUC de S9 (las 18); solo si todas pasan se predice el conjunto evaluado.
        imgs = bu.ImagenesGPU(cache, disp)
        modelos = {}
        for L in LS:
            conj_val = bu.ConjuntoGPU(cnn.ultimos_instantes(val, L), imgs, L)
            for m in REDES:
                for s in SEMILLAS:
                    nombre = f"{m}_L{L}_s{s}"
                    f = res9.query("modelo == @m and L == @L and semilla == @s").iloc[0]
                    ck = torch.load(dir_s9 / f"{nombre}.pt", map_location=disp, weights_only=False)
                    modelo = bu.construir_modelo(ck["cfg"]).to(disp)
                    modelo.load_state_dict(ck["estado"])
                    pv = bu.predecir_gpu(modelo, conj_val)
                    ap_v = ev.auc_pr(y_val, pv)[1]
                    assert abs(ap_v - f.val_pr_auc) < 1e-6, f"{nombre}: no reproduce su PR-AUC de validación ({ap_v} ≠ {f.val_pr_auc})"
                    umbral[nombre] = float(f.umbral)
                    modelos[nombre] = modelo.cpu()
                    if args.ensayo:
                        p[nombre] = pv
                    print(f"  {nombre}: PR-AUC val {ap_v:.4f} (reproduce S9) | umbral {umbral[nombre]:.2f}", flush=True)
            del conj_val
        print("Las 18 redes reproducen su validación de S9.", flush=True)

        # 3b. Predicciones del conjunto evaluado (aquí se abre prueba)
        if args.ensayo:
            for nombre, mod in refs.items():
                p[nombre] = rf.predecir(mod, x_ref(nombre, val))
        else:
            print("\n>>> Abriendo el conjunto de prueba", flush=True)
            for nombre, mod in refs.items():
                p[nombre] = rf.predecir(mod, x_ref(nombre, evl))
            for L in LS:
                conj_evl = bu.ConjuntoGPU(cnn.ultimos_instantes(evl, L), imgs, L)
                for m in REDES:
                    for s in SEMILLAS:
                        nombre = f"{m}_L{L}_s{s}"
                        p[nombre] = bu.predecir_gpu(modelos[nombre].to(disp), conj_evl)
                        modelos[nombre].cpu()
                del conj_evl
        p = {n: p[n] for n in [*refs, *modelos]}          # orden fijo: referencias y luego redes
        del imgs, modelos
        if disp.type == "cuda":
            torch.cuda.empty_cache()

        for nombre, pr in p.items():
            pd.DataFrame({"codigoestacion": evl.codigoestacion, "timestamp": evl.timestamp,
                          "etiqueta": y, "p": pr}).to_parquet(dir_sal / f"pred_{conjunto}_{nombre}.parquet")
        pd.Series(umbral, name="umbral").to_csv(dir_sal / "umbrales.csv")
        assert set(p) == {x for v in cfgs.values() for x in v}

        # 4. Métricas con umbral y su bootstrap (rápido)
        bloques = np.unique(ev.bloque_local(evl.timestamp))
        pesos = ev.pesos_bootstrap(len(bloques), B, SEMILLA)
        puntual, rep = {}, {}
        for nombre, pr in p.items():
            _, tablas = ev.tablas_por_bloque(evl.timestamp, y, pr, umbral[nombre], bloques=bloques)
            bs = ev.bootstrap(tablas, pesos)
            a, b, c, d = (int(v) for v in tablas.sum(0))
            puntual[nombre] = ev.metricas(a, b, c, d) | {"aciertos": a, "falsas_alarmas": b, "omisiones": c,
                                                          "umbral": umbral[nombre]}
            rep[nombre] = {k: bs.replicas[k] for k in METRICAS_UMBRAL}
            pd.DataFrame(tablas, columns=list("abcd"), index=pd.Index(bloques, name="bloque")).to_parquet(
                dir_sal / f"tablas_{nombre}.parquet")
        print(f"\nMétricas con umbral listas ({(time.time() - t_ini) / 60:.0f} min)", flush=True)

        def conclusion(ic_inf, ic_sup):
            if not (np.isfinite(ic_inf) and np.isfinite(ic_sup)):
                return "indefinida"
            return "A > B" if ic_inf > 0 else "A < B" if ic_sup < 0 else "sin diferencia"

        def parejas(A, Bc):
            """Comparación semilla por semilla (decisión del 15-oct-2026)."""
            ma, mb = cfgs[A], cfgs[Bc]
            if len(ma) == len(mb):
                return list(zip(ma, mb))                 # red − red (misma semilla) o referencia − referencia
            assert len(mb) == 1, (A, Bc)
            return [(a, mb[0]) for a in ma]              # red − referencia, con cada semilla

        def resumir(metricas_):
            filas_m, filas_c, filas_par, filas_comp = [], [], [], []
            for nombre in p:
                fila = {"modelo": nombre, **{k: float(v) for k, v in puntual[nombre].items()}}
                for k in metricas_:
                    fila[f"{k}_ic_inf"], fila[f"{k}_ic_sup"] = ev._ic(rep[nombre][k])
                    n_nan = int(np.isnan(rep[nombre][k]).sum())
                    fila[f"{k}_replicas_nan"] = n_nan
                    fila[f"{k}_ic_poco_confiable"] = n_nan / B > ev.UMBRAL_AVISO_NAN
                filas_m.append(fila)
            for cfg, mods in cfgs.items():
                fila = {"configuracion": cfg, "n_modelos": len(mods)}
                for k in metricas_:
                    vals = [puntual[x][k] for x in mods]
                    fila[k] = float(np.mean(vals))
                    fila[f"{k}_de_semillas"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                    if len(mods) == 1:
                        fila[f"{k}_ic_inf"], fila[f"{k}_ic_sup"] = ev._ic(rep[mods[0]][k])
                filas_c.append(fila)
            for A, Bc in comps:
                resumen_c = {"comparacion": f"{A} − {Bc}", "n_parejas": len(parejas(A, Bc))}
                por_k = {k: [] for k in metricas_}
                for a, b in parejas(A, Bc):
                    fila = {"comparacion": f"{A} − {Bc}", "pareja": f"{a} − {b}"}
                    for k in metricas_:
                        delta = rep[a][k] - rep[b][k]
                        validas = ~np.isnan(delta)
                        lo, hi = ev._ic(delta)
                        d = puntual[a][k] - puntual[b][k]
                        fila |= {f"delta_{k}": d, f"delta_{k}_ic_inf": lo, f"delta_{k}_ic_sup": hi,
                                 f"frac_gana_A_{k}": float(np.mean(delta[validas] > 0)) if validas.any() else np.nan,
                                 f"replicas_nan_{k}": int((~validas).sum()), f"conclusion_{k}": conclusion(lo, hi)}
                        por_k[k].append((d, conclusion(lo, hi)))
                    filas_par.append(fila)
                for k, v in por_k.items():
                    ds, cs = [x[0] for x in v], [x[1] for x in v]
                    resumen_c |= {f"delta_{k}_media": float(np.mean(ds)), f"delta_{k}_min": float(np.min(ds)),
                                  f"delta_{k}_max": float(np.max(ds)), f"conclusiones_{k}": " | ".join(cs),
                                  f"consistente_{k}": len(set(cs)) == 1}
                filas_comp.append(resumen_c)
            return pd.DataFrame(filas_m), pd.DataFrame(filas_c), pd.DataFrame(filas_par), pd.DataFrame(filas_comp)

        def guardar(metricas_):
            por_modelo, por_cfg_, por_par, comp = resumir(metricas_)
            por_modelo.to_csv(dir_sal / "resultados_por_modelo.csv", index=False)
            por_cfg_.to_csv(dir_sal / "resultados_por_configuracion.csv", index=False)
            por_par.to_csv(dir_sal / "comparaciones_por_semilla.csv", index=False)
            comp.to_csv(dir_sal / "comparaciones.csv", index=False)
            malos = [f"{r.modelo}:{k}" for _, r in por_modelo.iterrows() for k in metricas_ if r[f"{k}_ic_poco_confiable"]]
            if malos:
                print("AVISO: IC poco confiables (> 1 % de réplicas indefinidas): " + ", ".join(malos), flush=True)
            return por_modelo, por_cfg_, por_par, comp

        por_modelo, por_cfg_, por_par, comp = guardar(METRICAS_UMBRAL)
        print("\nPor configuración (media de semillas; CSI, POD, FAR):\n" +
              por_cfg_[["configuracion", "csi", "csi_de_semillas", "pod", "far"]].round(4).to_string(index=False), flush=True)
        print("\nComparaciones en CSI (semilla por semilla):\n" +
              comp[["comparacion", "delta_csi_media", "delta_csi_min", "delta_csi_max", "conclusiones_csi"]]
              .round(4).to_string(index=False), flush=True)

        # 4b. Ensayo: sensibilidad del largo de bloque con las redes finales (en validación)
        if args.ensayo:
            filas = []
            for nombre, pr in p.items():
                for h in (6, 12, 24, 48):
                    pl, _ = ev.evaluar(evl.timestamp, y, pr, umbral[nombre], conjunto="val", B=1000,
                                       semilla=SEMILLA, horas_bloque=h, con_auc=False)
                    filas.append({"modelo": nombre, "horas_bloque": h, "n_bloques": pl["val_n_bloques"],
                                  "ancho_ic_csi": pl["val_csi_ic_sup"] - pl["val_csi_ic_inf"]})
            sens = pd.DataFrame(filas)
            sens.to_csv(dir_sal / "sensibilidad_bloque.csv", index=False)
            sens["grupo"] = sens.modelo.str.replace(r"_L\d.*", "", regex=True)
            print("\nSensibilidad del bloque (ancho medio del IC del CSI en validación, B = 1,000):\n" +
                  sens.pivot_table(index="horas_bloque", columns="grupo", values="ancho_ic_csi").round(4).to_string(), flush=True)

        # 5. AUC y PR-AUC con bootstrap (lento, en paralelo, reanudable)
        pendientes = [n for n in p if not (dir_sal / f"replicas_auc_{n}.npz").exists()]
        print(f"\nBootstrap de AUC y PR-AUC: {len(pendientes)} modelos pendientes, {args.procesos} procesos", flush=True)
        _COMPARTIDO.update(ts=evl.timestamp, y=y, p=p, pesos=pesos, bloques=bloques, dir=dir_sal)
        t = time.time()
        if pendientes:
            # fork: los procesos heredan las predicciones y los pesos sin copiarlos por tubería
            with ProcessPoolExecutor(max_workers=args.procesos, mp_context=multiprocessing.get_context("fork")) as ex:
                for i, nombre in enumerate(ex.map(_auc_un_modelo, pendientes), 1):
                    print(f"  {nombre} listo ({i}/{len(pendientes)}, {(time.time() - t) / 60:.0f} min)", flush=True)
        for nombre, pr in p.items():
            auc, ap_ = ev.auc_pr(y, pr)
            puntual[nombre] |= {"auc": auc, "pr_auc": ap_}
            r = np.load(dir_sal / f"replicas_auc_{nombre}.npz")
            rep[nombre] |= {"auc": r["auc"], "pr_auc": r["pr_auc"]}

        por_modelo, por_cfg_, por_par, comp = guardar((*METRICAS_UMBRAL, "auc", "pr_auc"))
        print("\nPor configuración (media de semillas; PR-AUC, AUC, CSI):\n" +
              por_cfg_[["configuracion", "pr_auc", "pr_auc_de_semillas", "auc", "csi"]].round(4).to_string(index=False), flush=True)
        print("\nComparaciones en PR-AUC (semilla por semilla):\n" +
              comp[["comparacion", "delta_pr_auc_media", "delta_pr_auc_min", "delta_pr_auc_max", "conclusiones_pr_auc"]]
              .round(4).to_string(index=False), flush=True)

        # 6. MLflow
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlruns' / 'mlflow.db').as_posix()}")
            exp = "s11-prueba" if not args.ensayo else "s11-ensayo-val"
            if mlflow.get_experiment_by_name(exp) is None:
                mlflow.create_experiment(exp, artifact_location=(RAIZ / "mlruns" / "artefactos").as_uri())
            mlflow.set_experiment(exp)
            for _, f in por_modelo.iterrows():
                with mlflow.start_run(run_name=f.modelo):
                    mlflow.set_tags({"conjunto": conjunto, "B": str(B)})
                    mlflow.log_metrics({f"{conjunto}_{k}": float(v) for k, v in f.drop("modelo").items()
                                        if pd.notna(v) and np.isfinite(float(v))})
        except ImportError:
            pass
        print(f"\nFin: {(time.time() - t_ini) / 3600:.2f} h", flush=True)
    finally:
        if args.apagar:
            print("Apagando la VM en 1 minuto…", flush=True)
            subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)


if __name__ == "__main__":
    main()
