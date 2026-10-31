"""
Episodios de lluvia fuerte de prueba (2025) no evaluables — S4b, decisión 1: "se reportan aparte".

Script independiente: no necesita correr la nb 21. Lee solo las etiquetas, los umbrales y los
pares de prueba de una predicción de S11 (todas tienen los mismos pares).

Uso, desde la raíz del repo con el entorno `nowcasting`:
    python modelo/episodios_no_evaluables.py
Salida: modelo/salidas_21/episodios_no_evaluables.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".gitignore").exists())
sys.path.insert(0, str(RAIZ / "src"))
import evaluacion as ev  # noqa: E402

NG = "0021206600"  # Nueva Generación, excluida en el modelado
cod10 = lambda s: s.astype(str).str.zfill(10)

# Pares de prueba (los mismos en los 22 archivos de S11)
pares = pd.read_parquet(RAIZ / "modelo" / "salidas_s11" / "pred_test_umbral_c13.parquet",
                        columns=["codigoestacion", "timestamp", "etiqueta"])
pares["codigoestacion"] = cod10(pares["codigoestacion"])

# Episodios, igual que en la nb 21 (celda 4)
lab = pd.read_parquet(RAIZ / "data_ingestion" / "ground_truth" / "etiquetas_ground_truth.parquet")
lab["codigoestacion"] = cod10(lab["codigoestacion"])
lab = lab[lab.codigoestacion != NG]
umb = pd.read_csv(RAIZ / "data_ingestion" / "ground_truth" / "umbral_operativo_por_estacion.csv",
                  dtype={"codigoestacion": str})
umb["codigoestacion"] = cod10(umb["codigoestacion"])
umb = umb.set_index("codigoestacion")["umbral_mm_h"]
umb = umb[umb.index.isin(lab.codigoestacion.unique())]
print("zonas horarias:", pares.timestamp.dt.tz, lab.timestamp.dt.tz)

episodios, fuertes = ev.episodios_lluvia(lab, umb)
pares["id_episodio"], _ = ev.asignar_episodio(pares, fuertes)
assert ((pares.id_episodio >= 0) == (pares.etiqueta == 1)).all(), "La asignación no reproduce las etiquetas"

evaluables = set(pares.loc[pares.id_episodio >= 0, "id_episodio"])
print(f"Episodios evaluables: {len(evaluables):,} (S12 reportó 1,114)")

# Episodios de 2025 en estaciones del conjunto de prueba
ini = pares.timestamp.min() + pd.Timedelta(hours=2)
fin = pares.timestamp.max() + pd.Timedelta(hours=3)
ep = episodios[(episodios.inicio > ini) & (episodios.inicio <= fin)
               & episodios.codigoestacion.isin(pares.codigoestacion.unique())].copy()
ep["evaluable"] = ep.id_episodio.isin(evaluables)
assert ep.evaluable.sum() == len(evaluables), "El filtro de fechas no cubre todos los evaluables"

# Motivo: mirar los pares positivos del episodio en las etiquetas completas
pos = lab.loc[lab.etiqueta == 1, ["codigoestacion", "timestamp", "en_curso"]].copy()
pos["id_episodio"], _ = ev.asignar_episodio(pos, fuertes)
con_pos_sin_curso = set(pos.loc[~pos.en_curso.astype(bool), "id_episodio"])
con_pos = set(pos.id_episodio)

ep["motivo"] = np.select(
    [ep.evaluable,
     ep.id_episodio.isin(con_pos_sin_curso),
     ep.id_episodio.isin(con_pos)],
    ["evaluable",
     "positivo sin lluvia en curso, fuera del índice de prueba (imágenes / L_max = 4)",
     "solo positivos con lluvia en curso (excluidos por diseño)"],
    "sin par positivo en las etiquetas (cobertura)")

resumen = ep.motivo.value_counts().to_frame("n")
resumen["frac"] = (resumen.n / resumen.n.sum()).round(3)
print(f"\nEpisodios de lluvia fuerte en el período de prueba: {len(ep):,}")
print(resumen.to_string())
salida = RAIZ / "modelo" / "salidas_21" / "episodios_no_evaluables.csv"
resumen.to_csv(salida)
ep.to_csv(salida.with_name("episodios_prueba_detalle.csv"), index=False)
print(f"\nGuardado en {salida}")
