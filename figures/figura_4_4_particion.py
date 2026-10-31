"""Figura 4.4. Partición temporal del conjunto de datos y pares positivos por semana (hora local),
con los cortes candidatos, los tramos sin datos y el cambio de GOES-16 a GOES-19.
Datos: datasets/indice_completo.parquet."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")

d = pd.read_parquet(RAIZ / "datasets/indice_completo.parquet", columns=["timestamp", "etiqueta", "periodo"])
d["t"] = d.timestamp.dt.tz_convert("America/Bogota").dt.tz_localize(None)
pos = d[d.etiqueta == 1].set_index("t").etiqueta.resample("W-MON", label="left", closed="left").sum()
K = ROL["linea_ref"]
P = {"train": (ROL["split_train"], "Entrenamiento", "2024-01-01", "2024-11-15"),
     "val":   (ROL["split_val"], "Validación", "2024-11-15", "2025-01-01"),
     "test":  (ROL["split_test"], "Prueba", "2025-01-01", "2026-01-01")}

fig, ax = plt.subplots(figsize=(11, 4.6))
for k, (c, lab, a, b) in P.items():
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    ax.axvspan(a, b, color=c, alpha=0.10, lw=0)
    ax.text(a + (b - a) / 2, 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=c, fontweight="bold", fontsize=9.5)
# barras apiladas por conjunto, para respetar las fronteras dentro de una semana
base = None
for k in ["train", "val", "test"]:
    pk = d[(d.etiqueta == 1) & (d.periodo == k)].set_index("t").etiqueta.resample("W-MON", label="left", closed="left").sum().reindex(pos.index, fill_value=0)
    ax.bar(pos.index, pk.values, width=6, align="edge", color=P[k][0], alpha=0.85, bottom=base)
    base = pk.values if base is None else base + pk.values
# tramos sin pares en toda la red
for a, b, m in [("2024-06-01", "2024-06-22", "2024-06-11"), ("2025-02-05", "2025-02-25", "2025-02-15")]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#bdbdbd", alpha=0.45, lw=0, hatch="///")
    ax.text(pd.Timestamp(m), 0.82, "sin datos", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color="#666", rotation=90)
# cortes candidatos (el elegido, 15 nov, destacado) y corte de prueba
for f, el in [("2024-11-01", False), ("2024-11-15", True), ("2024-12-01", False)]:
    ax.axvline(pd.Timestamp(f), color=ROL["corte_val"] if el else "#9e9e9e", lw=2 if el else 1, ls="-" if el else "--", zorder=4)
ax.axvline(pd.Timestamp("2025-01-01"), color=ROL["corte_test"], lw=2, zorder=4)
ax.annotate("cortes candidatos:\n1 nov · 15 nov · 1 dic", xy=(pd.Timestamp("2024-11-01"), 0.62), xycoords=("data", "axes fraction"),
            xytext=(pd.Timestamp("2024-07-15"), 0.62), textcoords=("data", "axes fraction"), fontsize=8, color=K, va="center", ha="center",
            arrowprops=dict(arrowstyle="-", color="#9e9e9e", lw=0.8))
# transición de satélite
ts = pd.Timestamp("2025-04-07")
ax.axvline(ts, color=K, lw=0.8, ls=":", zorder=3)
ax.text(ts + pd.Timedelta(days=4), 0.93, "GOES-16 → GOES-19", transform=ax.get_xaxis_transform(), fontsize=7.8, color=K, va="top")

ax.set_ylabel("Pares positivos por semana", color=K)
ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
def fmt_mes(x, _):
    f = mdates.num2date(x)
    return f"{MES[f.month-1]}\n{f.year}" if f.month == 1 else MES[f.month - 1]
ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_mes))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
