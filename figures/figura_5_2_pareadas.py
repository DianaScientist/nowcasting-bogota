"""Figura 5.2. Comparaciones pareadas en prueba: Δ = A − B en CSI y AUC con IC 95 % por bootstrap de días
pluviométricos; las redes, una línea por semilla. A la derecha, Δ (redes: rango entre semillas).
Datos: modelo/salidas_s11/comparaciones_por_semilla.csv."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.8})
SALIDA = Path(__file__).with_suffix(".png")
d = pd.read_csv(RAIZ / "modelo/salidas_s11/comparaciones_por_semilla.csv")
NOM = {"umbral_c13": "umbral en C13", "logistica": "logística", "cnn_plana": "CNN plana", "resnet": "ResNet"}
def nom(s):
    s = s.replace("_L", " L")
    for k, v in NOM.items(): s = s.replace(k, v)
    return s
GR = [("Peldaño 2 frente a 1", d.index[0:3]), ("Peldaño 3 frente a 2", d.index[3:12]), ("Peldaño 4 frente a 3", d.index[12:21]),
      ("Historia L dentro de cada modelo", d.index[21:35]), ("Pregunta central: mejor modelo frente a la referencia", d.index[35:38])]
K = ROL["linea_ref"]; GAN = PALETA["wwu-green"]; PIER = ROL["positivo"]; NS = "#9e9e9e"

filas = []; y = 0; etq = []
for g, idx in GR:
    etq.append((y, g)); y += 1
    blq = d.loc[idx]
    for c in blq.comparacion.unique():
        sub = blq[blq.comparacion == c]
        n = len(sub); offs = [0] if n == 1 else [-0.25, 0, 0.25]
        for o, (_, r) in zip(offs, sub.iterrows()): filas.append((y + o, r))
        filas.append((y, None, nom(c), sub)); y += 1
    y += 0.3

fig, axs = plt.subplots(1, 2, figsize=(11.5, 9.5), sharey=True)
fm = lambda x: f"{x:+.3f}".replace(".", ",")
for ax, (m, tit) in zip(axs, [("csi", "Δ CSI"), ("auc", "Δ AUC")]):
    for f in filas:
        if f[1] is None: continue
        yy, r = f
        lo, hi, v, cn = r[f"delta_{m}_ic_inf"], r[f"delta_{m}_ic_sup"], r[f"delta_{m}"], r[f"conclusion_{m}"]
        col = GAN if cn == "A > B" else PIER if cn == "A < B" else NS
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.3); ax.plot(v, yy, "o", ms=4, color=col, zorder=3)
    for f in filas:
        if f[1] is not None: continue
        v = f[3][f"delta_{m}"]
        t = fm(v.iloc[0]) if len(v) == 1 else f"{fm(v.min())} a {fm(v.max())}"
        ax.text(1.02, f[0], t, transform=ax.get_yaxis_transform(), fontsize=7.4, color=K, va="center")
    ax.axvline(0, color=K, lw=1)
    ax.set_title(tit, fontsize=10)
    for yy, g in etq: ax.axhline(yy + 0.5, color="#e6e6e6", lw=0.6)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, ls="--", alpha=0.3); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}".replace(".", ",") if abs(v) > 1e-9 else "0"))
lab = [(f[0], f[2]) for f in filas if f[1] is None]
axs[0].set_yticks([a for a, _ in lab]); axs[0].set_yticklabels([b for _, b in lab], fontsize=8.3)
axs[0].tick_params(axis="y", length=0)
for yy, g in etq:
    axs[0].text(axs[0].get_xlim()[0], yy, g, fontsize=8.8, fontweight="bold", color=K, va="center", ha="right")
axs[0].invert_yaxis()
fig.legend(handles=[Line2D([], [], color=GAN, marker="o", label="A > B (el intervalo excluye el cero)"),
                    Line2D([], [], color=PIER, marker="o", label="A < B (el intervalo excluye el cero)"),
                    Line2D([], [], color=NS, marker="o", label="sin diferencia significativa")],
           loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.6, 0.0))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.subplots_adjust(wspace=0.5); fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
