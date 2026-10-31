"""Figura 6.1. Coeficientes estandarizados de la regresión logística L = 2, por banda, estadística del
parche e instante. Datos: modelo/salidas_21/coeficientes_logistica_L2.csv."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")
c = pd.read_csv(RAIZ / "modelo/salidas_21/coeficientes_logistica_L2.csv")
B = ["CMI_C08", "CMI_C09", "CMI_C10", "CMI_C13", "CMI_C14"]
BL = ["Banda 8 (6,2 µm)", "Banda 9 (6,9 µm)", "Banda 10 (7,3 µm)", "Banda 13 (10,3 µm)", "Banda 14 (11,2 µm)"]
E = ["p10", "p50", "p90", "desvio"]; EL = ["p10", "p50", "p90", "desviación"]
cm = LinearSegmentedColormap.from_list("div", [PALETA["wi-darkblue"], "#ffffff", ROL["positivo"]])
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True, constrained_layout=True)
for ax, (ins, tit) in zip(axs, [("t−10 min", "Instante t − 10 min"), ("t", "Instante t")]):
    M = c[c.instante == ins].pivot(index="banda", columns="estadistica", values="coef").loc[B, E].values
    im = ax.imshow(M, cmap=cm, vmin=-5.2, vmax=5.2, aspect="auto")
    for i in range(5):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}".replace(".", ","), ha="center", va="center", fontsize=8, color="white" if abs(v) > 3 else ROL["linea_ref"])
    ax.set_xticks(range(4)); ax.set_xticklabels(EL); ax.set_yticks(range(5)); ax.set_yticklabels(BL)
    ax.set_title(tit, fontsize=10, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
cb = fig.colorbar(im, ax=axs, shrink=0.85); cb.set_label("Coeficiente estandarizado")
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
