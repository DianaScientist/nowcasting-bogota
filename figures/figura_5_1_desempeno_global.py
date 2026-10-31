"""Figura 5.1. Desempeño global en el conjunto de prueba (CSI, AUC y PR-AUC) con IC 95 % por bootstrap
de días pluviométricos; las redes, una línea por semilla. A la derecha, el valor del modelo
(redes: media ± DE entre semillas). Datos: modelo/salidas_s11/resultados_por_modelo.csv."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import pandas as pd, matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")
r = pd.read_csv(RAIZ / "modelo/salidas_s11/resultados_por_modelo.csv")
COL = {"umbral": PALETA["wi-grey"], "logistica": PALETA["wi-ocher"], "cnn_plana": PALETA["wwu-blue"], "resnet": PALETA["wi-purple"]}
FIL = ([("umbral_c13", "Umbral en C13", "umbral")]
       + [(f"logistica_L{l}", f"Logística L = {l}", "logistica") for l in (2, 3, 4)]
       + [(f"cnn_plana_L{l}", f"CNN plana L = {l}", "cnn_plana") for l in (2, 3, 4)]
       + [(f"resnet_L{l}", f"ResNet L = {l}", "resnet") for l in (2, 3, 4)])
MET = [("csi", "CSI", None), ("auc", "AUC", 0.5), ("pr_auc", "PR-AUC", 0.00514)]  # 0,00514 = prevalencia en prueba
fmt = lambda x: f"{x:.3f}".replace(".", ",")
gris = "#555"

fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True)
for ax, (m, tit, ref) in zip(axs, MET):
    for i, (clave, lab, fam) in enumerate(FIL):
        y = len(FIL) - 1 - i; c = COL[fam]
        sub = r[r.modelo.str.startswith(clave)].reset_index(drop=True)
        n = len(sub); offs = [0] if n == 1 else [-0.22, 0, 0.22]
        v = sub[m]
        txt = fmt(v.iloc[0]) if n == 1 else f"{fmt(v.mean())} ± {fmt(v.std())}"
        ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), fontsize=7.6, color=c if fam != "umbral" else gris, va="center", ha="left")
        for k, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row[f"{m}_ic_inf"], row[f"{m}_ic_sup"]], [y + offs[k]] * 2, color=c, lw=1.6 if n == 1 else 1.1, alpha=0.9)
            ax.plot(row[m], y + offs[k], "o", ms=6 if n == 1 else 4.5, color=c, mec="white", mew=0.6, zorder=3)
    if ref:
        ax.axvline(ref, color=gris, ls=":", lw=1)
        ax.text(ref + (0.005 if m == "auc" else 0.0004), -0.55, "azar" if m == "auc" else "prevalencia", fontsize=7.5, va="center",
                bbox=dict(fc="white", ec="none", pad=0.5), color=gris, ha="left")
    ax.set_title(tit, fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",") if v < 0.1 else f"{v:.2f}".replace(".", ",")))
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, ls="--", alpha=0.3); ax.set_axisbelow(True)
    for yy in (2.5, 5.5, 8.5): ax.axhline(yy, color="#e6e6e6", lw=0.8)
lab = [l for _, l, _ in FIL][::-1]; fam = [f for _, _, f in FIL][::-1]
axs[0].set_yticks(range(len(FIL))); axs[0].set_yticklabels(lab)
for t, f in zip(axs[0].get_yticklabels(), fam): t.set_color(COL[f] if f != "umbral" else gris)
axs[1].set_xlim(0.45, 0.85); axs[1].set_xticks([0.5, 0.6, 0.7, 0.8])
axs[2].set_xticks([0.01, 0.02, 0.03]); axs[0].set_xticks([0.01, 0.02, 0.03, 0.04])
for a in axs: a.set_ylim(-1, len(FIL) - 0.4)
fig.tight_layout(); fig.subplots_adjust(wspace=0.55); fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
