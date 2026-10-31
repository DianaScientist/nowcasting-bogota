"""Figura 5.4. POD por evento en prueba por estrato (umbral en C13, logística L2 y las dos redes con L = 2).
Barras finas: IC 95 % por bootstrap de días (umbral y logística); gruesas: DE entre semillas (redes).
Valores de la tabla de POD por estrato del Anexo D (salida de modelo/21-analisis_errores.ipynb)."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")
MOD = [("Umbral en C13", PALETA["wi-grey"], "o"), ("Logística L2", PALETA["wi-ocher"], "s"),
       ("CNN plana L2", PALETA["wwu-blue"], "D"), ("ResNet L2", PALETA["wi-purple"], "^")]
# por categoría: (nombre, episodios, [(valor, (inf, sup) o ± DE)] × 4 modelos)
E = {"Hora del día": [("Día", 642, [(0.08, (0.02, 0.14)), (0.40, (0.29, 0.50)), (0.45, 0.01), (0.50, 0.05)]),
                      ("Noche", 472, [(0.19, (0.07, 0.33)), (0.22, (0.12, 0.33)), (0.05, 0.03), (0.10, 0.08)])],
     "Nube fría": [("Sin nube fría", 877, [(0.09, (0.03, 0.16)), (0.30, (0.21, 0.38)), (0.35, 0.01), (0.39, 0.06)]),
                   ("Con nube fría", 237, [(0.25, (0.09, 0.44)), (0.41, (0.26, 0.55)), (0.05, 0.04), (0.11, 0.10)])],
     "Temporada": [("DEF", 110, [(0.00, None), (0.34, (0.10, 0.49)), (0.52, 0.07), (0.49, 0.07)]),
                   ("MAM", 492, [(0.14, (0.03, 0.28)), (0.30, (0.19, 0.43)), (0.17, 0.01), (0.24, 0.05)]),
                   ("JJA", 101, [(0.25, (0.00, 0.55)), (0.38, (0.14, 0.58)), (0.14, 0.04), (0.21, 0.18)]),
                   ("SON", 411, [(0.11, (0.03, 0.21)), (0.32, (0.17, 0.48)), (0.39, 0.03), (0.42, 0.07)])],
     "Umbral de la estación": [("Tercil bajo", 511, [(0.17, (0.06, 0.29)), (0.37, (0.25, 0.49)), (0.27, 0.01), (0.31, 0.09)]),
                               ("Tercil medio", 364, [(0.09, (0.04, 0.14)), (0.21, (0.15, 0.27)), (0.18, 0.03), (0.25, 0.05)]),
                               ("Tercil alto", 239, [(0.09, (0.03, 0.17)), (0.38, (0.27, 0.50)), (0.45, 0.00), (0.49, 0.05)])]}
fig, axs = plt.subplots(4, 1, figsize=(8.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [2, 2, 4, 3]})
OFF = [-0.27, -0.09, 0.09, 0.27]
for ax, (tit, cats) in zip(axs, E.items()):
    for i, (nom, n, vals) in enumerate(cats):
        y = len(cats) - 1 - i
        for (v, u), (mlab, c, mk), o in zip(vals, MOD, OFF):
            if isinstance(u, tuple): ax.plot(u, [y + o] * 2, color=c, lw=1.2)
            elif u: ax.plot([v - u, v + u], [y + o] * 2, color=c, lw=2.2)
            ax.plot(v, y + o, mk, ms=6, color=c, mec="white", mew=0.6, zorder=3)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([f"{c[0]}  ({c[1]})" for c in cats][::-1])
    ax.set_ylim(-0.6, len(cats) - 0.4)
    ax.set_title(tit, loc="left", fontsize=9.5, fontweight="bold")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, ls="--", alpha=0.3); ax.set_axisbelow(True)
    for b in range(len(cats) - 1): ax.axhline(b + 0.5, color="#ececec", lw=0.7)
axs[-1].set_xlim(0, 0.62); axs[-1].set_xlabel("POD por evento (fracción de episodios avisados)")
axs[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
fig.legend(handles=[Line2D([], [], marker=mk, ls="", color=c, ms=7, label=l) for l, c, mk in MOD],
           loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
