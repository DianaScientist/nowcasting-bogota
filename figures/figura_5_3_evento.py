"""Figura 5.3. Métricas por evento en prueba. (a) POD por evento frente al número de avisos emitidos
(barras: IC 95 % por bootstrap en el umbral y la logística L2; DE entre semillas en las redes).
(b) Anticipación de los episodios avisados: mediana y rango intercuartílico.
Valores de la Tabla D.4 (Anexo D), salida de modelo/21-analisis_errores.ipynb."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SALIDA = Path(__file__).with_suffix(".png")
COL = {"u": PALETA["wi-grey"], "l": PALETA["wi-ocher"], "c": PALETA["wwu-blue"], "r": PALETA["wi-purple"]}
# (rótulo, familia, POD evento, DE entre semillas, avisos, anticipación mediana, p25, p75, IC 95 % de la POD)
M = [("Umbral en C13", "u", 0.127, None, 1892, 2.0, 1.7, 2.3, (0.055, 0.205)),
     ("Logística L = 2", "l", 0.321, None, 10432, 2.3, 1.8, 2.8, (0.239, 0.404)),
     ("Logística L = 3", "l", 0.268, None, 8450, 2.2, 1.7, 2.8, None),
     ("Logística L = 4", "l", 0.179, None, 5582, 2.0, 1.5, 2.7, None),
     ("CNN plana L = 2", "c", 0.282, 0.005, 7335, 2.4, 1.7, 3.0, None),
     ("CNN plana L = 3", "c", 0.265, 0.065, 7531, 2.3, 1.7, 3.0, None),
     ("CNN plana L = 4", "c", 0.317, 0.044, 9837, 2.4, 1.8, 2.9, None),
     ("ResNet L = 2", "r", 0.329, 0.066, 12287, 2.4, 1.8, 3.0, None),
     ("ResNet L = 3", "r", 0.305, 0.040, 10975, 2.4, 1.8, 3.0, None),
     ("ResNet L = 4", "r", 0.223, 0.065, 6882, 2.3, 1.7, 3.0, None)]
OFF = {"Logística L = 3": (8, -4), "CNN plana L = 3": (-8, -12), "ResNet L = 3": (8, -4), "CNN plana L = 4": (-8, 6),
       "Logística L = 2": (8, 10), "ResNet L = 2": (8, -2), "CNN plana L = 2": (-8, 4), "ResNet L = 4": (-8, -4)}
coma = lambda s: s.replace(".", ",")
gris = "#555"
txtcol = lambda f: COL[f] if f != "u" else gris

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
# (a) detección frente a número de avisos
for lab, f, p, de, n, med, q1, q3, ic in M:
    c = COL[f]
    if ic: a1.plot([n, n], ic, color=c, lw=1.4, zorder=2)
    if de: a1.errorbar(n, p, yerr=de, color=c, capsize=3, lw=1.4, zorder=2)
    a1.plot(n, p, "o", ms=8, color=c, mec="white", mew=0.8, zorder=3)
    dx, dy = OFF.get(lab, (6, 4))
    a1.annotate(coma(f"{lab} ({p:.2f})"), (n, p), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                color=txtcol(f), ha="right" if dx < 0 else "left", va="center")
a1.set_ylim(0, 0.45); a1.set_xlim(0, 14500)
a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: coma(f"{v:.2f}")))
a1.set_xlabel("Avisos emitidos en 2025"); a1.set_ylabel("POD por evento (fracción de episodios avisados)")
a1.text(0.97, 0.05, "fracción de avisos falsos: 0,93–0,97 en todos los modelos", transform=a1.transAxes, ha="right",
        fontsize=8, style="italic", color=gris)
a1.set_title("(a) Detección frente a número de avisos", loc="left", fontsize=10)
# (b) anticipación
for i, (lab, f, p, de, n, med, q1, q3, ic) in enumerate(M):
    y = len(M) - 1 - i; c = COL[f]
    a2.plot([q1, q3], [y, y], color=c, lw=5, alpha=0.45, solid_capstyle="butt")
    a2.plot(med, y, "|", ms=14, mew=2.2, color=c)
    a2.text(3.15, y, coma(f"{med:.1f} h"), fontsize=8, color=txtcol(f), va="center")
a2.set_yticks(range(len(M))); a2.set_yticklabels([m[0] for m in M][::-1])
for t, m in zip(a2.get_yticklabels(), M[::-1]): t.set_color(txtcol(m[1]))
a2.set_xlim(1, 3.4); a2.set_xticks([1, 1.5, 2, 2.5, 3]); a2.set_xticklabels(["1", "1,5", "2", "2,5", "3"])
a2.set_ylim(-0.6, len(M) - 0.4)
a2.set_xlabel("Anticipación (h): mediana y rango intercuartílico")
a2.set_title("(b) Anticipación de los episodios avisados", loc="left", fontsize=10)
for ax in (a1, a2):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(True, ls="--", alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
