"""Figura 4.5. Co-ocurrencia de positivos entre pares de estaciones según la distancia (entrenamiento).
Valores de datasets/11-dataset_training.ipynb, sección 3 (celda verificada con assert):
lift = P(positivo en B | positivo en A) / prevalencia de B; p_cond(0–5 km) = 0,376."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from paleta import PALETA, ROL

import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9.5})
SALIDA = Path(__file__).with_suffix(".png")
coma = lambda s: s.replace(".", ",")

# (desde km, hasta km, lift, pares de estaciones)
T = [(0, 5, 101.3, 212), (5, 10, 44.3, 432), (10, 15, 21.5, 428), (15, 20, 16.3, 298),
     (20, 25, 17.3, 229), (25, 30, 13.2, 153), (30, 40, 3.5, 110), (40, 58, 10.8, 59)]
K = ROL["linea_ref"]
fig, ax = plt.subplots(figsize=(8, 4.5))
for a, b, v, n in T:
    ax.bar(a, v, width=b - a, align="edge", color=ROL["serie"], alpha=0.8 if b <= 40 else 0.35, edgecolor="white", lw=1.2)
    ax.text((a + b) / 2, v * 1.12, f"{v:.0f}×" if v >= 10 else coma(f"{v:.1f}×"), ha="center", fontsize=8.3, color=K)
    ax.text((a + b) / 2, -0.16, f"{n}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#777")
ax.text(-1, -0.16, "pares de\nestaciones:", transform=ax.get_xaxis_transform(), ha="right", va="center", fontsize=8, color="#777")
ax.axhline(1, color=K, lw=1, ls="--"); ax.text(57.5, 1.12, "estaciones independientes (1×)", ha="right", fontsize=8.3, color=K)
ax.text(2.5, 140, "37,6 % frente a\n0,37 % de prevalencia", ha="left", fontsize=8, color=ROL["positivo"], style="italic")
ax.annotate("misma celda\nde tormenta", xy=(7.5, 46), xytext=(11, 70), fontsize=8.3, color=K, arrowprops=dict(arrowstyle="-", color=K, lw=0.7))
ax.annotate("misma tarde convectiva\n(alcance regional)", xy=(22, 20), xytext=(27, 45), fontsize=8.3, color=K, arrowprops=dict(arrowstyle="-", color=K, lw=0.7))
ax.set_yscale("log"); ax.set_ylim(0.8, 260); ax.set_xlim(-6, 58)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 40, 58]); ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30", "40", "58"])
ax.set_xlabel("Distancia entre estaciones (km)", color=K, labelpad=34)
ax.set_ylabel("¿Cuántas veces más probable es un\npositivo si otra estación lo registra?\n(escala logarítmica)", color=K)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.yaxis.grid(True, ls="--", alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(SALIDA, dpi=250)
print("ok", SALIDA.name)
