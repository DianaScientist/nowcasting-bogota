"""Figura 3.3. Ventana de la etiqueta (t + 1 h, t + 3 h] y rango de τ (esquema)."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9.5})
K=ROL["linea_ref"]; OK=PALETA["wwu-green"]; NO=ROL["positivo"]; W=ROL["serie"]
fig,ax=plt.subplots(figsize=(8.4,4.4)); ax.set_xlim(-0.15,3.95); ax.set_ylim(-1.35,4.35); ax.axis("off")
ax.add_patch(Rectangle((1,-0.25),2,4.3,fc=W,alpha=0.09,ec="none"))
ax.text(2,4.15,"ventana de la etiqueta (t + 1 h, t + 3 h]",ha="center",color=W,fontsize=9.5,fontweight="bold")
ax.annotate("",xy=(3.3,0),xytext=(-0.1,0),arrowprops=dict(arrowstyle="-|>",color=K,lw=1.2))
for x,l in [(0,"t"),(1,"t + 1 h"),(2,"t + 2 h"),(3,"t + 3 h")]:
    ax.plot([x,x],[-0.1,0.1],color=K,lw=1.2); ax.text(x,-0.38,l,ha="center",fontsize=9.5,color=K)
    ax.plot([x,x],[0.1,4.0],color="#d9d9d9",lw=0.7,ls=":",zorder=0)
ax.plot([2,3],[-0.85,-0.85],color=OK,lw=5,solid_capstyle="butt")
ax.text(2.5,-1.22,"rango de τ: (t + 2 h, t + 3 h]",ha="center",color=OK,fontsize=9.5,fontweight="bold")
C=[(1.5,NO,"τ = t + 1 h 30: la hora incluye lluvia\nanterior a t + 1 h; no se evalúa"),
   (2+1/6,OK,"τ = t + 2 h 10: hora completa\ndentro de la ventana"),
   (3.0,OK,"τ = t + 3 h: última hora\nde la ventana")]
for i,(tau,col,lab) in enumerate(C):
    y=0.85+i*1.1
    ax.add_patch(Rectangle((tau-1,y-0.17),1,0.34,fc=col,alpha=0.30,ec=col,lw=1.1))
    ax.plot(tau,y,"o",color=col,ms=7,zorder=5)
    ax.text(tau+0.07,y,lab,va="center",fontsize=8.6,color=col)
ax.annotate("I(e, τ): lluvia acumulada en (τ − 60 min, τ]",xy=(2+1/6-0.5,1.95+0.17),xytext=(0.05,2.95),
            fontsize=8.6,color=K,arrowprops=dict(arrowstyle="-",color=K,lw=0.7))
fig.tight_layout(); fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
