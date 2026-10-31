"""Figura 2.1. Una tarde convectiva típica sobre la Sabana de Bogotá (esquema).
Curva esquemática; horarios y precursores según Casallas-García et al. (2023)."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10})
S=ROL["serie"]; P=ROL["positivo"]; K=ROL["linea_ref"]; GR=PALETA["wi-grey"]; G=PALETA["wwu-green"]
fig=plt.figure(figsize=(10,5.2),facecolor="white")
ax=fig.add_axes([0.06,0.12,0.62,0.74])
t=np.linspace(6,18,400)
def pwv(x):
    base=0.18+0.03*(x-6)/3
    slow=0.06/(1+np.exp(-(x-10)/0.5))
    fast=0.55/(1+np.exp(-(x-12.5)/0.6))
    drop=-0.06/(1+np.exp(-(x-16.5)/1.0))   # descenso lento: el PWV no cae rápido tras el pico
    return base+slow+fast+drop
ax.axvspan(9,11,color=GR,alpha=0.12,lw=0)
ax.axvspan(11,14,color=S,alpha=0.10,lw=0)
ax.axvspan(14,15,color=P,alpha=0.18,lw=0)
ax.plot(t,pwv(t),color=S,lw=2.4)
ax.set_xlim(6,18); ax.set_ylim(0,1.25)
ax.set_xticks(range(6,19,2)); ax.set_xticklabels([f"{h:02d}:00" for h in range(6,19,2)])
ax.set_yticks([]); ax.set_ylabel("Vapor de agua precipitable\n(esquemático)",color=S)
ax.set_xlabel("Hora local",color=K)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.text(10,1.1,"Aumento lento\n9:00–11:00",ha="center",va="top",fontsize=9,color=K)
ax.text(12.5,1.1,"Aumento rápido\n~3 h",ha="center",va="top",fontsize=9,color=S)
ax.text(14.5,1.1,"Máximo de\nconvección\n14:00–15:00",ha="center",va="top",fontsize=9,color=P)
ax.annotate("",xy=(14,0.25),xytext=(11,0.25),arrowprops=dict(arrowstyle="<->",color=S,lw=1.2))
ax.text(12.5,0.28,"convergencia de humedad (~3 h)",ha="center",fontsize=8.5,color=S,style="italic")
def mark(x,yy,n):
    ax.plot(x,yy,"o",ms=15,mfc="white",mec=G,mew=1.6,zorder=5)
    ax.text(x,yy,str(n),ha="center",va="center",color=G,fontsize=9,fontweight="bold",zorder=6)
mark(8,0.8,1); ax.text(8,0.73,"Viento más débil\nque el promedio\n(toda la mañana)",ha="center",va="top",fontsize=8.3,color=K)
mark(10.6,0.8,2); ax.text(10.6,0.73,"Temperatura del aire\nsuperior a la normal\n(hacia el mediodía)",ha="center",va="top",fontsize=8.3,color=K)
mark(13.2,pwv(13.2),3); ax.text(13.5,pwv(13.2)-0.02,"Aumento rápido del\nvapor precipitable\ny de la velocidad\ndel viento",fontsize=8.3,va="top",color=K)
ia=fig.add_axes([0.72,0.12,0.26,0.74]); ia.axis("off"); ia.set_xlim(0,1); ia.set_ylim(0,1)
ia.add_patch(FancyBboxPatch((0.02,0.02),0.96,0.96,boxstyle="round,pad=0.01",fc="#f4f4f2",ec="#cccccc"))
ia.text(0.5,0.9,"Viento del oeste:\nnecesario, no suficiente",ha="center",va="top",fontsize=10,fontweight="bold",color=K)
def bar(yb,frac,col,lab,val):
    ia.add_patch(plt.Rectangle((0.1,yb),0.8,0.07,fc=ROL["negativo"],ec="none"))
    ia.add_patch(plt.Rectangle((0.1,yb),0.8*frac,0.07,fc=col,ec="none"))
    ia.text(0.1,yb+0.1,lab,fontsize=8.5,color=K,va="bottom")
    ia.text(0.1+0.8*frac-0.02,yb+0.035,val,ha="right",va="center",fontsize=9,color="white",fontweight="bold")
bar(0.52,0.78,S,"Días de temporada lluviosa con\nviento del oeste a inicio de la tarde","78 %")
bar(0.2,0.26,P,"Días de temporada lluviosa\ncon convección profunda","26 %")
fig.text(0.06,0.94,"Una tarde convectiva típica sobre la Sabana de Bogotá",fontsize=13,fontweight="bold",color=K)
fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
