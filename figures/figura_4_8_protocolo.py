"""Figura 4.8. Protocolo experimental, de la búsqueda a la apertura única de la prueba."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8.8})
K=ROL["linea_ref"]; V=ROL["split_val"]; T=ROL["split_test"]
fig,ax=plt.subplots(figsize=(11,2.7)); ax.set_xlim(0,110); ax.set_ylim(0,26); ax.axis("off")
def caja(x,y,w,h,tit,txt,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1",fc=col,alpha=0.14,ec="none"))
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1",fc="none",ec=col,lw=1.3))
    ax.text(x+w/2,y+h-2.2,tit,ha="center",va="top",fontsize=9.2,color=K,fontweight="bold")
    ax.text(x+w/2,y+h-6,txt,ha="center",va="top",fontsize=8,color=K,linespacing=1.35)
B=[("1. Búsqueda","Optuna (TPE), 50 config.\npor modelo y L;\nselección por PR-AUC",V),
   ("2. Reentrenamiento","mejor configuración\ncon 3 semillas\n(logística: 1 vez)",V),
   ("3. Umbral","CSI máximo en\nvalidación\n(grilla 0,01–0,99)",V),
   ("4. Congelamiento","ningún hiperparámetro,\numbral ni procedimiento\ncambia desde aquí","#9e9e9e"),
   ("5. Prueba (2025)","evaluación única:\npor par, por evento,\nestratos y estaciones",T)]
w=19; gap=3.2; x=1.5
for i,(a,b,c) in enumerate(B):
    caja(x,10,w,14,a,b,c)
    if i<len(B)-1: ax.add_patch(FancyArrowPatch((x+w+0.4,17),(x+w+gap-0.4,17),arrowstyle="-|>",mutation_scale=12,color=K,lw=1.2))
    x+=w+gap
ax.annotate("",xy=(1.5,5.2),xytext=(3*(w+gap)+1.5-gap,5.2),arrowprops=dict(arrowstyle="-",color=V,lw=2))
ax.text((3*(w+gap)-gap)/2+1.5,2.2,"conjunto de validación (prevalencia real)",ha="center",fontsize=8.3,color=V)
ax.annotate("",xy=(4*(w+gap)+1.5,5.2),xytext=(4*(w+gap)+1.5+w,5.2),arrowprops=dict(arrowstyle="-",color=T,lw=2))
ax.text(4*(w+gap)+1.5+w/2,2.2,"se abre una sola vez",ha="center",fontsize=8.3,color=T)
fig.tight_layout(); fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
