"""Figura 2.4. Etapas del ciclo de vida de una tormenta convectiva y fuentes de observación (esquema ilustrado).
Fondo sin texto generado con ChatGPT (figures/fondos/fondo_figura_2_4.png); títulos, barras y llaves encima."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import FancyArrowPatch, Rectangle
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif"})
im=Image.open(RAIZ / "figures" / "fondos" / "fondo_figura_2_4.png").convert("RGB"); W,H=im.size
K=ROL["linea_ref"]; WV=PALETA["wi-darkblue"]; IR=ROL["positivo"]; RAD=PALETA["wwu-green"]
HB=330  # alto de la franja inferior (px)
fig=plt.figure(figsize=(W/180,(H+HB)/180),dpi=180)
ax=fig.add_axes([0,HB/(H+HB),1,H/(H+HB)]); ax.imshow(im); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
bx=fig.add_axes([0,0,1,HB/(H+HB)]); bx.set_xlim(0,W); bx.set_ylim(HB,0); bx.axis("off")
cx=[220,660,1115,1680]
tit=["1. Acumulación de\nhumedad y convergencia","2. Crecimiento de cúmulos\ny enfriamiento del tope","3. Primeros ecos\nde precipitación","4. Lluvia fuerte\nen superficie"]
for x,t in zip(cx,tit): ax.text(x,40,t,ha="center",va="top",fontsize=10.5,color=K,fontweight="bold")
# etapa 1: convergencia y ascenso
for a,b in [((70,675),(185,675)),((370,675),(255,675)),((220,665),(220,545))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,color=WV,lw=2.2))
# etapa 2: crecimiento del tope
ax.add_patch(FancyArrowPatch((835,600),(835,410),arrowstyle="-|>",mutation_scale=16,color=IR,lw=2.2))
# etapa 3: hidrometeoros
rng=np.random.default_rng(2)
ax.scatter(1115+rng.uniform(-120,120,45),rng.uniform(380,575,45),s=14,color=RAD,edgecolors="white",linewidths=0.4,zorder=5)
# eje de tiempo, barras y llaves
bx.add_patch(FancyArrowPatch((15,25),(W-15,25),arrowstyle="-|>",mutation_scale=18,color=K,lw=1.6))
bx.text(W-15,45,"tiempo",ha="right",va="top",fontsize=9,color=K,style="italic")
def barra(y,x0,col,lab):
    bx.add_patch(Rectangle((x0,y-14),W-20-x0,28,fc=col,ec="none",alpha=0.9)); bx.text(x0+12,y,lab,va="center",fontsize=9.5,color="white",fontweight="bold")
barra(85,20,WV,"Satélite: bandas de vapor de agua")
barra(125,440,IR,"Satélite: bandas infrarrojas de ventana")
barra(165,890,RAD,"Radar")
def llave(y,x0,x1,col,lab):
    bx.annotate("",xy=(x0,y),xytext=(x1,y),arrowprops=dict(arrowstyle="<->",color=col,lw=1.8))
    for x in (x0,x1): bx.plot([x,x],[y-10,y+10],color=col,lw=1.8)
    bx.text((x0+x1)/2,y+14,lab,ha="center",va="top",fontsize=9.5,color=col)
llave(220,300,1550,K,"margen de anticipación de esta tesis")
llave(285,1000,1550,RAD,"margen de los métodos basados en radar")
fig.savefig(SALIDA,dpi=180,facecolor="white")
print("ok", SALIDA.name)
