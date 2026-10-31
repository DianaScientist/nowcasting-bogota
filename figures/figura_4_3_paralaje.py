"""Figura 4.3. Efecto de paralaje entre GOES-East y la Sabana de Bogotá (esquema ilustrado, ángulo exagerado).
Fondo sin texto generado con ChatGPT (figures/fondos/fondo_figura_4_3.png); geometría y rótulos encima."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch, FancyBboxPatch
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif"})
im=Image.open(RAIZ / "figures" / "fondos" / "fondo_figura_4_3.png").convert("RGB"); W,H=im.size
K=ROL["linea_ref"]; REAL=PALETA["wwu-green"]; AP=ROL["positivo"]; SAT=PALETA["wi-darkblue"]
GY=818; TOP=(1125,305); APX=1265
d=np.array([APX-TOP[0],GY-TOP[1]],float); s=(TOP[1]-60)/d[1]; SATP=(TOP[0]-d[0]*s,60)
fig=plt.figure(figsize=(W/150,H/150),dpi=150); ax=fig.add_axes([0,0,1,1]); ax.imshow(im); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
# satélite y línea de visión
ax.plot(*SATP,marker="^",ms=15,color=SAT,zorder=6)
ax.text(SATP[0]-25,SATP[1]+5,"GOES-East",ha="right",va="center",fontsize=11,color=SAT,fontweight="bold")
ax.plot([SATP[0],TOP[0]],[SATP[1],TOP[1]],ls="--",color=K,lw=1.8,zorder=5)
ax.plot([TOP[0],APX],[TOP[1],GY],ls=":",color=K,lw=1.8,zorder=5)
ax.plot([TOP[0],TOP[0]],[TOP[1],GY],ls="-",color=REAL,lw=1.4,alpha=0.9,zorder=5)
ax.plot(*TOP,"o",ms=9,color=K,mec="white",mew=1.2,zorder=7)
ax.text(TOP[0]-18,TOP[1]-22,"tope de la nube",ha="right",fontsize=10.5,color=K,fontweight="bold",bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.85))
# ángulo
ang=np.degrees(np.arctan2(d[0],d[1]))
ax.add_patch(Arc(TOP,320,320,theta1=90-ang,theta2=90,color="#e08a00",lw=2,zorder=6))
ax.text(TOP[0]+8,TOP[1]+200,"θ",fontsize=13,color="#e08a00",fontweight="bold",zorder=7)
# posiciones en superficie
ax.plot(TOP[0],GY,"o",ms=11,color=REAL,mec="white",mew=1.5,zorder=7)
ax.plot(APX,GY,"X",ms=13,color=AP,mec="white",mew=1.2,zorder=7)
ax.text(TOP[0]-15,GY+42,"posición real",ha="right",fontsize=10.5,color=REAL,fontweight="bold",bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.85))
ax.text(APX+15,GY+42,"posición aparente",ha="left",fontsize=10.5,color=AP,fontweight="bold",bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.85))
ax.add_patch(FancyArrowPatch((TOP[0],GY+82),(APX,GY+82),arrowstyle="<->",mutation_scale=14,color=K,lw=1.5))
ax.text(APX+12,GY+82,"Δ ≈ 1.17–1.37 km",ha="left",va="center",fontsize=10.5,color=K,bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.85))
# recuadro del píxel
bx,by,bw,bh=50,560,470,230
ax.text(bx+bw/2,by+28,"un píxel de la banda 13 (~2 km × 2 km)",ha="center",fontsize=10,color=K,style="italic",zorder=7)
px,py,ps=bx+bw/2-75,by+50,150
ax.add_patch(Rectangle((px,py),ps,ps,fc="white",alpha=0.35,ec=K,lw=1.2,ls="--",zorder=7))
s2=ps/2.0; cx=px+ps/2; cy=py+ps/2
ax.plot(cx-1.27*s2/2,cy,"o",ms=9,color=REAL,zorder=8); ax.plot(cx+1.27*s2/2,cy,"X",ms=10,color=AP,zorder=8)
ax.plot([cx-1.27*s2/2,cx+1.27*s2/2],[cy,cy],color=K,lw=1,zorder=7)
ax.text(cx,cy+32,"Δ ≈ 1.3 km",ha="center",fontsize=9,color=K,zorder=8)
ax.text(bx+bw/2,by+bh-12,"ambas posiciones caen en el mismo píxel",ha="center",fontsize=9,color=K,style="italic",zorder=7)
fig.savefig(SALIDA,dpi=200)
print("ok", SALIDA.name)
