"""Figura 2.2. Capa de la atmósfera a la que responde cada banda del ABI (esquema ilustrado).
Fondo sin texto generado con ChatGPT (figures/fondos/fondo_figura_2_2.png); haces, capas y rótulos encima."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import matplotlib.pyplot as plt
from PIL import Image
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif"})
im=Image.open(RAIZ / "figures" / "fondos" / "fondo_figura_2_2.png").convert("RGB")
W,H=im.size
Y0,KM0,S=805,2.55,39.0           # suelo de la Sabana = 2.55 km; 39 px/km (escala esquemática)
CROP=200
y=lambda km: Y0-(km-KM0)*S
K=ROL["linea_ref"]; B8,B9,B10=PALETA["wwu-blue"],PALETA["wi-darkblue"],PALETA["wwu-lightblue"]; RW=ROL["positivo"]
fig=plt.figure(figsize=(W/150,(H-CROP)/150),dpi=150); ax=fig.add_axes([0,0,1,1]); ax.imshow(im); ax.set_xlim(0,W); ax.set_ylim(H,CROP); ax.axis("off")
# capas
for lo,hi,c,lab in [(8,10,B8,"troposfera alta\n(~300–350 hPa)"),(5.8,7.2,B9,"troposfera media\n(~450 hPa)"),(3.6,4.6,B10,"troposfera media-baja\n(~600–650 hPa)")]:
    ax.fill_between([0,W],y(hi),y(lo),color=c,alpha=0.16,lw=0)
    ax.text(14,(y(lo)+y(hi))/2,lab,va="center",fontsize=8.5,color=c,fontweight="bold")
SAT=330
ax.text(890,SAT-95,"GOES — ABI",ha="center",fontsize=11,fontweight="bold",color=K)
ax.plot([800,1300],[SAT-15,SAT-15],color=K,lw=0.8,ls=":")
def haz(x,km,col,lab,dy=0):
    ax.annotate("",xy=(x,y(km)),xytext=(x,SAT),arrowprops=dict(arrowstyle="-|>",color=col,lw=2,mutation_scale=12))
    ax.plot(x,y(km),"o",color=col,ms=5.5,mec="white",mew=0.8)
    ax.text(x,SAT-22-dy,lab,ha="center",va="bottom",fontsize=7.8,color=col,fontweight="bold")
haz(840,9.0,B8,"B8")
haz(872,6.5,B9,"B9",dy=14)
haz(904,4.1,B10,"B10")
haz(938,KM0,RW,"B13·B14\ndespejado",dy=14)
haz(1250,KM0+(Y0-434)/S,RW,"B13·B14\ncon nube")
# espesor de atmósfera
TOP=y(11.5)
def esp(x,ybot,lab):
    ax.annotate("",xy=(x,ybot),xytext=(x,TOP),arrowprops=dict(arrowstyle="<->",color="#555",lw=1.2))
    ax.text(x-10,(TOP+ybot)/2,lab,rotation=90,va="center",ha="right",fontsize=7.8,color="#444")
esp(300,858,"espesor de atmósfera sobre el valle")
esp(790,800,"espesor de atmósfera\nsobre la Sabana")
ax.text(150,905,"Valle del Magdalena",fontsize=9,color="white",fontweight="bold",ha="center")
ax.text(1150,880,"Sabana de Bogotá (~2 550 m)",fontsize=9,color="white",fontweight="bold",ha="center")
ax.text(660,690,"Cordillera\nOriental",fontsize=8.5,color=K,ha="center")
ax.text(1440,420,"cumulonimbo",fontsize=8.5,color="#555",ha="left")
fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
