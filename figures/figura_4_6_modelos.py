"""Figura 4.6. Escalera de modelos y arquitecturas de la CNN plana y la ResNet (según src/cnn.py)."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8.6})
K=ROL["linea_ref"]
C1,C2,C3,C4=PALETA["wi-grey"],PALETA["wi-ocher"],PALETA["wwu-blue"],PALETA["wi-purple"]
fig,ax=plt.subplots(figsize=(11,7.2)); ax.set_xlim(0,110); ax.set_ylim(0,72); ax.axis("off")
def caja(x,y,w,h,txt,col,fs=8.6,alpha=0.16,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25,rounding_size=0.8",fc=col,alpha=alpha,ec=col,lw=1.3))
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25,rounding_size=0.8",fc="none",ec=col,lw=1.3))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=K,fontweight="bold" if bold else None)
def flecha(x0,y0,x1,y1,col=K):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,color=col,lw=1.1))
# --- (a) escalera
ax.text(1,70,"(a) Escalera de modelos",fontsize=10,color=K)
P=[("1. Umbral en C13",C1,"¿basta con nubes\nfrías cerca?"),
   ("2. Regresión logística\n(20·L estadísticas)",C2,"¿aportan todas las\nbandas y su evolución?"),
   ("3. CNN plana",C3,"¿aporta la estructura\nespacial?"),
   ("4. ResNet",C4,"¿aportan la profundidad\ny los residuales?")]
for i,(t,c,q) in enumerate(P):
    x=2+i*27; caja(x,57,22,8,t,c,fs=9,alpha=0.22,bold=True)
    ax.text(x+11,53.3,q,ha="center",va="top",fontsize=8,color=c if c!=C1 else "#666",style="italic")
    if i<3: flecha(x+22.6,61,x+26.4,61)
ax.plot([2,107],[45.5,45.5],color="#dddddd",lw=0.8)
# --- (b) CNN plana
ax.text(1,43,"(b) CNN plana",fontsize=10,color=C3)
y=32; h=7
B=[("Entrada\n32×32×(5·L)",8.5),("Bloque 1\n32 filtros\n32→16",9),("Bloque 2\n64 filtros\n16→8",9),("Bloque 3\n128 filtros\n8→4",9),
   ("Promedio\nglobal\n(128)",9),("Dropout\n+ lineal",9),("Sigmoide\n→ p",8.5)]
x=2
for j,(t,w) in enumerate(B):
    caja(x,y,w,h,t,C3 if 0<j<4 else "#9e9e9e",fs=7.8,alpha=0.22 if 0<j<4 else 0.10)
    if j<len(B)-1: flecha(x+w+0.3,y+h/2,x+w+3.2,y+h/2)
    x+=w+3.5
ax.text(36,28.3,"cada bloque: [conv 3×3 → BN → ReLU] × 2 → max-pooling 2×2",ha="center",fontsize=7.8,color="#555",style="italic")
# --- (c) ResNet
ax.text(1,24,"(c) ResNet (adaptada de Fan et al., 2024)",fontsize=10,color=C4)
y=8; x=2
R=[("Entrada\n32×32×(5·L)",8.5,0),("Conv 7×7\nF",6.5,1),("Res\nF",5,1),("Res\nF",5,1),("Res 2F\n↓ 16",5.6,1),("Res\n2F",5,1),("Res 4F\n↓ 8",5.6,1),("Res\n4F",5,1),
   ("Prom. 2×2\n→ 4×4",7,0),("Densa\n64",6,0),("Densa\n1",5.5,0),("Sigm.\n→ p",5.5,0)]
pos=[]
for j,(t,w,isr) in enumerate(R):
    caja(x,y,w,h,t,C4 if isr else "#9e9e9e",fs=7.4,alpha=0.22 if isr else 0.10)
    pos.append((x,w))
    if j<len(R)-1: flecha(x+w+0.3,y+h/2,x+w+2.4,y+h/2)
    x+=w+2.7
# atajo residual sobre un bloque
xr,wr=pos[3]
ax.add_patch(FancyArrowPatch((xr-1.2,y+h+0.2),(xr+wr+1.2,y+h+0.2),connectionstyle="arc3,rad=-0.6",arrowstyle="-|>",mutation_scale=9,color=C4,lw=1))
ax.text(xr+wr/2,y+h+4.6,"atajo: F(x) + x",ha="center",fontsize=7.6,color=C4)
ax.text(30,3.5,"bloque residual: dos convoluciones 3×3 (conv → Leaky ReLU → BN → dropout) más el atajo;\nlos bloques 3.º y 5.º reducen la resolución a la mitad (atajo con conv 1×1, paso 2)",
        ha="left",fontsize=7.6,color="#555",style="italic",va="center")
fig.tight_layout(); fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
