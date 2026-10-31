"""Figura 2.5. Limitación de principio de la extrapolación (campos sintéticos)."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.ndimage import gaussian_filter
from paleta import ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9.5})
K=ROL["linea_ref"]
n=400; X,Y=np.meshgrid(np.linspace(0,100,n),np.linspace(0,100,n))
rng=np.random.default_rng(7); tex=gaussian_filter(rng.normal(size=(n,n)),10); tex/=tex.std()
def celda(x0,y0,r,amp,ang=0,el=1.4):
    c,s=np.cos(ang),np.sin(ang); dx,dy=X-x0,Y-y0
    u=(c*dx+s*dy)/el; v=-s*dx+c*dy
    return amp*np.exp(-(u**2+v**2)/(2*r**2))*(1+0.18*tex)
A=(22,70); B=(30,32); V=(30,8)   # desplazamiento común (flujo óptico)
f0=celda(*A,5,35,0.5)+celda(*B,6,45,0.3)
f1=celda(A[0]+V[0],A[1]+V[1],5,35,0.5)+celda(B[0]+V[0],B[1]+V[1],6,45,0.3)
C=(24,22)
f2=celda(A[0]+V[0],A[1]+V[1],5,2.5,0.5)+celda(B[0]+V[0],B[1]+V[1],7.5,75,0.3)+celda(*C,4.5,40,1.0)
lev=[1,2,5,10,20,30,50,80]; cols=["#c6e6f5","#7cc4e8","#2b8cbe","#41ab5d","#fed976","#fd8d3c","#e31a1c","#800026"]
cm=ListedColormap(cols); cm.set_under((1,1,1,0)); norm=BoundaryNorm(lev+[200],cm.N)
fig,axs=plt.subplots(1,3,figsize=(11,4.4),constrained_layout=True)
tit=["Observación en t","Pronóstico por extrapolación en t + Δt","Observación real en t + Δt"]
for ax,f,t in zip(axs,[f0,f1,f2],tit):
    ax.set_facecolor("#f4f4f2"); im=ax.imshow(np.where(f<1,np.nan,f),extent=[0,100,0,100],origin="lower",cmap=cm,norm=norm)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,fontsize=10,color=K)
for p in (A,B): axs[0].add_patch(FancyArrowPatch(p,(p[0]+V[0],p[1]+V[1]),arrowstyle="-|>",mutation_scale=14,color=K,lw=1.6))
axs[0].text(52,52,"desplazamiento\nestimado",fontsize=8.3,color=K)
# panel 3: anotaciones
pa=(A[0]+V[0],A[1]+V[1]); axs[2].add_patch(Circle(pa,10,fill=False,ec=K,lw=1,ls=":"))
axs[2].text(pa[0],pa[1]+12,"se disipó",ha="center",fontsize=8.3,color=K)
pb=(B[0]+V[0],B[1]+V[1]); axs[2].text(pb[0],pb[1]-22,"se intensificó",ha="center",fontsize=8.3,color=K)
axs[2].add_patch(Circle(C,11,fill=False,ec=ROL["positivo"],lw=1.6,ls="--"))
axs[2].text(C[0],C[1]+13,"iniciación\nno anticipable",ha="center",va="bottom",fontsize=8.5,color=ROL["positivo"],fontweight="bold")
cb=fig.colorbar(im,ax=axs,orientation="vertical",shrink=0.85,pad=0.01,ticks=lev); cb.set_label("Intensidad de lluvia (mm/h)")
fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
