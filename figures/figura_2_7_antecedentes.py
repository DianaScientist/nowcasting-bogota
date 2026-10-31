"""Figura 2.7. Posición de los antecedentes colombianos según horizonte y tipo de tarea."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9.5})
K=ROL["linea_ref"]; C=PALETA["wwu-blue"]; P=ROL["positivo"]
XS=["Presente","30 min","1 h","3 h","24 h"]
YS=["Estimación o detección\ndel presente","Pronóstico de la imagen\no del campo de lluvia","Pronóstico\ncualitativo","Clasificación de evento contra\numbral de intensidad"]
MK={"radar":"^","satélite":"o","combinación":"s"}
# (etiqueta, x, y, fuente, operativo, dx, dy, ha)
D=[("Sistema de Alerta de Bogotá (IDIGER)",0,0,"combinación",True,0.13,0.17,"left"),
   ("Estimación satelital del IDEAM",0,0,"satélite",True,0.13,-0.02,"left"),
   ("García y Rodas Arias (2020)",0,0,"satélite",False,0.13,-0.2,"left"),
   ("Bohórquez (2025)",1,1,"radar",False,0,-0.2,"center"),
   ("Salazar y García Diaz (2025)",2,1,"satélite",False,0,0.17,"center"),
   ("Salazar (2019)",3,1,"combinación",False,0,-0.2,"center"),
   ("Pronósticos IDEAM–IDIGER",4,2,"combinación",True,-0.13,0.15,"right"),
   ("SIATA, pronóstico diario",4,2,"combinación",True,-0.13,-0.15,"right")]
OFF={("Sistema de Alerta de Bogotá (IDIGER)"):0.15,"Estimación satelital del IDEAM":0,"García y Rodas Arias (2020)":-0.15,
     "Pronósticos IDEAM–IDIGER":0.12,"SIATA, pronóstico diario":-0.12}
fig,ax=plt.subplots(figsize=(10,5.6))
for y in range(4): ax.axhline(y,color="#eeeeee",lw=0.8,zorder=0)
ax.add_patch(Rectangle((2,2.72),1,0.56,fc=P,alpha=0.15,ec=P,lw=1.2))
ax.plot(3,3,"D",ms=10,color=P); ax.text(2.5,3,"esta tesis",ha="center",va="center",fontsize=10,color=P,fontweight="bold")
import numpy as np
xr=2+np.log(2)/np.log(3); xs=3+np.log(2)/np.log(8)   # 2 h y 6 h en el eje no lineal
ax.plot([xr,xs],[1.14,1.14],color=C,lw=1.2,zorder=2)
ax.plot(xr,1.14,"^",ms=10,color=C,zorder=3); ax.plot(xs,1.14,"o",ms=10,color=C,zorder=3)
ax.text((xr+xs)/2,1.36,"SIATA, extrapolación\n(radar 2 h, satélite 6 h)",ha="center",va="center",fontsize=8.6,color=K)
for lab,x,y,src,op,dx,dy,ha in D:
    yy=y+OFF.get(lab,0)
    ax.plot(x,yy,MK[src],ms=10,mfc=C if op else "white",mec=C,mew=1.8,zorder=3)
    ax.text(x+dx,yy+(dy if ha=="center" else 0),lab,ha=ha,va="center",fontsize=8.6,color=K)
ax.set_xticks(list(range(5))+[xs]); ax.set_xticklabels(XS+["6 h"]); ax.set_yticks(range(4)); ax.set_yticklabels(YS,fontsize=9)
ax.set_xlim(-0.35,4.35); ax.set_ylim(-0.45,3.45)
ax.set_xlabel("Horizonte de anticipación (escala no lineal)",color=K)
for s in ["top","right"]: ax.spines[s].set_visible(False)
h=[Line2D([],[],marker=m,ls="",mfc="white",mec=K,mew=1.6,ms=8,label=f"Fuente: {k}") for k,m in MK.items()]
h+=[Line2D([],[],marker="o",ls="",color=C,ms=8,label="Sistema operativo"),Line2D([],[],marker="o",ls="",mfc="white",mec=C,mew=1.6,ms=8,label="Trabajo académico")]
ax.legend(handles=h,loc="upper left",fontsize=8.2,frameon=True,framealpha=0.95,ncol=1)
fig.tight_layout(); fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
