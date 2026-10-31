"""Figura 2.6. Horizonte máximo y tipo de salida de los modelos de nowcasting revisados."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import matplotlib.pyplot as plt
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9.5})
K=ROL["linea_ref"]
FAM={"Extrapolación":PALETA["wi-grey"],"Generativos y de difusión":PALETA["wi-darkblue"],"Satélite como fuente principal o complementaria":PALETA["wwu-green"]}
# (modelo, familia, horizonte máximo en h, salida: 'campo' o 'clase')  -- VERIFICAR cada horizonte en su paper
M=[("pySTEPS (Pulkkinen et al., 2019)","Extrapolación",1.0,"campo"),
   ("ConvLSTM (Shi et al., 2015)","Extrapolación",1.5,"campo"),
   ("TrajGRU (Shi et al., 2017)","Extrapolación",2.0,"campo"),
   ("DGMR (Ravuri et al., 2021)","Generativos y de difusión",1.5,"campo"),
   ("NowcastNet (Zhang et al., 2023)","Generativos y de difusión",3.0,"campo"),
   ("Dai et al. (2025)","Generativos y de difusión",4.0,"campo"),
   ("DaYu (Wei et al., 2024)","Generativos y de difusión",12.0,"campo"),
   ("SATcast (Chen et al., 2025)","Generativos y de difusión",24.0,"campo"),
   ("MetNet (Sønderby et al., 2020)","Satélite como fuente principal o complementaria",8.0,"campo"),
   ("MetNet-2 (Espeholt et al., 2022)","Satélite como fuente principal o complementaria",12.0,"campo"),
   ("ProbSevere (Cintineo et al., 2018)","Satélite como fuente principal o complementaria",1.5,"clase"),
   ("ThunderCast (Ortland et al., 2023)","Satélite como fuente principal o complementaria",1.0,"clase"),
   ("Lagerquist et al. (2021)","Satélite como fuente principal o complementaria",2.0,"clase"),
   ("Fan et al. (2024)","Satélite como fuente principal o complementaria",1.0,"clase")]
fig,ax=plt.subplots(figsize=(9,6.6))
ax.axvspan(1,3,color=ROL["positivo"],alpha=0.10,lw=0)
n=len(M); ys=list(range(n+1,1,-1))
for (m,f,h,s),y in zip(M,ys):
    ax.plot([0.5,h],[y,y],color=FAM[f],lw=6,solid_capstyle="butt")
    ax.plot(h,y,"D" if s=="clase" else "o",ms=9,mfc="white" if s=="campo" else FAM[f],mec=FAM[f],mew=2)
# esta tesis
ax.plot([1,3],[0.6,0.6],color=ROL["positivo"],lw=6,solid_capstyle="butt"); ax.plot(3,0.6,"D",ms=10,color=ROL["positivo"])
ax.set_yticks(ys+[0.6]); ax.set_yticklabels([m for m,*_ in M]+["Esta tesis"])
ax.get_yticklabels()[-1].set_fontweight("bold"); ax.get_yticklabels()[-1].set_color(ROL["positivo"])
ax.set_xscale("log"); ax.set_xlim(0.5,30); ax.set_xticks([0.5,1,2,3,6,12,24]); ax.set_xticklabels(["0.5","1","2","3","6","12","24"])
ax.set_xlabel("Horizonte máximo de pronóstico (horas, escala logarítmica)",color=K); ax.set_ylim(-0.2,n+2)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
from matplotlib.lines import Line2D
h=[Line2D([],[],color=c,lw=6,label=f) for f,c in FAM.items()]
h+= [Line2D([],[],marker="o",ls="",mfc="white",mec=K,mew=2,ms=8,label="Salida: campo continuo"),
     Line2D([],[],marker="D",ls="",color=K,ms=8,label="Salida: clasificación o probabilidad de evento")]
ax.legend(handles=h,loc="lower right",frameon=True,fontsize=8.3,framealpha=0.95)
fig.tight_layout(); fig.savefig(SALIDA,dpi=250)
print("ok", SALIDA.name)
