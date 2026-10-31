"""Figura 4.7. Métricas por evento en una estación: episodio, aviso y anticipación (esquema)."""
from pathlib import Path; import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
SALIDA = Path(__file__).with_suffix(".png")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from paleta import PALETA, ROL
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9})
K=ROL["linea_ref"]; S=ROL["serie"]; P=ROL["positivo"]; OK=PALETA["wwu-green"]; U=ROL["umbral"]
t=np.linspace(0,10,1000)   # horas desde las 10:00
I=8.5*np.exp(-((t-6.0)/0.45)**2)+7.5*np.exp(-((t-7.0)/0.4)**2)+0.4
u=6.0
fig,(a1,a2,a3)=plt.subplots(3,1,figsize=(10,6.2),sharex=True,gridspec_kw={"height_ratios":[2.4,0.7,1.3],"hspace":0.12})
# (1) lluvia
a1.plot(t,I,color=S,lw=1.8); a1.axhline(u,color=U,ls="--",lw=1.2)
a1.text(0.05,u+0.35,"umbral de la estación",fontsize=8.3,color=U)
sob=I>=u; ini=t[sob][0]; fin=t[sob][-1]
a1.axvspan(ini,fin,color=P,alpha=0.12,lw=0)
a1.text((ini+fin)/2,10.3,"episodio",ha="center",fontsize=9,color=P,fontweight="bold")
gap=t[(t>6.2)&(t<6.9)&(I<u)]
if len(gap): a1.annotate("pausa < 30 min:\nmismo episodio",xy=(gap.mean(),u-0.3),xytext=(8.1,3.2),fontsize=8,color=K,arrowprops=dict(arrowstyle="-",color=K,lw=0.7))
a1.axvline(ini,color=P,lw=1.4); a1.text(ini-0.08,9.3,"inicio",ha="right",fontsize=8.6,color=P)
a1.set_ylabel("Intensidad\nmóvil 60 min",color=K); a1.set_ylim(0,11.5); a1.set_yticks([])
# (2) etiquetas
pos0,pos1=ini-3,fin-2
a2.add_patch(Rectangle((pos0,0.2),pos1-pos0,0.6,fc=P,alpha=0.35,ec=P))
a2.text(pos1+0.12,0.5,"pares positivos: su ventana (t + 1 h, t + 3 h]\ncontiene una hora sobre el umbral",ha="left",va="center",fontsize=7.8,color=K)
a2.set_ylim(0,1); a2.set_yticks([]); a2.set_ylabel("Etiqueta",color=K)
# (3) alertas
def alertas(ts,col,y):
    for x in ts: a3.add_patch(Rectangle((x,y-0.18),0.15,0.36,fc=col,ec="white",lw=0.6))
fa=np.arange(0.6,1.5,1/6); alertas(fa,"#9e9e9e",1.2)
a3.text(fa.mean()+0.07,1.62,"aviso falso",ha="center",fontsize=8.6,color="#666")
a3.text(fa.mean()+0.07,0.72,"ninguna alerta\nen par positivo",ha="center",va="top",fontsize=7.6,color="#777")
co=np.arange(ini-2.5,ini-1.1,1/6); alertas(co,OK,1.2)
a3.text(co.mean()+0.07,1.62,"aviso correcto",ha="center",fontsize=8.6,color=OK,fontweight="bold")
a3.add_patch(FancyArrowPatch((co[0],0.55),(ini,0.55),arrowstyle="<->",mutation_scale=11,color=K,lw=1.2))
a3.text((co[0]+ini)/2,0.25,"anticipación: primera alerta válida → inicio del episodio",ha="center",fontsize=8.2,color=K)
a3.axvline(ini,color=P,lw=1.4,ymax=0.95)
a3.set_ylim(0,2); a3.set_yticks([]); a3.set_ylabel("Alertas\ndel modelo",color=K)
for a in (a1,a2,a3):
    for s in ["top","right","left"]: a.spines[s].set_visible(False)
a3.set_xlim(0,10); a3.set_xticks(range(0,11)); a3.set_xticklabels([f"{10+h}:00" for h in range(11)])
a3.set_xlabel("Hora local (esquema)",color=K)
fig.savefig(SALIDA,dpi=250,bbox_inches="tight")
print("ok", SALIDA.name)
