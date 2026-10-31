"""Figura 4.1. Flujo del sistema de nowcasting, de los datos a la alerta.
Se dibuja como SVG y se rasteriza con Playwright (Chromium). Genera figura_4_1_flujo_sistema.svg y .png."""
from pathlib import Path
SALIDA = Path(__file__).with_suffix(".png")
SVG = Path(__file__).with_suffix(".svg")

W=1200
boxes=[]
def box(x,y,w,h,lines,sec,fill="#ffffff",stroke="#4a4a4a",bold_first=True):
    s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    n=len(lines); lh=19; y0=y+h/2-(n-1)*lh/2+6
    for i,t in enumerate(lines):
        wgt='600' if (i==0 and bold_first) else '400'; sz=16 if i==0 else 14
        col='#1a1a1a' if i==0 else '#3d3d3d'
        s+=f'<text x="{x+w/2}" y="{y0+i*lh}" text-anchor="middle" font-size="{sz}" font-weight="{wgt}" fill="{col}">{t}</text>'
    if sec: s+=f'<text x="{x+w-8}" y="{y+16}" text-anchor="end" font-size="11.5" fill="#6b6b6b">{sec}</text>'
    boxes.append(s)
arrows=[]
def arrow(x1,y1,x2,y2):
    arrows.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4a4a4a" stroke-width="1.6" marker-end="url(#m)"/>')
def path(d):
    arrows.append(f'<path d="{d}" fill="none" stroke="#4a4a4a" stroke-width="1.6" marker-end="url(#m)"/>')
SAT="#eef3f8"; GT="#f3f1ea"; MOD="#ffffff"; ACC="#e6f0ec"
L,R,cw,bh,gap=60,640,500,70,26
y=70
heads=[f'<text x="{L+cw/2}" y="48" text-anchor="middle" font-size="17" font-weight="700" fill="#1f3a5a">Cadena satelital · GOES-16/19 ABI</text>',
       f'<text x="{R+cw/2}" y="48" text-anchor="middle" font-size="17" font-weight="700" fill="#5a4a1f">Cadena de verdad de campo · IDEAM</text>']
sat=[(["Imágenes ABI L2 CMIP","bandas 8, 9, 10, 13 y 14 · cada 10 min · 2024–2025"],"3.1"),
     (["Recorte del dominio y control de calidad","re-descarga y reconstrucción desde L1b"],"3.1"),
     (["Intercalibración GOES-16 / GOES-19","comparación simultánea · sin corrección"],"3.1"),
     (["Normalización y extracción del parche","32 × 32 píxeles · imagen t y L − 1 previas"],"4.2")]
gt=[(["Pluviómetros subhorarios","red de la Sabana de Bogotá · 2024–2025"],"3.2"),
    (["Depuración y control de calidad","duplicados, criterios de exclusión"],"3.2"),
    (["Umbral de lluvia fuerte por estación","P95 empírico de los instantes húmedos de 2024"],"3.3"),
    (["Etiqueta de cada par (estación, t)","lluvia fuerte en (t + 1 h, t + 3 h]"],"3.4")]
for i,((a,b),(c,d)) in enumerate(zip(sat,gt)):
    yy=y+i*(bh+gap)
    box(L,yy,cw,bh,a,b,fill=SAT,stroke="#5b7ea3"); box(R,yy,cw,bh,c,d,fill=GT,stroke="#a38f5b")
    if i: arrow(L+cw/2,yy-gap,L+cw/2,yy-2); arrow(R+cw/2,yy-gap,R+cw/2,yy-2)
ylast=y+3*(bh+gap)+bh
yi=ylast+50
box(L,yi,W-2*L,bh+4,["Índice de pares (estación, t)","se excluyen los pares con lluvia fuerte en curso o inminente en t · historia de L = 2 a 4 imágenes"],"3.5")
path(f"M{L+cw/2},{ylast} L{L+cw/2},{yi-2}"); path(f"M{R+cw/2},{ylast} L{R+cw/2},{yi-2}")
yp=yi+bh+4+gap+4
box(L,yp,W-2*L,bh,["Partición temporal","entrenamiento 1-ene → 14-nov-2024 · validación 15-nov → 31-dic-2024 · prueba 2025"],"4.3")
arrow(W/2,yi+bh+4,W/2,yp-2)
ym=yp+bh+gap+4; mh=118
box(L,ym,W-2*L,mh,["",""],"4.4–4.6",fill=MOD)
boxes.append(f'<text x="{W/2}" y="{ym+28}" text-anchor="middle" font-size="16" font-weight="600" fill="#1a1a1a">Escalera de complejidad · entrenamiento</text>')
steps=["1 · Umbral en C13","2 · Regresión logística","3 · CNN plana","4 · ResNet"]
sw,sg=230,22; sx=(W-(4*sw+3*sg))/2
for i,t in enumerate(steps):
    xx=sx+i*(sw+sg)
    boxes.append(f'<rect x="{xx}" y="{ym+44}" width="{sw}" height="38" rx="6" fill="#f6f6f6" stroke="#8a8a8a"/>'
                 f'<text x="{xx+sw/2}" y="{ym+68}" text-anchor="middle" font-size="14.5" fill="#1a1a1a">{t}</text>')
    if i: arrow(xx-sg+2,ym+63,xx-3,ym+63)
boxes.append(f'<text x="{W/2}" y="{ym+104}" text-anchor="middle" font-size="13.5" fill="#3d3d3d">peldaños 2–4 con L ∈ {{2, 3, 4}} · submuestreo por época 1 : n : n con negativos difíciles · pérdida BCE + λ · CSI</text>')
arrow(W/2,yp+bh,W/2,ym-2)
ys=ym+mh+gap+4
box(L,ys,W-2*L,bh,["Selección en validación (prevalencia real)","hiperparámetros y mejor modelo por PR-AUC · umbral de decisión por CSI máximo, congelado"],"4.4, 4.6")
arrow(W/2,ym+mh,W/2,ys-2)
yo=ys+bh+gap+4; ow=(W-2*L-30)/2
box(L,yo,ow,bh+6,["Salida operativa","probabilidad por estación → alerta si p ≥ umbral"],"",fill=ACC,stroke="#4f8a72")
box(L+ow+30,yo,ow,bh+6,["Evaluación en prueba (una sola apertura)","por par, por evento, por estrato · bootstrap por días"],"4.7",fill=ACC,stroke="#4f8a72")
path(f"M{W/2},{ys+bh} L{W/2},{ys+bh+14} L{L+ow/2},{ys+bh+14} L{L+ow/2},{yo-2}")
path(f"M{W/2},{ys+bh+14} L{L+ow+30+ow/2},{ys+bh+14} L{L+ow+30+ow/2},{yo-2}")
H=yo+bh+6+40
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="DejaVu Sans, Arial, sans-serif">
<defs><marker id="m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4a4a4a"/></marker></defs>
<rect width="100%" height="100%" fill="#ffffff"/>{"".join(heads)}{"".join(arrows)}{"".join(boxes)}</svg>'''
SVG.write_text(svg, encoding="utf-8")

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1200, "height": 1200}, device_scale_factor=2.5)
    pg.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
    pg.query_selector("svg").screenshot(path=str(SALIDA)); b.close()
print("ok", SALIDA.name)
