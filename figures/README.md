# figures/

Scripts que generan las figuras de la tesis. Cada `figura_X_Y_*.py` produce, al lado, el `.png` del mismo nombre.

```bash
python figures/figura_5_1_desempeno_global.py
```

Todos los scripts toman la raíz del repositorio a partir de su propia ubicación (`RAIZ = Path(__file__).resolve().parents[1]`), importan la paleta desde `src/paleta.py` y usan DejaVu Serif, coma decimal y 250 ppp.

## Figuras con datos

| Figura | Script | Datos |
|---|---|---|
| 2.3 | `figura_2_3_bandas_goes.py` | `data/cmip_cropped/CMIPC_20241106-{170000,200000}.nc`, `data/cache/estaciones.parquet`, `data/cartografia/perimetro_urbano_bogota.geojson` |
| 4.4 | `figura_4_4_particion.py` | `datasets/indice_completo.parquet` |
| 4.5 | `figura_4_5_coocurrencia.py` | valores de `datasets/11-dataset_training.ipynb`, sección 3 (escritos en el script) |
| 5.1 | `figura_5_1_desempeno_global.py` | `modelo/salidas_s11/resultados_por_modelo.csv` |
| 5.2 | `figura_5_2_pareadas.py` | `modelo/salidas_s11/comparaciones_por_semilla.csv` |
| 5.3 | `figura_5_3_evento.py` | valores de la tabla de métricas por evento del Anexo D (escritos en el script) |
| 5.4 | `figura_5_4_estratos.py` | valores de la tabla de POD por estrato del Anexo D (escritos en el script) |
| 5.5 | `figura_5_5_mapa_pod.py` | `modelo/salidas_21/por_estacion.csv`, `data_ingestion/ground_truth/anexo_A_estaciones.csv` |
| 5.6–5.8 | `figura_5_6_a_5_8_casos.py` | `modelo/salidas_s11/pred_test_*.parquet`, `data/cmip_cropped/` (25-abr, 11-mar y 25-oct-2025) |
| 6.1 | `figura_6_1_coeficientes_logistica.py` | `modelo/salidas_21/coeficientes_logistica_L2.csv` |

## Esquemas

| Figura | Script |
|---|---|
| 2.1 | `figura_2_1_tarde_convectiva.py` |
| 2.2 | `figura_2_2_bandas_capas.py` (fondo ilustrado) |
| 2.4 | `figura_2_4_ciclo_tormenta.py` (fondo ilustrado) |
| 2.5 | `figura_2_5_extrapolacion.py` |
| 2.6 | `figura_2_6_horizontes.py` |
| 2.7 | `figura_2_7_antecedentes.py` |
| 3.3 | `figura_3_3_ventana_etiqueta.py` |
| 4.1 | `figura_4_1_flujo_sistema.py` (SVG rasterizado con Playwright) |
| 4.3 | `figura_4_3_paralaje.py` (fondo ilustrado) |
| 4.6 | `figura_4_6_modelos.py` |
| 4.7 | `figura_4_7_evento.py` |
| 4.8 | `figura_4_8_protocolo.py` |

Las figuras 2.2, 2.4 y 4.3 superponen haces, rótulos y geometría sobre un fondo sin texto generado con ChatGPT. Los scripts esperan esos fondos en `figures/fondos/` (`fondo_figura_2_2.png`, `fondo_figura_2_4.png`, `fondo_figura_4_3.png`); las coordenadas en píxeles están calibradas para esas imágenes. Los `.png` de estas tres figuras que hay en la carpeta son los de la versión entregada de la tesis.

## Figuras de las notebooks

Las Figuras 3.1, 3.2, 3.4–3.7, 4.2, A.1 y B.1 se generan dentro de las notebooks del repositorio y no se duplican aquí.
