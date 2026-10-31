# nowcasting-bogota

Código de la tesis de Maestría en Ciencia de Datos (Universidad Austral) de Diana Bohórquez: **clasificación de la ocurrencia de lluvia fuerte con una a tres horas de anticipación en las estaciones de la Sabana de Bogotá, a partir de imágenes del satélite GOES-16/19**.

Cada ejemplo es un par (estación, instante t). La entrada son parches de 32 × 32 píxeles de las bandas 8, 9, 10, 13 y 14 del ABI en t y en las L − 1 imágenes previas (L = 2, 3 o 4). La etiqueta indica si en la estación ocurre lluvia fuerte en la ventana (t + 1 h, t + 3 h], según un umbral propio de cada estación (percentil 95 de 2024). Se comparan cuatro modelos de complejidad creciente: umbral sobre el percentil 10 de C13, regresión logística, CNN plana y ResNet. Entrenamiento: 1-ene a 14-nov-2024; validación: 15-nov a 31-dic-2024; prueba: 2025, abierta una sola vez.

> **Versión citable:** `v1.0-tesis`, archivada en Zenodo · DOI: [10.5281/zenodo.23848908](https://doi.org/10.5281/zenodo.23848908)
>
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23848908.svg)](https://doi.org/10.5281/zenodo.23848908)

## Estructura

Las notebooks están numeradas en el orden en que se ejecutan.

| Carpeta | Contenido |
|---|---|
| `data_ingestion/ground_truth/` | 01–06: exploración y depuración de los pluviómetros del IDEAM, control de calidad, umbral de lluvia fuerte por estación, etiquetas y cuantificación del paralaje. `anexo_A_estaciones.csv` es la tabla del Anexo A. |
| `data_ingestion/satellite/` | `download_cmip.py` y `get_cut_compress.py` (descarga y recorte del producto ABI-L2-CMIPF); 07–10: control de calidad, inventario y reconstrucción desde L1b, intercalibración GOES-16/19 y extracción de parches. |
| `datasets/` | 11–15: índice de pares (estación, t), análisis exploratorio, caché de imágenes, negativos difíciles y cargador. |
| `modelo/` | 16–18 y 21: evaluación, modelos de referencia, CNN plana y análisis de errores. Scripts de análisis complementarios posteriores a la apertura de prueba. |
| `src/` | Módulos usados por las notebooks: etiquetado, control de calidad GOES, caché, cargador, modelos (`referencias.py`, `cnn.py`), búsqueda de hiperparámetros (`buscar.py`), congelamiento (`congelar.py`), apertura de prueba (`abrir_prueba.py`), métricas y bootstrap (`evaluacion.py`), análisis de errores (`analisis.py`) y paleta de colores (`paleta.py`). |
| `figures/` | Scripts de las figuras de la tesis (ver `figures/README.md`). |
| `exploracion/`, `prework/`, `docs/` | Exploración inicial previa al diseño. |

## Datos

Los datos de entrada no se incluyen en el repositorio por su tamaño (ver `.gitignore`):

- **Imágenes GOES-16/19 (NOAA)**, producto ABI-L2-CMIPF, bandas 8, 9, 10, 13 y 14, cada 10 minutos, 2024–2025. Se descargan del repositorio público de NOAA en AWS con `data_ingestion/satellite/download_cmip.py`.
- **Precipitación subhoraria del IDEAM**, catálogo de datos abiertos de Colombia (datos.gov.co), estaciones de la Sabana de Bogotá. Se descargan en `data_ingestion/ground_truth/01-ground_truth_exploration.ipynb`.

Ambas fuentes se rigen por sus propias condiciones de uso.

## Entorno

- Python 3.10 en un entorno conda dedicado (`nowcasting`).
- Librerías: `pip install -r requirements.txt`. En Windows con GPU, instalar primero PyTorch como indica la primera línea de `requirements.txt`.
- Herramientas NetCDF-C para la compresión de archivos: `conda install -c conda-forge libnetcdf`.
- `entorno.py` corrige el PATH de DLL de conda en Windows cuando el kernel se lanza sin activar el entorno. Las notebooks lo usan además para ubicar la raíz del repositorio, de modo que corren desde cualquier carpeta donde se clone.
- Solo para la Figura 4.1: `playwright install chromium`.

La búsqueda de hiperparámetros y el congelamiento de las redes se corrieron en una máquina virtual con GPU (`src/cadena_s8b_s9.sh`). Las semillas están fijas, pero el entrenamiento en GPU no es exactamente determinista: al reentrenar, los resultados se reproducen dentro de la variabilidad entre semillas reportada en la tesis, no necesariamente al último decimal.

## Licencia y cita

Código bajo licencia MIT (ver `LICENSE`). Si usas este trabajo, cita la versión archivada en Zenodo (DOI arriba).
