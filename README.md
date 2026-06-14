# nowcasting-bogota
Nowcasting de lluvia fuerte en Bogotá usando imágenes GOES-16

## Prerequisitos

- Miniconda o Anaconda instalado (https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10 (se recomienda crear un entorno dedicado)

## Requisitos del entorno

Además de las librerías de Python (requirements.txt), se necesita
instalar las herramientas de línea de comandos de NetCDF-C para la
compresión de archivos:

    conda install -c conda-forge libnetcdf
