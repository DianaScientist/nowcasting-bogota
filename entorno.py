import os
import sys

def asegurar_dlls_conda():
    """En Windows, agrega las carpetas de DLL del entorno conda al PATH
    si el entorno no fue activado (p. ej., kernel lanzado por VS Code sin activación).
    En otros sistemas, o si el entorno ya está activado, no hace nada."""
    if sys.platform != "win32":
        return
    rutas = [os.path.join(sys.prefix, p) for p in
             (r"Library\mingw-w64\bin", r"Library\usr\bin", r"Library\bin", "Scripts", "bin")]
    actuales = os.environ.get("PATH", "").lower().split(";")
    faltantes = [r for r in rutas if r.lower() not in actuales and os.path.isdir(r)]
    if faltantes:
        os.environ["PATH"] = ";".join(faltantes) + ";" + os.environ["PATH"]

asegurar_dlls_conda()