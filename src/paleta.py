# paleta.py — paleta institucional para todas las figuras de la tesis
import matplotlib as mpl

PALETA = {
    "wi-green1":       "#5bbb56", "wi-mud":          "#b5bb56",
    "wi-iceblue":      "#56bbb7", "wi-grey":         "#929292",
    "wi-darkblue":     "#567cbb", "wwu-black":       "#3e3e3b",
    "wi-purple":       "#880085", "wwu-lightgreen":  "#7ab516",
    "wi-pink":         "#f36196", "wwu-green":       "#008e96",
    "wi-coral":        "#f88379", "wwu-lightblue":   "#009dd1",
    "wi-rose":         "#ffc0cb", "wwu-blue":        "#006e89",
    "wi-ocher":        "#bbb056", "ercis-red":       "#852339",
}

# Roles semánticos: los scripts usan estos nombres, nunca hex sueltos
ROL = {
    "positivo":     PALETA["ercis-red"],
    "negativo":     "#d9d9d9",               # gris claro de fondo (fuera de paleta, solo relleno)
    "hard_neg":     PALETA["wi-mud"],
    "random_neg":   PALETA["wi-grey"],
    "serie":        PALETA["wi-darkblue"],
    "estacional":   PALETA["wi-green1"],
    "linea_ref":    PALETA["wwu-black"],
    "umbral":       PALETA["wi-coral"],
    # Conjuntos del split y sus fronteras
    "split_train":  PALETA["wwu-blue"],
    "split_val":    PALETA["wi-ocher"],
    "split_test":   PALETA["wi-purple"],
    "corte_val":    PALETA["wi-ocher"],      # frontera train → val
    "corte_test":   PALETA["wi-purple"],     # frontera val → test
}

ORIGEN_UMBRAL = {
    "propio":                          PALETA["wi-darkblue"],
    "respaldo pooled":                 PALETA["wi-ocher"],
    "propio (diagnóstico, excepción)": PALETA["ercis-red"],
}

def aplicar_estilo():
    """Ciclo de colores por defecto = paleta, para gráficos sin color explícito."""
    ciclo = [PALETA[k] for k in ["wi-darkblue", "wi-coral", "wi-green1", "wi-ocher",
                                  "wi-purple", "wwu-green", "wi-pink", "wwu-black"]]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=ciclo)