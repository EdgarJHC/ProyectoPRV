"""Diagrama del detector adaptativo (ventanas Elgendi + 2a pasada)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = r"D:\Escuela IPN\Servicio\LecturaEDF\diagrama_filtro_adaptativo.png"

# Paleta
C_IN = "#1f4e79"
C_CLEAN = "#2e75b6"
C_WIN = "#3A8F01"
C_LOOP = "#c45911"
C_OUT = "#a30bdf"
C_NEUT = "#5b5b5b"
BG = "#f7f5f2"
CARD = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "text.color": "#1a1a1a",
})

fig, ax = plt.subplots(figsize=(12.5, 8.2), dpi=180)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 125)
ax.set_ylim(0, 82)
ax.set_axis_off()
ax.set_facecolor(BG)


def box(x, y, w, h, title, lines, color, lw=1.6):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=0.6",
        facecolor=CARD,
        edgecolor=color,
        linewidth=lw,
        mutation_aspect=0.6,
        zorder=2,
    )
    ax.add_patch(p)
    ax.add_patch(Rectangle((x, y + h - 0.22), w, 0.22, color=color, zorder=3, clip_on=False))
    ax.text(x + w / 2, y + h - 1.15, title, ha="center", va="center",
            fontsize=10, fontweight="medium", color=color, zorder=4)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + h / 2 - 0.55, body, ha="center", va="center",
            fontsize=8.6, color="#222", linespacing=1.35, zorder=4)
    return (x + w / 2, y, x + w / 2, y + h)


def diamond(cx, cy, w, h, text, color):
    xs = [cx, cx + w / 2, cx, cx - w / 2, cx]
    ys = [cy + h / 2, cy, cy - h / 2, cy, cy + h / 2]
    ax.fill(xs, ys, facecolor=CARD, edgecolor=color, linewidth=1.6, zorder=2)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.4,
            color=color, fontweight="medium", zorder=4)
    return cx, cy


def arrow(x1, y1, x2, y2, color=C_NEUT):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.35,
            mutation_scale=11,
            connectionstyle="arc3,rad=0",
            shrinkA=1,
            shrinkB=1,
        ),
        zorder=1,
    )


def wire(points, color=C_NEUT, head=True):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if head and len(points) >= 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=1.35, zorder=1, solid_capstyle="round")
        arrow(xs[-2], ys[-2], xs[-1], ys[-1], color)
    else:
        ax.plot(xs, ys, color=color, lw=1.35, zorder=1, solid_capstyle="round")


# Encabezado
ax.text(62.5, 79.2, "Detector adaptativo de picos PPG (PRV)",
        ha="center", va="center", fontsize=15, fontweight="medium", color="#1a1a1a")
ax.text(62.5, 76.4, "Elgendi por ventanas de 20 s  ·  retroalimentación hasta error ≈ 0",
        ha="center", va="center", fontsize=10, color=C_NEUT)

# Columna izquierda: adquisicion y pasada 1
box(4, 64.5, 36, 9.2, "1. Entrada",
    ["EDF Alice NightOne  ·  canal Pleth", "fs = 100 Hz   recorte 5–10 min"],
    C_IN)

box(4, 51.5, 36, 9.8, "2. Limpieza (global, no adaptativa)",
    ["ppg_clean Elgendi", "pasa-banda 0.5–8 Hz en todo el recorte"],
    C_CLEAN)

box(4, 32.2, 36, 16.0, "3. Pasada 1 — umbral local",
    ["Ventana 20 s, solape 5 s (paso 15 s)",
     "Elgendi: MA 111 ms y 667 ms",
     "umbral = MA_beat + 0.02·mean(sqrd)",
     "mean(sqrd) se calcula en ESA ventana",
     "salida: sístoles de la ventana i"],
    C_WIN, lw=2.0)

# Mini ventanas ilustrativas
ax.text(22, 30.3, "avance:  V1 → V2 → … → V20", ha="center", va="center",
        fontsize=8, color=C_WIN)
for i, lab in enumerate(["20 s", "20 s", "20 s"]):
    x0 = 8.5 + i * 9.2
    ax.add_patch(FancyBboxPatch(
        (x0, 25.6), 8.4, 3.4,
        boxstyle="round,pad=0.15,rounding_size=0.3",
        facecolor="#eaf5e3", edgecolor=C_WIN, linewidth=1.1, zorder=2,
    ))
    ax.text(x0 + 4.2, 27.3, lab, ha="center", va="center", fontsize=8, color=C_WIN)
arrow(16.9, 27.3, 17.7, 27.3, C_WIN)
arrow(26.1, 27.3, 26.9, 27.3, C_WIN)

box(4, 13.6, 36, 9.4, "4. Unión de ventanas",
    ["Picos del solape a < 0.3 s", "se fusionan; se queda el más alto"],
    C_WIN)

# Columna derecha: pasada 2
box(48, 51.2, 40, 22.6, "5. Pasada 2 — lazo de error",
    ["PPI mediano de todas las ventanas",
     "error = extras + huecos",
     "extra: round(PPI/med) = 0  → quitar el más bajo",
     "hueco: round(PPI/med) ≥ 2  → insertar máximo local",
     "si el error baja, se guarda esa serie",
     "si empeora 2 vueltas, se restaura la mejor"],
    C_LOOP, lw=2.0)

cx, cy = diamond(68, 42.6, 22, 8.4, "¿error ≈ 0?", C_LOOP)

box(48, 22.8, 40, 11.2, "Corrección",
    ["Quitar extras  /  rellenar huecos",
     "volver a medir el error",
     "tope 80 vueltas; para si no cambia"],
    C_LOOP)

box(48, 8.4, 40, 10.6, "6. Fiduciales y PRV",
    ["Diástoles = mínimo entre sístoles",
     "tacograma PPI → interpolación cúbica 4 Hz",
     "FFT  ·  Welch  ·  STFT"],
    C_OUT)

# Columna izquierda, en orden
arrow(22, 64.5, 22, 61.4, C_IN)
arrow(22, 51.5, 22, 48.3, C_CLEAN)
arrow(22, 32.2, 22, 30.8, C_WIN)
arrow(22, 25.6, 22, 23.1, C_WIN)

# Union -> pasada 2 (sube por el pasillo, no cruza cajas)
wire([(40, 18.3), (44.2, 18.3), (44.2, 62.5), (48, 62.5)], C_WIN)

# Lazo de error
arrow(68, 51.2, 68, 46.8, C_LOOP)
arrow(68, 38.4, 68, 34.1, C_LOOP)
ax.text(70.4, 36.0, "no", fontsize=8, color=C_LOOP)
wire([(48, 28.4), (45.6, 28.4), (45.6, 66.5), (48, 66.5)], C_LOOP)
ax.text(43.3, 47.5, "vuelve", fontsize=8, color=C_LOOP, rotation=90,
        va="center", ha="center")

# error ~ 0 -> fiduciales
wire([(79, 42.6), (91.5, 42.6), (91.5, 13.7), (88.1, 13.7)], C_OUT)
ax.text(84.4, 44.1, "sí / no mejora", fontsize=8, color=C_OUT)

# Panel ejemplo numerico
ax.add_patch(FancyBboxPatch(
    (93, 51.2), 28, 22.6,
    boxstyle="round,pad=0.35,rounding_size=0.6",
    facecolor=CARD, edgecolor="#8a8a8a", linewidth=1.1, zorder=2,
))
ax.add_patch(Rectangle((93, 73.58), 28, 0.22, color="#8a8a8a", zorder=3))
ax.text(107, 72.5, "Ejemplo 5–10 min", ha="center", va="center",
        fontsize=10, fontweight="medium", color="#555")
ax.text(107, 61.2,
        "20 ventanas\n"
        "PPI mediano 800 ms\n\n"
        "iter 0:  error = 4\n"
        "         2 extras, 2 huecos\n\n"
        "iter 1:  error = 0\n\n"
        "376 sístoles\n375 diástoles",
        ha="center", va="center", fontsize=8.6, color="#222", linespacing=1.38)

# Leyenda
ax.add_patch(Rectangle((4, 3.2), 2.2, 1.5, color=C_IN))
ax.text(7.0, 3.95, "Entrada", va="center", fontsize=8, color=C_IN)
ax.add_patch(Rectangle((22, 3.2), 2.2, 1.5, color=C_CLEAN))
ax.text(25.0, 3.95, "Limpieza global", va="center", fontsize=8, color=C_CLEAN)
ax.add_patch(Rectangle((48, 3.2), 2.2, 1.5, color=C_WIN))
ax.text(51.0, 3.95, "Adaptación local (20 s)", va="center", fontsize=8, color=C_WIN)
ax.add_patch(Rectangle((80, 3.2), 2.2, 1.5, color=C_LOOP))
ax.text(83.0, 3.95, "Retroalimentación", va="center", fontsize=8, color=C_LOOP)
ax.add_patch(Rectangle((108, 3.2), 2.2, 1.5, color=C_OUT))
ax.text(111.0, 3.95, "Salida PRV", va="center", fontsize=8, color=C_OUT)

ax.text(62.5, 1.2,
        "El umbral se adapta a cada ventana; el error no aplasta la PRV (solo extras y huecos).",
        ha="center", va="center", fontsize=8, color=C_NEUT)

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(OUT)
