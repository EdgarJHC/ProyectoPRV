"""Genera el PDF de investigacion sobre identificacion de picos de pulso."""

from __future__ import annotations

import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pyedflib
from matplotlib.patches import FancyBboxPatch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

EDF_PATH = (
    r"D:\Escuela IPN\Servicio\archivos\00000069-AN1PD2004512"
    r"\00000069-AN1PD2004512[001].edf"
)
PDF_PATH = (
    r"D:\Escuela IPN\Servicio\PDFs"
    r"\Identificacion_picos_de_pulso_Pleth_PRV.pdf"
)
PLOTH_IDX = 5
T0 = 180.0
WIN_S = 20.0
ZOOM_S = 8.0

NAVY = HexColor("#1B365D")
TEAL = HexColor("#1F6F8B")
RED = HexColor("#A93226")
INK = HexColor("#2C2C2C")
MUTED = HexColor("#5D6D7E")
LINE = HexColor("#D5D8DC")
BAND = HexColor("#EAF0F6")

C_SIG = "#1F4E79"
C_FILT = "#1F6F8B"
C_PEAK = "#A93226"
C_ON = "#1D8348"
C_DER = "#6C3483"
C_NAIVE = "#B7950B"
C_MA1 = "#1F6F8B"
C_MA2 = "#A04000"


def register_fonts() -> str:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts = os.path.join(windir, "Fonts")
    arial = os.path.join(fonts, "arial.ttf")
    if os.path.exists(arial):
        pdfmetrics.registerFont(TTFont("DocSans", arial))
        pdfmetrics.registerFont(
            TTFont("DocSans-Bold", os.path.join(fonts, "arialbd.ttf"))
        )
        italic = os.path.join(fonts, "ariali.ttf")
        bolditalic = os.path.join(fonts, "arialbi.ttf")
        if os.path.exists(italic):
            pdfmetrics.registerFont(TTFont("DocSans-Italic", italic))
        if os.path.exists(bolditalic):
            pdfmetrics.registerFont(TTFont("DocSans-BoldItalic", bolditalic))
        pdfmetrics.registerFontFamily(
            "DocSans",
            normal="DocSans",
            bold="DocSans-Bold",
            italic="DocSans-Italic" if os.path.exists(italic) else "DocSans",
            boldItalic="DocSans-BoldItalic"
            if os.path.exists(bolditalic)
            else "DocSans-Bold",
        )
        return "DocSans"
    return "Helvetica"


def styles(font: str) -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["cover_kicker"] = ParagraphStyle(
        "cover_kicker",
        parent=base["Normal"],
        fontName=f"{font}-Bold" if font == "DocSans" else "Helvetica-Bold",
        fontSize=10,
        textColor=TEAL,
        alignment=TA_CENTER,
        tracking=1.2,
        spaceAfter=8,
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title",
        parent=base["Title"],
        fontName=f"{font}-Bold" if font == "DocSans" else "Helvetica-Bold",
        fontSize=20,
        leading=26,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub",
        parent=base["Normal"],
        fontName=font,
        fontSize=12,
        leading=16,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    s["h1"] = ParagraphStyle(
        "h1",
        parent=base["Heading1"],
        fontName=f"{font}-Bold" if font == "DocSans" else "Helvetica-Bold",
        fontSize=14,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8,
        borderPadding=0,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        parent=base["Heading2"],
        fontName=f"{font}-Bold" if font == "DocSans" else "Helvetica-Bold",
        fontSize=12,
        textColor=TEAL,
        spaceBefore=10,
        spaceAfter=6,
    )
    s["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontName=font,
        fontSize=10,
        leading=14,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        parent=base["Normal"],
        fontName=f"{font}-Italic" if font == "DocSans" else "Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=MUTED,
        alignment=TA_LEFT,
        spaceBefore=2,
        spaceAfter=10,
    )
    s["bullet"] = ParagraphStyle(
        "bullet",
        parent=base["Normal"],
        fontName=font,
        fontSize=10,
        leading=13.5,
        textColor=INK,
        leftIndent=8,
        spaceAfter=3,
    )
    s["footer"] = ParagraphStyle(
        "footer",
        parent=base["Normal"],
        fontName=font,
        fontSize=8,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
    s["cell"] = ParagraphStyle(
        "cell",
        parent=base["Normal"],
        fontName=font,
        fontSize=8,
        leading=11,
        textColor=INK,
    )
    s["cell_h"] = ParagraphStyle(
        "cell_h",
        parent=base["Normal"],
        fontName=f"{font}-Bold" if font == "DocSans" else "Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=white,
    )
    s["ref"] = ParagraphStyle(
        "ref",
        parent=base["Normal"],
        fontName=font,
        fontSize=8.5,
        leading=12,
        textColor=INK,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=4,
    )
    return s


def load_pleth(path: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    reader = pyedflib.EdfReader(path)
    try:
        labels = [lab.strip() for lab in reader.getSignalLabels()]
        idx = PLOTH_IDX
        if "Pleth" in labels:
            idx = labels.index("Pleth")
        fs = float(reader.getSampleFrequency(idx))
        x = np.asarray(reader.readSignal(idx), dtype=float)
        label = labels[idx]
    finally:
        reader.close()
    t = np.arange(len(x)) / fs
    return t, x, fs, label


def bandpass(x: np.ndarray, fs: float, lo=0.5, hi=8.0, order=2) -> np.ndarray:
    ny = 0.5 * fs
    b, a = butter(order, [lo / ny, hi / ny], btype="band")
    return filtfilt(b, a, x)


def maybe_invert(x: np.ndarray) -> tuple[np.ndarray, bool]:
    d = np.diff(x)
    if np.max(d) < -np.min(d):
        return -x, True
    return x, False


def elgendi_peaks(
    x: np.ndarray,
    fs: float,
    peakwindow=0.111,
    beatwindow=0.667,
    beatoffset=0.02,
    mindelay=0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sqrd = np.maximum(x, 0.0) ** 2
    k_peak = max(3, int(round(peakwindow * fs)))
    k_beat = max(k_peak + 2, int(round(beatwindow * fs)))
    if k_peak % 2 == 0:
        k_peak += 1
    if k_beat % 2 == 0:
        k_beat += 1
    ma_peak = uniform_filter1d(sqrd, size=k_peak, mode="nearest")
    ma_beat = uniform_filter1d(sqrd, size=k_beat, mode="nearest")
    thr = ma_beat + beatoffset * np.mean(sqrd)
    waves = ma_peak > thr
    edges = np.diff(waves.astype(int))
    begs = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if waves[0]:
        begs = np.r_[0, begs]
    if waves[-1]:
        ends = np.r_[ends, len(x) - 1]
    n = min(len(begs), len(ends))
    begs, ends = begs[:n], ends[:n]
    min_len = int(round(peakwindow * fs))
    min_delay = int(round(mindelay * fs))
    peaks = []
    last = -10**9
    for a, b in zip(begs, ends):
        if b - a < min_len:
            continue
        seg = x[a:b]
        loc, props = find_peaks(seg, prominence=(None, None))
        if loc.size == 0:
            pk = a + int(np.argmax(seg))
        else:
            pk = a + loc[np.argmax(props["prominences"])]
        if pk - last > min_delay:
            peaks.append(pk)
            last = pk
    return np.asarray(peaks, dtype=int), ma_peak, ma_beat


def onset_and_slope(x: np.ndarray, peaks: np.ndarray, fs: float):
    dx = np.gradient(x, 1.0 / fs)
    onsets = []
    slopes = []
    search = int(0.35 * fs)
    for p in peaks:
        a = max(0, p - search)
        win = x[a:p]
        if win.size < 5:
            continue
        on = a + int(np.argmin(win))
        onsets.append(on)
        sl_win = dx[on:p]
        if sl_win.size:
            slopes.append(on + int(np.argmax(sl_win)))
    return np.asarray(onsets, dtype=int), np.asarray(slopes, dtype=int), dx


def setup_mpl():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def savefig(fig, folder: str, name: str) -> str:
    path = os.path.join(folder, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_raw(t, x, path_dir):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(t, x, color=C_SIG, lw=0.9)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Pleth (u.a.)")
    ax.set_title("Canal Pleth crudo — Alice NightOne, 100 Hz")
    ax.grid(True, ls=":", alpha=0.45)
    return savefig(fig, path_dir, "fig01_pleth_cruda.png")


def fig_naive_vs_elgendi(t, x, naive, good, path_dir):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(t, x, color=C_SIG, lw=0.9, label="Pleth filtrada")
    ax.plot(
        t[naive],
        x[naive],
        "o",
        ms=4.2,
        color=C_NAIVE,
        label="find_peaks sin restricciones",
        zorder=3,
    )
    ax.plot(
        t[good],
        x[good],
        "x",
        ms=7,
        mew=1.6,
        color=C_PEAK,
        label="Elgendi (pico sistólico)",
        zorder=4,
    )
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (u.a.)")
    ax.set_title("Máximos locales vs picos de pulso")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(True, ls=":", alpha=0.45)
    return savefig(fig, path_dir, "fig02_naive_vs_elgendi.png")


def fig_schematic(path_dir):
    t = np.linspace(0, 1.05, 800)
    sys_up = 1.0 / (1 + np.exp(-28 * (t - 0.18)))
    sys_peak = np.exp(-((t - 0.28) ** 2) / (2 * 0.018**2))
    notch = -0.22 * np.exp(-((t - 0.46) ** 2) / (2 * 0.012**2))
    dia = 0.28 * np.exp(-((t - 0.58) ** 2) / (2 * 0.035**2))
    decay = np.exp(-np.maximum(t - 0.30, 0) / 0.42)
    y = 0.12 + 0.55 * sys_up * decay + 0.55 * sys_peak + notch + dia
    y = y - y[0]
    dy = np.gradient(y, t)
    i_on = int(np.argmin(y[: np.argmax(y)]))
    i_sp = int(np.argmax(y))
    i_u = int(np.argmax(dy[:i_sp]))
    i_dn = int(0.46 / t[-1] * (len(t) - 1))
    i_dp = int(0.58 / t[-1] * (len(t) - 1))

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(t, y, color=C_SIG, lw=2.0)
    marks = [
        (i_on, "onset (on)", C_ON, (0.02, -0.12)),
        (i_u, "máx. pendiente (u)", C_DER, (-0.02, 0.10)),
        (i_sp, "pico sistólico (sp)", C_PEAK, (0.02, 0.08)),
        (i_dn, "muesca dicrota (dn)", "#7D3C98", (0.02, -0.14)),
        (i_dp, "pico diastólico (dp)", "#1A5276", (0.02, 0.10)),
    ]
    for i, lab, col, off in marks:
        ax.scatter(t[i], y[i], s=36, zorder=4, color=col)
        ax.annotate(
            lab,
            (t[i], y[i]),
            xytext=(t[i] + off[0], y[i] + off[1]),
            fontsize=8,
            color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
        )
    ax.set_xlabel("Tiempo de un ciclo (s, esquema)")
    ax.set_ylabel("Amplitud (u.a.)")
    ax.set_title("Puntos fiduciales de una onda de pulso (nomenclatura pyPPG)")
    ax.grid(True, ls=":", alpha=0.4)
    return savefig(fig, path_dir, "fig03_fiduciales_esquema.png")


def fig_real_pulse(t, x, onsets, slopes, peaks, path_dir):
    if len(peaks) < 3:
        return None
    p = peaks[len(peaks) // 2]
    a = max(0, p - int(0.45 * 100))
    b = min(len(x), p + int(0.70 * 100))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(t[a:b], x[a:b], color=C_SIG, lw=1.6)
    items = []
    on = onsets[np.argmin(np.abs(onsets - p))] if len(onsets) else None
    sl = slopes[np.argmin(np.abs(slopes - p))] if len(slopes) else None
    if on is not None and a <= on < b:
        items.append((on, "onset", C_ON))
    if sl is not None and a <= sl < b:
        items.append((sl, "máx. dP/dt", C_DER))
    items.append((p, "pico sistólico", C_PEAK))
    for i, lab, col in items:
        ax.scatter(t[i], x[i], s=42, color=col, zorder=4)
        ax.annotate(lab, (t[i], x[i]), textcoords="offset points", xytext=(6, 8),
                    fontsize=8, color=col)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Pleth filtrada (u.a.)")
    ax.set_title("Un pulso real del estudio 00000069, anotado")
    ax.grid(True, ls=":", alpha=0.45)
    return savefig(fig, path_dir, "fig04_pulso_real.png")


def fig_filter(t, raw, filt, path_dir):
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.plot(t, raw, color="#95A5A6", lw=0.8, label="Cruda")
    ax.plot(t, filt, color=C_FILT, lw=1.15, label="Pasabanda 0.5–8 Hz")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (u.a.)")
    ax.set_title("Preprocesamiento: conservar la banda cardiaca")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.45)
    return savefig(fig, path_dir, "fig05_filtro.png")


def fig_elgendi_ma(t, x, ma_peak, ma_beat, peaks, path_dir):
    sqrd = np.maximum(x, 0.0) ** 2
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
    axes[0].plot(t, sqrd, color="#7F8C8D", lw=0.7, label="x² recortada")
    axes[0].plot(t, ma_peak, color=C_MA1, lw=1.2, label="MA pico (111 ms)")
    axes[0].plot(t, ma_beat, color=C_MA2, lw=1.2, label="MA latido (667 ms)")
    axes[0].set_ylabel("Energía")
    axes[0].set_title("Algoritmo de Elgendi: dos medias móviles")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(True, ls=":", alpha=0.45)
    axes[1].plot(t, x, color=C_SIG, lw=0.9)
    axes[1].plot(t[peaks], x[peaks], "o", color=C_PEAK, ms=4.5, label="sp")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Pleth filtrada")
    axes[1].legend(frameon=False)
    axes[1].grid(True, ls=":", alpha=0.45)
    fig.tight_layout()
    return savefig(fig, path_dir, "fig06_elgendi.png")


def fig_derivative(t, x, dx, slopes, peaks, path_dir):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
    axes[0].plot(t, x, color=C_SIG, lw=0.95)
    axes[0].plot(t[peaks], x[peaks], "o", color=C_PEAK, ms=4, label="pico sistólico")
    if len(slopes):
        axes[0].plot(t[slopes], x[slopes], "s", color=C_DER, ms=3.5, label="máx. pendiente")
    axes[0].set_ylabel("Pleth")
    axes[0].set_title("Fiducial de pendiente (base de qppg / pyPPG u)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, ls=":", alpha=0.45)
    axes[1].plot(t, dx, color=C_DER, lw=0.9)
    if len(slopes):
        axes[1].plot(t[slopes], dx[slopes], "s", color=C_DER, ms=3.5)
    axes[1].axhline(0, color="#B0B0B0", lw=0.6)
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("dPleth/dt")
    axes[1].grid(True, ls=":", alpha=0.45)
    fig.tight_layout()
    return savefig(fig, path_dir, "fig07_derivada.png")


def fig_tacho_hist(peaks, fs, path_dir):
    ppi = np.diff(peaks) / fs * 1000.0
    t_ppi = peaks[1:] / fs
    bpm = 60000.0 / ppi
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(t_ppi, ppi, "-o", color=C_SIG, ms=3.5, lw=1.0)
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("PPI (ms)")
    axes[0].set_title("Tacograma (intervalos pulso a pulso)")
    axes[0].grid(True, ls=":", alpha=0.45)
    axes[1].hist(
        ppi,
        bins=min(12, max(6, len(ppi) // 3)),
        color="#1F6F8B",
        edgecolor="white",
    )
    axes[1].set_xlabel("PPI (ms)")
    axes[1].set_ylabel("Conteo")
    axes[1].set_title(f"Distribución de PPI  |  media {np.mean(bpm):.0f} lpm")
    fig.tight_layout()
    p1 = savefig(fig, path_dir, "fig08_tacograma.png")
    if len(ppi) > 3:
        fig2, ax = plt.subplots(figsize=(4.4, 4.0))
        ax.plot(ppi[:-1], ppi[1:], "o", color=C_PEAK, ms=5, alpha=0.85)
        lims = [np.min(ppi) - 20, np.max(ppi) + 20]
        ax.plot(lims, lims, "--", color="#95A5A6", lw=0.8)
        ax.set_xlabel("PPI[n] (ms)")
        ax.set_ylabel("PPI[n+1] (ms)")
        ax.set_title("Poincaré del tacograma")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.45)
        p2 = savefig(fig2, path_dir, "fig09_poincare.png")
    else:
        p2 = None
    return p1, p2, ppi, bpm


def fig_stack(path_dir):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Stack de software para picos de pulso y PRV", pad=8)
    boxes = [
        (0.3, 4.5, 9.4, 1.2, "#1B365D", "Aplicación PRV\ntacograma · SDNN · RMSSD · LF/HF", "white"),
        (0.3, 3.15, 4.5, 1.15, "#1F6F8B", "NeuroKit2\nElgendi / MSPTD + HRV", "white"),
        (5.05, 3.15, 4.65, 1.15, "#1F6F8B", "pyPPG\nfiduciales sp, on, dn, u, a–e", "white"),
        (0.3, 1.8, 3.0, 1.1, "#5D6D7E", "HeartPy\nopcional / prototipo", "white"),
        (3.5, 1.8, 6.2, 1.1, "#2C3E50", "SciPy  ·  NumPy  ·  pandas\nfind_peaks, filtros, PPI", "white"),
        (0.3, 0.35, 9.4, 1.2, "#A93226", "pyedflib  →  canal Pleth del EDF Alice NightOne", "white"),
    ]
    for x, y, w, h, color, text, tc in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            color=tc,
            fontsize=8.5,
        )
    return savefig(fig, path_dir, "fig10_stack.png")


def header_footer(canvas, doc, font):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 12 * mm, w, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont(font, 8)
    canvas.drawString(
        18 * mm,
        h - 7.5 * mm,
        "Identificación de picos de pulso en Pleth  ·  PRV / PPV",
    )
    canvas.drawRightString(w - 18 * mm, h - 7.5 * mm, "IPN · Servicio social")
    canvas.setFillColor(LINE)
    canvas.rect(0, 0, w, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont(font, 8)
    canvas.drawString(18 * mm, 5 * mm, "Documento de investigación previa a la implementación")
    canvas.drawRightString(w - 18 * mm, 5 * mm, f"{doc.page}")
    canvas.restoreState()


def cover_footer(canvas, doc, font):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, 28 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont(font, 8)
    canvas.drawCentredString(
        w / 2, 12 * mm, "Instituto Politécnico Nacional  ·  Servicio social  ·  2026"
    )
    canvas.restoreState()


def p(text, style):
    return Paragraph(text, style)


def img(path, width=16.4 * cm):
    im = Image(path)
    iw, ih = im.imageWidth, im.imageHeight
    im.drawWidth = width
    im.drawHeight = width * ih / iw
    return im


def make_table(headers, rows, s, col_widths):
    head = [Paragraph(h, s["cell_h"]) for h in headers]
    body = [[Paragraph(str(c), s["cell"]) for c in row] for row in rows]
    data = [head] + body
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 1), (-1, -1), white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, BAND]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE),
            ]
        )
    )
    return tbl


def build_figures(figdir: str):
    t_all, x_all, fs, label = load_pleth(EDF_PATH)
    i0 = int(T0 * fs)
    i1 = int((T0 + WIN_S) * fs)
    t = t_all[i0:i1] - t_all[i0]
    raw = x_all[i0:i1].copy()
    raw, inverted = maybe_invert(raw)
    filt = bandpass(raw, fs)
    naive, _ = find_peaks(filt)
    peaks, ma_peak, ma_beat = elgendi_peaks(filt, fs)
    onsets, slopes, dx = onset_and_slope(filt, peaks, fs)

    z1 = int(ZOOM_S * fs)
    paths = {}
    paths["raw"] = fig_raw(t[:z1], raw[:z1], figdir)
    paths["naive"] = fig_naive_vs_elgendi(
        t[:z1], filt[:z1], naive[naive < z1], peaks[peaks < z1], figdir
    )
    paths["schema"] = fig_schematic(figdir)
    paths["pulse"] = fig_real_pulse(
        t[:z1], filt[:z1],
        onsets[onsets < z1], slopes[slopes < z1], peaks[peaks < z1], figdir
    )
    paths["filt"] = fig_filter(t[:z1], raw[:z1] - np.mean(raw[:z1]), filt[:z1], figdir)
    paths["elg"] = fig_elgendi_ma(
        t[:z1], filt[:z1], ma_peak[:z1], ma_beat[:z1], peaks[peaks < z1], figdir
    )
    paths["der"] = fig_derivative(
        t[:z1], filt[:z1], dx[:z1], slopes[slopes < z1], peaks[peaks < z1], figdir
    )
    tacho, poincare, ppi, bpm = fig_tacho_hist(peaks, fs, figdir)
    paths["tacho"] = tacho
    paths["poincare"] = poincare
    paths["stack"] = fig_stack(figdir)

    stats = {
        "fs": fs,
        "label": label,
        "inverted": inverted,
        "n_naive": int(np.sum(naive < z1)),
        "n_elg_zoom": int(np.sum(peaks < z1)),
        "n_elg": int(len(peaks)),
        "mean_ppi": float(np.mean(ppi)) if len(ppi) else np.nan,
        "sd_ppi": float(np.std(ppi, ddof=1)) if len(ppi) > 1 else np.nan,
        "mean_bpm": float(np.mean(bpm)) if len(bpm) else np.nan,
        "n_total_samples": int(len(x_all)),
        "duration_min": float(len(x_all) / fs / 60.0),
    }
    return paths, stats


def build_pdf(paths, stats, font, s):
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="Identificación de picos de pulso en Pleth para PRV",
        author="Servicio social IPN",
        subject="Investigación de detección de picos de pulso, pyPPG y librerías",
    )
    story = []
    cw = 17.4 * cm

    story.append(Spacer(1, 28 * mm))
    story.append(p("SERVICIO SOCIAL · INSTITUTO POLITÉCNICO NACIONAL", s["cover_kicker"]))
    story.append(
        p(
            "Identificación de picos de pulso en señales de pletismografía (Pleth) para análisis PRV",
            s["cover_title"],
        )
    )
    story.append(
        p(
            "Investigación previa a la implementación · canal Pleth de Alice NightOne (Philips Respironics)",
            s["cover_sub"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        p(
            "Señal de trabajo: estudio 00000069-AN1PD2004512 · 100 Hz · pyedflib + SciPy",
            s["cover_sub"],
        )
    )
    story.append(p("Septiembre 2026", s["cover_sub"]))
    story.append(PageBreak())

    story.append(p("1. Objetivo de esta investigación", s["h1"]))
    story.append(
        p(
            "El análisis de variabilidad de la frecuencia de pulso (PRV) no parte de la "
            "frecuencia cardiaca ya promediada, sino de un tacograma: la serie de intervalos "
            "entre pulsos consecutivos (PPI). Para construirlo hay que identificar, en la "
            "onda de pletismografía, <b>un punto fiducial por ciclo</b>. Esta nota responde "
            "tres preguntas del asesor: qué es un pico de pulso, cómo se identifica de forma "
            "reproducible, y qué librerías de Python hacen falta. La intuición de usar "
            "<b>pyPPG</b> es correcta para morfología y fiduciales; no es, por sí sola, "
            "la librería de PRV.",
            s["body"],
        )
    )

    story.append(p("2. La señal con la que se trabaja", s["h1"]))
    story.append(
        p(
            f"Alice NightOne exporta el canal <b>{stats['label']}</b> a {stats['fs']:.0f} Hz "
            f"en EDF. En el segmento [001] hay {stats['n_total_samples']:,} muestras "
            f"({stats['duration_min']:.1f} min). El canal PulseRate (1 Hz) <b>no</b> sirve "
            "para PRV: ya es un promedio. El flujo nasal (Flow Patient) tampoco es un pulso "
            "arterial. El objeto de detección es la onda de volumen pulsátil de la Pleth. "
            f"Las figuras usan 20 s a partir de t = {T0:.0f} s, después del ChannelFail "
            "inicial de SpO2/PulseRate reportado en el .rml.",
            s["body"],
        )
    )
    story.append(img(paths["raw"]))
    story.append(
        p(
            "Figura 1. Pleth cruda del estudio de prueba. Alice NightOne, 100 Hz. "
            "Cada cresta corresponde, en principio, a un pulso; el ruido de línea base "
            "y la muesca dicrota impiden usar un máximo local sin reglas.",
            s["caption"],
        )
    )

    story.append(p("3. Qué se identifica: no todo máximo es un pulso", s["h1"]))
    story.append(
        p(
            "Una onda de pulso tiene varios puntos con nombre en la literatura y en pyPPG "
            "(Goda, Charlton y Behar, 2023):",
            s["body"],
        )
    )
    story.append(
        make_table(
            ["Fiducial", "Símbolo pyPPG", "Qué es", "¿Sirve para el tacograma?"],
            [
                [
                    "Onset / pie",
                    "on",
                    "Inicio del flanco sistólico (mínimo local previo)",
                    "Sí, si se usa siempre el mismo",
                ],
                [
                    "Máxima pendiente",
                    "u",
                    "Máximo de la 1.ª derivada (VPG)",
                    "Sí; suele dar mejor timing",
                ],
                [
                    "Pico sistólico",
                    "sp",
                    "Máximo de la onda de pulso",
                    "Sí; el más intuitivo",
                ],
                [
                    "Muesca dicrota",
                    "dn",
                    "Cierre aórtico, no es un latido nuevo",
                    "No: marcarlo duplica PPI",
                ],
                [
                    "Pico diastólico",
                    "dp",
                    "Máximo secundario tras la muesca",
                    "No para PRV",
                ],
            ],
            s,
            [3.2 * cm, 2.6 * cm, 6.2 * cm, 5.4 * cm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(img(paths["schema"]))
    story.append(
        p(
            "Figura 2. Esquema de una onda PPG con la nomenclatura de pyPPG. El error "
            "clásico es tomar dn o dp como un segundo pico sistólico.",
            s["caption"],
        )
    )
    story.append(
        p(
            "Para PRV basta <b>un</b> fiducial por ciclo, siempre el mismo. El pico "
            "sistólico (sp) es el más fácil de explicar. El punto de máxima pendiente (u) "
            "o el punto medio del flanco suelen dar intervalos más estables a 100 Hz "
            "(Peralta et al., 2019). Mezclar sp en un pulso y dn en el siguiente invalida "
            "RMSSD.",
            s["body"],
        )
    )

    story.append(p("4. Pipeline de detección", s["h1"]))
    story.append(
        p(
            "Identificar picos no es un único <font face='Courier'>find_peaks</font>. "
            "La cadena que usan pyPPG, NeuroKit2 y Elgendi es la misma en esencia:",
            s["body"],
        )
    )
    story.append(
        ListFlowable(
            [
                ListItem(p("<b>Leer</b> el EDF y extraer solo Pleth (pyedflib).", s["bullet"])),
                ListItem(p("<b>Filtrar</b> pasabanda ~0.5–8 Hz para dejar la banda cardiaca.", s["bullet"])),
                ListItem(p("<b>Detectar</b> un candidato por ciclo (Elgendi, Aboy/pyPPG, MSPTD).", s["bullet"])),
                ListItem(p("<b>Elegir fiducial</b> dentro del ciclo (sp, u u onset).", s["bullet"])),
                ListItem(p("<b>Refractario</b> ~0.3 s (~200 lpm) para no marcar la muesca dicrota.", s["bullet"])),
                ListItem(p("<b>Calidad</b>: tirar segmentos planos, saturados o con PPI imposibles.", s["bullet"])),
                ListItem(p("<b>Tacograma</b>: PPI[n] = t[n+1] − t[n], en milisegundos.", s["bullet"])),
            ],
            bulletType="1",
            start="1",
            leftIndent=12,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(img(paths["filt"]))
    story.append(
        p(
            "Figura 3. Mismo segmento, crudo vs filtrado (Butterworth 0.5–8 Hz, "
            "filtfilt). El filtro no detecta picos: solo quita deriva y temblor de alta "
            "frecuencia para que el detector no persiga ruido.",
            s["caption"],
        )
    )

    story.append(p("5. Métodos para los picos máximos", s["h1"]))
    story.append(p("5.1 Máximos locales (SciPy)", s["h2"]))
    story.append(
        p(
            "<font face='Courier'>scipy.signal.find_peaks</font> marca todo máximo local. "
            f"En 8 s de Pleth filtrada encontró <b>{stats['n_naive']} máximos</b> frente a "
            f"<b>{stats['n_elg_zoom']} picos sistólicos</b> de Elgendi. La diferencia son "
            "muescas, rizados y hombros diastólicos. Sin <i>distance</i> (p. ej. 0.4·fs) y "
            "<i>prominence</i>, no es un detector de pulso.",
            s["body"],
        )
    )
    story.append(img(paths["naive"]))
    story.append(
        p(
            "Figura 4. Círculos amarillos: todos los máximos locales. Cruces rojas: "
            "picos sistólicos con Elgendi. El asesor pide lo segundo, no lo primero.",
            s["caption"],
        )
    )

    story.append(p("5.2 Elgendi (2013): el método didáctico", s["h2"]))
    story.append(
        p(
            "Elgendi filtra, recorta valores negativos, eleva al cuadrado y compara dos "
            "medias móviles: 111 ms (énfasis del pico) y 667 ms (escala del latido). Donde "
            "la media corta supera a la larga más un umbral, busca el máximo más prominente "
            "y exige 0.3 s entre picos. Es el default de NeuroKit2 "
            "(<font face='Courier'>nk.ppg_peaks(..., method='elgendi')</font>). "
            "En el segmento de 20 s se obtuvieron "
            f"{stats['n_elg']} pulsos, PPI medio {stats['mean_ppi']:.0f} ms "
            f"({stats['mean_bpm']:.0f} lpm).",
            s["body"],
        )
    )
    story.append(img(paths["elg"]))
    story.append(
        p(
            "Figura 5. Arriba: señal al cuadrado y las dos medias móviles de Elgendi. "
            "Abajo: picos sistólicos sobre la Pleth filtrada. Implementación propia con "
            "SciPy, equivalente a la descrita en Elgendi et al., 2013.",
            s["caption"],
        )
    )

    story.append(p("5.3 Fiducial de pendiente (qppg / punto u de pyPPG)", s["h2"]))
    story.append(
        p(
            "La primera derivada (VPG) localiza el flanco, no la cresta. pyPPG llama "
            "<b>u</b> al máximo de PPG′ entre onset y onset. Detectores tipo qppg "
            "(Zong; Vest et al., 2018) y el M1D usan esa idea. A 100 Hz este punto, "
            "interpolado, suele ser más estable para PRV que el máximo de la onda.",
            s["body"],
        )
    )
    story.append(img(paths["der"]))
    story.append(
        p(
            "Figura 6. Pleth y su derivada. Cuadros: máximo de dP/dt. Círculos: pico "
            "sistólico. Ambos marcan el mismo latido; no se deben mezclar en un mismo tacograma.",
            s["caption"],
        )
    )
    if paths["pulse"]:
        story.append(img(paths["pulse"]))
        story.append(
            p(
                "Figura 7. Un pulso real del estudio 00000069 con onset, máxima pendiente "
                "y pico sistólico. Esta es la unidad que hay que detectar miles de veces "
                "a lo largo de la noche.",
                s["caption"],
            )
        )

    story.append(p("5.4 pyPPG (Aboy mejorado) y MSPTD", s["h2"]))
    story.append(
        p(
            "<b>pyPPG</b> no usa Elgendi. Detecta el pico sistólico con una versión "
            "mejorada de Aboy et al. (2005): filtra alrededor de una FC estimada, "
            "deriva y umbraliza; el pico en la PPG débilmente filtrada sigue a ese "
            "evento. Para esa etapa <b>remuestrea a 75 Hz</b>. Luego, sobre cada "
            "onda, marca on, off, dn, dp y los puntos de PPG′, PPG′′ y PPG′′′ "
            "(u, v, w, a–f, p1, p2). Charlton et al. (2022) compararon 15 detectores "
            "open-source a 100 Hz: los mejores fueron <b>MSPTD</b> (Bishop y Ercole, 2018) "
            "y <b>qppg</b>. NeuroKit2 expone MSPTD como "
            "<font face='Courier'>method='bishop'</font> o "
            "<font face='Courier'>'charlton'</font>.",
            s["body"],
        )
    )

    story.append(p("6. Del pico al tacograma", s["h1"]))
    story.append(
        p(
            "Con los tiempos de sp (o de u) se arma PPI en milisegundos. Eso es el "
            "tacograma. A 100 Hz cada muestra son 10 ms: RMSSD es el índice que más "
            "se estropea si no se interpola el fiducial (parábola o punto medio del "
            "flanco). PulseRate a 1 Hz no puede producir esta figura.",
            s["body"],
        )
    )
    story.append(img(paths["tacho"]))
    story.append(
        p(
            f"Figura 8. Tacograma e histograma de PPI en 20 s de Pleth. Media "
            f"{stats['mean_ppi']:.0f} ms, DE {stats['sd_ppi']:.0f} ms. Ventana corta: "
            "solo ilustra el método, no un PRV clínico.",
            s["caption"],
        )
    )
    if paths["poincare"]:
        story.append(img(paths["poincare"], width=9.5 * cm))
        story.append(
            p(
                "Figura 9. Nube de Poincaré PPI[n] vs PPI[n+1] del mismo segmento. "
                "Es el gráfico que se espera al final de la cadena pico → intervalo. "
                "Un detector que marca muescas dicrotas abre nubes imposibles.",
                s["caption"],
            )
        )

    story.append(p("7. Librerías: ¿pyPPG basta?", s["h1"]))
    story.append(
        p(
            "Sí conviene pyPPG para <b>identificar picos y el resto de fiduciales</b>. "
            "No basta como único paquete del proyecto PRV. pyPPG está pensado para "
            "morfología de la onda (74 biomarcadores, SQI de Li 2015) y lee EDF "
            "buscando por defecto el canal llamado <font face='Courier'>Pleth</font>, "
            "que es exactamente la etiqueta de Alice NightOne. El PRV (SDNN, RMSSD, "
            "LF/HF, Poincaré) lo resuelven mejor NeuroKit2 o cálculo propio sobre "
            "los PPI con NumPy/SciPy.",
            s["body"],
        )
    )
    story.append(img(paths["stack"]))
    story.append(
        p(
            "Figura 10. Stack recomendado. La base ya está en el venv (pyedflib, NumPy, "
            "SciPy, Matplotlib, pandas). Encima: pyPPG para fiduciales y NeuroKit2 para "
            "picos + métricas de variabilidad.",
            s["caption"],
        )
    )

    story.append(
        make_table(
            ["Librería", "Rol en picos de pulso", "¿Instalarla?"],
            [
                [
                    "pyedflib",
                    "Lee el EDF y el canal Pleth. Ya está en el venv.",
                    "Sí (ya instalada)",
                ],
                [
                    "NumPy / SciPy",
                    "Filtros, derivadas, find_peaks, PPI. Base de todo detector.",
                    "Sí (ya instaladas)",
                ],
                [
                    "Matplotlib",
                    "Inspección visual de picos, tacograma y Poincaré.",
                    "Sí (ya instalada)",
                ],
                [
                    "pandas",
                    "Tabla de tiempos, PPI y calidad por pulso.",
                    "Sí (ya instalada)",
                ],
                [
                    "pyPPG",
                    "Detector Aboy mejorado + fiduciales sp/on/dn/u/a–e + 74 biomarcadores + SQI. Acepta EDF.",
                    "Sí, para morfología. Cuidado: declara Python 3.10 y pines viejos de SciPy.",
                ],
                [
                    "NeuroKit2",
                    "ppg_clean, ppg_peaks (Elgendi, Bishop/MSPTD, Charlton) y funciones HRV/PRV sobre esos picos.",
                    "Sí, para el tacograma y el PRV.",
                ],
                [
                    "HeartPy",
                    "Detector por media móvil adaptativa. Ligero, menos fiduciales.",
                    "Opcional, como control.",
                ],
                [
                    "wfdb / mne",
                    "Dependencias de pyPPG para I/O fisiológico. No se usan directo si ya hay pyedflib.",
                    "Solo si se instala pyPPG",
                ],
            ],
            s,
            [3.4 * cm, 8.6 * cm, 5.4 * cm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(p("Detalle de pyPPG respecto a este proyecto", s["h2"]))
    story.append(
        p(
            "Autores: Goda, Charlton y Behar, <i>Physiological Measurement</i>, 2023. "
            "Instalación: <font face='Courier'>pip install pyPPG</font>. Flujo típico: "
            "<font face='Courier'>load_data</font> → "
            "<font face='Courier'>Preprocess.get_signals</font> (PPG, VPG, APG, JPG) → "
            "<font face='Courier'>FpCollection.get_fiducials</font> → "
            "<font face='Courier'>plot_fiducials</font>. El pico sistólico es "
            "<font face='Courier'>fp.sp</font>. Documentación: pyppg.readthedocs.io. "
            "Limitación práctica: el paquete fija Python 3.10, SciPy 1.9 y NumPy 1.23; "
            "el venv actual es Python 3.11 con SciPy 1.17. Conviene un entorno aparte "
            "o probar si la rueda reciente admite 3.11, sin pisar el venv de lectura EDF.",
            s["body"],
        )
    )
    story.append(p("Detalle de NeuroKit2", s["h2"]))
    story.append(
        p(
            "Para el objetivo del asesor (picos) y el del proyecto (PRV) es la API más "
            "directa: <font face='Courier'>nk.ppg_clean(pleth, sampling_rate=100)</font> "
            "y <font face='Courier'>nk.ppg_peaks(..., method='elgendi')</font> o "
            "<font face='Courier'>method='bishop'</font> (MSPTD, el mejor del benchmark "
            "de 2022). Los índices de picos alimentan "
            "<font face='Courier'>nk.hrv</font>. No extrae la muesca dicrota ni los 74 "
            "biomarcadores de pyPPG.",
            s["body"],
        )
    )

    story.append(p("8. Recomendación para el equipo", s["h1"]))
    story.append(
        p(
            "<b>Para identificar picos máximos / fiduciales:</b> pyPPG es la librería "
            "adecuada (intuición correcta), complementada con SciPy para control y "
            "gráficas. <b>Para el PRV:</b> NeuroKit2 sobre esos picos, o PPI calculado "
            "con NumPy. <b>No hace falta</b> HeartPy si ya están las dos anteriores, "
            "salvo como segundo detector de contraste. <b>No usar</b> PulseRate a 1 Hz "
            "ni máximos locales sin prominencia y refractario. A 100 Hz, interpolar el "
            "fiducial antes de RMSSD.",
            s["body"],
        )
    )
    story.append(
        p(
            "Respuesta corta para el asesor: los picos de pulso se marcan en la Pleth "
            "filtrada, un fiducial por ciclo (sp o u), con detector tipo Elgendi, Aboy "
            "(pyPPG) o MSPTD (NeuroKit2), nunca con un máximo local desnudo. Las "
            "librerías necesarias son pyedflib, NumPy, SciPy, Matplotlib, pyPPG y "
            "NeuroKit2.",
            s["body"],
        )
    )

    story.append(p("Referencias", s["h1"]))
    refs = [
        "Aboy M. et al. (2005). An automatic beat detection algorithm for pressure signals. IEEE Trans Biomed Eng.",
        "Allen J. (2007). Photoplethysmography and its application in clinical physiological measurement. Physiol Meas.",
        "Bishop S. M., Ercole A. (2018). Multi-scale peak and trough detection (MSPTD). Acta Neurochir Suppl.",
        "Charlton P. H. et al. (2022). Detecting beats in the photoplethysmogram: benchmarking open-source algorithms. Physiol Meas, 43, 085007.",
        "Elgendi M. et al. (2013). Systolic peak detection in acceleration photoplethysmograms. PLoS ONE, 8(10), e76585.",
        "Goda M. A., Charlton P. H., Behar J. A. (2023). pyPPG: A Python toolbox for comprehensive photoplethysmography signal analysis. Physiol Meas. DOI 10.1088/1361-6579/ad33a2.",
        "Li Q., Clifford G. D. (2012/2015). Signal quality indices para PPG (base del SQI de pyPPG).",
        "Peralta E. et al. (2019). Fiducial point del flanco medio en PPG para PRV a frecuencias de muestreo reducidas.",
        "Vest A. N. et al. (2018). An open source benchmarked toolbox for cardiovascular waveform and interval analysis (incluye qppg). Physiol Meas.",
        "van Gent P. et al. (2019). HeartPy: A novel heart rate algorithm for the analysis of noisy signals. Transp Res Part F.",
        "Makowski D. et al. NeuroKit2: documentación de ppg_peaks / ppg_findpeaks (métodos elgendi, bishop, charlton).",
        "pyPPG documentation. https://pyppg.readthedocs.io/en/latest/",
    ]
    for r in refs:
        story.append(p(r, s["ref"]))

    def first_page(c, d):
        cover_footer(c, d, font)

    def later(c, d):
        header_footer(c, d, font)

    doc.build(story, onFirstPage=first_page, onLaterPages=later)
    return PDF_PATH


def main():
    setup_mpl()
    font = register_fonts()
    s = styles(font)
    figdir = tempfile.mkdtemp(prefix="picos_pulso_")
    try:
        paths, stats = build_figures(figdir)
        out = build_pdf(paths, stats, font, s)
        print("PDF:", out)
        print("stats:", stats)
    finally:
        shutil.rmtree(figdir, ignore_errors=True)


if __name__ == "__main__":
    main()
