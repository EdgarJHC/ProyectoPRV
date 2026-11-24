import os
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
import neurokit2 as nk
from scipy.signal import find_peaks, periodogram, welch, spectrogram
from scipy.interpolate import PchipInterpolator, interp1d
import math


def _elgendi_sistole_ventana(seg, fs):
    """Elgendi en un recorte; el umbral mean(sqrd) es local a esa ventana."""
    try:
        info = nk.ppg_findpeaks(seg, sampling_rate=fs, method="elgendi")
        return np.asarray(info["PPG_Peaks"], dtype=int)
    except (IndexError, ValueError, KeyError):
        return np.array([], dtype=int)


def detectar_sistole_por_ventanas(ppg, fs, win_s=20.0, overlap_s=5.0):
    """Primera pasada: Elgendi cada 20 s con solape para no perder el borde."""
    n = len(ppg)
    win = int(win_s * fs)
    step = int((win_s - overlap_s) * fs)
    if step < 1:
        step = max(win, 1)
    picos = []
    n_ventanas = 0
    start = 0
    while start < n:
        end = min(start + win, n)
        if end - start < int(2 * fs):
            break
        loc = _elgendi_sistole_ventana(ppg[start:end], fs)
        if loc.size:
            picos.append(loc + start)
        n_ventanas += 1
        if end >= n:
            break
        start += step
    if not picos:
        return np.array([], dtype=int), n_ventanas
    return np.concatenate(picos), n_ventanas


def fusionar_picos_cercanos(picos, ppg, fs, min_delay_s=0.30):
    """Une duplicados del solape: si dos picos están a < 0.3 s, se queda el más alto."""
    picos = np.unique(np.sort(np.asarray(picos, dtype=int)))
    if picos.size == 0:
        return picos
    min_d = int(min_delay_s * fs)
    kept = [int(picos[0])]
    for p in picos[1:]:
        p = int(p)
        if p - kept[-1] > min_d:
            kept.append(p)
        elif ppg[p] >= ppg[kept[-1]]:
            kept[-1] = p
    return np.asarray(kept, dtype=int)


def error_deteccion_ppi(picos, fs):
    """
    Error de deteccion: latidos de mas (extras) o de menos (huecos)
    respecto al PPI mediano. 0 = cada intervalo es ~1 latido (la PRV real se conserva).
    """
    if picos.size < 2:
        return 0.0, np.nan, 0, 0
    ppi = np.diff(picos) / float(fs)
    med = float(np.median(ppi))
    if med <= 0:
        return 0.0, med, 0, 0
    n_lat = np.round(ppi / med)
    extras = int(np.sum(np.maximum(0.0, 1.0 - n_lat)))
    huecos = int(np.sum(np.maximum(0.0, n_lat - 1.0)))
    return float(extras + huecos), med, extras, huecos


def _mejor_pico_en(ppg, lo, hi):
    if hi - lo < 3:
        return None
    mid = ppg[lo:hi]
    loc, _ = find_peaks(mid)
    if loc.size == 0:
        return None
    return lo + int(loc[int(np.argmax(mid[loc]))])


def corregir_picos_segunda_pasada(picos, ppg, fs, max_iter=80):
    """
    Retroalimentacion hasta que el error (extras + huecos) deje de bajar o llegue a 0.
    Conserva la mejor serie si una vuelta empeora.
    """
    picos = fusionar_picos_cercanos(picos, ppg, fs)
    min_d = int(0.30 * fs)
    err, med, extras, huecos = error_deteccion_ppi(picos, fs)
    print(
        f"  sistole iter 0: error={err:.0f} "
        f"({extras} extras, {huecos} huecos), PPI med={med * 1000:.1f} ms"
    )
    mejor = picos.copy()
    mejor_err = err
    estancado = 0

    for it in range(1, max_iter + 1):
        if picos.size < 4 or err < 0.5:
            break
        ppi = np.diff(picos) / fs
        med = float(np.median(ppi))
        if med <= 0:
            break
        cambiado = False

        n_lat = np.round(ppi / med)
        keep = np.ones(picos.size, dtype=bool)
        for i in range(len(ppi)):
            if not keep[i]:
                continue
            if n_lat[i] < 1 or ppi[i] < 0.35:
                if ppg[picos[i]] < ppg[picos[i + 1]]:
                    keep[i] = False
                else:
                    keep[i + 1] = False
                cambiado = True
        picos = picos[keep]
        if picos.size < 4:
            break

        ppi = np.diff(picos) / fs
        med = float(np.median(ppi))
        n_lat = np.round(ppi / med)
        nuevos = []
        for i in range(len(ppi)):
            n_miss = int(n_lat[i]) - 1
            if n_miss < 1:
                continue
            a, b = int(picos[i]), int(picos[i + 1])
            span = b - a
            for k in range(1, n_miss + 1):
                centro = a + int(round(span * k / (n_miss + 1)))
                lo = max(a + min_d, centro - min_d)
                hi = min(b - min_d, centro + min_d)
                cand = _mejor_pico_en(ppg, lo, hi)
                if cand is None:
                    continue
                if cand - a > min_d and b - cand > min_d:
                    nuevos.append(cand)
                    cambiado = True
        if nuevos:
            picos = fusionar_picos_cercanos(
                np.concatenate([picos, np.asarray(nuevos, dtype=int)]), ppg, fs
            )

        err, med, extras, huecos = error_deteccion_ppi(picos, fs)
        print(
            f"  sistole iter {it}: error={err:.0f} "
            f"({extras} extras, {huecos} huecos), PPI med={med * 1000:.1f} ms"
        )

        if err + 1e-9 < mejor_err:
            mejor = picos.copy()
            mejor_err = err
            estancado = 0
        else:
            estancado += 1

        if not cambiado or err < 0.5 or estancado >= 2:
            break

    picos = mejor
    err, med, extras, huecos = error_deteccion_ppi(picos, fs)
    print(
        f"  sistole final: error={err:.0f} "
        f"({extras} extras, {huecos} huecos), PPI med={med * 1000:.1f} ms"
    )
    return picos, err


def _valle_en(ppg, lo, hi, fs):
    """Ultimo minimo local en [lo, hi): pie diastolico, no la muesca dicrota."""
    if hi - lo < 3:
        return None
    seg = ppg[lo:hi]
    dist = max(3, int(0.05 * fs))
    mins, _ = find_peaks(-seg, distance=dist)
    if mins.size:
        return lo + int(mins[-1])
    return lo + int(np.argmin(seg))


def _reubicar_distole(ppg, sistole, k, fs, frac_inicio=0.50, margen_fin=0.08):
    if k < 0 or k + 1 >= len(sistole):
        return None
    a, b = int(sistole[k]), int(sistole[k + 1])
    n = b - a
    if n < 4:
        return a + 1 if n > 1 else None
    lo = a + int(frac_inicio * n)
    hi = b - max(2, int(margen_fin * n))
    if hi - lo < 3:
        lo, hi = a + n // 2, b - 1
    cand = _valle_en(ppg, lo, hi, fs)
    if cand is None:
        return a + int(np.argmin(ppg[a:b]))
    return int(np.clip(cand, a + 1, b - 1))


def distoles_entre_sistoles(ppg, sistole, fs, frac_inicio=0.50):
    """Un valle por ciclo, en la segunda mitad (evita argmin global = dicrota)."""
    distole = []
    for k in range(len(sistole) - 1):
        cand = _reubicar_distole(ppg, sistole, k, fs, frac_inicio=frac_inicio)
        if cand is not None:
            distole.append(cand)
    return np.asarray(distole, dtype=int)


def corregir_distoles_segunda_pasada(distole, sistole, ppg, fs, max_iter=80):
    """
    Mismo lazo de error que la sistole, pero reubica valles (no borra ni inserta).
    Si el PPI sistolico esta sano y el diastolico no, el valle estaba en la dicrota.
    """
    distole = np.asarray(distole, dtype=int)
    if distole.size < 4:
        return distole, 0.0
    err, med, extras, huecos = error_deteccion_ppi(distole, fs)
    print(
        f"  distole iter 0: error={err:.0f} "
        f"({extras} extras, {huecos} huecos), PPI med={med * 1000:.1f} ms"
    )
    mejor = distole.copy()
    mejor_err = err
    estancado = 0
    fracs = (0.50, 0.55, 0.60, 0.65, 0.70)

    for it in range(1, max_iter + 1):
        if distole.size < 4 or err < 0.5:
            break
        ppi = np.diff(distole) / fs
        med = float(np.median(ppi))
        if med <= 0:
            break
        n_lat = np.round(ppi / med)
        mover = set()
        for i in range(len(ppi) - 1):
            if n_lat[i] < 1 and n_lat[i + 1] > 1:
                mover.add(i + 1)
            elif n_lat[i] > 1 and n_lat[i + 1] < 1:
                mover.add(i)
        for i in range(len(ppi)):
            if n_lat[i] != 1:
                mover.add(i)
                mover.add(i + 1)
        if not mover:
            break

        frac = fracs[min(it - 1, len(fracs) - 1)]
        cambiado = False
        nuevo = distole.copy()
        for k in mover:
            cand = _reubicar_distole(ppg, sistole, k, fs, frac_inicio=frac)
            if cand is None or k >= len(nuevo):
                continue
            if cand != nuevo[k]:
                nuevo[k] = cand
                cambiado = True
        distole = nuevo

        err, med, extras, huecos = error_deteccion_ppi(distole, fs)
        print(
            f"  distole iter {it}: error={err:.0f} "
            f"({extras} extras, {huecos} huecos), PPI med={med * 1000:.1f} ms"
            f", frac={frac:.2f}"
        )

        if err + 1e-9 < mejor_err:
            mejor = distole.copy()
            mejor_err = err
            estancado = 0
        else:
            estancado += 1

        if not cambiado or err < 0.5 or estancado >= 3:
            break

    distole = mejor
    err, med, extras, huecos = error_deteccion_ppi(distole, fs)
    print(
        f"  distole final: error={err:.0f} "
        f"({extras} extras, {huecos} huecos), PPI med={med * 1000:.1f} ms"
    )
    return distole, err


def unir_extremos(t_ext, y_ext, t_fino):
    """Curva que pasa por cada extremo. Une máximos con máximos, o valles con valles."""
    t_ext = np.asarray(t_ext, dtype=float)
    y_ext = np.asarray(y_ext, dtype=float)
    t_fino = np.asarray(t_fino, dtype=float)
    curva = np.full(t_fino.shape, np.nan)
    if t_ext.size < 2:
        return curva
    orden = np.argsort(t_ext)
    t_ext, y_ext = t_ext[orden], y_ext[orden]
    t_ext, unicos = np.unique(t_ext, return_index=True)
    y_ext = y_ext[unicos]
    if t_ext.size < 2:
        return curva
    if t_ext.size == 2:
        curva = np.interp(t_fino, t_ext, y_ext).astype(float)
    else:
        curva = np.asarray(PchipInterpolator(t_ext, y_ext)(t_fino), dtype=float)
    curva[t_fino < t_ext[0]] = y_ext[0]
    curva[t_fino > t_ext[-1]] = y_ext[-1]
    return curva


def primer_paso_huang(emd, t, senal):
    """Máximos, valles, envolventes que los unen y su media."""
    t = np.asarray(t, dtype=float)
    senal = np.asarray(senal, dtype=float)
    max_pos, max_val, min_pos, min_val, _ = emd.find_extrema(t, senal)
    env_sup = unir_extremos(max_pos, max_val, t)
    env_inf = unir_extremos(min_pos, min_val, t)
    media = (env_sup + env_inf) / 2.0
    return max_pos, max_val, min_pos, min_val, env_sup, env_inf, media


def dibujar_envolventes(ax, t, senal, t_env, max_pos, max_val, min_pos, min_val, env_sup, env_inf, media, titulo, nombre_senal):
    """Tacograma con envolvente superior, inferior y media."""
    ax.plot(t, senal, color="#1f4e79", linewidth=1.15, label=nombre_senal, zorder=2)
    ax.plot(t_env, env_sup, color="#2e7d32", linewidth=2.0, label="Envolvente superior", zorder=3)
    ax.plot(t_env, env_inf, color="#6a1b9a", linewidth=2.0, label="Envolvente inferior", zorder=3)
    ax.plot(t_env, media, color="#e65100", linewidth=1.8, label="Media", zorder=4)
    ax.plot(max_pos, max_val, linestyle="none", marker="^", color="#2e7d32", markersize=5, markeredgewidth=0, label=f"Máximos n={len(max_pos)}", zorder=5)
    ax.plot(min_pos, min_val, linestyle="none", marker="v", color="#6a1b9a", markersize=5, markeredgewidth=0, label=f"Valles n={len(min_pos)}", zorder=5)
    serie = np.concatenate([
        np.asarray(senal, dtype=float).ravel(),
        np.asarray(env_sup, dtype=float).ravel(),
        np.asarray(env_inf, dtype=float).ravel(),
    ])
    serie = serie[np.isfinite(serie)]
    if serie.size:
        bajo, alto = float(np.min(serie)), float(np.max(serie))
        margen = 0.06 * (alto - bajo if alto > bajo else 1.0)
        ax.set_ylim(bajo - margen, alto + margen)
    ax.set_title(titulo, loc="left")
    ax.set_ylabel("Intervalo PP (ms)")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.margins(x=0.01)


# Ruta del archivo
file_name = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "archivos",
    "00000069-AN1PD2004512",
    "00000069-AN1PD2004512[001].edf",
))















# Metodos de analisis de señal



print("¿Qué metodo deseas realizar")
print("a) Método Fourier Lineal")
print("b) Método EMD No Lineal")
opcion = input("Opcion: ").strip().lower()

if not (
    opcion.startswith("a") or opcion == "1"
    or opcion.startswith("b") or opcion == "2"
):
    print("Opcion no valida. Elige a o b.")
elif not os.path.exists(file_name):
    print(f"Error: No se encontró el archivo en:\n{file_name}")
else:
    archivo = pyedflib.EdfReader(file_name)
    canal_idx = 5
    sig_values = archivo.readSignal(canal_idx)
    fs = int(archivo.getSampleFrequency(canal_idx))
    archivo.close()




#Parametros de la señal


    # 1. Ventana corta Task Force: minutos 5 a 10 del EDF [001]
    t_inicio_min = 5
    t_fin_min = 10
    inicio = int(t_inicio_min * 60 * fs)
    fin = int(t_fin_min * 60 * fs)

    sig_ventana = sig_values[inicio:fin]
    t_minutos = np.arange(len(sig_ventana)) / (fs * 60)
    t_segundos = np.arange(len(sig_ventana)) / fs




#Detección de sístole y diástole globales


    ppg_limpia = nk.ppg_clean(sig_ventana, sampling_rate=fs, method="elgendi")
    info_picos = nk.ppg_findpeaks(ppg_limpia, sampling_rate=fs, method="elgendi")
    sistole = np.asarray(info_picos["PPG_Peaks"], dtype=int)

    distole = []
    for i in range(len(sistole) - 1):
        a, b = int(sistole[i]), int(sistole[i + 1])
        if b - a > 1:
            distole.append(a + int(np.argmin(ppg_limpia[a:b])))
    distole = np.asarray(distole, dtype=int)
    n_sis = len(sistole)
    n_dis = len(distole)
    print(
        f"Elgendi: {n_sis} sistoles, {n_dis} distoles "
        f"en min {t_inicio_min:.0f}-{t_fin_min:.0f}"
    )

    intervalos_ms = (np.diff(sistole) / fs) * 1000
    t_taco_minutos = t_minutos[sistole][1:]
    t_taco_segundos = t_segundos[sistole][1:]

    intervalos_ms_distole = (np.diff(distole) / fs) * 1000
    t_taco_distole_minutos = t_minutos[distole][1:]
    t_taco_distole_segundos = t_segundos[distole][1:]

    fs_interp = 4.0
    t_reg_sis = np.arange(t_taco_segundos[0], t_taco_segundos[-1], 1.0 / fs_interp)
    taco_sis_interp = interp1d(t_taco_segundos, intervalos_ms, kind='cubic')(t_reg_sis)
    t_reg_sis_min = t_reg_sis / 60

    t_reg_dis = np.arange(t_taco_distole_segundos[0], t_taco_distole_segundos[-1], 1.0 / fs_interp)
    taco_dis_interp = interp1d(t_taco_distole_segundos, intervalos_ms_distole, kind='cubic')(t_reg_dis)
    t_reg_dis_min = t_reg_dis / 60










#Método Fourier




    if opcion.startswith("a") or opcion == "1":
        # 5. VISUALIZACIÓN EN TRES ETAPAS (Fourier) 
   
        Grafico_Tacograma, (ax_onda, ax_taco_crudo, ax_senoidal) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    
        paso = max(1, len(ppg_limpia) // 100000)
        ax_onda.plot(t_minutos[::paso], ppg_limpia[::paso], label="Señal PPG Limpia", color="#00459e", linewidth=0.8)
        ax_onda.plot(t_minutos[sistole], ppg_limpia[sistole], 'ro', markersize=4, label=f"Sístole (Picos) n={n_sis}", color="#3A8F01")
        ax_onda.plot(t_minutos[distole], ppg_limpia[distole], 'go', markersize=4, label=f"Diástole (Valles) n={n_dis}", color="#a30bdf")
        ax_onda.set_ylabel("Amplitud normalizada")
        ax_onda.set_title(f"PPG  |  Sístole: {n_sis} puntos   Diástole: {n_dis} puntos")
        ax_onda.text(
        0.02, 0.95,
        f"Sístole: {n_sis}\nDiástole: {n_dis}",
        transform=ax_onda.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax_onda.legend(loc="upper right")
        ax_onda.grid(True, linestyle=":", alpha=0.6)

        ax_taco_crudo.plot(t_taco_minutos, intervalos_ms, color="#3A8F01", marker="o", markersize=3, linewidth=1, label="Tacograma Sistólico Crudo")
        ax_taco_crudo.plot(t_taco_distole_minutos, intervalos_ms_distole, color="#a30bdf", marker="o", markersize=3, linewidth=1, alpha=0.8, label="Tacograma Diastólico Crudo")
        ax_taco_crudo.set_ylabel("Intervalo PP (ms)")
        ax_taco_crudo.legend(loc="upper right")
        ax_taco_crudo.grid(True, linestyle=":", alpha=0.6)

        ax_senoidal.plot(t_reg_sis_min, taco_sis_interp, color="#3A8F01", linewidth=1.5, label="Onda PRV Sistólica (4 Hz)")
        ax_senoidal.plot(t_reg_dis_min, taco_dis_interp, color="#a30bdf", linewidth=1.5, alpha=0.8, label="Onda PRV Diastólica (4 Hz)")
        ax_senoidal.set_xlabel("Tiempo (minutos)")
        ax_senoidal.set_ylabel("Intervalo PP (ms)")
        ax_senoidal.legend(loc="upper right")
        ax_senoidal.grid(True, linestyle=":", alpha=0.6)
    
        Grafico_Tacograma.suptitle(
        f"Evolución de PRV: Detección -> Tacograma -> Onda Senoidal "
        f"(min {t_inicio_min:.0f}–{t_fin_min:.0f}, {t_fin_min - t_inicio_min:.0f} min)",
        fontweight="bold",
        fontsize=13,
        )
        Grafico_Tacograma.tight_layout()

        # 6. ESPECTRO DE FRECUENCIAS APLICANDO FFT
        N_sis = len(taco_sis_interp)
        N_dis = len(taco_dis_interp)

        fft_sis_cruda = np.fft.fft(taco_sis_interp - np.mean(taco_sis_interp))
        fft_dis_cruda = np.fft.fft(taco_dis_interp - np.mean(taco_dis_interp))

        mag_sis_cruda = np.abs(fft_sis_cruda) / N_sis
        mag_dis_cruda = np.abs(fft_dis_cruda) / N_dis

        freqs_sis_cruda = np.fft.fftfreq(N_sis, d=1.0/fs_interp)
        freqs_dis_cruda = np.fft.fftfreq(N_dis, d=1.0/fs_interp)

        mitad_sis = N_sis // 2
        mitad_dis = N_dis // 2

        freqs_sis = freqs_sis_cruda[:mitad_sis]
        mag_sis = mag_sis_cruda[:mitad_sis] * 2  

        freqs_dis = freqs_dis_cruda[:mitad_dis]
        mag_dis = mag_dis_cruda[:mitad_dis] * 2

        Grafico_Espectro_Frecuencias, (ax_juntas, ax_sis, ax_dis) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        def formatear_eje_espectro(ax, titulo):
            ax.axvspan(0.003, 0.04, color='red', alpha=0.2, label='VLF')
            ax.axvspan(0.04, 0.15, color='blue', alpha=0.2, label='LF')
            ax.axvspan(0.15, 0.40, color='green', alpha=0.2, label='HF')
            ax.set_title(titulo, fontweight="bold")
            ax.set_ylabel("ms")
            ax.set_xlim(0, 0.5)
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.6)

        ax_juntas.plot(freqs_sis, mag_sis, color="#3A8F01", linewidth=1.2, label="Espectro Sistólico")
        ax_juntas.plot(freqs_dis, mag_dis, color="#a30bdf", linewidth=1.2, alpha=0.7, label="Espectro Diastólico")
        formatear_eje_espectro(ax_juntas, "Espectros de Amplitud Superpuestos (FFT)")

        ax_sis.plot(freqs_sis, mag_sis, color="#3A8F01", linewidth=1.5, label="Espectro Sistólico")
        formatear_eje_espectro(ax_sis,"Espectro Sístole")

        ax_dis.plot(freqs_dis, mag_dis, color="#a30bdf", linewidth=1.5, label="Espectro Diastólico")
        formatear_eje_espectro(ax_dis,"Espectro Diástole")

        ax_dis.set_xlabel("Frecuencia (Hz)")
        Grafico_Espectro_Frecuencias.tight_layout()

        # 7. ESPECTRO DE POTENCIA (MÉTODO DE WELCH)

        GraficaPotencia, (ax_potencias_juntas, ax_sis_potencia, ax_dis_potencia) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # 1. Configuración de Welch para capturar la banda VLF (0.003 Hz)
        nperseg_VLF = min(len(taco_sis_interp), int(fs_interp * 334))
        noverlap_VLF = nperseg_VLF // 2


        # Cálculo formal de Densidad Espectral de Potencia (PSD)
        freqs_sis_w, potencia_sis_w = welch(
        taco_sis_interp - np.mean(taco_sis_interp), 
        fs=fs_interp, window='hann', nperseg=nperseg_VLF, noverlap=noverlap_VLF, scaling='density'
        )
        freqs_dis_w, potencia_dis_w = welch(
        taco_dis_interp - np.mean(taco_dis_interp), 
        fs=fs_interp, window='hann', nperseg=nperseg_VLF, noverlap=noverlap_VLF, scaling='density'
        )

        # 2. Graficar contra el eje de frecuencias generado por Welch 
        ax_sis_potencia.plot(freqs_sis_w, potencia_sis_w, color="#3A8F01", linewidth=1.5, label="Potencia Sistólica (Welch)")
        ax_sis_potencia.set_title("Densidad Espectral de Potencia - Sístole", fontweight="bold")
        ax_sis_potencia.set_ylabel("ms²/Hz")
        ax_sis_potencia.set_xlim(0, 0.5)
        ax_sis_potencia.legend(loc="upper right")
        ax_sis_potencia.grid(True, linestyle=":", alpha=0.6)
        # Sombreado de bandas
        ax_sis_potencia.axvspan(0.003, 0.04, color='red', alpha=0.1)
        ax_sis_potencia.axvspan(0.04, 0.15, color='blue', alpha=0.1)
        ax_sis_potencia.axvspan(0.15, 0.40, color='green', alpha=0.1)

        ax_dis_potencia.plot(freqs_dis_w, potencia_dis_w, color="#a30bdf", linewidth=1.5, label="Potencia Diastólica (Welch)")
        ax_dis_potencia.set_title("Densidad Espectral de Potencia - Diástole", fontweight="bold")
        ax_dis_potencia.set_ylabel("ms²/Hz")
        ax_dis_potencia.set_xlim(0, 0.5)
        ax_dis_potencia.legend(loc="upper right")
        ax_dis_potencia.grid(True, linestyle=":", alpha=0.6)
        # Sombreado de bandas
        ax_dis_potencia.axvspan(0.003, 0.04, color='red', alpha=0.1)
        ax_dis_potencia.axvspan(0.04, 0.15, color='blue', alpha=0.1)
        ax_dis_potencia.axvspan(0.15, 0.40, color='green', alpha=0.1)
    
        ax_potencias_juntas.plot(freqs_sis_w, potencia_sis_w, color="#3A8F01", linewidth=1.5, label="Potencia Sistólica")
        ax_potencias_juntas.plot(freqs_dis_w, potencia_dis_w, color="#a30bdf", linewidth=1.5, alpha=0.7, label="Potencia Diastólica")
        ax_potencias_juntas.set_title("Espectros de Potencia Superpuestos", fontweight="bold")
        ax_potencias_juntas.set_xlabel("Frecuencia (Hz)")
        ax_potencias_juntas.set_ylabel("ms²/Hz")
        ax_potencias_juntas.set_xlim(0, 0.5)
        ax_potencias_juntas.legend(loc="upper right")
        ax_potencias_juntas.grid(True, linestyle=":", alpha=0.6)
    
        # Sombreado de bandas
        ax_potencias_juntas.axvspan(0.003, 0.04, color='red', alpha=0.1)
        ax_potencias_juntas.axvspan(0.04, 0.15, color='blue', alpha=0.1)
        ax_potencias_juntas.axvspan(0.15, 0.40, color='green', alpha=0.1)

        GraficaPotencia.tight_layout()
    

        # ESPECTROGRAMA (STFT): solo enfoque temporal (ventana de 1 min)
        nperseg_stft = min(len(taco_sis_interp), int(fs_interp * 60))
        paso_segundos = 2
        titulo_stft = "Espectrograma Sistólico (STFT) - Enfoque TEMPORAL"

  
        noverlap_stft = max(0, nperseg_stft - int(fs_interp * paso_segundos))

  
        freqs_stft, tiempos_stft, Sxx_sis = spectrogram(
        taco_sis_interp - np.mean(taco_sis_interp),
        fs=fs_interp,
        window='hann',
        nperseg=nperseg_stft,
        noverlap=noverlap_stft,
        scaling='density'
        )


        tiempos_stft_min = tiempos_stft / 60


        fig_stft, ax_stft = plt.subplots(figsize=(14, 6))


        im = ax_stft.pcolormesh(tiempos_stft_min, freqs_stft, Sxx_sis, shading='gouraud', cmap='jet')

        ax_stft.set_title(titulo_stft, fontweight="bold")
        ax_stft.set_ylabel("Frecuencia (Hz)")
        ax_stft.set_xlabel("Tiempo (minutos)")
        ax_stft.set_ylim(0, 0.5) 

  
        ax_stft.axhline(0.04, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_stft.text(tiempos_stft_min[0], 0.02, ' VLF', color='white', fontweight='bold')
    
        ax_stft.axhline(0.15, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_stft.text(tiempos_stft_min[0], 0.09, ' LF', color='white', fontweight='bold')
    
        ax_stft.axhline(0.40, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_stft.text(tiempos_stft_min[0], 0.27, ' HF', color='white', fontweight='bold')

        cbar = fig_stft.colorbar(im, ax=ax_stft)
        cbar.set_label("Densidad de Potencia (ms²/Hz)")

        fig_stft.tight_layout()
    
    
    
        fig_resumen, (ax_onda, ax_taco, ax_espectro) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        paso = max(1, len(ppg_limpia) // 100000)
        ax_onda.plot(t_minutos[::paso], ppg_limpia[::paso], label="Señal PPG Limpia", color="#00459e", linewidth=0.8)
        ax_onda.plot(t_minutos[sistole], ppg_limpia[sistole], 'ro', markersize=4, label='Sístole (Picos)', color="#3A8F01")
        ax_onda.plot(t_minutos[distole], ppg_limpia[distole], 'go', markersize=4, label='Diástole (Valles)',color="#a30bdf" )
        ax_onda.set_ylabel("Amplitud normalizada")
        ax_onda.legend(loc="upper right")
        ax_onda.grid(True, linestyle=":", alpha=0.6)
        
        ax_taco.plot(t_taco_minutos, intervalos_ms, color="#3A8F01", marker="o", markersize=3, linewidth=1, label="Tacograma Sistólico")
        ax_taco.set_title("2. Tacograma Sistole", fontweight="bold")
        ax_taco.set_ylabel("Intervalo PP (ms)")
        ax_taco.legend(loc="upper right")
        ax_taco.grid(True, linestyle=":", alpha=0.6)
        
        im = ax_espectro.pcolormesh(tiempos_stft_min, freqs_stft, Sxx_sis, shading='gouraud', cmap='jet')
        ax_espectro.set_title("3. Espectrograma", fontweight="bold")
        ax_espectro.set_ylabel("Frecuencia (Hz)")
        ax_espectro.set_xlabel("Tiempo (minutos)")
        ax_espectro.set_ylim(0, 0.5) 
        
        cbar = fig_resumen.colorbar(im, ax=ax_espectro, pad=0.02)
        cbar.set_label("Potencia (ms²/Hz)")

        fig_resumen.suptitle("Resumen: PPG -> Tacograma -> Espectrograma", fontsize=15, fontweight="bold")
        
        fig_resumen.canvas.draw()
        pos_base = ax_espectro.get_position()
        
        pos_onda = ax_onda.get_position()
        ax_onda.set_position([pos_base.x0, pos_onda.y0, pos_base.width, pos_onda.height])
        
        pos_taco = ax_taco.get_position()
        ax_taco.set_position([pos_base.x0, pos_taco.y0, pos_base.width, pos_taco.height])

        plt.show()







    # Un solo EMD. La envolvente superior pasa por los máximos y la inferior por los valles.

    elif opcion.startswith("b") or opcion == "2":
        from PyEMD import EMD
        from scipy.interpolate import PchipInterpolator

        t_taco = np.asarray(t_taco_minutos, dtype=float)
        y_taco = np.asarray(intervalos_ms, dtype=float)
        t_prv = np.asarray(t_reg_sis_min, dtype=float)
        # PCHIP pasa por cada latido y no inventa picos, a diferencia del spline cúbico.
        y_prv = np.asarray(PchipInterpolator(t_taco, y_taco)(t_prv), dtype=float)

        emd = EMD(spline_kind="pchip")
        max_pos, max_val, min_pos, min_val, env_sup, env_inf, media = primer_paso_huang(emd, t_prv, y_prv)
        print(f"EMD: {len(max_pos)} máximos, {len(min_pos)} valles")

        fig1, (ax_ppg, ax_taco) = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)
        paso_ppg = max(1, int(round(fs / 80.0)))
        ax_ppg.plot(
            t_minutos[::paso_ppg], ppg_limpia[::paso_ppg],
            color="#1f4e79", linewidth=0.6, label="PPG",
        )
        ax_ppg.plot(
            t_minutos[sistole], ppg_limpia[sistole],
            linestyle="none", marker="^", color="#2e7d32", markersize=3,
            label=f"Máximos n={len(sistole)}",
        )
        ax_ppg.plot(
            t_minutos[distole], ppg_limpia[distole],
            linestyle="none", marker="v", color="#6a1b9a", markersize=3,
            label=f"Valles n={len(distole)}",
        )
        ax_ppg.set_title("PPG", loc="left")
        ax_ppg.set_ylabel("Amplitud")
        ax_ppg.grid(True, linestyle=":", alpha=0.45)
        ax_ppg.margins(x=0.01)
        ax_ppg.legend(loc="upper right", fontsize=8, frameon=False)

        dibujar_envolventes(
            ax_taco, t_taco, y_taco, t_prv, max_pos, max_val, min_pos, min_val,
            env_sup, env_inf, media, "Tacograma", "Tacograma",
        )
        ax_taco.set_xlabel("Tiempo (minutos)")
        handles, labels = ax_taco.get_legend_handles_labels()
        ax_taco.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8, frameon=False)
        fig1.suptitle("PPG y envolventes del tacograma", fontweight="bold")
        fig1.tight_layout(rect=(0, 0, 0.86, 0.96))

        # La media de las dos envolventes es lo que el tamizado resta. EMD repite ese paso.
        emd(y_prv, t_prv)
        imfs, residuo = emd.get_imfs_and_residue()
        imfs = np.atleast_2d(np.asarray(imfs, dtype=float))
        residuo = np.asarray(residuo, dtype=float)
        print(f"Descomposición: {len(imfs)} IMFs + residuo, {len(y_prv)} muestras a {fs_interp} Hz")

        n_filas = len(imfs) + 1
        fig2, axes = plt.subplots(n_filas, 1, figsize=(13, 1.45 * n_filas), sharex=True)
        if n_filas == 1:
            axes = [axes]
        for i, imf in enumerate(imfs):
            axes[i].plot(t_prv, imf, color="#1f4e79", linewidth=0.8)
            axes[i].set_ylabel(f"IMF {i + 1}")
            axes[i].grid(True, linestyle=":", alpha=0.55)
            axes[i].margins(x=0.01)
        axes[-1].plot(t_prv, residuo, color="#e65100", linewidth=1.0)
        axes[-1].set_ylabel("Residuo")
        axes[-1].set_xlabel("Tiempo (minutos)")
        axes[-1].grid(True, linestyle=":", alpha=0.55)
        axes[-1].margins(x=0.01)
        fig2.suptitle("Descomposición empírica", fontweight="bold")
        fig2.tight_layout()
        plt.show()
    else:
        print("Opcion no valida. Elige a o b.")
