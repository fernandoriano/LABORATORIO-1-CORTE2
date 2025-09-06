# -----------------------------------------------------------------------------
# Parte 2 - Muestreo "estable" por espera activa
--------------

# --- Importaciones de módulos del firmware (MicroPython) y de Python ---------
from machine import ADC, Pin       # Periféricos: convertidor ADC y control de pines
import utime                       # Tiempos con resolución de microsegundos
import array                       # Estructura eficiente para buffers (enteros sin signo)
import math                        # Funciones matemáticas reales
import cmath                       # Números complejos (para FFT manual)

# ------------------------ Parámetros de adquisición --------------------------
adc = ADC(Pin(27))                 # Instancia de ADC en GP27 (canal ADC1); usar GP26 si prefieres ADC0
N = 512                            # Número de muestras a capturar en el dominio del tiempo
N_FFT = 1024                       # Tamaño de la FFT (si N < N_FFT, se aplica zero-padding)
f_muestreo = 2000                  # Frecuencia de muestreo objetivo (Hz) 
dt_us = int(1_000_000 / f_muestreo)# Periodo de muestreo en microsegundos; 1e6/f_s para obtener Δt

# ---------------- Adquisición con "tiempos estables" (espera activa) --------
def acquire_data():
    # Lista para almacenar el tiempo relativo (en segundos) de cada muestra
    tiempos = []
    # Lista para almacenar el valor leído por el ADC (cuentas 0..65535)
    muestras = []

    # Marca temporal en microsegundos al inicio de la captura (t=0)
    start = utime.ticks_us()
    # Próximo instante de muestreo (en microsegundos, relativo al reloj de utime)
    next_t = start

    # Bucle de adquisición de N muestras
    for i in range(N):
        # Espera activa (busy-wait) hasta que el reloj alcance "next_t"
        # ticks_diff(next_t, ahora) > 0 implica que aún no llegamos al instante objetivo
        while utime.ticks_diff(next_t, utime.ticks_us()) > 0:
            pass

        # Toma tiempo actual (microsegundos)
        now = utime.ticks_us()
        # Convierte a tiempo relativo en segundos desde "start"
        t_rel = utime.ticks_diff(now, start) / 1_000_000.0

        # Guarda tiempo relativo y lectura ADC (entero sin signo 16 bits escalado desde 12 bits reales)
        tiempos.append(t_rel)
        muestras.append(adc.read_u16())

        # Programa el siguiente instante de muestreo sumando Δt en microsegundos
        next_t = utime.ticks_add(next_t, dt_us)

    # Estima f_s real usando la duración entre la primera y la última muestra
    elapsed_time = tiempos[-1] - tiempos[0]     # segundos entre la muestra 0 y la N-1
    fs_real = (N - 1) / elapsed_time if elapsed_time > 0 else f_muestreo

    # Reporte por consola de f_s objetivo y f_s real alcanzada
    print(f"[ACQ] f_deseada: {f_muestreo} Hz | f_real: {fs_real:.2f} Hz | dt_us: {dt_us} us")

    # --- Guarda archivo de muestras en voltios para graficar en MATLAB/Python ---
    with open("muestras.txt", "w") as f:
        f.write("Tiempo(s)\tVoltaje(V)\n")
        for t, u in zip(tiempos, muestras):
            # Conversión a voltios asumiendo Vref=3.3 V y escala 0..65535
            v = (u / 65535.0) * 3.3
            f.write(f"{t:.6f}\t{v:.6f}\n")

    # --- Cálculo de dt entre muestras y métricas de jitter ----------------------
    # Δt_k = t_k - t_{k-1} en segundos
    dts = [ (tiempos[i] - tiempos[i-1]) for i in range(1, len(tiempos)) ]

    if dts:
        # Promedio de Δt
        mean_dt = sum(dts) / len(dts)
        # Jitter pico a pico = max(Δt) - min(Δt)
        jitter_pp = (max(dts) - min(dts))
        # Jitter RMS = sqrt( mean( (Δt - mean(Δt))^2 ) )
        jitter_rms = (sum((dt - mean_dt)**2 for dt in dts) / len(dts))**0.5

        # Reporte por consola en microsegundos
        print(f"[ACQ] Jitter p-p: {jitter_pp*1e6:.2f} us | Jitter RMS: {jitter_rms*1e6:.2f} us")

        # Guarda archivo de Δt en microsegundos para análisis/figuras
        with open("dt_us.txt", "w") as fdt:
            fdt.write("dt_us\n")
            for dt in dts:
                fdt.write(f"{dt*1e6:.3f}\n")

    # Devuelve listas y f_s real para etapas posteriores
    return muestras, tiempos, fs_real

# -------------------------- Utilidades de preprocesado -----------------------
def convert_to_voltage(data_u16, VREF=3.3):
    # Convierte la lista de cuentas ADC (0..65535) a voltios asumiendo referencia VREF
    return [(x / 65535.0) * VREF for x in data_u16]

def remove_offset(x):
    # Calcula la media (DC) y la resta a cada muestra para centrar la señal en 0 V
    m = sum(x) / len(x)
    print(f"[PROC] DC estimada: {m:.4f} V")
    return [xi - m for xi in x], m

def apply_hanning_window(x):
    # Aplica ventana de Hanning: w[n] = 0.5*(1 - cos(2πn/(N-1)))
    # Reduce la fuga espectral; OJO: atenúa amplitud (ganancia coherente ≈ 0.5)
    Nloc = len(x)
    w = [0.5 * (1 - math.cos(2 * math.pi * i / (Nloc - 1))) for i in range(Nloc)]
    return [xi * wi for xi, wi in zip(x, w)]

# ------------------------------ FFT radix-2 ----------------------------------
def fft_manual(x, NFFT):
    # Reversa de bits para reordenamiento inicial (Cooley–Tukey)
    def bit_reversal(n, logN):
        r = 0
        for i in range(logN):
            if (n >> i) & 1:
                r |= 1 << (logN - 1 - i)
        return r

    # Construye vector complejo con zero-padding si hace falta
    X = [complex(v, 0.0) for v in x[:NFFT]] + [0.0] * (NFFT - len(x))
    logN = int(math.log2(NFFT))     # Número de etapas (NFFT debe ser potencia de 2)

    # Reordenamiento por reversa de bits
    for i in range(NFFT):
        j = bit_reversal(i, logN)
        if j > i:
            X[i], X[j] = X[j], X[i]

    # Etapas "mariposa" iterativas
    m = 2
    for s in range(logN):
        half_m = m // 2
        w_m = cmath.exp(-2j * math.pi / m)  # Factor giratorio e^{-j2π/m}
        for k in range(0, NFFT, m):
            w = 1 + 0j
            for j in range(half_m):
                t = w * X[k + j + half_m]   # rama superior rotada
                u = X[k + j]                # rama inferior
                X[k + j] = u + t            # salida "par"
                X[k + j + half_m] = u - t   # salida "impar"
                w *= w_m                    # actualiza factor giratorio
        m <<= 1                             # duplica tamaño de grupo (m = 2,4,8,...)

    return X

# ----------------------------- Análisis espectral ----------------------------
def analyze_fft(fft_result, fs_real, NFFT):
    # Magnitud lineal normalizada por (NFFT/2) para estimar amplitud de tono puro
    mags = [abs(c) / (NFFT / 2.0) for c in fft_result[:NFFT // 2]]
    # Vector de frecuencias correspondiente a bins 0..NFFT/2-1
    freqs = [i * fs_real / NFFT for i in range(NFFT // 2)]

    # Busca pico dominante omitiendo DC (índice 0)
    if len(mags) > 1:
        kmax = 1 + max(range(1, len(mags)), key=lambda i: mags[i])
    else:
        kmax = 0
    f_dom = freqs[kmax]             # Frecuencia estimada del tono dominante
    A = mags[kmax]                  # Amplitud en el bin (si usaste Hanning, real ≈ A/0.5)

    # Estimación simple del "piso de ruido" como promedio de magnitudes quitando el bin pico
    noise = mags[:]
    if kmax < len(noise):
        noise[kmax] = 0.0
    noise_floor_v = sum(noise) / max(1, (len(noise) - 1))

    # Cálculo de SNR (dB) y ENOB (bits) con fórmulas estándar aproximadas
    if noise_floor_v > 0:
        SNR = 20 * math.log10(A / noise_floor_v)
    else:
        SNR = float('inf')
    ENOB = (SNR - 1.76) / 6.02 if SNR != float('inf') else float('inf')

    # Reporte en consola
    print(f"[FFT] f_dom ~ {f_dom:.2f} Hz | A(bin) ~ {A:.4f} V | SNR ~ {SNR:.2f} dB | ENOB ~ {ENOB:.2f} bits")

    # Guarda archivo de espectro (frecuencia vs magnitud)
    with open("fft.txt", "w") as f:
        f.write("Frecuencia(Hz)\tMagnitud(V)\n")
        for fr, mg in zip(freqs, mags):
            f.write(f"{fr:.2f}\t{mg:.6f}\n")

    # Devuelve datos útiles por si quieres graficar en otra parte
    return freqs, mags, f_dom, A, SNR, ENOB

# --------------------------------- Programa principal ------------------------
def main():
    # Mensaje inicial
    print("[INFO] Iniciando adquisición estable (espera activa)...")

    # Captura de datos crudos (u16), tiempos relativos y f_s real
    muestras_u16, tiempos_s, fs_real = acquire_data()

    # Conversión a voltios y retiro de componente DC (centra señal en 0 V)
    v = convert_to_voltage(muestras_u16)
    v_ac, dc = remove_offset(v)

    # Aplicar ventana de Hanning para reducir fuga espectral
    v_win = apply_hanning_window(v_ac)

    # Ejecutar FFT y analizar resultados (pico, SNR, ENOB)
    X = fft_manual(v_win, N_FFT)
    analyze_fft(X, fs_real, N_FFT)

    # Mensaje de cierre y lista de archivos generados
    print("[OK] Archivos generados: muestras.txt, dt_us.txt (jitter), fft.txt")

# Punto de entrada del script
if __name__ == "__main__":
    main()
