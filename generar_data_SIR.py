import pandas as pd
import numpy as np
import time
import random
import os

# =========================
# CONFIGURACIÓN GENERAL
# =========================
# Número de simulaciones
N_S = 130_000

# Checkpoints
CHECKPOINT_EVERY = 1000                      # cada cuántas simulaciones guardar un checkpoint
OUT_DIR = "checkpoints"                      # carpeta de salida para checkpoints
OUT_BASE = "simulaciones_SIR_V7"             # prefijo de archivo para checkpoints

# Guardar también las trayectorias completas (S, I, R, u) en memoria (cuidado: muy pesado para N_S grande)
SAVE_SISTEM = False

# Crear carpeta de checkpoints
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# UTILIDADES
# =========================
def human_count(n: int) -> str:
    """Convierte n en sufijo legible: 100_000 -> '100k', 1_000_000 -> '1M'."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)

def final_filename(base="simulaciones_SIR_V7", total=N_S):
    """Genera el nombre final a partir de N_S."""
    return f"{base}_{human_count(total)}.csv"

# =========================
# MEDICIÓN DE TIEMPO
# =========================
tiempo_inicial = time.time()

# =========================
# MODELO Y ÓPTIMO
# =========================
# Función del modelo SIR con control no lineal
def SIR_control(S, I, R, u, delta_beta, delta_gamma):
    beta_eff = delta_beta * (1.0 - u)
    dS = -beta_eff * S * I
    dI = beta_eff * S * I - delta_gamma * I
    dR = delta_gamma * I
    return dS, dI, dR

# Funciones de coestados
def coestados(S, I, R, lambdaS, lambdaI, lambdaR, u, delta_beta, delta_gamma, C_I):
    beta_eff = delta_beta * (1.0 - u)
    d_lambdaS = beta_eff * I * (lambdaS - lambdaI)
    d_lambdaI = - 2 * C_I * I + beta_eff * S * (lambdaS - lambdaI) + delta_gamma * (lambdaI - lambdaR)
    d_lambdaR = 0.0
    return d_lambdaS, d_lambdaI, d_lambdaR

# Condición de optimalidad
def optimalidad(S, I, lambdaS, lambdaI, R_0, delta_beta, C_u, tol=1e-8):
    if S <= 0.0 or I <= 0.0:
        return 0.0
    A = delta_beta * S * I * (lambdaS - lambdaI)
    p  = lambda C: 2 * C_u * C * (1 + C)**2 + A
    dp = lambda C: 2 * C_u * (1 + C) * (1 + 3 * C)
    u = np.clip(1 - (1 / (R_0 * S)), 0, 0.999)

    if A >= 0:
        return 0.0

    C = u / (1 - u)
    for _ in range(100):
        f = p(C)
        df = dp(C)
        C_nueva = max(0.0, C - f / df)
        if abs(C_nueva - C) < tol:
            C = C_nueva
            break
        C = C_nueva
    return C / (1 + C)

# =========================
# PARÁMETROS DEL MODELO
# =========================
C_I = 1000.0  # 5.0 -> 3.0 -> 1500
C_u = 1.0     # 1.0 -> 4.0
C_LI = 50.0     # 20 antes -> 50

T_total = 120
dt = 1.0  # 0.5
N = int(T_total / dt)

print(f"Inicio: N_S={N_S}, N={N}, dt={dt}", flush=True)

# =========================
# BUCLE PRINCIPAL
# =========================
rows = []
sistem = [] if SAVE_SISTEM else None

# buffers para checkpoint
rows_chunk = []
sistem_chunk = [] if SAVE_SISTEM else None

for sim_idx in range(1, N_S + 1):
    delta_S0 = random.uniform(0.85, 0.99999)
    delta_I0 = 1 - delta_S0
    delta_R0 = 0
    delta_NRB = random.uniform(1.1, 3.0)
    delta_gamma = random.uniform(0.05, 0.15)
    delta_beta = delta_NRB * delta_gamma
    inicio_control = random.randint(14, 60)

    # Parámetros del horizonte errante
    T_horizonte = 14.0
    N_H = int(T_horizonte / dt)

    # Iniciar condiciones generales
    S, I, R, u = [delta_S0], [delta_I0], [delta_R0], []
    S_entrada, I_entrada, R_entrada, T_entrada = [], [], [], []
    I_pronostico, u_pronostico, T_pronostico = [], [], []
    T_restante, I_restante, u_restante = [], [], []

    for t_step in range(N):
        tiempo_actual = t_step * dt
        S_ini, I_ini, R_ini = S[-1], I[-1], R[-1]

        if tiempo_actual >= inicio_control:  # Condición para aplicar el control
            # N_H+1 estados (índices 0..N_H), N_H controles (índices 0..N_H-1)
            S_h = np.zeros(N_H + 1)
            I_h = np.zeros(N_H + 1)
            R_h = np.zeros(N_H + 1)
            u_h = np.ones(N_H) * 0.5
            lambdaS = np.zeros(N_H + 1)
            lambdaI = np.zeros(N_H + 1)
            lambdaR = np.zeros(N_H + 1)

            S_h[0], I_h[0], R_h[0] = S_ini, I_ini, R_ini

            for _ in range(100):
                # forward
                for i in range(N_H):
                    dS, dI, dR = SIR_control(S_h[i], I_h[i], R_h[i], u_h[i], delta_beta, delta_gamma)
                    S_h[i+1] = np.clip(S_h[i] + dt * dS, 0.0, 1.0)
                    I_h[i+1] = np.clip(I_h[i] + dt * dI, 0.0, 1.0)
                    R_h[i+1] = np.clip(R_h[i] + dt * dR, 0.0, 1.0)

                # condición terminal con I_h[-1] actualizado del forward
                lambdaS[-1], lambdaI[-1], lambdaR[-1] = 0.0, 2 * C_LI * I_h[-1], 0.0

                # backward
                for i in reversed(range(N_H)):
                    d_lambdaS, d_lambdaI, d_lambdaR = coestados(
                        S_h[i], I_h[i], R_h[i],
                        lambdaS[i+1], lambdaI[i+1], lambdaR[i+1],
                        u_h[i], delta_beta, delta_gamma, C_I
                    )
                    lambdaS[i] = lambdaS[i+1] - dt * d_lambdaS
                    lambdaI[i] = lambdaI[i+1] - dt * d_lambdaI
                    lambdaR[i] = lambdaR[i+1] - dt * d_lambdaR

                # actualización del control con damping
                u_old = u_h.copy()
                for i in range(N_H):
                    u_nueva = optimalidad(S_h[i], I_h[i], lambdaS[i], lambdaI[i], delta_NRB, delta_beta, C_u)
                    u_h[i] = np.clip(0.7 * u_h[i] + 0.3 * u_nueva, 0.0, 0.999)

                # criterio de paro global dentro de la ventana MPC
                if np.max(np.abs(u_h - u_old)) < 1e-6:
                    break

            u_aplicado = u_h[0]
        else:
            u_aplicado = 0.0

        # Actualización del sistema global
        dS, dI, dR = SIR_control(S_ini, I_ini, R_ini, u_aplicado, delta_beta, delta_gamma)
        S_next = np.clip(S_ini + dt * dS, 0.0, 1.0)
        I_next = np.clip(I_ini + dt * dI, 0.0, 1.0)
        R_next = np.clip(R_ini + dt * dR, 0.0, 1.0)

        S.append(S_next)
        I.append(I_next)
        R.append(R_next)
        u.append(u_aplicado)

        # Ventanas de muestreo
        if (inicio_control - 14 <= tiempo_actual < inicio_control):  # precontrol
            T_entrada.append(tiempo_actual)
            S_entrada.append(S_ini)
            I_entrada.append(I_ini)
            R_entrada.append(R_ini)
        if (inicio_control <= tiempo_actual < inicio_control + 14):  # primeras 2 semanas de control
            T_pronostico.append(tiempo_actual)
            I_pronostico.append(I_ini)
            u_pronostico.append(u_aplicado)
        if (inicio_control + 14 <= tiempo_actual):  # resto
            T_restante.append(tiempo_actual)
            I_restante.append(I_ini)
            u_restante.append(u_aplicado)


        # Dinámica del pico de infectados (clasificación)
        altura_pico = max(I)
        if (altura_pico <= 0.005):
            estado = 'brote_incipiente'
        elif (altura_pico <= 0.01):
            estado = 'crecimiento_claro'
        elif (altura_pico <= 0.03):
            estado = 'inicia_presion'
        elif (altura_pico <= 0.05):
            estado = 'ola_en_marcha'
        elif (altura_pico <= 0.10):
            estado = 'ola_severa'
        elif (altura_pico <= 0.20):
            estado = 'critica'
        elif (altura_pico <= 0.30):
            estado = 'extrema'
        else:
            estado = 'irreal'

        momento_pico = np.argmax(I) * dt
        if momento_pico < inicio_control:
            momento = 'pre_pico'
        else:
            momento = 'post_pico'

    # Registro de la simulación
    row = {
        'R_0': delta_NRB,
        'gamma': delta_gamma,
        'beta': delta_beta,
        'S0': delta_S0,
        'I0': delta_I0,
        'Altura_pico': estado,
        'Tiempo_pico': momento,
        'T_entrada': T_entrada,
        'S_entrada': S_entrada,
        'I_entrada': I_entrada,
        'R_entrada': R_entrada,
        'T_pronostico': T_pronostico,
        'I_pronostico': I_pronostico,
        'u_pronostico': u_pronostico,
        'T_restante': T_restante,
        'I_restante': I_restante,
        'u_restante': u_restante
    }
    rows.append(row)
    rows_chunk.append(row)

    if SAVE_SISTEM:
        sysrow = {'S': S, 'I': I, 'R': R, 'u': u}
        sistem.append(sysrow)
        sistem_chunk.append(sysrow)

    # Prints de progreso (vida)
    if sim_idx % 100 == 0 or sim_idx == 1:
        elapsed = time.time() - tiempo_inicial
        print(f"[{sim_idx}/{N_S}] filas={len(rows)}  t={elapsed:.1f}s", flush=True)

    # Checkpoint cada CHECKPOINT_EVERY simulaciones
    if CHECKPOINT_EVERY and sim_idx % CHECKPOINT_EVERY == 0:
        start_id = sim_idx - CHECKPOINT_EVERY + 1
        end_id = sim_idx
        df_chunk = pd.DataFrame(rows_chunk)
        out_path = os.path.join(OUT_DIR, f"{OUT_BASE}_chunk_{start_id}_{end_id}.csv")
        df_chunk.to_csv(out_path, index=False, encoding='utf-8')
        elapsed = time.time() - tiempo_inicial
        print(f"[Checkpoint] {out_path}  (simulaciones {start_id}-{end_id})  t={elapsed:.1f}s", flush=True)
        rows_chunk.clear()
        if SAVE_SISTEM:
            sistem_chunk.clear()

# =========================
# GUARDADO FINAL
# =========================
df = pd.DataFrame(rows)
# Nota: df_sistem puede ser enorme; solo lo construimos si SAVE_SISTEM=True
df_sistem = pd.DataFrame(sistem) if SAVE_SISTEM else None

OUT_FINAL = final_filename(base="simulaciones_SIR_V7", total=N_S)  # p.ej. simulaciones_SIR_V5_100k.csv
df.to_csv(OUT_FINAL, index=False, encoding='utf-8')

tiempo_final = time.time()
print("\n✅ Simulación completada", flush=True)
print(f"Archivo final guardado como: {OUT_FINAL}", flush=True)
print(f"Tiempo total: {tiempo_final - tiempo_inicial:.2f} s (~{(tiempo_final - tiempo_inicial)/60:.2f} min)", flush=True)
print(f"Total de filas generadas: {len(df)}", flush=True)
