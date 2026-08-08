import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. System & Reactor Setup ---
V = 0.058             # Reactor volume (L) [58 mL]
Q = 0.01 / 1000.0     # Flow rate (L/min) [0.01 mL/min]
S_calcite_0 = 4.0     # Initial calcite loading (g/L)
SSA_calcite = 3.0     # Calcite specific surface area (m^2/g)
MW_calcite = 100.09   # g/mol

# Inflow Concentrations
C_in_F = 0.21 / 1000.0# Influent F: 0.21 mM
C_in_Ca = 1.0e-5      # Influent Ca: 0.01 mM
C_in_CO3 = 1.0e-5     # Influent CO3: 0.01 mM

# --- 2. Dissolution & Adsorption Kinetic Constants ---
k_diss_calcite = 10**(-5.90)  # mol/m^2/min (calcite dissolution rate)
Ksp_calcite = 10**(-8.48)     # Calcite solubility product
k_gl = 10**(-4.06)            # CO2 gas-liquid transfer rate (min^-1)

# Fitted Adsorption Parameters
k_F_ad = 1.565                # L/mol/min
q_max_F = 8.968e-5            # mol/g

# --- 3. Numerical Simulation Loop ---
dt = 0.2                       # min
t_end = 12750.0                # min
steps = int(t_end / dt)

C_F, C_Ca, C_CO3 = 0.0, C_in_Ca, C_in_CO3
S_calcite = S_calcite_0
q_F = 0.0

results = []

for step in range(steps):
    t = step * dt
    
    # 1. Calcite Saturation Index & Dissolution Rate
    IAP_calcite = C_Ca * C_CO3
    Omega_calcite = IAP_calcite / Ksp_calcite
    
    # Dissolution rate (mol/L/min) occurs when undersaturated (Omega < 1)
    if Omega_calcite < 1.0 and S_calcite > 0:
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega_calcite)
    else:
        r_diss = 0.0
        
    # 2. Fluoride Adsorption Rate (mol/L/min)
    r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
    
    # 3. Coupled Liquid Mass Balances
    dC_F_dt = (Q / V) * (C_in_F - C_F) - r_ad_F
    dC_Ca_dt = (Q / V) * (C_in_Ca - C_Ca) + r_diss
    dC_CO3_dt = (Q / V) * (C_in_CO3 - C_CO3) + r_diss
    
    # 4. Forward Euler Updates
    C_F += dC_F_dt * dt
    C_Ca += dC_Ca_dt * dt
    C_CO3 += dC_CO3_dt * dt
    q_F += (r_ad_F / S_calcite) * dt
    S_calcite -= (r_diss * MW_calcite / 1000.0) * dt # Calcite mass loss
    
    if step % 500 == 0:
        results.append({
            "time_min": t,
            "C_F_mM": C_F * 1000,
            "C_Ca_mM": C_Ca * 1000,
            "C_CO3_mM": C_CO3 * 1000,
            "S_calcite_gL": S_calcite
        })

df = pd.DataFrame(results)
df.to_csv("coupled_dissolution_adsorption_results.csv", index=False)
print("Coupled calcite dissolution + fluoride adsorption simulation completed successfully!")