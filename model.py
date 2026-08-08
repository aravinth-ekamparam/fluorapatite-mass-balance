import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Experimental Observed Data Points ---
obs_data = pd.DataFrame({
    "time_min": [
        150, 450, 750, 1050, 1350, 1650, 1950, 2250, 2550, 2850, 3150, 3450, 3750, 4050, 
        4350, 4650, 4950, 5250, 5550, 5850, 6150, 6450, 6750, 7050, 7350, 7650, 7950, 
        8250, 8550, 8850, 9150, 9450, 9750, 10200, 10650, 10950, 11250, 11550, 11850, 
        12150, 12450, 12750
    ],
    "F_obs_M": [
        2.02E-05, 1.92E-05, 2.11E-05, 2.44E-05, 2.74E-05, 3.05E-05, 3.37E-05, 3.52E-05, 
        3.90E-05, 4.15E-05, 4.32E-05, 4.53E-05, 4.80E-05, 4.86E-05, 5.12E-05, 5.39E-05, 
        5.28E-05, 5.60E-05, 5.79E-05, 5.90E-05, 6.34E-05, 6.17E-05, 6.34E-05, 6.34E-05, 
        6.34E-05, 6.88E-05, 6.76E-05, 6.65E-05, 7.31E-05, 7.68E-05, 7.52E-05, 7.52E-05, 
        7.79E-05, 7.81E-05, 8.08E-05, 8.23E-05, 8.13E-05, 7.90E-05, 8.08E-05, 7.96E-05, 
        8.34E-05, 8.23E-05
    ]
})

# --- 2. System & Kinetic Parameters ---
V = 0.058             # Reactor volume (L) [58 mL]
Q = 0.01 / 1000.0     # Flow rate (L/min) [0.01 mL/min]
S_calcite_0 = 4.0     # Calcite loading (g/L)
SSA_calcite = 3.0     # Specific surface area (m^2/g)
MW_calcite = 100.09   # g/mol

C_in_F = 0.21 / 1000.0# Influent F (0.21 mM -> M)
C_in_Ca = 1.0e-5      # Influent Ca (M)
C_in_CO3 = 1.0e-5     # Influent CO3 (M)

k_diss_calcite = 10**(-5.90)  # mol/m^2/min (Calcite dissolution rate)
Ksp_calcite = 10**(-8.48)     # Calcite solubility product
k_F_ad = 1.565                # L/mol/min (Adsorption rate constant)
q_max_F = 8.968e-5            # mol/g (Adsorption capacity)

dt = 0.5                      # Integration time step (min)
t_obs = obs_data["time_min"].values
F_obs = obs_data["F_obs_M"].values
tR = V / Q                    # Residence time = 5800 min


def system_derivatives(Y):
    """
    Computes dy/dt vector for the coupled mass balance system:
    Y = [C_F, C_Ca, C_CO3, q_F, S_calcite]
    """
    C_F, C_Ca, C_CO3, q_F, S_calcite = Y
    
    # 1. Calcite Dissolution Rate
    IAP = C_Ca * C_CO3
    Omega = IAP / Ksp_calcite
    if Omega < 1.0 and S_calcite > 0:
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega)
    else:
        r_diss = 0.0
        
    # 2. Fluoride Adsorption Rate
    r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
    
    # Differential Equations
    dC_F_dt = (Q / V) * (C_in_F - C_F) - r_ad_F
    dC_Ca_dt = (Q / V) * (C_in_Ca - C_Ca) + r_diss
    dC_CO3_dt = (Q / V) * (C_in_CO3 - C_CO3) + r_diss
    dq_F_dt = r_ad_F / max(1e-6, S_calcite)
    dS_calcite_dt = -r_diss * MW_calcite / 1000.0
    
    return np.array([dC_F_dt, dC_Ca_dt, dC_CO3_dt, dq_F_dt, dS_calcite_dt])


def run_rk4_simulation():
    print("Running 4th-Order Runge-Kutta (RK4) coupled mass balance solver...")
    
    t_end = t_obs[-1]
    steps = int(t_end / dt) + 1
    t_grid = np.linspace(0, t_end, steps)
    
    # Array to store solution [C_F, C_Ca, C_CO3, q_F, S_calcite]
    Y = np.zeros((steps, 5))
    Y[0] = [0.0, C_in_Ca, C_in_CO3, 0.0, S_calcite_0]
    
    # RK4 Integration Loop
    for i in range(steps - 1):
        y_curr = Y[i]
        
        k1 = system_derivatives(y_curr)
        k2 = system_derivatives(y_curr + 0.5 * dt * k1)
        k3 = system_derivatives(y_curr + 0.5 * dt * k2)
        k4 = system_derivatives(y_curr + dt * k3)
        
        Y[i + 1] = y_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    C_F_arr = Y[:, 0]
    C_Ca_arr = Y[:, 1]
    
    # Interpolate predicted values at exact observed points
    C_F_pred_obs = np.interp(t_obs, t_grid, C_F_arr)
    C_Ca_pred_obs = np.interp(t_obs, t_grid, C_Ca_arr)
    
    # --- 3. Plotting Dual Axis Figure ---
    fig, ax1 = plt.subplots(figsize=(9.5, 5.8), dpi=150)

    # Primary Y-Axis: Fluoride
    ax1.set_xlabel('Dimensionless Time ($t / t_R$)', fontsize=12)
    ax1.set_ylabel('Normalized Fluoride ($C / C_{in}$)', color='darkblue', fontsize=12)
    line1 = ax1.plot(t_obs / tR, F_obs / C_in_F, 'ro', markersize=5, label='Experimental Fluoride Data ($C/C_{in}$)')
    line2 = ax1.plot(t_grid / tR, C_F_arr / C_in_F, 'b-', linewidth=2, label='Fluoride Adsorption (RK4 Fit)')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Secondary Y-Axis: Calcium / Dissolution
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Dissolved Calcium from Calcite ($C_{Ca}$, mM)', color='darkred', fontsize=12)
    line3 = ax2.plot(t_grid / tR, C_Ca_arr * 1000, 'r--', linewidth=2, label='Calcite Dissolution (RK4 $C_{Ca}$)')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(-0.005, max(C_Ca_arr * 1000) * 1.2)

    # Combine Legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10.5, framealpha=0.9)

    plt.title('RK4 Numerical Scheme: Coupled Calcite Dissolution & Fluoride Adsorption Fit\n(Q = 0.01 mL/min, $C_{in,F}$ = 0.21 mM, $t_R$ = 5800 min)', fontsize=12)
    fig.tight_layout()
    plt.savefig('rk4_coupled_model_fit.png')
    print("Saved 'rk4_coupled_model_fit.png'")

    # --- 4. Update README.md for GitHub ---
    readme_content = f"""# Fluorapatite Mass Balance & Adsorption Kinetics

This repository runs the coupled mass balance kinetic model using a **4th-Order Runge-Kutta (RK4)** numerical integration scheme.

## RK4 Model Fit with Experimental Points & Calcite Dissolution

![RK4 Model Fit](rk4_coupled_model_fit.png)

## Numerical Scheme & Model Parameters

| Property | Value | Description |
| :--- | :--- | :--- |
| **Numerical Integration Scheme** | **4th-Order Runge-Kutta (RK4)** | $\mathcal{{O}}(\Delta t^4)$ local truncation error accuracy |
| **Adsorption Rate Constant ($k_{{\\text{{F, ad}}}}$)** | **{k_F_ad:.3f} $\\text{{L}} \\cdot \\text{{mol}}^{{-1}} \\cdot \\text{{min}}^{{-1}}$** | Fitted second-order adsorption rate constant |
| **Max Capacity ($q_{{\\text{{max, F}}}}$)** | **{q_max_F:.3e} $\\text{{mol}} \\cdot \\text{{g}}^{{-1}}$** | Maximum surface site concentration |
| **Calcite Dissolution Rate ($k_{{\\text{{calcite}}}}$)** | **$10^{{-5.90}} \\text{{mol}} \\cdot \\text{{m}}^{{-2}} \\cdot \\text{{min}}^{{-1}}$** | Surface normalized dissolution rate constant |
| **Hydraulic Residence Time ($t_R$)** | **5800 minutes** | $V / Q$ (~96.7 hours) |

## Summary Data Table (First 10 Points)

| Time (min) | $t/t_R$ | $F_{{\\text{{obs}}}}$ (mM) | $F_{{\\text{{pred, RK4}}}}$ (mM) | $C/C_{{\\text{{in, obs}}}}$ | $C/C_{{\\text{{in, pred}}}}$ | $C_{{\\text{{Ca, RK4}}}}$ (mM) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for i in range(10):
        t_m = t_obs[i]
        t_tr = t_m / tR
        f_o = F_obs[i] * 1000
        f_p = C_F_pred_obs[i] * 1000
        c_o = F_obs[i] / C_in_F
        c_p = C_F_pred_obs[i] / C_in_F
        ca_p = C_Ca_pred_obs[i] * 1000
        readme_content += f"| {t_m} | {t_tr:.4f} | {f_o:.3f} | {f_p:.3f} | {c_o:.4f} | {c_p:.4f} | {ca_p:.4f} |\n"

    with open("README.md", "w") as f:
        f.write(readme_content)
    print("Updated README.md successfully!")

if __name__ == "__main__":
    run_rk4_simulation()