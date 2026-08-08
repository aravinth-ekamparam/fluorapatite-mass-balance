import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Experimental Observed Data ---
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

# --- 2. System Constants ---
V = 0.058             # Reactor volume (L) [58 mL]
Q = 0.01 / 1000.0     # Flow rate (L/min) [0.01 mL/min]
S_calcite_0 = 4.0     # Calcite loading (g/L)
C_in_F = 0.21 / 1000.0# Influent F concentration (0.21 mM -> M)
dt = 0.5              # Time step (min)

t_obs = obs_data["time_min"].values
F_obs = obs_data["F_obs_M"].values
tR = V / Q            # Residence time = 5800 minutes


def forward_euler_sim(params):
    """
    Forward Euler integration tracking fluoride uptake in CFSTR reactor.
    """
    log_k_ad, log_q_max = params
    k_F_ad = 10**log_k_ad
    q_max_F = 10**log_q_max
    
    t_end = t_obs[-1]
    steps = int(t_end / dt) + 1
    t_grid = np.linspace(0, t_end, steps)
    
    C_F_arr = np.zeros(steps)
    q_F_arr = np.zeros(steps)
    
    C_F = 0.0
    q_F = 0.0
    
    for i in range(1, steps):
        r_ad_F = S_calcite_0 * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        dC_F_dt = (Q / V) * (C_in_F - C_F) - r_ad_F
        
        C_F += dC_F_dt * dt
        q_F += (r_ad_F / S_calcite_0) * dt
        
        C_F_arr[i] = C_F
        q_F_arr[i] = q_F
        
    return np.interp(t_obs, t_grid, C_F_arr)


def objective(params):
    pred_F = forward_euler_sim(params)
    return np.sum((pred_F - F_obs)**2)


def fit_and_update_github():
    print("Fitting adsorption rate parameters to observed data...")
    
    initial_guess = [0.0, -4.5]
    res = minimize(objective, x0=initial_guess, method='Nelder-Mead')
    
    opt_log_k_ad, opt_log_q_max = res.x
    k_ad_opt = 10**opt_log_k_ad
    q_max_opt = 10**opt_log_q_max
    
    pred_F = forward_euler_sim(res.x)
    
    # 1. Generate and save plot image
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(t_obs / tR, F_obs / C_in_F, 'ro', markersize=5, label='Experimental Data (C+F)')
    plt.plot(t_obs / tR, pred_F / C_in_F, 'b-', linewidth=2, 
             label=f'Fitted Model (k_ad={k_ad_opt:.2f} L/mol/min, q_max={q_max_opt:.2e} mol/g)')
    
    plt.xlabel('Dimensionless Time (t / t_R)', fontsize=12)
    plt.ylabel('Normalized Fluoride (C / C_in)', fontsize=12)
    plt.title('Calcite + Fluoride Adsorption Kinetic Fit\n(Q = 0.01 mL/min, C_in,F = 0.21 mM)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('fluoride_adsorption_fit.png')
    print("Saved 'fluoride_adsorption_fit.png'")

    # 2. Automatically update README.md so results show directly on GitHub
    readme_content = f"""# Fluorapatite Mass Balance & Adsorption Kinetics

This repository runs the mass balance kinetic model for fluoride adsorption onto calcite in a continuous flow-stirred tank reactor (CFSTR).

## Model Fit Result

![Fluoride Adsorption Fit](fluoride_adsorption_fit.png)

## Optimized Kinetic Parameters

| Parameter | Fitted Value | Unit |
| :--- | :--- | :--- |
| **Adsorption Rate Constant ($k_{{\\text{{F, ad}}}}$)** | **{k_ad_opt:.3f}** | $\\text{{L}} \\cdot \\text{{mol}}^{{-1}} \\cdot \\text{{min}}^{{-1}}$ |
| **Max Capacity ($q_{{\\text{{max, F}}}}$)** | **{q_max_opt:.3e}** | $\\text{{mol}} \\cdot \\text{{g}}^{{-1}}$ |
| **$\log_{{10}}(k_{{\\text{{F, ad}}}})$** | **{opt_log_k_ad:.4f}** | — |
| **$\log_{{10}}(q_{{\\text{{max, F}}}})$** | **{opt_log_q_max:.4f}** | — |

## Summary Data Table (First 10 Data Points)

| Time (min) | $t/t_R$ | $F_{{\\text{{obs}}}}$ (mM) | $F_{{\\text{{pred}}}}$ (mM) | $C/C_{{\\text{{in, obs}}}}$ | $C/C_{{\\text{{in, pred}}}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for i in range(10):
        t_m = t_obs[i]
        t_tr = t_m / tR
        f_o = F_obs[i] * 1000
        f_p = pred_F[i] * 1000
        c_o = F_obs[i] / C_in_F
        c_p = pred_F[i] / C_in_F
        readme_content += f"| {t_m} | {t_tr:.4f} | {f_o:.3f} | {f_p:.3f} | {c_o:.4f} | {c_p:.4f} |\n"

    readme_content += "\n\n*Full dataset exported to `cfstr_fitted_adsorption_results.csv`.*"

    with open("README.md", "w") as f:
        f.write(readme_content)
    print("Successfully updated README.md with fitted plot and parameter table!")

if __name__ == "__main__":
    fit_and_update_github()