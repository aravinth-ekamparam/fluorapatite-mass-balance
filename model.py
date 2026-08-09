import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Load Experimental Datasets ---
excel_file = 'CF_Exp data.xlsx'
df_raw = pd.read_excel(excel_file)

# Parse C+F Conditions
col_pairs_cf = [(1, 2), (3, 4), (5, 6), (7, 8)]
q_vals_cf = [0.01, 0.04, 0.03, 0.04]  # mL/min

cf_conditions = []
for idx, (t_col, f_col) in enumerate(col_pairs_cf):
    q_val = q_vals_cf[idx]
    fin_ppm = float(df_raw.iloc[0, t_col])
    data_sub = df_raw.iloc[3:, [t_col, f_col]].dropna().copy()
    data_sub.columns = ['Time_min', 'F_M']
    data_sub['Time_min'] = pd.to_numeric(data_sub['Time_min'], errors='coerce')
    data_sub['F_M'] = pd.to_numeric(data_sub['F_M'], errors='coerce')
    data_sub = data_sub.dropna()
    
    cf_conditions.append({
        'Q_L_min': q_val / 1000.0,
        'Fin_ppm': fin_ppm,
        'Fin_M': (fin_ppm / 1000.0) / 18.9984,
        't_obs': data_sub['Time_min'].values.astype(np.float64),
        'f_obs': data_sub['F_M'].values.astype(np.float64)
    })

# C+P Datasets (Pin = 1.0 mM, Fin = 0)
t_cp_1 = np.array([450, 750, 1050, 1350, 1650, 1950, 2250, 2550, 2850, 3150, 3450, 3750, 4050, 4350, 4650, 4950, 5250, 5550, 5850, 6150, 6450, 6750, 7050, 7350, 7650, 7950, 8250], dtype=np.float64)
p_cp_1 = np.array([9.91E-06, 3.07E-05, 4.96E-05, 6.12E-05, 6.44E-05, 6.46E-05, 6.55E-05, 6.38E-05, 6.52E-05, 6.75E-05, 7.26E-05, 7.1E-05, 6.98E-05, 7.12E-05, 7.2E-05, 7.85E-05, 8.25E-05, 8.23E-05, 8.26E-05, 8.42E-05, 8.63E-05, 8.83E-05, 8.81E-05, 8.83E-05, 8.68E-05, 9.37E-05, 9.37E-05], dtype=np.float64)

t_cp_2 = np.array([120, 360, 600, 840, 1080, 1320, 1560, 1800, 2040, 2280, 2520, 2760], dtype=np.float64)
p_cp_2 = np.array([0.000102, 0.000195, 0.000217, 0.000179, 0.000166, 0.000167, 0.000177, 0.000189, 0.0002, 0.000217, 0.000202, 0.000203], dtype=np.float64)

t_cp_3 = np.array([269.2597, 448.7516, 628.2581, 807.7645, 987.271, 1166.777, 1346.269, 1525.776, 1705.282, 1884.789, 2064.295, 2243.787, 2423.294, 2602.8, 2782.306, 2961.813], dtype=np.float64)
p_cp_3 = np.array([3.52E-05, 0.000168, 0.000273, 0.000344, 0.000363, 0.000345, 0.000318, 0.000311, 0.000297, 0.000304, 0.000319, 0.000333, 0.000342, 0.000345, 0.000142, 0.000338], dtype=np.float64)

cp_conditions = [
    {'Q_L_min': 0.01/1000.0, 'Pin_M': 0.001, 't_obs': t_cp_1, 'p_obs': p_cp_1},
    {'Q_L_min': 0.03/1000.0, 'Pin_M': 0.001, 't_obs': t_cp_2, 'p_obs': p_cp_2},
    {'Q_L_min': 0.04/1000.0, 'Pin_M': 0.001, 't_obs': t_cp_3, 'p_obs': p_cp_3}
]

# --- 2. System Constants ---
V = 0.058             # Reactor volume (L)
S_calcite_0 = 4.0     # g/L
SSA_calcite = 3.0     # m^2/g
MW_calcite = 100.09   # g/mol
k_diss_calcite = 10**(-5.90)
Ksp_calcite = 10**(-8.48)
Ksp_HA = 10**(-58.33)
dt = 2.0


# --- 3. System 1 (C+F) RK4 Simulator ---
def sim_cf(params, cond):
    log_k_ad, log_q_max = params
    k_F_ad = 10**log_k_ad
    q_max_F = 10**log_q_max
    Q = cond['Q_L_min']
    fin_M = cond['Fin_M']
    t_obs = cond['t_obs']
    
    t_end = float(t_obs.max())
    steps = int(t_end / dt) + 1
    t_grid = np.linspace(0, t_end, steps)
    
    Y = np.zeros((steps, 5))
    Y[0] = [0.0, 1.0e-5, 1.0e-5, 0.0, S_calcite_0]
    
    for i in range(steps - 1):
        C_F, C_Ca, C_CO3, q_F, S_calcite = Y[i]
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        
        k1 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])
        
        y2 = Y[i] + 0.5 * dt * k1
        C_F, C_Ca, C_CO3, q_F, S_calcite = y2
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k2 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])
        
        y3 = Y[i] + 0.5 * dt * k2
        C_F, C_Ca, C_CO3, q_F, S_calcite = y3
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k3 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])

        y4 = Y[i] + dt * k3
        C_F, C_Ca, C_CO3, q_F, S_calcite = y4
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k4 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])

        Y[i + 1] = Y[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return np.interp(t_obs, t_grid, Y[:, 0])


# --- 4. System 2 (C+P) RK4 Simulator with Corrected Critical Length Scaling ---
def sim_cp_crit_length(params, cond):
    log_k_prec, Omega_star_HA, log_k_nuc = params
    k_prec = 10**log_k_prec
    k_nuc = 10**log_k_nuc
    
    Q = cond['Q_L_min']
    pin_M = cond['Pin_M']
    t_obs = cond['t_obs']
    
    t_end = float(t_obs.max())
    steps = int(t_end / dt) + 1
    t_grid = np.linspace(0, t_end, steps)
    
    # State Y: [C_P, C_Ca, C_CO3, L_nuc, S_calcite]
    Y = np.zeros((steps, 5))
    Y[0] = [0.0, 1.0e-5, 1.0e-5, 0.0, S_calcite_0]
    
    for i in range(steps - 1):
        C_P, C_Ca, C_CO3, L_nuc, S_calcite = Y[i]
        
        IAP_calc = max(1e-12, C_Ca * C_CO3)
        Omega_calc = IAP_calc / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * max(0.0, 1.0 - Omega_calc) if S_calcite > 0 else 0.0
        
        if C_Ca > 0 and C_P > 0:
            log_IAP_HA = 5.0 * np.log10(max(1e-12, C_Ca)) + 3.0 * np.log10(max(1e-12, C_P))
            log_Omega_HA = (log_IAP_HA - (-58.33)) / 8.0
            Omega_HA = 10**min(3.0, log_Omega_HA)
        else:
            Omega_HA = 0.0
            
        # Nucleus size growth
        dL_dt = k_nuc * max(0.0, Omega_HA - Omega_star_HA)
        
        # Smooth growth factor (L_nuc / (1 + L_nuc))
        growth_activation = L_nuc / (1.0 + L_nuc) if L_nuc > 0 else 0.0
        r_prec_HA = k_prec * SSA_calcite * max(0.0, S_calcite) * max(0.0, Omega_HA - Omega_star_HA) * growth_activation
        
        dC_P_dt = (Q/V)*(pin_M - C_P) - 3.0 * r_prec_HA
        dC_Ca_dt = (Q/V)*(1e-5 - C_Ca) + r_diss - 5.0 * r_prec_HA
        dC_CO3_dt = (Q/V)*(1e-5 - C_CO3) + r_diss
        dL_nuc_dt = dL_dt
        dS_calcite_dt = -r_diss * MW_calcite / 1000.0
        
        k1 = np.array([dC_P_dt, dC_Ca_dt, dC_CO3_dt, dL_nuc_dt, dS_calcite_dt])
        
        y2 = Y[i] + 0.5 * dt * k1
        C_P, C_Ca, C_CO3, L_nuc, S_calcite = y2
        IAP_calc = max(1e-12, C_Ca * C_CO3)
        Omega_calc = IAP_calc / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * max(0.0, 1.0 - Omega_calc) if S_calcite > 0 else 0.0
        if C_Ca > 0 and C_P > 0:
            log_IAP_HA = 5.0 * np.log10(max(1e-12, C_Ca)) + 3.0 * np.log10(max(1e-12, C_P))
            log_Omega_HA = (log_IAP_HA - (-58.33)) / 8.0
            Omega_HA = 10**min(3.0, log_Omega_HA)
        else:
            Omega_HA = 0.0
        dL_dt = k_nuc * max(0.0, Omega_HA - Omega_star_HA)
        growth_activation = L_nuc / (1.0 + L_nuc) if L_nuc > 0 else 0.0
        r_prec_HA = k_prec * SSA_calcite * max(0.0, S_calcite) * max(0.0, Omega_HA - Omega_star_HA) * growth_activation
        k2 = np.array([(Q/V)*(pin_M-C_P)-3*r_prec_HA, (Q/V)*(1e-5-C_Ca)+r_diss-5*r_prec_HA, (Q/V)*(1e-5-C_CO3)+r_diss, dL_dt, -r_diss*MW_calcite/1000.0])

        y3 = Y[i] + 0.5 * dt * k2
        C_P, C_Ca, C_CO3, L_nuc, S_calcite = y3
        IAP_calc = max(1e-12, C_Ca * C_CO3)
        Omega_calc = IAP_calc / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * max(0.0, 1.0 - Omega_calc) if S_calcite > 0 else 0.0
        if C_Ca > 0 and C_P > 0:
            log_IAP_HA = 5.0 * np.log10(max(1e-12, C_Ca)) + 3.0 * np.log10(max(1e-12, C_P))
            log_Omega_HA = (log_IAP_HA - (-58.33)) / 8.0
            Omega_HA = 10**min(3.0, log_Omega_HA)
        else:
            Omega_HA = 0.0
        dL_dt = k_nuc * max(0.0, Omega_HA - Omega_star_HA)
        growth_activation = L_nuc / (1.0 + L_nuc) if L_nuc > 0 else 0.0
        r_prec_HA = k_prec * SSA_calcite * max(0.0, S_calcite) * max(0.0, Omega_HA - Omega_star_HA) * growth_activation
        k3 = np.array([(Q/V)*(pin_M-C_P)-3*r_prec_HA, (Q/V)*(1e-5-C_Ca)+r_diss-5*r_prec_HA, (Q/V)*(1e-5-C_CO3)+r_diss, dL_dt, -r_diss*MW_calcite/1000.0])

        y4 = Y[i] + dt * k3
        C_P, C_Ca, C_CO3, L_nuc, S_calcite = y4
        IAP_calc = max(1e-12, C_Ca * C_CO3)
        Omega_calc = IAP_calc / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * max(0.0, 1.0 - Omega_calc) if S_calcite > 0 else 0.0
        if C_Ca > 0 and C_P > 0:
            log_IAP_HA = 5.0 * np.log10(max(1e-12, C_Ca)) + 3.0 * np.log10(max(1e-12, C_P))
            log_Omega_HA = (log_IAP_HA - (-58.33)) / 8.0
            Omega_HA = 10**min(3.0, log_Omega_HA)
        else:
            Omega_HA = 0.0
        dL_dt = k_nuc * max(0.0, Omega_HA - Omega_star_HA)
        growth_activation = L_nuc / (1.0 + L_nuc) if L_nuc > 0 else 0.0
        r_prec_HA = k_prec * SSA_calcite * max(0.0, S_calcite) * max(0.0, Omega_HA - Omega_star_HA) * growth_activation
        k4 = np.array([(Q/V)*(pin_M-C_P)-3*r_prec_HA, (Q/V)*(1e-5-C_Ca)+r_diss-5*r_prec_HA, (Q/V)*(1e-5-C_CO3)+r_diss, dL_dt, -r_diss*MW_calcite/1000.0])

        Y[i + 1] = Y[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return np.interp(t_obs, t_grid, Y[:, 0])


# --- 5. Objectives ---
def obj_cf(params):
    sse = 0.0
    for cond in cf_conditions:
        pred = sim_cf(params, cond)
        sse += np.sum(((cond['f_obs'] - pred) / cond['Fin_M'])**2)
    return sse

def obj_cp_crit_length(params):
    sse = 0.0
    for cond in cp_conditions:
        pred = sim_cp_crit_length(params, cond)
        sse += np.sum(((cond['p_obs'] - pred) / cond['Pin_M'])**2)
    return sse


def main():
    print("Fitting System 1 (Calcite + Fluoride Adsorption)...")
    res_cf = minimize(obj_cf, x0=[0.5, -4.0], method='Nelder-Mead')
    k_F_ad_opt = 10**res_cf.x[0]
    q_max_F_opt = 10**res_cf.x[1]

    print("Fitting System 2 (Calcite + Phosphate via Nucleation Growth & Critical Length Scaling)...")
    res_cp = minimize(obj_cp_crit_length, x0=[-10.5, 1.05, -3.0], method='Nelder-Mead')
    k_prec_opt = 10**res_cp.x[0]
    Omega_star_HA_opt = res_cp.x[1]
    k_nuc_opt = 10**res_cp.x[2]

    print("\n================ SYSTEM 1 (C+F) FITTED KINETICS ================")
    print(f"k_F_ad  = {k_F_ad_opt:.4f} L/(mol*min)")
    print(f"q_max_F = {q_max_F_opt:.4e} mol/g ({q_max_F_opt*1e6:.2f} umol/g)")

    print("\n================ SYSTEM 2 (C+P) FITTED KINETICS ================")
    print(f"k_prec (Precipitation Rate Constant) = {k_prec_opt:.4e} mol/m2/min")
    print(f"Omega*_HA (Critical Supersaturation) = {Omega_star_HA_opt:.4f}")
    print(f"k_nuc (Nucleus Growth Rate Constant)  = {k_nuc_opt:.4e} min^-1")
    print("===============================================================\n")

    # --- 6. Plotting Two Systems Comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)

    # C+F Plots
    for idx in range(3):
        ax = axes[0, idx]
        cond = cf_conditions[idx]
        t_obs = cond['t_obs']
        f_obs = cond['f_obs']
        fin_m = cond['Fin_M']
        q_ml = cond['Q_L_min'] * 1000.0
        tR_val = V / cond['Q_L_min']
        
        t_fine = np.linspace(0, t_obs.max(), 300)
        f_pred = sim_cf(res_cf.x, cond)
        tracer = 1.0 - np.exp(-t_fine / tR_val)
        
        ax.plot(t_obs / tR_val, f_obs / fin_m, 'ro', markersize=5, label=r'Experimental ($C_{\mathcal{F}}/C_{in}$)')
        ax.plot(t_obs / tR_val, f_pred / fin_m, 'b-', linewidth=2, label=f'C+F Model ($k_{{ad}}$={k_F_ad_opt:.2f})')
        ax.plot(t_fine / tR_val, tracer, 'g--', linewidth=1.5, label='Ideal Non-Reactive Tracer')
        
        ax.set_xlabel('Dimensionless Time ($t / t_R$)', fontsize=10)
        ax.set_ylabel(r'Normalized Fluoride ($C_{\mathcal{F}} / C_{in}$)', fontsize=10)
        ax.set_title(f"System 1 (C+F): Q={q_ml} mL/min, $F_{{in}}$={fin_m*1000*18.9984:.1f} ppm", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8.5, loc='lower right')

    # C+P Plots
    for idx in range(3):
        ax = axes[1, idx]
        cond = cp_conditions[idx]
        t_obs = cond['t_obs']
        p_obs = cond['p_obs']
        pin_m = cond['Pin_M']
        q_ml = cond['Q_L_min'] * 1000.0
        tR_val = V / cond['Q_L_min']
        
        t_fine = np.linspace(0, t_obs.max(), 300)
        p_pred = sim_cp_crit_length(res_cp.x, cond)
        tracer = 1.0 - np.exp(-t_fine / tR_val)
        
        ax.plot(t_obs / tR_val, p_obs / pin_m, 'mo', markersize=5, label=r'Experimental ($C_P/C_{in}$)')
        ax.plot(t_obs / tR_val, p_pred / pin_m, 'k-', linewidth=2, label=r'Critical Length Growth Fit' + f'\n($\Omega^*={Omega_star_HA_opt:.2f}$)')
        ax.plot(t_fine / tR_val, tracer, 'g--', linewidth=1.5, label='Ideal Non-Reactive Tracer')
        
        ax.set_xlabel('Dimensionless Time ($t / t_R$)', fontsize=10)
        ax.set_ylabel(r'Normalized Phosphate ($C_P / C_{in}$)', fontsize=10)
        ax.set_title(f"System 2 (C+P): Q={q_ml} mL/min, $P_{{in}}$=1.0 mM", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8.5, loc='upper left')

    plt.suptitle(r'Isolated Systems: System 1 (Calcite + Fluoride Adsorption) & System 2 (Calcite + Phosphate Critical Length $L_{\mathrm{crit}}$ Nucleation Growth)', fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig('two_systems_comparison.png')
    print("Saved plot as 'two_systems_comparison.png'")

    # --- 7. Update README.md ---
    readme_content = f"""# Fluorapatite Mass Balance: System 1 (C+F) vs. System 2 (C+P)

This repository models isolated single-solute continuous flow-stirred tank reactor (CFSTR) dynamics:
1. **System 1 (Calcite + Fluoride):** Decoupled fluoride adsorption and calcite dissolution.
2. **System 2 (Calcite + Phosphate):** Classical Nucleation Theory (CNT) using Critical Nucleus Length ($L_{{\\text{{crit}}}}$) and Critical Supersaturation Ratio ($\Omega^*$).

## Two Systems Model Comparison

![Two Systems Comparison](two_systems_comparison.png)

## Fitted Kinetic Parameters

| System | Process / Mechanism | Parameter | Fitted Value | Unit |
| :--- | :--- | :---: | :---: | :--- |
| **System 1 (C+F)** | Fluoride Adsorption Rate | $k_{{\\text{{F, ad}}}}$ | **{k_F_ad_opt:.4f}** | $\\text{{L}} \\cdot \\text{{mol}}^{{-1}} \\cdot \\text{{min}}^{{-1}}$ |
| **System 1 (C+F)** | Fluoride Surface Capacity | $q_{{\\text{{max, F}}}}$ | **{q_max_F_opt:.4e}** | $\\text{{mol}} \\cdot \\text{{g}}^{{-1}}$ |
| **System 2 (C+P)** | Precipitation Rate Constant | $k_{{\\text{{prec}}}}$ | **{k_prec_opt:.4e}** | $\\text{{mol}} \\cdot \\text{{m}}^{{-2}} \\cdot \\text{{min}}^{{-1}}$ |
| **System 2 (C+P)** | HA Critical Supersaturation | $\\Omega^*_{{\\text{{HA}}}}$ | **{Omega_star_HA_opt:.4f}** | — |
| **System 2 (C+P)** | Nucleus Growth Rate Constant | $k_{{\\text{{nuc}}}}$ | **{k_nuc_opt:.4e}** | $\\text{{min}}^{{-1}}$ |
"""
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("Updated README.md successfully!")

if __name__ == "__main__":
    main()