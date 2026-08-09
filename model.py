import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Experimental Data ---
excel_file = 'CF_Exp data.xlsx'
df_raw = pd.read_excel(excel_file)

col_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
q_values = [0.01, 0.04, 0.03, 0.04]  # mL/min

conditions = []
for idx, (time_col_idx, f_col_idx) in enumerate(col_pairs):
    q_val = q_values[idx]
    fin_ppm = float(df_raw.iloc[0, time_col_idx])
    
    data_sub = df_raw.iloc[3:, [time_col_idx, f_col_idx]].dropna()
    data_sub.columns = ['Time_min', 'F_M']
    
    conditions.append({
        'id': idx + 1,
        'label': f"Dataset {idx+1}: Q={q_val} mL/min, Fin={fin_ppm} ppm",
        'Q_L_min': q_val / 1000.0,
        'Fin_ppm': fin_ppm,
        'Fin_M': (fin_ppm / 1000.0) / 18.9984,
        'data': data_sub
    })

# --- 2. Constants ---
V = 0.058             # L
S_calcite_0 = 4.0     # g/L
SSA_calcite = 3.0     # m^2/g
MW_calcite = 100.09   # g/mol
k_diss_calcite = 10**(-5.90)
Ksp_calcite = 10**(-8.48)
dt = 5.0              # Fast integration step (min)


# --- 3. 2nd-Order RK4 Forward Simulator ---
def rk4_solve(q_L_min, fin_M, t_obs, log_k_ad, log_q_max):
    k_F_ad = 10**log_k_ad      # 2nd-order rate constant (L/mol/min)
    q_max_F = 10**log_q_max    # Capacity (mol/g)
    Q = q_L_min
    
    t_end = t_obs.max()
    steps = int(t_end / dt) + 1
    t_grid = np.linspace(0, t_end, steps)
    
    Y = np.zeros((steps, 5))
    Y[0] = [0.0, 1.0e-5, 1.0e-5, 0.0, S_calcite_0]
    
    for i in range(steps - 1):
        C_F, C_Ca, C_CO3, q_F, S_calcite = Y[i]
        
        # k1
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        # 2nd-order adsorption rate law
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k1 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])
        
        # k2
        y2 = Y[i] + 0.5 * dt * k1
        C_F, C_Ca, C_CO3, q_F, S_calcite = y2
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k2 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])
        
        # k3
        y3 = Y[i] + 0.5 * dt * k2
        C_F, C_Ca, C_CO3, q_F, S_calcite = y3
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k3 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])

        # k4
        y4 = Y[i] + dt * k3
        C_F, C_Ca, C_CO3, q_F, S_calcite = y4
        IAP = C_Ca * C_CO3
        Omega = IAP / Ksp_calcite
        r_diss = S_calcite * SSA_calcite * k_diss_calcite * (1.0 - Omega) if (Omega < 1.0 and S_calcite > 0) else 0.0
        r_ad_F = S_calcite * k_F_ad * C_F * max(0.0, q_max_F - q_F)
        k4 = np.array([(Q/V)*(fin_M-C_F)-r_ad_F, (Q/V)*(1e-5-C_Ca)+r_diss, (Q/V)*(1e-5-C_CO3)+r_diss, r_ad_F/max(1e-6, S_calcite), -r_diss*MW_calcite/1000.0])

        Y[i + 1] = Y[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return np.interp(t_obs, t_grid, Y[:, 0])


# --- 4. Global Bayesian Log-Likelihood Function ---
def log_posterior(theta):
    log_k_ad, log_q_max, log_sigma = theta
    if not (-2.0 < log_k_ad < 2.0 and -6.0 < log_q_max < -2.0 and -5.0 < log_sigma < -1.0):
        return -np.inf
    
    sigma = 10**log_sigma
    total_log_like = 0.0
    
    for cond in conditions:
        t_obs = cond['data']['Time_min'].values
        f_obs = cond['data']['F_M'].values
        f_pred = rk4_solve(cond['Q_L_min'], cond['Fin_M'], t_obs, log_k_ad, log_q_max)
        
        # Normalized error term across conditions
        norm_res = (f_obs - f_pred) / cond['Fin_M']
        total_log_like += -0.5 * np.sum((norm_res / sigma)**2) - len(f_obs) * np.log(sigma * np.sqrt(2 * np.pi))
        
    return total_log_like


# --- 5. Bayesian MCMC Sampler ---
def run_multi_condition_mcmc(n_samples=1000):
    print("Running Multi-Condition Bayesian MCMC Sampling...")
    start_t = time.time()
    
    current_theta = np.array([-0.86, -4.36, -1.8])  # Starting guess from global optimization
    current_log_post = log_posterior(current_theta)
    
    samples = []
    proposal_std = np.array([0.015, 0.015, 0.015])
    
    for i in range(n_samples):
        proposal = current_theta + np.random.normal(0, proposal_std)
        prop_log_post = log_posterior(proposal)
        
        if np.log(np.random.rand()) < (prop_log_post - current_log_post):
            current_theta = proposal
            current_log_post = prop_log_post
            
        samples.append(current_theta)
        
    chain = np.array(samples)[int(n_samples * 0.3):]  # Apply 30% burn-in
    
    k_ad_mcmc = 10**chain[:, 0]
    q_max_mcmc = 10**chain[:, 1]
    
    print(f"\nCompleted in {time.time() - start_t:.1f} seconds.")
    print("=== BAYESIAN POSTERIOR ESTIMATES (95% Credible Intervals) ===")
    print(f"k_F_ad  : {np.median(k_ad_mcmc):.3f} L/(mol*min)  (95% CI: [{np.percentile(k_ad_mcmc, 2.5):.3f}, {np.percentile(k_ad_mcmc, 97.5):.3f}])")
    print(f"q_max_F : {np.median(q_max_mcmc):.3e} mol/g       (95% CI: [{np.percentile(q_max_mcmc, 2.5):.3e}, {np.percentile(q_max_mcmc, 97.5):.3e}])")
    print("=============================================================\n")

if __name__ == "__main__":
    run_multi_condition_mcmc(n_samples=1000)