import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import emcee
import corner

# ==============================================================================
# 1. EXPERIMENTAL CONDITIONS FROM EXCEL (CF_Exp data.xlsx)
# ==============================================================================
CONDITIONS = [
    {"name": "Q = 0.01 mL/min, Fin = 2 mg/L", "col_t": 1, "col_f": 2, "id": "cond1"},
    {"name": "Q = 0.04 mL/min, Fin = 4 mg/L", "col_t": 3, "col_f": 4, "id": "cond2"},
    {"name": "Q = 0.03 mL/min, Fin = 8 mg/L", "col_t": 5, "col_f": 6, "id": "cond3"},
    {"name": "Q = 0.04 mL/min, Fin = 2 mg/L", "col_t": 7, "col_f": 8, "id": "cond4"}
]

excel_file = "CF_Exp data.xlsx"
df = pd.read_excel(excel_file, header=None)

# ==============================================================================
# 2. CFSTR BASE CONFIGURATION (Thesis Section 5.2)
# ==============================================================================
BASE_CONFIG = {
    "V": 0.058,              # Reactor Volume (L) [58 mL]
    "Ccal_0": 4.0,           # Calcite loading (g/L)
    "SSA_cal": 3.0,          # Specific surface area (m2/g)
    "MW_cal": 100.0869,      # g/mol
    "Ksp_cal": 10**(-8.48),
    "gamma_Ca": 0.87,        # Activity coefficient at I = 1 mM
    "gamma_CO3": 0.90,
    "gamma_F": 0.87,
    "alpha0": 0.0967,
    "kgl": 10**(-2.145),     # CO2 mass transfer rate (min^-1)
    "CO3_sat": 10**(-5.0)
}

# ==============================================================================
# 3. ODE MASS BALANCES
# ==============================================================================
def cfstr_cf_odes(t, y, params, config):
    F, Ca, CO3, Ccal, Fs = y
    Q = config["Q"]
    Fin = config["Fin"]
    kF = 10**params["log10_kF"]      # min^-1
    kcal = 10**params["log10_kcal"]  # mol/(m2*min)
    
    # 1. Langmuir Adsorption (Thesis Eq. 5.35)
    # F in M; convert to uM for isotherm: F_uM = F * 1e6
    F_uM = max(0.0, F * 1e6)
    Fs_star = (625.0 * F_uM) / (1.0 + 0.061 * F_uM)
    dFs_dt = kF * (Fs_star - Fs)
    
    # 2. Calcite Dissolution
    IAP_cal = (config["gamma_Ca"] * max(0.0, Ca)) * (config["gamma_CO3"] * max(0.0, CO3))
    f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
    r_diss_cal = -kcal * config["SSA_cal"] * max(0.0, Ccal) * f_G_cal
    
    # 3. Liquid & Solid Mass Balances
    dF_dt = (Q / config["V"]) * (Fin - F) - max(0.0, Ccal) * dFs_dt * 1e-6
    dCa_dt = -(Q / config["V"]) * Ca + r_diss_cal
    r_gas = config["kgl"] * ((config["CO3_sat"] / config["alpha0"]) - CO3)
    dCO3_dt = -(Q / config["V"]) * CO3 + r_diss_cal + r_gas
    dCcal_dt = -r_diss_cal * config["MW_cal"]
    
    return [dF_dt, dCa_dt, dCO3_dt, dCcal_dt, dFs_dt]

def run_forward_model(params, time_points, config):
    y0 = [0.0, 10**(-3.69), 10**(-5.0), config["Ccal_0"], 0.0]
    sol = solve_ivp(
        fun=lambda t, y: cfstr_cf_odes(t, y, params, config),
        t_span=[0.0, max(time_points)],
        y0=y0,
        t_eval=time_points,
        method="BDF",
        rtol=1e-5,
        atol=1e-8
    )
    return sol.y[0]

# ==============================================================================
# 4. BAYESIAN LOG-PRIOR & LOG-LIKELIHOOD
# ==============================================================================
def log_prior(theta):
    log10_kF, log10_kcal, log_sigma_F = theta
    if not (-9.0 <= log10_kF <= -4.0):
        return -np.inf
    if not (-7.0 <= log10_kcal <= -5.0):
        return -np.inf
    if not (-8.0 <= log_sigma_F <= -3.0):
        return -np.inf
    # Informative prior on kcal from independent batch dissolution test: N(-5.90, 0.15)
    return -0.5 * ((log10_kcal - (-5.90)) / 0.15)**2

def log_likelihood(theta, exp_data, config):
    log10_kF, log10_kcal, log_sigma_F = theta
    sigma_F = 10**log_sigma_F
    params = {"log10_kF": log10_kF, "log10_kcal": log10_kcal}
    try:
        F_sim = run_forward_model(params, exp_data["t_min"], config)
    except Exception:
        return -np.inf
    res = exp_data["F_meas"] - F_sim
    n = len(exp_data["t_min"])
    return -0.5 * np.sum((res / sigma_F)**2) - n * np.log(sigma_F * np.sqrt(2 * np.pi))

def log_posterior(theta, exp_data, config):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, exp_data, config)

# ==============================================================================
# 5. EXECUTION ACROSS ALL FOUR CONDITIONS
# ==============================================================================
if __name__ == "__main__":
    results = {}
    ndim = 3
    nwalkers = 16
    n_burnin = 200
    n_steps = 400
    
    # Setup 2x2 grid for the 4 conditions
    fig_all, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes_flat = axes.flatten()
    
    for idx, cond in enumerate(CONDITIONS):
        print(f"\n=======================================================")
        print(f"RUNNING CONDITION {idx+1}: {cond['name']}")
        print(f"=======================================================")
        
        # Parse experimental time and Fluoride
        t_raw = pd.to_numeric(df.iloc[4:, cond["col_t"]], errors="coerce")
        f_raw = pd.to_numeric(df.iloc[4:, cond["col_f"]], errors="coerce")
        mask = t_raw.notna() & f_raw.notna()
        
        t_data = t_raw[mask].values
        f_data = f_raw[mask].values
        
        Q_ml = float(df.iloc[0, cond["col_t"]])
        Fin_mg = float(df.iloc[1, cond["col_t"]])
        Fin_M = (Fin_mg * 1e-3) / 18.9984
        
        cfg = BASE_CONFIG.copy()
        cfg["Q"] = Q_ml * 1e-3
        cfg["Fin"] = Fin_M
        
        exp_data = {"t_min": t_data, "F_meas": f_data}
        
        # Initialize walkers
        init_pos = np.array([-7.0, -5.9, -5.0]) + 0.05 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(exp_data, cfg))
        sampler.run_mcmc(init_pos, n_steps + n_burnin, progress=True)
        
        samples = sampler.get_chain(discard=n_burnin, flat=True)
        results[cond["id"]] = {"samples": samples, "cfg": cfg, "data": exp_data, "name": cond["name"]}
        
        # Parameter summary
        kF_p16, kF_p50, kF_p84 = np.percentile(samples[:, 0], [16, 50, 84])
        print(f"Inferred log10(kF): {kF_p50:.3f} (+{kF_p84-kF_p50:.3f} / -{kF_p50-kF_p16:.3f})")
        
        # Save corner plot for this condition
        fig_c = corner.corner(
            samples, 
            labels=[r"$\log_{10}(k_F)$", r"$\log_{10}(k_{\mathrm{cal}})$", r"$\log_{10}(\sigma_F)$"],
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True
        )
        fig_c.savefig(f"case1_corner_{cond['id']}.png", dpi=300)
        plt.close(fig_c)
        
        # Plot on 2x2 subplot
        ax = axes_flat[idx]
        t_fine = np.linspace(0, max(t_data), 150)
        trajectories = []
        for s_idx in np.random.randint(len(samples), size=80):
            th = samples[s_idx]
            f_sim = run_forward_model({"log10_kF": th[0], "log10_kcal": th[1]}, t_fine, cfg)
            trajectories.append(f_sim)
        trajectories = np.array(trajectories)
        
        f_low = np.percentile(trajectories, 2.5, axis=0) * 1e3
        f_med = np.percentile(trajectories, 50.0, axis=0) * 1e3
        f_high = np.percentile(trajectories, 97.5, axis=0) * 1e3
        
        ax.plot(t_data, f_data * 1e3, "ro", ms=4, label="Data")
        ax.plot(t_fine, f_med, "b-", lw=2, label="Median Posterior")
        ax.fill_between(t_fine, f_low, f_high, color="blue", alpha=0.2, label="95% CI")
        ax.axhline(Fin_M * 1e3, color="gray", linestyle=":", label=f"Fin ({Fin_mg} mg/L)")
        ax.set_title(cond["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Effluent Fluoride (mM)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        
    fig_all.tight_layout()
    fig_all.savefig("case1_all_fits_comparison.png", dpi=300)
    print("\nSaved 4-panel comparison -> case1_all_fits_comparison.png")
    
    # ==============================================================================
    # 6. OVERLAID POSTERIOR COMPARISON PLOT FOR kF
    # ==============================================================================
    plt.figure(figsize=(7.5, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, cond in enumerate(CONDITIONS):
        kF_samples = results[cond["id"]]["samples"][:, 0]
        plt.hist(
            kF_samples, 
            bins=25, 
            density=True, 
            alpha=0.40, 
            color=colors[idx], 
            label=cond["name"]
        )
    plt.xlabel(r"$\log_{10}(k_F)$  $[\mathrm{min}^{-1}]$", fontsize=11)
    plt.ylabel("Posterior Probability Density", fontsize=11)
    plt.title("Comparison of Inferred Adsorption Rate Constant Across All 4 Conditions", fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("case1_kF_posteriors_comparison.png", dpi=300)
    print("Saved posterior comparison plot -> case1_kF_posteriors_comparison.png")