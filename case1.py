import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import emcee
import corner

# ==============================================================================
# 1. LOAD EXPERIMENTAL DATA FROM EXCEL (CF_Exp data.xlsx)
# ==============================================================================
# Column Mapping in your Excel sheet:
#   Cols (1, 2) -> B & C: Q = 0.01 mL/min, Fin = 2 mg/L (~0.105 mM)
#   Cols (3, 4) -> D & E: Q = 0.04 mL/min, Fin = 4 mg/L (~0.211 mM)
#   Cols (5, 6) -> F & G: Q = 0.03 mL/min, Fin = 8 mg/L (~0.421 mM)
#   Cols (7, 8) -> H & I: Q = 0.04 mL/min, Fin = 2 mg/L (~0.105 mM)

excel_file = "CF_Exp data.xlsx"
df = pd.read_excel(excel_file, header=None)

# Select dataset (e.g., Cols B & C: Q=0.01, Fin=2 mg/L)
COL_TIME = 1  # Col B
COL_F = 2     # Col C

# Parse experimental Time (min) and Effluent Fluoride (M)
t_raw = pd.to_numeric(df.iloc[4:, COL_TIME], errors="coerce")
f_raw = pd.to_numeric(df.iloc[4:, COL_F], errors="coerce")
valid_mask = t_raw.notna() & f_raw.notna()

exp_dataset = {
    "t_min": t_raw[valid_mask].values,
    "F_meas": f_raw[valid_mask].values
}

# Read inflow conditions from header rows
Q_ml_min = float(df.iloc[0, COL_TIME])       # 0.01 mL/min
Fin_mg_L = float(df.iloc[1, COL_TIME])       # 2 mg/L
Fin_M = (Fin_mg_L * 1e-3) / 18.9984          # Convert mg/L to M (MW_F = 18.9984 g/mol)

print(f"Loaded Condition: Q = {Q_ml_min} mL/min | Fin = {Fin_mg_L} mg/L ({Fin_M*1e3:.3f} mM)")
print(f"Total experimental points: {len(exp_dataset['t_min'])}")

# ==============================================================================
# 2. PHYSICAL CONSTANTS & REACTOR CONFIGURATION (Thesis Section 5.2)
# ==============================================================================
REACTOR_CONFIG = {
    "V": 0.058,                     # Volume (L) [58 mL]
    "Q": Q_ml_min * 1e-3,           # Convert mL/min to L/min
    "Fin": Fin_M,                   # Influent F (M)
    "Ccal_0": 4.0,                  # Initial calcite (g/L)
    "SSA_cal": 3.0,                 # Calcite area (m2/g)
    "MW_cal": 100.0869,             # g/mol
    "Ksp_cal": 10**(-8.48),
    "gamma_Ca": 0.87,               # Davies activity coeff at I=1 mM
    "gamma_CO3": 0.90,
    "gamma_F": 0.87,
    "alpha0": 0.0967,
    "kgl": 10**(-2.145),            # CO2 mass transfer (min^-1)
    "CO3_sat": 10**(-5.0)
}

# ==============================================================================
# 3. CFSTR GOVERNING EQUATIONS (ODEs)
# ==============================================================================
def cfstr_cf_odes(t, y, params, config):
    F, Ca, CO3, Ccal, Fs = y
    
    Q = config["Q"]
    Fin = config["Fin"]
    kF = 10**params["log10_kF"]      # min^-1
    kcal = 10**params["log10_kcal"]  # mol/(m2*min)
    
    # 1. Langmuir Adsorption (Thesis Eq. 5.35)
    # F in M; Fs* in umol/g
    F_uM = max(0.0, F * 1e6)
    Fs_star = (625.0 * F_uM) / (1.0 + 0.061 * F_uM)
    dFs_dt = kF * (Fs_star - Fs)
    
    # 2. Calcite Dissolution
    IAP_cal = (config["gamma_Ca"] * max(0.0, Ca)) * (config["gamma_CO3"] * max(0.0, CO3))
    f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
    r_diss_cal = -kcal * config["SSA_cal"] * max(0.0, Ccal) * f_G_cal
    
    # 3. Liquid Mass Balances
    dF_dt = (Q / config["V"]) * (Fin - F) - max(0.0, Ccal) * dFs_dt * 1e-6
    dCa_dt = -(Q / config["V"]) * Ca + r_diss_cal
    r_gas = config["kgl"] * ((config["CO3_sat"] / config["alpha0"]) - CO3)
    dCO3_dt = -(Q / config["V"]) * CO3 + r_diss_cal + r_gas
    dCcal_dt = -r_diss_cal * config["MW_cal"]
    
    return [dF_dt, dCa_dt, dCO3_dt, dCcal_dt, dFs_dt]

def run_forward_model(params, time_points, config):
    # Initial Conditions: F=0, Ca=10^-3.69 M, CO3=10^-5 M, Ccal=4 g/L, Fs=0
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
    return sol.y[0]  # Return simulated F(t)

# ==============================================================================
# 4. BAYESIAN LIKELIHOOD & MCMC SAMPLER
# ==============================================================================
# Parameter vector: theta = [log10_kF, log10_kcal, log_sigma_F]

def log_prior(theta):
    log10_kF, log10_kcal, log_sigma_F = theta
    if not (-8.0 <= log10_kF <= -4.0):
        return -np.inf
    if not (-7.0 <= log10_kcal <= -5.0):
        return -np.inf
    if not (-8.0 <= log_sigma_F <= -4.0):
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
# 5. EXECUTION & INFERENCE
# ==============================================================================
if __name__ == "__main__":
    ndim = 3
    nwalkers = 12
    n_burnin = 100
    n_steps = 300
    
    # Initialize walkers around prior center
    init_pos = np.array([-5.70, -5.90, -5.50]) + 0.05 * np.random.randn(nwalkers, ndim)
    
    print("\nStarting MCMC sampling with emcee...")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(exp_dataset, REACTOR_CONFIG)
    )
    sampler.run_mcmc(init_pos, n_steps + n_burnin, progress=True)
    
    samples = sampler.get_chain(discard=n_burnin, flat=True)
    
    labels = [r"$\log_{10}(k_F)$", r"$\log_{10}(k_{\mathrm{cal}})$", r"$\log_{10}(\sigma_F)$"]
    print("\n=== POSTERIOR ESTIMATES (16th, 50th, 84th percentiles) ===")
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{labels[i]}: {mcmc[1]:.3f} (+{q[1]:.3f} / -{q[0]:.3f})")
        
    # Generate Corner Plot
    fig_corner = corner.corner(samples, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True)
    fig_corner.savefig("case1_corner.png", dpi=300)
    print("\nSaved parameter corner plot -> case1_corner.png")
    
    # Generate Posterior Predictive Check Plot
    t_fine = np.linspace(0, max(exp_dataset["t_min"]), 150)
    trajectories = []
    for idx in np.random.randint(len(samples), size=80):
        th = samples[idx]
        f_sim = run_forward_model({"log10_kF": th[0], "log10_kcal": th[1]}, t_fine, REACTOR_CONFIG)
        trajectories.append(f_sim)
        
    trajectories = np.array(trajectories)
    f_low = np.percentile(trajectories, 2.5, axis=0) * 1e3
    f_med = np.percentile(trajectories, 50.0, axis=0) * 1e3
    f_high = np.percentile(trajectories, 97.5, axis=0) * 1e3
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(exp_dataset["t_min"], exp_dataset["F_meas"] * 1e3, "ro", label="Experimental F Data")
    plt.plot(t_fine, f_med, "b-", lw=2, label="Median Posterior Model")
    plt.fill_between(t_fine, f_low, f_high, color="blue", alpha=0.2, label="95% Credible Interval")
    plt.axhline(REACTOR_CONFIG["Fin"] * 1e3, color="gray", linestyle=":", label="Inflow Fin")
    plt.xlabel("Time (min)")
    plt.ylabel("Effluent Fluoride (mM)")
    plt.title(f"Case 1 (C+F): Q = {Q_ml_min} mL/min, Fin = {Fin_mg_L} mg/L")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("case1_posterior_fit.png", dpi=300)
    print("Saved posterior fit plot -> case1_posterior_fit.png")