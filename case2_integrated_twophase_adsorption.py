import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner

# ==============================================================================
# 1. EXPERIMENTAL CONDITIONS & DATA SELECTION
# ==============================================================================
CONDITIONS = {
    "cond1": {"name": "Q = 0.01 mL/min, Pin = 1 mM", "col_t": 9,  "col_p": 10, "Q_val": 0.01},
    "cond2": {"name": "Q = 0.03 mL/min, Pin = 1 mM", "col_t": 11, "col_p": 12, "Q_val": 0.03},
    "cond3": {"name": "Q = 0.04 mL/min, Pin = 1 mM", "col_t": 13, "col_p": 14, "Q_val": 0.04},
}

ACTIVE_COND = "cond1"  # Switch between "cond1", "cond2", "cond3"
excel_file = "CF_Exp data.xlsx"

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file, header=None)
    c_info = CONDITIONS[ACTIVE_COND]
    t_raw = pd.to_numeric(df.iloc[4:, c_info["col_t"]], errors="coerce")
    p_raw = pd.to_numeric(df.iloc[4:, c_info["col_p"]], errors="coerce")
    mask = t_raw.notna() & p_raw.notna()
    exp_data = {"t_min": t_raw[mask].values, "P_meas": p_raw[mask].values}
    flow_rate_ml = float(df.iloc[0, c_info["col_t"]]) if pd.notna(df.iloc[0, c_info["col_t"]]) else c_info["Q_val"]
else:
    print(f"Warning: '{excel_file}' not found. Generating dummy experimental data for execution.")
    exp_data = {
        "t_min": np.linspace(0, 15000, 30),
        "P_meas": 1.0e-3 / (1.0 + np.exp(-0.0006 * (np.linspace(0, 15000, 30) - 6000)))
    }
    flow_rate_ml = CONDITIONS[ACTIVE_COND]["Q_val"]

# ==============================================================================
# 2. INTEGRATED THERMODYNAMIC & REACTOR CONFIGURATION
# ==============================================================================
CONFIG = {
    # Reactor geometry & flow
    "V": 0.058,                     # Volume (L)
    "Q": flow_rate_ml * 1e-3,       # Flow rate (L/min)
    "Pin": 1.0e-3,                  # Influent PO4 (M)
    
    # Calcite properties & dissolution
    "Ccal_0": 4.0,                  # Calcite loading (g/L)
    "SSA_cal": 3.0,                 # Specific surface area (m^2/g)
    "MW_cal": 100.0869,             # g/mol
    "Ksp_cal": 10**(-8.48),         # Calcite solubility product
    "kcal": 10**(-5.90),            # Dissolution rate constant (mol/(m^2*min))
    "gamma_Ca": 0.87,
    "gamma_CO3": 0.90,
    "alpha0": 0.0967,               # Carbonate distribution factor at pH 7.5
    "kgl": 10**(-4.06),             # Gas-liquid exchange rate (min^-1)
    "CO3_sat": 10**(-5.0),          # Carbonate saturation (M)
    
    # Aqueous chemistry & speciation
    "gamma_PO4": 0.73,
    "OH_act": 10**(-6.50),          # pH 7.5 (pOH = 6.50)
    "H_act": 10**(-7.50),           # pH 7.5
    "PO4_alpha3": 9.4e-6,           # PO4^3- fraction of total dissolved P at pH 7.5
    
    # Octacalcium Phosphate (OCP) precursor parameters
    "Ksp_OCP": 10**(-96.60),        # Ca8H2(PO4)6*5H2O
    "MW_OCP": 982.52,               # g/mol
    "Omega_star_OCP": 1.02,         # Low nucleation barrier for precursor
    
    # Hydroxyapatite (HAP) parameters
    "Ksp_HA": 10**(-58.333),        # Ca5(PO4)3OH
    "MW_HA": 502.31,                # g/mol
    
    # Initial conditions
    "Ca_init": 10**(-3.69),         # Calcite equilibrium buffer ~2.04e-4 M
    "CO3_init": 10**(-5.0),         # Equilibrium carbonate ~1.0e-5 M
}

# ==============================================================================
# 3. COUPLED FORWARD CFSTR SIMULATOR
# ==============================================================================
def simulate_integrated(theta, config, t_max):
    """
    Solves dynamic Calcite dissolution + Freundlich Adsorption + Two-Phase (OCP -> HA) Precipitation.
    theta = [log10_kP, log10_kOCP, log10_kHA, log10_ktrans, Omega_star_HA]
    """
    dt = 1.0  # Integration time step (min)
    n_steps = int(t_max / dt) + 1
    
    kP = 10**theta[0]
    kOCP = 10**theta[1]
    kHA = 10**theta[2]
    ktrans = 10**theta[3]
    Omega_star_HA = theta[4]
    
    Q = config["Q"]
    V = config["V"]
    Pin = config["Pin"]
    Ccal = config["Ccal_0"]
    
    # Dynamic state variables
    P = 0.0
    Ca = config["Ca_init"]
    CO3 = config["CO3_init"]
    Ps = 0.0
    C_OCP = 0.0
    C_HA = 0.0
    ha_nucleated = False
    
    P_out = np.zeros(n_steps)
    t_out = np.arange(n_steps) * dt
    
    for step in range(n_steps):
        P_out[step] = P
        
        # 1. Freundlich kinetic adsorption on calcite
        P_uM = max(0.0, P * 1e6)
        Ps_star = 0.084 * (P_uM**0.31)
        dPs_dt = kP * (Ps_star - Ps)
        Ps = max(0.0, Ps + dPs_dt * dt)
        
        # 2. Dynamic calcite dissolution & gas-liquid CO3 transfer
        IAP_cal = (config["gamma_Ca"] * Ca) * (config["gamma_CO3"] * CO3)
        f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
        # Dissolution occurs when undersaturated (f_G_cal < 0)
        r_diss_cal = max(0.0, -config["kcal"] * config["SSA_cal"] * Ccal * f_G_cal)
        r_gas = config["kgl"] * ((config["CO3_sat"] / config["alpha0"]) - CO3)
        
        # 3. Speciation and Saturation Indices
        PO4_act = config["gamma_PO4"] * P * config["PO4_alpha3"]
        Ca_act = config["gamma_Ca"] * Ca
        
        # Hydroxyapatite saturation
        IAP_HA = (Ca_act**5) * (PO4_act**3) * config["OH_act"]
        Omega_HA = (IAP_HA / config["Ksp_HA"])**(1.0 / 9.0) if IAP_HA > 0 else 0.0
        f_G_HA = max(0.0, Omega_HA - 1.0)
        
        # Octacalcium phosphate saturation
        IAP_OCP = (Ca_act**8) * (config["H_act"]**2) * (PO4_act**6)
        Omega_OCP = (IAP_OCP / config["Ksp_OCP"])**(1.0 / 16.0) if IAP_OCP > 0 else 0.0
        f_G_OCP = max(0.0, Omega_OCP - 1.0)
        
        # 4. Precipitation & Transformation Rates
        # OCP precursor formation
        r_precip_OCP = kOCP * f_G_OCP * P if Omega_OCP >= config["Omega_star_OCP"] else 0.0
        
        # HAP nucleation barrier and growth
        if (not ha_nucleated) and (Omega_HA >= Omega_star_HA):
            ha_nucleated = True
            
        r_precip_HA = kHA * f_G_HA * P if (ha_nucleated and f_G_HA > 0.0) else 0.0
        
        # OCP -> HAP dissolution-reprecipitation transformation
        r_trans = ktrans * C_OCP * f_G_HA
        
        # 5. Fluid & Solid Phase Mass Balances
        # Aqueous P (mol/L/min)
        dP_dt = (Q / V) * (Pin - P) - Ccal * dPs_dt * 1e-6 - 6.0 * r_precip_OCP - 3.0 * r_precip_HA
        # Aqueous Ca (mol/L/min)
        dCa_dt = -(Q / V) * Ca + r_diss_cal - 8.0 * r_precip_OCP - 5.0 * r_precip_HA
        # Aqueous CO3 (mol/L/min)
        dCO3_dt = -(Q / V) * CO3 + r_diss_cal + r_gas
        # OCP Solid Mass (g/L/min)
        dCOCP_dt = r_precip_OCP * config["MW_OCP"] - r_trans * config["MW_OCP"]
        # HAP Solid Mass (g/L/min)
        dCHA_dt = r_precip_HA * config["MW_HA"] + (5.0 / 8.0) * r_trans * config["MW_HA"]
        
        # State updates
        P = max(0.0, P + dP_dt * dt)
        Ca = max(0.0, Ca + dCa_dt * dt)
        CO3 = max(0.0, CO3 + dCO3_dt * dt)
        C_OCP = max(0.0, C_OCP + dCOCP_dt * dt)
        C_HA = max(0.0, C_HA + dCHA_dt * dt)
        
    return t_out, P_out

def run_forward_model(theta, time_points, config):
    t_sim, P_sim = simulate_integrated(theta, config, max(time_points))
    return np.interp(time_points, t_sim, P_sim)

# ==============================================================================
# 4. BAYESIAN MCMC ESTIMATION
# ==============================================================================
# Parameter vector: theta = [log10_kP, log10_kOCP, log10_kHA, log10_ktrans, Omega_star_HA, log10_sigma]

def log_prior(theta):
    log_kP, log_kOCP, log_kHA, log_ktrans, Omega_star_HA, log_sigma = theta
    if not (-6.5 <= log_kP <= -2.0):
        return -np.inf
    if not (-5.0 <= log_kOCP <= 0.5):
        return -np.inf
    if not (-5.5 <= log_kHA <= -0.5):
        return -np.inf
    if not (-6.0 <= log_ktrans <= -1.0):
        return -np.inf
    if not (1.01 <= Omega_star_HA <= 1.30):
        return -np.inf
    if not (-6.0 <= log_sigma <= -2.0):
        return -np.inf
    return 0.0

def log_likelihood(theta, exp_data, config):
    sigma = 10**theta[5]
    P_sim = run_forward_model(theta[:5], exp_data["t_min"], config)
    res = exp_data["P_meas"] - P_sim
    n = len(exp_data["t_min"])
    ll = -0.5 * np.sum((res / sigma)**2) - n * np.log(sigma * np.sqrt(2.0 * np.pi))
    return ll if np.isfinite(ll) else -np.inf

def log_posterior(theta, exp_data, config):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, exp_data, config)

# ==============================================================================
# 5. EXECUTION & POST-PROCESSING
# ==============================================================================
if __name__ == "__main__":
    ndim = 6
    nwalkers = 24
    n_burnin = 150
    n_steps = 350
    
    # Realistic initial parameter guess
    init_guess = np.array([-4.2, -1.8, -3.2, -3.6, 1.09, -4.5])
    init_pos = init_guess + 0.03 * np.random.randn(nwalkers, ndim)
    
    print(f"Executing Integrated Model (Adsorption + Calcite Dissolution + OCP/HAP) on {ACTIVE_COND}...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(exp_data, CONFIG))
    sampler.run_mcmc(init_pos, n_steps + n_burnin, progress=True)
    
    samples = sampler.get_chain(discard=n_burnin, flat=True)
    
    labels = [
        r"$\log_{10}(k_P)$",
        r"$\log_{10}(k_{\mathrm{OCP}})$",
        r"$\log_{10}(k_{\mathrm{HA}})$",
        r"$\log_{10}(k_{\mathrm{trans}})$",
        r"$\Omega^*_{\mathrm{HA}}$",
        r"$\log_{10}(\sigma_P)$"
    ]
    
    print("\n=== POSTERIOR ESTIMATES FOR INTEGRATED MODEL ===")
    for i in range(ndim):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{labels[i]}: {p50:.3f} (+{p84-p50:.3f} / -{p50-p16:.3f})")
        
    # Generate Corner Plot
    fig_corner = corner.corner(samples, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True)
    fig_corner.savefig("case2_integrated_corner.png", dpi=300)
    plt.close(fig_corner)
    print("Saved MCMC posterior corner plot -> case2_integrated_corner.png")
    
    # Generate Predictive Fit Plot
    t_fine = np.linspace(0, max(exp_data["t_min"]), 250)
    trajectories = []
    for idx in np.random.randint(len(samples), size=60):
        trajectories.append(run_forward_model(samples[idx][:5], t_fine, CONFIG))
    trajectories = np.array(trajectories)
    
    p_low = np.percentile(trajectories, 2.5, axis=0) * 1e3
    p_med = np.percentile(trajectories, 50.0, axis=0) * 1e3
    p_high = np.percentile(trajectories, 97.5, axis=0) * 1e3
    
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(exp_data["t_min"], exp_data["P_meas"] * 1e3, "ro", ms=4, label="Experimental Data")
    plt.plot(t_fine, p_med, "b-", lw=2, label="Integrated Model (Adsorption + OCP + HAP)")
    plt.fill_between(t_fine, p_low, p_high, color="blue", alpha=0.2, label="95% Credible Interval")
    plt.axhline(CONFIG["Pin"] * 1e3, color="gray", linestyle=":", label=f"Influent Pin ({CONFIG['Pin']*1e3:.1f} mM)")
    plt.xlabel("Time (min)")
    plt.ylabel("Effluent Phosphate (mM)")
    plt.title(f"Integrated Model Breakthrough Fit ({CONDITIONS[ACTIVE_COND]['name']})", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("case2_integrated_fit.png", dpi=300)
    print("Saved effluent breakthrough fit plot -> case2_integrated_fit.png")