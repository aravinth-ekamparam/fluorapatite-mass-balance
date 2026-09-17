import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner

# ==============================================================================
# 1. LOAD EXPERIMENTAL DATA (Q = 0.01 mL/min, Pin = 1.0 mM)
# ==============================================================================
excel_file = "CF_Exp data.xlsx"
df = pd.read_excel(excel_file, header=None)

# Column J (index 9) = Time (min), Column K (index 10) = Effluent P (M)
t_raw = pd.to_numeric(df.iloc[4:, 9], errors="coerce")
p_raw = pd.to_numeric(df.iloc[4:, 10], errors="coerce")
valid = t_raw.notna() & p_raw.notna()

exp_data = {
    "t_min": t_raw[valid].values,
    "P_meas": p_raw[valid].values
}

# ==============================================================================
# 2. CFSTR CONFIGURATION & THERMODYNAMIC CONSTANTS
# ==============================================================================
CONFIG = {
    "V": 0.058,              # Volume (L)
    "Q": 0.01 * 1e-3,        # Flow rate (L/min)
    "Pin": 1.0e-3,           # Influent P (M)
    "Ccal_0": 4.0,           # Calcite loading (g/L)
    "Ca_eff": 1.75e-4,       # Measured steady-state Ca buffer (M)
    "gamma_Ca": 0.87,
    "gamma_PO4": 0.73,
    "OH_act": 10**(-6.50),   # pH 7.5
    "H_act": 10**(-7.50),
    
    # Solubility products
    "Ksp_HA": 10**(-58.333), # Ca5(PO4)3OH
    "Ksp_OCP": 10**(-96.60), # Ca8H2(PO4)6*5H2O
    
    # Molecular weights
    "MW_HA": 502.31,
    "MW_OCP": 982.52,
    
    # Precursor parameters
    "Omega_star_OCP": 1.02   # Very low nucleation barrier due to low surface energy
}

# ==============================================================================
# 3. COUPLED TWO-PHASE (OCP -> HA) FORWARD SIMULATOR
# ==============================================================================
def simulate_twophase(theta, config, t_max):
    """
    Solves coupled two-phase precipitation with Freundlich adsorption.
    theta = [log10_kP, log10_kOCP, log10_kHA, log10_ktrans, Omega_star_HA]
    """
    dt = 1.0
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
    Ca = config["Ca_eff"]
    
    P = 0.0
    Ps = 0.0
    C_OCP = 0.0
    C_HA = 0.0
    ha_nucleated = False
    
    P_out = np.zeros(n_steps)
    t_out = np.arange(n_steps) * dt
    
    for step in range(n_steps):
        P_out[step] = P
        
        # 1. Freundlich Adsorption
        P_uM = max(0.0, P * 1e6)
        Ps_star = 0.084 * (P_uM**0.31)
        dPs_dt = kP * (Ps_star - Ps)
        Ps = max(0.0, Ps + dPs_dt * dt)
        
        # 2. Speciation & Thermodynamic Driving Forces
        PO4_act = config["gamma_PO4"] * P * 9.4e-6
        Ca_act = config["gamma_Ca"] * Ca
        
        # Hydroxyapatite Saturation
        IAP_HA = (Ca_act**5) * (PO4_act**3) * config["OH_act"]
        Omega_HA = (IAP_HA / config["Ksp_HA"])**(1.0 / 9.0) if IAP_HA > 0 else 0.0
        f_G_HA = max(0.0, Omega_HA - 1.0)
        
        # Octacalcium Phosphate Saturation (16 ions: 8 Ca, 2 H, 6 PO4)
        IAP_OCP = (Ca_act**8) * (config["H_act"]**2) * (PO4_act**6)
        Omega_OCP = (IAP_OCP / config["Ksp_OCP"])**(1.0 / 16.0) if IAP_OCP > 0 else 0.0
        f_G_OCP = max(0.0, Omega_OCP - 1.0)
        
        # 3. Kinetic Rates
        # A. OCP Precipitation
        if Omega_OCP >= config["Omega_star_OCP"]:
            r_precip_OCP = kOCP * f_G_OCP * P
        else:
            r_precip_OCP = 0.0
            
        # B. OCP to HA Transformation (Dissolution-Reprecipitation)
        r_trans = ktrans * C_OCP * f_G_HA
        
        # C. Direct HA Nucleation & Growth
        if (not ha_nucleated) and (Omega_HA >= Omega_star_HA):
            ha_nucleated = True
            
        if ha_nucleated and (f_G_HA > 0.0):
            r_precip_HA = kHA * f_G_HA * P
        else:
            r_precip_HA = 0.0
            
        # 4. Mass Balances
        # Aqueous P Balance
        dP_dt = (Q / V) * (Pin - P) - Ccal * dPs_dt * 1e-6 - 6.0 * r_precip_OCP - 3.0 * r_precip_HA
        # OCP Solid Balance
        dCOCP_dt = r_precip_OCP * config["MW_OCP"] - r_trans * config["MW_OCP"]
        # HA Solid Balance
        dCHA_dt = r_precip_HA * config["MW_HA"] + (5.0 / 8.0) * r_trans * config["MW_HA"]
        
        P = max(0.0, P + dP_dt * dt)
        C_OCP = max(0.0, C_OCP + dCOCP_dt * dt)
        C_HA = max(0.0, C_HA + dCHA_dt * dt)
        
    return t_out, P_out

def run_forward_model(theta, time_points, config):
    t_sim, P_sim = simulate_twophase(theta, config, max(time_points))
    return np.interp(time_points, t_sim, P_sim)

# ==============================================================================
# 4. BAYESIAN MCMC SAMPLER
# ==============================================================================
# Parameter vector: theta = [log10_kP, log10_kOCP, log10_kHA, log10_ktrans, Omega_star_HA, log_sigma]

def log_prior(theta):
    log_kP, log_kOCP, log_kHA, log_ktrans, Omega_star_HA, log_sigma = theta
    
    if not (-6.0 <= log_kP <= -2.0):
        return -np.inf
    if not (-4.0 <= log_kOCP <= 0.0):
        return -np.inf
    if not (-5.0 <= log_kHA <= -1.0):
        return -np.inf
    if not (-6.0 <= log_ktrans <= -1.0):
        return -np.inf
    if not (1.01 <= Omega_star_HA <= 1.25):
        return -np.inf
    if not (-6.0 <= log_sigma <= -2.0):
        return -np.inf
    return 0.0

def log_likelihood(theta, exp_data, config):
    sigma = 10**theta[5]
    P_sim = run_forward_model(theta[:5], exp_data["t_min"], config)
    
    res = exp_data["P_meas"] - P_sim
    n = len(exp_data["t_min"])
    ll = -0.5 * np.sum((res / sigma)**2) - n * np.log(sigma * np.sqrt(2 * np.pi))
    return ll if np.isfinite(ll) else -np.inf

def log_posterior(theta, exp_data, config):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, exp_data, config)

# ==============================================================================
# 5. EXECUTION & VISUALIZATION
# ==============================================================================
if __name__ == "__main__":
    ndim = 6
    nwalkers = 20
    n_burnin = 150
    n_steps = 350
    
    # Starting positions
    init_pos = np.array([-4.0, -2.0, -3.5, -3.5, 1.08, -4.6]) + 0.02 * np.random.randn(nwalkers, ndim)
    
    print("Running Two-Phase (OCP -> HA) Bayesian MCMC Calibration...")
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
    
    print("\n=== POSTERIOR ESTIMATES FOR TWO-PHASE MODEL (Q = 0.01 mL/min) ===")
    for i in range(ndim):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{labels[i]}: {p50:.3f} (+{p84-p50:.3f} / -{p50-p16:.3f})")
        
    # Save Corner Plot
    fig_corner = corner.corner(samples, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True)
    fig_corner.savefig("case2_twophase_corner.png", dpi=300)
    plt.close(fig_corner)
    
    # Generate Fit Plot
    t_fine = np.linspace(0, max(exp_data["t_min"]), 200)
    trajectories = []
    for idx in np.random.randint(len(samples), size=60):
        th = samples[idx][:5]
        trajectories.append(run_forward_model(th, t_fine, CONFIG))
    trajectories = np.array(trajectories)
    
    p_low = np.percentile(trajectories, 2.5, axis=0) * 1e3
    p_med = np.percentile(trajectories, 50.0, axis=0) * 1e3
    p_high = np.percentile(trajectories, 97.5, axis=0) * 1e3
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(exp_data["t_min"], exp_data["P_meas"] * 1e3, "ro", ms=4, label="Experimental Data")
    plt.plot(t_fine, p_med, "b-", lw=2, label="Two-Phase Model (OCP -> HA)")
    plt.fill_between(t_fine, p_low, p_high, color="blue", alpha=0.2, label="95% CI")
    plt.axhline(CONFIG["Pin"] * 1e3, color="gray", linestyle=":", label="Inflow Pin (1.0 mM)")
    plt.xlabel("Time (min)")
    plt.ylabel("Effluent Phosphate (mM)")
    plt.title("Two-Phase Precipitation Fit (Q = 0.01 mL/min)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("case2_twophase_fit.png", dpi=300)
    print("\nSaved two-phase fit plot -> case2_twophase_fit.png")