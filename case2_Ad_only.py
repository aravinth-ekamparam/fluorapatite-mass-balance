import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 0. MECHANISM SWITCH
# ==============================================================================
# Set to False to run pure adsorption; set to True when testing precipitation
ENABLE_PRECIPITATION = False

# ==============================================================================
# 1. EXPERIMENTAL CONDITIONS FROM EXCEL (CF_Exp data.xlsx)
# ==============================================================================
CONDITIONS = [
    {
        "name": "Q = 0.01 mL/min, Pin = 1 mM",
        "col_t": 9,
        "col_p": 10,
        "id": "cond1",
    },
    {
        "name": "Q = 0.03 mL/min, Pin = 1 mM",
        "col_t": 11,
        "col_p": 12,
        "id": "cond2",
    },
    {
        "name": "Q = 0.04 mL/min, Pin = 1 mM",
        "col_t": 13,
        "col_p": 14,
        "id": "cond3",
    },
]

excel_file = "CF_Exp data.xlsx"
df = pd.read_excel(excel_file, header=None)

# ==============================================================================
# 2. CFSTR BASE PARAMETERS (Thesis Section 5.2 & Appendix 5G)
# ==============================================================================
BASE_CONFIG = {
    "V": 0.058,  # Volume (L) [58 mL]
    "Pin": 1.0e-3,  # Influent PO4 (M) [1.0 mM]
    "Ccal_0": 4.0,  # Calcite loading (g/L)
    "SSA_cal": 3.0,  # Calcite specific surface area (m2/g)
    "MW_cal": 100.0869,  # CaCO3 MW (g/mol)
    "Ksp_cal": 10 ** (-8.48),
    # Calcite Dissolution
    "kcal": 10 ** (-5.90),  # mol/(m2*min)
    "gamma_Ca": 0.87,
    "gamma_CO3": 0.90,
    "alpha0": 0.0967,
    "kgl": 10 ** (-4.06),  # min^-1
    "CO3_sat": 10 ** (-5.0),
    # Hydroxyapatite parameters (inactive when ENABLE_PRECIPITATION = False)
    "MW_HA": 502.31,
    "rho_HA": 3100.0,
    "sigma_HA": 0.087,
    "a_HA": 0.8,
    "b_HA": 0.047,
    "Ksp3": 10 ** (-58.333),
    "Np_HA": 8.62e6,
    "gamma_PO4": 0.73,
    "OH_act": 10 ** (-6.50),
}


# ==============================================================================
# 3. EXPLICIT EULER FORWARD CFSTR SIMULATOR
# ==============================================================================
def simulate_cp_euler(params, config, t_max):
  dt = 1.0  # Time step h = 1.0 min
  n_steps = int(t_max / dt) + 1

  Q = config["Q"]
  V = config["V"]
  Pin = config["Pin"]
  Ccal = config["Ccal_0"]

  kP = 10 ** params["log10_kP"]  # min^-1
  kcal = config["kcal"]

  # State variables
  P = 0.0
  Ca = 10 ** (-3.69)
  CO3 = 10 ** (-5.0)
  Ps = 0.0

  P_out = np.zeros(n_steps)
  t_out = np.arange(n_steps) * dt

  for step in range(n_steps):
    P_out[step] = P

    # 1. Freundlich Adsorption on Calcite (Thesis Eq. 5.40 & Appendix 5G)
    P_uM = max(0.0, P * 1e6)
    Ps_star = 0.084 * (P_uM**0.31)
    dPs_dt = kP * (Ps_star - Ps)
    Ps = max(0.0, Ps + dPs_dt * dt)

    # 2. Calcite Dissolution
    IAP_cal = (config["gamma_Ca"] * Ca) * (config["gamma_CO3"] * CO3)
    f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
    r_diss_cal = -kcal * config["SSA_cal"] * Ccal * f_G_cal

    # 3. Precipitation Term (Masked when ENABLE_PRECIPITATION = False)
    if ENABLE_PRECIPITATION:
      # Placeholder for active precipitation
      r_precip_HA = 0.0
    else:
      r_precip_HA = 0.0

    # 4. Fluid Mass Balances (Euler Step)
    dP_dt = (
        (Q / V) * (Pin - P) - Ccal * dPs_dt * 1e-6 - 3.0 * r_precip_HA
    )
    dCa_dt = -(Q / V) * Ca + r_diss_cal - 5.0 * r_precip_HA
    r_gas = config["kgl"] * ((config["CO3_sat"] / config["alpha0"]) - CO3)
    dCO3_dt = -(Q / V) * CO3 + r_diss_cal + r_gas

    P = max(0.0, P + dP_dt * dt)
    Ca = max(0.0, Ca + dCa_dt * dt)
    CO3 = max(0.0, CO3 + dCO3_dt * dt)

  return t_out, P_out


def run_forward_model(params, time_points, config):
  t_sim, P_sim = simulate_cp_euler(params, config, max(time_points))
  return np.interp(time_points, t_sim, P_sim)


# ==============================================================================
# 4. BAYESIAN INFERENCE (1D Parameter Estimation for kP)
# ==============================================================================
# Parameter vector: theta = [log10_kP, log_sigma_P]


def log_prior(theta):
  log10_kP, log_sigma_P = theta
  if not (-7.0 <= log10_kP <= -2.0):
    return -np.inf
  if not (-6.0 <= log_sigma_P <= -1.0):
    return -np.inf
  return 0.0


def log_likelihood(theta, exp_data, config):
  log10_kP, log_sigma_P = theta
  sigma_P = 10**log_sigma_P

  params = {"log10_kP": log10_kP}
  P_sim = run_forward_model(params, exp_data["t_min"], config)

  res = exp_data["P_meas"] - P_sim
  n = len(exp_data["t_min"])
  ll = -0.5 * np.sum((res / sigma_P) ** 2) - n * np.log(
      sigma_P * np.sqrt(2 * np.pi)
  )

  if not np.isfinite(ll):
    return -np.inf
  return ll


def log_posterior(theta, exp_data, config):
  lp = log_prior(theta)
  if not np.isfinite(lp):
    return -np.inf
  ll = log_likelihood(theta, exp_data, config)
  if not np.isfinite(ll):
    return -np.inf
  return lp + ll


# ==============================================================================
# 5. EXECUTION ACROSS ALL 3 FLOW RATES
# ==============================================================================
if __name__ == "__main__":
  results = {}
  ndim = 2
  nwalkers = 12
  n_burnin = 100
  n_steps = 250

  fig_all, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

  for idx, cond in enumerate(CONDITIONS):
    print(f"\n=======================================================")
    print(f"RUNNING ADSORPTION-ONLY FOR: {cond['name']}")
    print(f"=======================================================")

    t_raw = pd.to_numeric(df.iloc[4:, cond["col_t"]], errors="coerce")
    p_raw = pd.to_numeric(df.iloc[4:, cond["col_p"]], errors="coerce")
    mask = t_raw.notna() & p_raw.notna()

    t_data = t_raw[mask].values
    p_data = p_raw[mask].values

    Q_ml = float(df.iloc[0, cond["col_t"]])
    cfg = BASE_CONFIG.copy()
    cfg["Q"] = Q_ml * 1e-3

    exp_data = {"t_min": t_data, "P_meas": p_data}

    # Initialize walkers
    init_pos = np.array([-4.5, -3.5]) + 0.05 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(exp_data, cfg)
    )
    sampler.run_mcmc(init_pos, n_steps + n_burnin, progress=True)

    samples = sampler.get_chain(discard=n_burnin, flat=True)
    results[cond["id"]] = {
        "samples": samples,
        "cfg": cfg,
        "data": exp_data,
        "name": cond["name"],
    }

    labels = [r"$\log_{10}(k_P)$", r"$\log_{10}(\sigma_P)$"]
    print(f"--- INFERRED PARAMETERS FOR {cond['name']} ---")
    for p_i in range(ndim):
      mcmc = np.percentile(samples[:, p_i], [16, 50, 84])
      diff = np.diff(mcmc)
      print(f"{labels[p_i]}: {mcmc[1]:.3f} (+{diff[1]:.3f} / -{diff[0]:.3f})")

    # Plot model fit on 3-panel figure
    ax = axes[idx]
    t_fine = np.linspace(0, max(t_data), 150)
    trajectories = []
    for s_idx in np.random.randint(len(samples), size=50):
      th = samples[s_idx]
      p_dict = {"log10_kP": th[0]}
      p_sim = run_forward_model(p_dict, t_fine, cfg)
      trajectories.append(p_sim)
    trajectories = np.array(trajectories)

    p_low = np.percentile(trajectories, 2.5, axis=0) * 1e3
    p_med = np.percentile(trajectories, 50.0, axis=0) * 1e3
    p_high = np.percentile(trajectories, 97.5, axis=0) * 1e3

    ax.plot(t_data, p_data * 1e3, "ro", ms=4, label="Experimental Data")
    ax.plot(
        t_fine,
        p_med,
        "b-",
        lw=2,
        label="Adsorption-Only Model ($M_1$)",
    )
    ax.fill_between(
        t_fine, p_low, p_high, color="blue", alpha=0.2, label="95% CI"
    )
    ax.axhline(
        cfg["Pin"] * 1e3,
        color="gray",
        linestyle=":",
        label="Inflow Pin (1.0 mM)",
    )
    ax.set_title(cond["name"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Effluent Phosphate (mM)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

  fig_all.tight_layout()
  fig_all.savefig("case2_adsorption_only_fit.png", dpi=300)
  print("\nSaved adsorption-only fit -> case2_adsorption_only_fit.png")