import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# 2. CFSTR BASE PARAMETERS (Thesis Section 5.2 & Paper ES&T Eng. 2024)
# ==============================================================================
BASE_CONFIG = {
    "V": 0.058,  # Volume (L) [58 mL]
    "Pin": 1.0e-3,  # Influent PO4 (M) [1.0 mM]
    "Ccal_0": 4.0,  # Calcite loading (g/L)
    "SSA_cal": 3.0,  # Calcite SSA (m2/g)
    "MW_cal": 100.0869,  # CaCO3 MW (g/mol)
    "Ksp_cal": 10 ** (-8.48),
    "MW_HA": 502.31,  # g/mol
    "Ksp3": 10 ** (-58.333),  # Ca5(PO4)3OH Ksp
    # Activities & Speciation at pH ~ 7.5
    "gamma_Ca": 0.87,
    "gamma_PO4": 0.73,
    "gamma_CO3": 0.90,
    "alpha0": 0.0967,
    "kgl": 10 ** (-4.06),  # min^-1
    "CO3_sat": 10 ** (-5.0),
    "OH_act": 10 ** (-6.50),  # pH 7.5
    "kcal": 10 ** (-5.90),  # Calcite dissolution rate
}


# ==============================================================================
# 3. PRECIPITATION-ONLY EULER SIMULATOR
# ==============================================================================
def simulate_cp_precip(params, config, t_max):
  dt = 1.0  # Step size h = 1.0 min
  n_steps = int(t_max / dt) + 1

  Q = config["Q"]
  V = config["V"]
  Pin = config["Pin"]
  Ccal = config["Ccal_0"]

  # Rate parameters
  k_precip = 10 ** params["log10_kprecip"]  # Effective rate in min^-1
  Omega_star_HA = params["Omega_star_HA"]
  kcal = config["kcal"]

  # State variables
  P = 0.0
  Ca = 10 ** (-3.69)
  CO3 = 10 ** (-5.0)
  nucleated = False

  P_out = np.zeros(n_steps)
  t_out = np.arange(n_steps) * dt

  for step in range(n_steps):
    P_out[step] = P

    # 1. Calcite Dissolution
    IAP_cal = (config["gamma_Ca"] * Ca) * (config["gamma_CO3"] * CO3)
    f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
    r_diss_cal = -kcal * config["SSA_cal"] * Ccal * f_G_cal

    # 2. Hydroxyapatite Saturation Ratio
    PO4_act = config["gamma_PO4"] * P * 9.4e-6
    Ca_act = config["gamma_Ca"] * Ca
    IAP_HA = (Ca_act**5) * (PO4_act**3) * config["OH_act"]

    Omega_HA = (IAP_HA / config["Ksp3"]) ** (1.0 / 9.0) if IAP_HA > 0 else 0.0
    f_G_HA = max(0.0, Omega_HA - 1.0)

    # 3. Nucleation Barrier Check
    if (not nucleated) and (Omega_HA >= Omega_star_HA):
      nucleated = True

    # 4. Precipitation Sink (Continuous Crystal Growth Once Triggered)
    if nucleated and (f_G_HA > 0.0):
      r_precip_HA = k_precip * f_G_HA * P  # mol/(L*min)
    else:
      r_precip_HA = 0.0

    # 5. Liquid Mass Balances (Adsorption is MASKED)
    dP_dt = (Q / V) * (Pin - P) - 3.0 * r_precip_HA
    dCa_dt = -(Q / V) * Ca + r_diss_cal - 5.0 * r_precip_HA
    r_gas = config["kgl"] * ((config["CO3_sat"] / config["alpha0"]) - CO3)
    dCO3_dt = -(Q / V) * CO3 + r_diss_cal + r_gas

    P = max(0.0, P + dP_dt * dt)
    Ca = max(0.0, Ca + dCa_dt * dt)
    CO3 = max(0.0, CO3 + dCO3_dt * dt)

  return t_out, P_out


def run_forward_model(params, time_points, config):
  t_sim, P_sim = simulate_cp_precip(params, config, max(time_points))
  return np.interp(time_points, t_sim, P_sim)


# ==============================================================================
# 4. BAYESIAN LIKELIHOOD & PRIORS
# ==============================================================================
def log_prior(theta):
  log10_kprecip, Omega_star_HA, log_sigma_P = theta

  # Priors on effective rate constant and nucleation threshold
  if not (-4.0 <= log10_kprecip <= 0.0):
    return -np.inf
  if not (1.01 <= Omega_star_HA <= 1.25):
    return -np.inf
  if not (-6.0 <= log_sigma_P <= -2.0):
    return -np.inf
  return 0.0


def log_likelihood(theta, exp_data, config):
  log10_kprecip, Omega_star_HA, log_sigma_P = theta
  sigma_P = 10**log_sigma_P

  params = {"log10_kprecip": log10_kprecip, "Omega_star_HA": Omega_star_HA}

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
  ndim = 3
  nwalkers = 16
  n_burnin = 150
  n_steps = 300

  fig_all, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

  for idx, cond in enumerate(CONDITIONS):
    print(f"\n=======================================================")
    print(f"RUNNING PRECIPITATION-ONLY FOR: {cond['name']}")
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

    # Initialize walkers around expected values
    init_pos = np.array([-2.0, 1.10, -4.5]) + 0.05 * np.random.randn(
        nwalkers, ndim
    )

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

    labels = [
        r"$\log_{10}(k_{\mathrm{precip}})$",
        r"$\Omega^*_{\mathrm{HA}}$",
        r"$\log_{10}(\sigma_P)$",
    ]
    print(f"--- INFERRED PARAMETERS FOR {cond['name']} ---")
    for p_i in range(ndim):
      mcmc = np.percentile(samples[:, p_i], [16, 50, 84])
      diff = np.diff(mcmc)
      print(f"{labels[p_i]}: {mcmc[1]:.3f} (+{diff[1]:.3f} / -{diff[0]:.3f})")

    # Corner plot
    fig_c = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
    )
    fig_c.savefig(f"case2_corner_precip_{cond['id']}.png", dpi=300)
    plt.close(fig_c)

    # Plot model fit on 3-panel figure
    ax = axes[idx]
    t_fine = np.linspace(0, max(t_data), 150)
    trajectories = []
    for s_idx in np.random.randint(len(samples), size=60):
      th = samples[s_idx]
      p_dict = {"log10_kprecip": th[0], "Omega_star_HA": th[1]}
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
        label="Precipitation-Only Model ($M_2$)",
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
  fig_all.savefig("case2_precip_only_fit.png", dpi=300)
  print("\nSaved precipitation-only fit -> case2_precip_only_fit.png")