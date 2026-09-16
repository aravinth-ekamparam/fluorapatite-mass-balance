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
    "SSA_cal": 3.0,  # Calcite specific surface area (m2/g)
    "MW_cal": 100.0869,  # CaCO3 MW (g/mol)
    "Ksp_cal": 10 ** (-8.48),
    # Hydroxyapatite (HA) Properties (Thesis Table 5.3)
    "MW_HA": 502.31,  # g/mol
    "rho_HA": 3100.0,  # kg/m3
    "sigma_HA": 0.087,  # Surface free energy (J/m2)
    "a_HA": 0.8,
    "b_HA": 0.047,
    "Ksp3": 10 ** (-58.333),  # Ca5(PO4)3OH Ksp
    "Np_HA": 8.62e6,  # Nucleated particles (Section 5.3.9.1)
    # Speciation & Mass Transfer
    "gamma_Ca": 0.87,
    "gamma_PO4": 0.73,
    "gamma_CO3": 0.90,
    "alpha0": 0.0967,
    "kgl": 10 ** (-4.06),  # min^-1 (ES&T Eng. 2024)
    "CO3_sat": 10 ** (-5.0),
    "OH_act": 10 ** (-6.50),  # pH 7.5
}


# ==============================================================================
# 3. EXPLICIT EULER FORWARD CFSTR SIMULATOR (Thesis Appendix 5D & Excel Model)
# ==============================================================================
def simulate_cp_euler(params, config, t_max):
  dt = 1.0  # Time step h = 1.0 min (matches thesis Appendix 5D)
  n_steps = int(t_max / dt) + 1

  Q = config["Q"]
  V = config["V"]
  Pin = config["Pin"]
  Ccal = config["Ccal_0"]

  kP = 10 ** params["log10_kP"]  # min^-1
  kHA = 10 ** params["log10_kHA"]  # umol/(m2*min)
  Omega_star_HA = params["Omega_star_HA"]
  kcal = 10 ** (-5.90)  # Invariant calcite dissolution

  v_HA = config["MW_HA"] / (config["rho_HA"] * 1000.0)  # m3/mol
  geom = 1.0 + (1.0 / config["a_HA"]) + (1.0 / config["b_HA"])

  # Initial conditions (Thesis Section 5.2.5)
  P = 0.0
  Ca = 10 ** (-3.69)
  CO3 = 10 ** (-5.0)
  Ps = 0.0
  L_HA = 0.0
  nucleated = False

  P_out = np.zeros(n_steps)
  t_out = np.arange(n_steps) * dt

  for step in range(n_steps):
    P_out[step] = P

    # 1. Freundlich Adsorption on Calcite (Thesis Eq. 5.40)
    P_uM = max(0.0, P * 1e6)
    Ps_star = 0.084 * (P_uM**0.31)
    dPs_dt = kP * (Ps_star - Ps)
    Ps = max(0.0, Ps + dPs_dt * dt)

    # 2. Calcite Dissolution
    IAP_cal = (config["gamma_Ca"] * Ca) * (config["gamma_CO3"] * CO3)
    f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
    r_diss_cal = -kcal * config["SSA_cal"] * Ccal * f_G_cal  # mol/(L*min)

    # 3. Hydroxyapatite Saturation & Precipitation
    PO4_act = config["gamma_PO4"] * P * 9.4e-6
    Ca_act = config["gamma_Ca"] * Ca
    IAP_HA = (Ca_act**5) * (PO4_act**3) * config["OH_act"]

    # Mean ionic saturation state
    Omega_HA = (IAP_HA / config["Ksp3"]) ** (1.0 / 9.0) if IAP_HA > 0 else 0.0
    f_G_HA = max(0.0, Omega_HA - 1.0)

    # Nucleation burst trigger (CNT)
    if (not nucleated) and (Omega_HA >= Omega_star_HA):
      nucleated = True
      ln_om = np.log(max(1.001, Omega_star_HA))
      Lc = geom * (v_HA * config["sigma_HA"]) / (8.314 * 298.15 * ln_om)
      L_HA = Lc

    # Crystal growth and precipitation
    if nucleated and (Omega_HA > 1.0):
      # Linear face growth along L (m/min)
      growth_coeff = (
          2.0 * (config["a_HA"] * config["b_HA"] + config["a_HA"] + config["b_HA"])
      ) / (3000.0 * config["rho_HA"] * config["a_HA"] * config["b_HA"])
      dL_dt = kHA * growth_coeff * f_G_HA * config["MW_HA"] * 1e-6
      L_HA += dL_dt * dt

      # Solid mass concentration (g/L) and dynamic SSA (m2/g)
      CHA = (
          1000.0
          * config["Np_HA"]
          * config["rho_HA"]
          * config["a_HA"]
          * config["b_HA"]
          * (L_HA**3)
          / V
      )
      SSA_HA = (
          2.0
          * (config["a_HA"] * config["b_HA"] + config["a_HA"] + config["b_HA"])
      ) / (
          1000.0
          * config["rho_HA"]
          * config["a_HA"]
          * config["b_HA"]
          * max(1e-9, L_HA)
      )

      # Volumetric precipitation rate: mol/(L*min)
      r_precip_HA = kHA * SSA_HA * CHA * f_G_HA * 1e-6
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
  if np.any(np.isnan(P_sim)) or np.any(np.isinf(P_sim)):
    return None
  return np.interp(time_points, t_sim, P_sim)


# ==============================================================================
# 4. BAYESIAN LIKELIHOOD & PRIORS
# ==============================================================================
def log_prior(theta):
  log10_kP, log10_kHA, Omega_star_HA, log_sigma_P = theta

  # Priors covering Thesis Table 5.5 and Paper Table 1 ranges
  if not (-7.0 <= log10_kP <= -3.0):
    return -np.inf
  if not (-17.5 <= log10_kHA <= -12.0):
    return -np.inf
  if not (1.05 <= Omega_star_HA <= 1.45):
    return -np.inf
  if not (-6.0 <= log_sigma_P <= -3.0):
    return -np.inf

  return 0.0


def log_likelihood(theta, exp_data, config):
  log10_kP, log10_kHA, Omega_star_HA, log_sigma_P = theta
  sigma_P = 10**log_sigma_P

  params = {
      "log10_kP": log10_kP,
      "log10_kHA": log10_kHA,
      "Omega_star_HA": Omega_star_HA,
  }

  P_sim = run_forward_model(params, exp_data["t_min"], config)
  if P_sim is None:
    return -np.inf

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
  ndim = 4
  nwalkers = 16
  n_burnin = 150
  n_steps = 300

  fig_all, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

  for idx, cond in enumerate(CONDITIONS):
    print(f"\n=======================================================")
    print(f"RUNNING CASE 2 (C+P) CONDITION {idx+1}: {cond['name']}")
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

    # Initial positions centered around thesis Table 5.5
    init_pos = np.array([-5.0, -15.5, 1.16, -4.5]) + 0.05 * np.random.randn(
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
        r"$\log_{10}(k_P)$",
        r"$\log_{10}(k_{\mathrm{HA}})$",
        r"$\Omega^*_{\mathrm{HA}}$",
        r"$\log_{10}(\sigma_P)$",
    ]
    print(f"\n--- INFERRED PARAMETERS FOR {cond['name']} ---")
    for p_i in range(ndim):
      mcmc = np.percentile(samples[:, p_i], [16, 50, 84])
      diff = np.diff(mcmc)
      print(f"{labels[p_i]}: {mcmc[1]:.3f} (+{diff[1]:.3f} / -{diff[0]:.3f})")

    fig_c = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
    )
    fig_c.savefig(f"case2_corner_{cond['id']}.png", dpi=300)
    plt.close(fig_c)

    # Plot model fit on 3-panel figure
    ax = axes[idx]
    t_fine = np.linspace(0, max(t_data), 150)
    trajectories = []
    for s_idx in np.random.randint(len(samples), size=60):
      th = samples[s_idx]
      p_dict = {
          "log10_kP": th[0],
          "log10_kHA": th[1],
          "Omega_star_HA": th[2],
      }
      p_sim = run_forward_model(p_dict, t_fine, cfg)
      if p_sim is not None:
        trajectories.append(p_sim)
    trajectories = np.array(trajectories)

    p_low = np.percentile(trajectories, 2.5, axis=0) * 1e3
    p_med = np.percentile(trajectories, 50.0, axis=0) * 1e3
    p_high = np.percentile(trajectories, 97.5, axis=0) * 1e3

    ax.plot(t_data, p_data * 1e3, "ro", ms=4, label="Data")
    ax.plot(t_fine, p_med, "b-", lw=2, label="Median Posterior")
    ax.fill_between(t_fine, p_low, p_high, color="blue", alpha=0.2, label="95% CI")
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
  fig_all.savefig("case2_all_fits_comparison.png", dpi=300)
  print("\nSaved 3-panel fit -> case2_all_fits_comparison.png")

  # Overlaid Posteriors Comparison
  fig_comp, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
  colors = ["#1f77b4", "#2ca02c", "#d62728"]
  for idx, cond in enumerate(CONDITIONS):
    s = results[cond["id"]]["samples"]
    ax1.hist(
        s[:, 0],
        bins=25,
        density=True,
        alpha=0.45,
        color=colors[idx],
        label=cond["name"],
    )
    ax2.hist(
        s[:, 1],
        bins=25,
        density=True,
        alpha=0.45,
        color=colors[idx],
        label=cond["name"],
    )
    ax3.hist(
        s[:, 2],
        bins=25,
        density=True,
        alpha=0.45,
        color=colors[idx],
        label=cond["name"],
    )

  ax1.set_xlabel(r"$\log_{10}(k_P)$  $[\mathrm{min}^{-1}]$", fontsize=11)
  ax1.set_ylabel("Posterior Density", fontsize=11)
  ax1.set_title("Phosphate Adsorption Rate", fontweight="bold")
  ax1.legend(fontsize=8)
  ax1.grid(True, linestyle="--", alpha=0.4)

  ax2.set_xlabel(
      r"$\log_{10}(k_{\mathrm{HA}})$  $[\mu\mathrm{mol}/(\mathrm{m}^2\cdot\mathrm{min})]$",
      fontsize=11,
  )
  ax2.set_title("HA Precipitation Rate", fontweight="bold")
  ax2.grid(True, linestyle="--", alpha=0.4)

  ax3.set_xlabel(r"$\Omega^*_{\mathrm{HA}}$", fontsize=11)
  ax3.set_title("HA Nucleation Barrier", fontweight="bold")
  ax3.grid(True, linestyle="--", alpha=0.4)

  fig_comp.tight_layout()
  fig_comp.savefig("case2_posteriors_comparison.png", dpi=300)
  print("Saved posteriors comparison -> case2_posteriors_comparison.png")