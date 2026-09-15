import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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
# 2. CFSTR BASE PARAMETERS (Thesis Section 5.2)
# ==============================================================================
BASE_CONFIG = {
    "V": 0.058,  # Volume (L) [58 mL]
    "Pin": 1.0e-3,  # Influent PO4 (M) [1 mM]
    "Ccal_0": 4.0,  # Calcite loading (g/L)
    "SSA_cal": 3.0,  # Calcite SSA (m2/g)
    "MW_cal": 100.0869,  # CaCO3 MW (g/mol)
    "Ksp_cal": 10 ** (-8.48),
    # Hydroxyapatite (HA) Properties (Thesis Chapter 5, Table 5.3)
    "MW_HA": 502.31,  # g/mol
    "rho_HA": 3100.0,  # kg/m3
    "sigma_HA": 0.087,  # Surface free energy (J/m2)
    "a_HA": 0.8,
    "b_HA": 0.047,
    "Ksp3": 10 ** (-58.333),  # Ca5(PO4)3OH Ksp (Thesis Eq. 5.15 & 5.27)
    "Np_HA": 8.62e6,  # Nucleated particles (Section 5.3.9.1)
    # Speciation at pH 7.5
    "gamma_Ca": 0.87,
    "gamma_PO4": 0.73,
    "gamma_CO3": 0.90,
    "alpha0": 0.0967,
    "kgl": 10 ** (-2.145),  # min^-1
    "CO3_sat": 10 ** (-5.0),
    "OH_act": 10 ** (-6.50),  # {OH-} at pH 7.5
}


# ==============================================================================
# 3. CFSTR GOVERNING EQUATIONS
# ==============================================================================
def cfstr_cp_odes(t, y, params, config):
  P, Ca, CO3, Ccal, Ps, CHA = y

  P = max(0.0, P)
  Ca = max(0.0, Ca)
  CO3 = max(0.0, CO3)
  Ccal = max(0.0, Ccal)
  Ps = max(0.0, Ps)
  CHA = max(0.0, CHA)

  Q = config["Q"]
  Pin = config["Pin"]

  kP = 10 ** params["log10_kP"]  # min^-1
  kHA = 10 ** params["log10_kHA"]  # umol/(m2*min)
  Omega_star_HA = params["Omega_star_HA"]
  kcal = 10 ** params["log10_kcal"]  # mol/(m2*min)

  # 1. Freundlich Adsorption on Calcite (Thesis Eq. 5.40)
  P_uM = P * 1e6
  Ps_star = 0.084 * (P_uM**0.31)
  dPs_dt = kP * (Ps_star - Ps)

  # 2. Calcite Dissolution
  IAP_cal = (config["gamma_Ca"] * Ca) * (config["gamma_CO3"] * CO3)
  f_G_cal = (IAP_cal / config["Ksp_cal"]) - 1.0
  r_diss_cal = -kcal * config["SSA_cal"] * Ccal * f_G_cal

  # 3. Hydroxyapatite Precipitation (Thesis Eq. 5.15, 5.27)
  # Free PO4^3- fraction at pH 7.5 is ~ 9.4e-6
  PO4_act = config["gamma_PO4"] * P * 9.4e-6
  Ca_act = config["gamma_Ca"] * Ca

  IAP_HA = (Ca_act**5) * (PO4_act**3) * config["OH_act"]

  # Driving force (Thesis Eq. 5.27)
  f_G_HA = max(0.0, (IAP_HA / config["Ksp3"]) - 1.0)

  # Mean ionic saturation state for nucleation threshold
  Omega_mean = (IAP_HA / config["Ksp3"]) ** (1.0 / 9.0) if IAP_HA > 0 else 0.0

  if Omega_mean >= Omega_star_HA or CHA > 1e-7:
    # Dynamic surface area of growing particles
    v_HA = config["MW_HA"] / (config["rho_HA"] * 1000.0)
    geom = 1.0 + (1.0 / config["a_HA"]) + (1.0 / config["b_HA"])
    ln_om = np.log(max(1.001, Omega_star_HA))
    Lc = geom * (v_HA * config["sigma_HA"]) / (8.314 * 298.15 * ln_om)

    # Effective mass & crystal size
    CHA_eff = max(
        CHA,
        1000.0
        * config["Np_HA"]
        * config["rho_HA"]
        * config["a_HA"]
        * config["b_HA"]
        * (Lc**3)
        / config["V"],
    )
    L = (
        CHA_eff
        * config["V"]
        / (
            1000.0
            * config["Np_HA"]
            * config["rho_HA"]
            * config["a_HA"]
            * config["b_HA"]
        )
    ) ** (1.0 / 3.0)
    SSA_HA = (
        2.0 * (config["a_HA"] * config["b_HA"] + config["a_HA"] + config["b_HA"])
    ) / (
        1000.0
        * config["rho_HA"]
        * config["a_HA"]
        * config["b_HA"]
        * max(1e-9, L)
    )

    # Precipitation rate (mol/(L*min))
    r_precip_HA = max(0.0, kHA * SSA_HA * CHA_eff * f_G_HA * 1e-6)
  else:
    r_precip_HA = 0.0

  # 4. Fluid Mass Balances (Thesis Eq. 5.37-5.42)
  dP_dt = (
      (Q / config["V"]) * (Pin - P)
      - Ccal * dPs_dt * 1e-6
      - 3.0 * r_precip_HA
  )
  dCa_dt = -(Q / config["V"]) * Ca + r_diss_cal - 5.0 * r_precip_HA
  r_gas = config["kgl"] * ((config["CO3_sat"] / config["alpha0"]) - CO3)
  dCO3_dt = -(Q / config["V"]) * CO3 + r_diss_cal + r_gas
  dCcal_dt = -r_diss_cal * config["MW_cal"]
  dCHA_dt = r_precip_HA * config["MW_HA"]

  return [dP_dt, dCa_dt, dCO3_dt, dCcal_dt, dPs_dt, dCHA_dt]


def run_forward_model(params, time_points, config):
  y0 = [0.0, 10 ** (-3.69), 10 ** (-5.0), config["Ccal_0"], 0.0, 0.0]

  sol = solve_ivp(
      fun=lambda t, y: cfstr_cp_odes(t, y, params, config),
      t_span=[0.0, max(time_points)],
      y0=y0,
      t_eval=time_points,
      method="BDF",
      rtol=1e-4,
      atol=1e-7,
  )
  if not sol.success:
    return None
  return sol.y[0]


# ==============================================================================
# 4. BAYESIAN INFERENCE (PRIORS & LIKELIHOOD)
# ==============================================================================
def log_prior(theta):
  log10_kP, log10_kHA, Omega_star_HA, log_sigma_P = theta

  # Priors centered on Thesis Table 5.5
  if not (-7.5 <= log10_kP <= -3.0):
    return -np.inf
  if not (-18.0 <= log10_kHA <= -13.0):
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
      "log10_kcal": -5.90,
  }

  P_sim = run_forward_model(params, exp_data["t_min"], config)
  if P_sim is None or np.any(np.isnan(P_sim)):
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
# 5. EXECUTION ACROSS ALL THREE FLOW RATES
# ==============================================================================
if __name__ == "__main__":
  results = {}
  ndim = 4
  nwalkers = 16
  n_burnin = 150
  n_steps = 350

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

    # Initial positions centered around Thesis Table 5.5
    init_pos = np.array([-5.20, -15.50, 1.16, -4.50]) + 0.05 * np.random.randn(
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

    # Subplot model fit
    ax = axes[idx]
    t_fine = np.linspace(0, max(t_data), 150)
    trajectories = []
    for s_idx in np.random.randint(len(samples), size=60):
      th = samples[s_idx]
      p_dict = {
          "log10_kP": th[0],
          "log10_kHA": th[1],
          "Omega_star_HA": th[2],
          "log10_kcal": -5.90,
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