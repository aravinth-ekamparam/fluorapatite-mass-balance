import numpy as np
import pandas as pd

def run_cfstr_mass_balance():
    # --- 1. Reactor Setup & Fixed Parameters ---
    V = 0.058          # Reactor volume (L)[cite: 1]
    Q = 0.03 / 1000.0  # Flow rate (L/min) [0.03 mL/min][cite: 1]
    S_calcite_0 = 4.0  # Calcite loading (g/L)[cite: 1]
    SSA_calcite = 3.0  # Specific surface area of calcite (m^2/g)[cite: 1]
    
    # Inlet Influent Concentrations (mM converted to moles/L)[cite: 1]
    C_in_F = 0.21 / 1000.0   # 0.21 mM F[cite: 1]
    C_in_P = 1.00 / 1000.0   # 1.00 mM PO4[cite: 1]
    C_in_Ca = 0.10 / 1000.0  # Initial dissolved Ca in equilibrium with calcite[cite: 1]
    
    # Kinetic Parameters (Log values from published fit)[cite: 1]
    k_gl = 10**(-4.06)      # CO2 gas-liquid transfer rate coefficient (min^-1)[cite: 1]
    k_F_ad = 10**(-6.47)    # Fluoride adsorption rate constant (min^-1)[cite: 1]
    k_P_ad = 10**(-4.07)    # Phosphate adsorption rate constant (min^-1)[cite: 1]
    k_HA = 10**(-21.74) * 1e-6 # HA precipitation rate constant (mol/m^2/min)[cite: 1]
    k_FA = 10**(-28.24) * 1e-6 # FA precipitation rate constant (mol/m^2/min)[cite: 1]
    
    Omega_star_HA = 1.38   # Critical supersaturation ratio for HA[cite: 1]
    Omega_star_FA = 1.37   # Critical supersaturation ratio for FA[cite: 1]
    
    # Initial Aqueous Concentrations in Reactor (moles/L)[cite: 1]
    C_F = 0.0
    C_P = 0.0
    C_Ca = C_in_Ca
    
    # Adsorbed Concentrations (moles/g calcite)[cite: 1]
    S_ad_F = 0.0
    S_ad_P = 0.0
    S_ad_F_eq = 1e-5 # Equilibrium capacity target[cite: 1]
    S_ad_P_eq = 5e-5 # Equilibrium capacity target[cite: 1]
    
    # --- 2. Forward Euler Numerical Integration ---
    t_end = 300.0    # Total runtime (minutes)[cite: 1]
    dt = 0.1         # Euler time step (minutes)[cite: 1]
    steps = int(t_end / dt)
    
    results = []
    
    for step in range(steps):
        t = step * dt
        
        # Approximate driving forces for precipitation[cite: 1]
        # In full model, SI is computed via speciation (Visual MINTEQ equilibrium logic)[cite: 1]
        IAP_HA = (C_Ca**5) * (C_P**3)
        Ksp_HA = 10**(-58.33)[cite: 1]
        Omega_HA = (IAP_HA / Ksp_HA)**(1/9)
        
        IAP_FA = (C_Ca**5) * (C_P**3) * C_F
        Ksp_FA = 10**(-59.74)[cite: 1]
        Omega_FA = (IAP_FA / Ksp_FA)**(1/9)
        
        # Classical Nucleation Rate Activation check[cite: 1]
        r_prec_HA = k_HA * SSA_calcite * S_calcite_0 * (Omega_HA - 1) if Omega_HA > Omega_star_HA else 0.0[cite: 1]
        r_prec_FA = k_FA * SSA_calcite * S_calcite_0 * (Omega_FA - 1) if Omega_FA > Omega_star_FA else 0.0[cite: 1]
        
        # Rates of Adsorption[cite: 1]
        r_ad_F = S_calcite_0 * k_F_ad * (S_ad_F_eq - S_ad_F)[cite: 1]
        r_ad_P = S_calcite_0 * k_P_ad * (S_ad_P_eq - S_ad_P)[cite: 1]
        
        # Forward Euler Liquid Mass Balances (dC/dt = Inflow - Outflow - Reaction Sinks)[cite: 1]
        dC_F_dt = (Q/V) * (C_in_F - C_F) - r_ad_F - r_prec_FA[cite: 1]
        dC_P_dt = (Q/V) * (C_in_P - C_P) - r_ad_P - (3.0 * r_prec_HA) - (3.0 * r_prec_FA)[cite: 1]
        dC_Ca_dt = (Q/V) * (C_in_Ca - C_Ca) - (5.0 * r_prec_HA) - (5.0 * r_prec_FA)[cite: 1]
        
        # Forward Euler Updates
        C_F += dC_F_dt * dt
        C_P += dC_P_dt * dt
        C_Ca += dC_Ca_dt * dt
        S_ad_F += (r_ad_F / S_calcite_0) * dt
        S_ad_P += (r_ad_P / S_calcite_0) * dt
        
        # Save output every 10 steps
        if step % 10 == 0:
            results.append({
                "time_min": t,
                "C_F_mM": C_F * 1000.0,
                "C_P_mM": C_P * 1000.0,
                "C_Ca_mM": C_Ca * 1000.0,
                "C_F_normalized": C_F / C_in_F,
                "C_P_normalized": C_P / C_in_P
            })
            
    df = pd.DataFrame(results)
    df.to_csv("cfstr_model_results.csv", index=False)
    print("Simulation completed successfully. Output saved to cfstr_model_results.csv.")

if __name__ == "__main__":
    run_cfstr_mass_balance()
