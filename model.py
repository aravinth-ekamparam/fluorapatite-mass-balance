import numpy as np
import pandas as pd

def run_cfstr_calcite_fluoride_model():
    """
    Forward Euler CFSTR Mass Balance Model tuned for:
    Experiment: Calcite + Fluoride (C+F)
    Flow Rate (Q): 0.01 mL/min
    Influent F Concentration: 0.21 mM
    Calcite Loading: 4 g/L
    Simulated duration: 25 residence times (t/tR = 25)
    """
    # --- 1. Reactor Setup & Experimental Constants ---
    V = 0.058             # Reactor volume (L) [58 mL]
    Q = 0.01 / 1000.0     # Flow rate (L/min) [0.01 mL/min]
    S_calcite_0 = 4.0     # Calcite loading (g/L)
    SSA_calcite = 3.0     # Specific surface area of calcite (m^2/g)
    
    # Residence Time Calculation
    tR = V / Q             # Residence time tR = 5800 minutes
    
    # Inflow Influent Concentrations (moles/L)
    C_in_F = 0.21 / 1000.0   # Influent Fluoride: 0.21 mM
    C_in_Ca = 0.10 / 1000.0  # Initial dissolved Calcium in equilibrium with calcite
    
    # Fitted Kinetic Parameters from C+F Control System
    k_gl = 10**(-4.06)      # Gas-liquid CO2 exchange coefficient (min^-1)
    k_F_ad = 10**(-6.47)    # Fluoride adsorption rate constant onto calcite (min^-1)
    
    # Langmuir Adsorption Isotherm Constants for F on Calcite
    q_max_F = 1.0e-5        # Maximum adsorption capacity (moles F / g calcite)
    
    # --- 2. Initial State Variables ---
    C_F = 0.0               # Initial reactor fluoride (moles/L)
    C_Ca = C_in_Ca          # Initial reactor calcium (moles/L)
    q_F = 0.0               # Initial sorbed fluoride on calcite (moles/g)
    
    # --- 3. Forward Euler Time Discretization ---
    dt = 0.1                # Integration time step (minutes)
    t_end = 25.0 * tR       # Total simulation time = 25 * tR = 145,000 minutes
    steps = int(t_end / dt)
    
    results = []
    
    # --- 4. Simulation Loop ---
    for step in range(steps):
        t = step * dt
        
        # Rate of Fluoride Adsorption onto Calcite (moles/L/min)
        r_ad_F = S_calcite_0 * k_F_ad * (q_max_F - q_F)
        
        # Liquid Mass Balance Equations (Forward Euler)
        dC_F_dt = (Q / V) * (C_in_F - C_F) - r_ad_F
        dC_Ca_dt = (Q / V) * (C_in_Ca - C_Ca)
        
        # State Updates
        C_F += dC_F_dt * dt
        C_Ca += dC_Ca_dt * dt
        q_F += (r_ad_F / S_calcite_0) * dt
        
        # Record Output every 10 minutes (100 steps)
        if step % 100 == 0:
            results.append({
                "time_min": round(t, 2),
                "t_over_tR": round(t / tR, 4),
                "C_F_mM": C_F * 1000.0,
                "C_Ca_mM": C_Ca * 1000.0,
                "C_F_normalized": C_F / C_in_F,
                "sorbed_F_umol_g": q_F * 1e6
            })
            
    # Save Results to CSV
    df = pd.DataFrame(results)
    df.to_csv("cfstr_model_results.csv", index=False)
    print("Simulation completed successfully!")
    print(f"Total simulated time: {t_end / tR:.1f} t/tR ({t_end:.0f} minutes)")
    print(f"Final normalized fluoride concentration (C/Cin): {df['C_F_normalized'].iloc[-1]:.4f}")

if __name__ == "__main__":
    run_cfstr_calcite_fluoride_model()