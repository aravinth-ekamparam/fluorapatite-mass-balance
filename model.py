import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Experimental data provided in image
time_obs = np.array([
    150, 450, 750, 1050, 1350, 1650, 1950, 2250, 2550, 2850, 3150, 3450, 3750, 4050, 
    4350, 4650, 4950, 5250, 5550, 5850, 6150, 6450, 6750, 7050, 7350, 7650, 7950, 
    8250, 8550, 8850, 9150, 9450, 9750, 10200, 10650, 10950, 11250, 11550, 11850, 
    12150, 12450, 12750
])

F_obs = np.array([
    2.02E-05, 1.92E-05, 2.11E-05, 2.44E-05, 2.74E-05, 3.05E-05, 3.37E-05, 3.52E-05, 
    3.90E-05, 4.15E-05, 4.32E-05, 4.53E-05, 4.80E-05, 4.86E-05, 5.12E-05, 5.39E-05, 
    5.28E-05, 5.60E-05, 5.79E-05, 5.90E-05, 6.34E-05, 6.17E-05, 6.34E-05, 6.34E-05, 
    6.34E-05, 6.88E-05, 6.76E-05, 6.65E-05, 7.31E-05, 7.68E-05, 7.52E-05, 7.52E-05, 
    7.79E-05, 7.81E-05, 8.08E-05, 8.23E-05, 8.13E-05, 7.90E-05, 8.08E-05, 7.96E-05, 
    8.34E-05, 8.23E-05
])

# Fixed parameters
V = 0.058             # L
Q = 0.01 / 1000.0     # L/min
S_calcite_0 = 4.0     # g/L
C_in_F = 0.21 / 1000.0# M (0.21 mM)
dt = 0.1              # min

def simulate(log_k_F, q_max_F):
    k_F_ad = 10**log_k_F
    t_max = time_obs[-1]
    steps = int(t_max / dt) + 1
    
    C_F = 0.0
    q_F = 0.0
    
    sim_t = []
    sim_C_F = []
    
    for step in range(steps):
        t = step * dt
        if t in time_obs or any(abs(t - to) < dt/2 for to in time_obs):
            sim_t.append(t)
            sim_C_F.append(C_F)
            
        r_ad_F = S_calcite_0 * k_F_ad * (q_max_F - q_F)
        dC_F_dt = (Q / V) * (C_in_F - C_F) - r_ad_F
        
        C_F += dC_F_dt * dt
        q_F += (r_ad_F / S_calcite_0) * dt
        
    # Interpolate C_F at exact observed times
    C_F_interp = np.interp(time_obs, np.arange(steps)*dt, [0.0] + list(np.cumsum([0.0]*steps)) if False else np.zeros(steps))
    return sim_C_F

# Let's write a proper numerical simulation for optimization
def run_model_for_times(params):
    log_k_F, log_q_max = params
    k_F_ad = 10**log_k_F
    q_max_F = 10**log_q_max
    
    t_eval = np.arange(0, time_obs[-1] + dt, dt)
    n_steps = len(t_eval)
    
    C_F_arr = np.zeros(n_steps)
    q_F_arr = np.zeros(n_steps)
    
    C_F = 0.0
    q_F = 0.0
    
    for i in range(1, n_steps):
        r_ad_F = S_calcite_0 * k_F_ad * max(0.0, q_max_F - q_F)
        dC_F_dt = (Q / V) * (C_in_F - C_F) - r_ad_F
        
        C_F += dC_F_dt * dt
        q_F += (r_ad_F / S_calcite_0) * dt
        
        C_F_arr[i] = C_F
        q_F_arr[i] = q_F
        
    # Interpolate to time_obs
    predicted = np.interp(time_obs, t_eval, C_F_arr)
    return predicted

def objective(params):
    pred = run_model_for_times(params)
    res = np.sum((pred - F_obs)**2)
    return res

res = minimize(objective, x0=[-6.47, -5.0], bounds=[(-10.0, -2.0), (-7.0, -3.0)], method='L-BFGS-B')
print("Optimal params:", res.x)
print("k_F_ad = 10^", res.x[0], "min^-1")
print("q_max_F = 10^", res.x[1], "mol/g")