# Fluorapatite Mass Balance & Adsorption Kinetics

This repository runs the mass balance kinetic model for fluoride adsorption onto calcite in a continuous flow-stirred tank reactor (CFSTR).

## Normalized Model Fit vs. Non-Reactive Tracer

![Normalized Tracer Fit](normalized_tracer_fit.png)

## Optimized Kinetic Parameters

| Parameter | Fitted Value | Unit |
| :--- | :--- | :--- |
| **Adsorption Rate Constant ($k_{\text{F, ad}}$)** | **1.564** | $\text{L} \cdot \text{mol}^{-1} \cdot \text{min}^{-1}$ |
| **Max Capacity ($q_{\text{max, F}}$)** | **8.969e-05** | $\text{mol} \cdot \text{g}^{-1}$ |
| **$\log_{10}(k_{\text{F, ad}})$** | **0.1943** | — |
| **$\log_{10}(q_{\text{max, F}})$** | **-4.0472** | — |

## Summary Data Table

| Time (min) | $t/t_R$ | $F_{\text{obs}}$ (mM) | $F_{\text{pred}}$ (mM) | $C/C_{\text{in, obs}}$ | $C/C_{\text{in, pred}}$ | Non-Reactive Tracer ($C/C_{\text{in}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 150 | 0.0259 | 0.020 | 0.005 | 0.0962 | 0.0245 | 0.0255 |
| 450 | 0.0776 | 0.019 | 0.014 | 0.0914 | 0.0661 | 0.0747 |
| 750 | 0.1293 | 0.021 | 0.021 | 0.1005 | 0.0996 | 0.1213 |
| 1050 | 0.1810 | 0.024 | 0.027 | 0.1162 | 0.1267 | 0.1656 |
| 1350 | 0.2328 | 0.027 | 0.031 | 0.1305 | 0.1487 | 0.2077 |
| 1650 | 0.2845 | 0.030 | 0.035 | 0.1452 | 0.1669 | 0.2476 |
| 1950 | 0.3362 | 0.034 | 0.038 | 0.1605 | 0.1819 | 0.2855 |
| 2250 | 0.3879 | 0.035 | 0.041 | 0.1676 | 0.1946 | 0.3215 |
| 2550 | 0.4397 | 0.039 | 0.043 | 0.1857 | 0.2055 | 0.3557 |
| 2850 | 0.4914 | 0.042 | 0.045 | 0.1976 | 0.2149 | 0.3882 |
| 3150 | 0.5431 | 0.043 | 0.047 | 0.2057 | 0.2232 | 0.4191 |
| 3450 | 0.5948 | 0.045 | 0.048 | 0.2157 | 0.2307 | 0.4483 |


*Full dataset exported to `cfstr_fitted_adsorption_results.csv`.*