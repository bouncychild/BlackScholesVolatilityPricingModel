# 📊 Black-Scholes Volatility Pricing Model

A **Streamlit** web app that calculates European option prices using the **Black-Scholes model** and visualizes Greeks, volatility surfaces, and price sensitivities.

---

## ✨ Features

### ✅ Option Pricing
- Calculates **Call & Put prices** using the Black-Scholes formula  
- Supports **dividend yield** (extended Black-Scholes)

### 📊 Interactive Visualizations
- **3D Surface Plots** for Call/Put prices vs. Spot Price & Volatility  
- **Greeks Analysis**: Delta, Gamma, Vega, Theta, Rho  
- **Heatmaps** for quick price comparisons

### 📈 Sensitivity Analysis
- Adjust **spot price**, **strike**, **volatility**, **interest rates**, and **time to maturity**  
- Real-time updates on **option prices** and **Greeks**

### 🧮 Advanced Features
- Full Greeks calculation (**first** and **second-order** sensitivities)  
- Handles edge cases (e.g., expiration, zero volatility)  
- Responsive UI with clean, modern styling

---

## 🚀 Usage

### Input Parameters
| Parameter | Description |
|-----------|-------------|
| **Current Asset Price (S)** | Underlying asset's current price |
| **Strike Price (K)** | Option strike price |
| **Time to Maturity (T)** | Years until expiration |
| **Volatility (σ)** | Annualized volatility (%) |
| **Risk-Free Rate (r)** | Annual interest rate (%) |

### Outputs
- **Call & Put Prices**: Calculated option values
- **Option Greeks**:
  - **Delta (Δ)**: Price sensitivity to underlying asset
  - **Gamma (Γ)**: Delta's sensitivity to spot price
  - **Vega (ν)**: Sensitivity to volatility changes
  - **Theta (Θ)**: Time decay effect
  - **Rho (ρ)**: Sensitivity to interest rates

---

## 📂 Tabs
- **3D Price Surface** – Interactive 3D plots for Call/Put prices
- **Greeks Analysis** – Visualize Greeks vs. spot price
- **Heatmaps** – Price matrices for different spot/volatility levels

---

## 📐 Model Formulas

The model calculates the theoretical price of options using the following formula for a call option:

$$
C = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

Where:
- \( C \) = Call option pr
