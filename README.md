# volatility-path-dependence

Modular Python implementation of the implied volatility model inspired by  
**Guyon & Lekeufack (2023) — _"Volatility is Mostly Path-Dependent"_**

---

##  Objective

Predict the **implied volatility index (VIX)** using past returns of the S&P 500.  
This is achieved by constructing **long-memory features** via TSPL kernels:

- **R₁**: trend (captures short-term memory)
- **R₂**: volatility (captures long-term memory)

---

## Scientific Motivation

The model expresses the VIX as a linear function of:

- **R₁** → trend from past returns (delayed response)
- **√R₂** → smoothed realized volatility (instantaneous response)

This simple approach:
- Achieves high out-of-sample performance (**R² ≈ 87%**),
- Generalizes well to other indices (**NDX**, **DJI**, etc.).

---


## Installation

```bash
git clone https://github.com/Omartfz/volatility-path-dependence.git
cd volatility-path-dependence
pip install -r requirements.txt
