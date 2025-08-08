# Path-Dependent Volatility Forecasting (TSPL VIX Modeling)

**Modular Python toolkit implementing the path-dependent volatility model from _Guyon & Lekeufack (2023)_, reproducing and extending their results for VIX forecasting.**  
Outperforms baseline realized volatility and HAR models on multiple indices (S&P 500, NDX, DJI).

---

## Objective

Predict the **implied volatility index (VIX)** from past returns of an underlying index (e.g. S&P 500) using  
**time-shifted power-law (TSPL) kernels** to build long-memory features:

- **R₁** → trend (short-term memory component)  
- **R₂** → volatility (long-term memory component)

The model fits:

\[
\text{VIX}_t \approx a \cdot R₁_t + b \cdot \sqrt{\tilde{R₂_t}}
\]

where \(\tilde{R₂_t}\) is an exponentially smoothed version of \(R₂\).

---

## Scientific Motivation

The approach from Guyon & Lekeufack (_"Volatility is Mostly Path-Dependent"_, 2023):

- Captures the **persistent memory structure** in volatility.
- Encodes **delayed and instantaneous responses** of implied vol to past returns.
- Achieves high **out-of-sample R² ≈ 0.87** on S&P 500 VIX.
- **Generalizes** to other implied vol indices (VXN, VXD…).

---

## Project Structure

```
data.py          # Data download, caching & preprocessing (Yahoo Finance)
features.py      # TSPL kernel construction & feature computation (R₁, R₂)
models.py        # TSPL model, Realized Vol baseline, HAR model
evaluate.py      # Train/test split & cross-validation utilities
optimization.py  # Parameter calibration via scipy.optimize.minimize
plotting.py      # Visualization utilities (kernels, fits, residuals…)
demo.ipynb       # End-to-end example & key result reproduction
requirements.txt # Dependencies
README.md        # This file
```

---

##  Quick Start

### Clone & install
```bash
git clone https://github.com/Omartfz/pathdependent-volatility.git
cd pathdependent-volatility
pip install -r requirements.txt
```

### Run the demo notebook
```bash
jupyter notebook demo.ipynb
```
The notebook downloads market data, computes TSPL features, calibrates the model, and compares it against baselines.

---

## Key Results

| Model                   | RMSE (test) | R² (test) |
|-------------------------|-------------|-----------|
| **TSPL (optimized)**    | 2,82        | 0.87      |
| HAR                     | 4,34        | 0.70      |
| Realized Vol (30-day)   | 5,30        | 0.56      |

> **Note**: Numbers from S&P 500 / VIX split 2000-2018 (train) vs 2019-2025 (test).  
> TSPL consistently outperforms in both accuracy and stability.


---

## References

- J.-P. Guyon & F. Lekeufack (2023). [_Volatility is Mostly Path-Dependent_]([https://doi.org/10.1080/14697688.2023.2221281]).


---

##  License

MIT License – see [LICENSE](LICENSE) for details.
