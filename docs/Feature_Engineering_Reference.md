# 📒 The Book of Mirror: Volume V — Feature Engineering Reference

Mirror treats "Time" as a dynamic dimension. This volume documents the math and logic behind our automated feature extraction engine.

## 1. Autocorrelation (ACF) Synchronization

Traditional rolling windows (e.g., "last 3 days") are often misaligned with reality. If a signal cycles every 5 days, a 3-day window introduces noise.

### 1.1 The Synchronization Algorithm
1.  **Resampling**: Data is smoothly resampled to a daily frequency using forward-filling.
2.  **ACF Calculation**: We compute the Autocorrelation function for lags 1 through 15.
3.  **Optimal Lag**: The system identifies the first significant peak in the ACF.
4.  **Dynamic Windows**: Statistical features (slope, acceleration) are then computed using a window size equal to that peak.

## 2. Abstract Geometric Features

For any numeric column `X` in any role `R`, Mirror generates:

| Feature Name | Description |
|--------------|-------------|
| `{R}_{X}_mean` | Central tendency of the signal. |
| `{R}_{X}_std` | Volatility and stability measure. |
| `{R}_{X}_slope` | Linear trend (Regression coefficient over time). |
| `{R}_{X}_velocity` | Rate of change relative to the temporal span. |
| `{R}_{X}_ma3` | Short-term moving average (configured via `default_window_ma3`). |
| `{R}_{X}_deviation` | Current value delta from moving average (Anomaly score). |

## 3. Profile Semantic Embedding

Static profile data (JSON) is not just flattened; it is **contextualized**.
- **Compact Form**: Key-value pairs are converted into a Markdown list for the LLM.
- **Reasoning Grounding**: The Profile acts as a "Prior" for the Bayesian-like reasoning of the agents (e.g., "This heart rate spike is normal *for this specific patient age/weight*").

## 4. Extensibility Guide for Engineers

To add a new feature:
1.  Open `feature_engineer.py`.
2.  Add your math to `_extract_generic_features`.
3.  The feature will automatically be available to **all** manifests and **all** domains instantly.

---
*Developed for the Google Deepmind Agentic Challenge 2026.*
