# ROCKET Reproduction — COMP41850: AI for Time Series

Reproduction and analysis of ROCKET (RandOm Convolutional KErnel Transform)
by Dempster et al. (2020) for time series classification.

**Report:** included in this repository as `AI_Time_Series.pdf`  
**Student:** Kalyan Kumar Bhumaiahgari (25234940), UCD

---

## Repository Structure

```
TImeseries.ipynb          # Full experiment notebook (A–J + scalability)
results/
  results.csv             # All results: 21 datasets x 10 methods x 10 seeds
  scalability.csv         # HandOutlines + ElectricDevices, Method A, 3 seeds
  accuracy_comparison.png
  ppv_vs_max.png
  pooling_comparison.png
  adaptive_kernel.png
  time_comparison.png
```

---

## How to Reproduce

### 1. Install dependencies
```bash
pip install aeon==1.4.0 scikit-learn==1.6.1 xgboost seaborn matplotlib
```

### 2. Run in Google Colab
Open `TImeseries.ipynb` in Google Colab and run all cells.
Results are checkpointed automatically to `results/results_in_progress.csv`
after every method-dataset pair, so the run can be interrupted and resumed.


---

## Experimental Methods

| Method | Description |
|--------|-------------|
| A | ROCKET + Ridge (baseline, reproduces paper) |
| B | ROCKET + Random Forest (head ablation) |
| C | ROCKET + XGBoost (head ablation) |
| D | MiniROCKET (efficiency comparison) |
| E | ROCKET PPV-only (feature ablation) |
| F | ROCKET max-only (feature ablation) |
| G | ROCKET PPP-only (novel pooling operator) |
| H | ROCKET PAM-only (novel pooling operator) |
| I | ROCKET all-features: PPV+max+PPP+PAM |
| J | ROCKET adaptive-k (novel efficiency heuristic) |

---

## Key Results

- Baseline reproduction mean delta vs paper: **-0.0016** across 21 datasets
- PPV wins over max on **12 of 21** datasets
- Adaptive-K matches baseline within **0.0011** while reducing kernels by up to 90%
- PPP-only achieves **+0.0008** over baseline (marginal improvement)

---

## Notes

- Method G is labelled `ppp-only` throughout the report and results CSV.
  An earlier version of the notebook used `mpv-only` before the naming
  was corrected to Peak-Prevalence Product (PPP). The fix cell at the
  bottom of the notebook regenerates the pooling_comparison figure
  with the correct label.
- XGBoost (Method C) required additional constraints to converge on
  high-dimensional datasets: `max_depth=3`, `reg_lambda=10`,
  `colsample_bytree=0.1`, `tree_method='hist'`.
- PAM (Method H) computes thresholds on training data only to avoid
  data leakage. See Section III-E of the report.
