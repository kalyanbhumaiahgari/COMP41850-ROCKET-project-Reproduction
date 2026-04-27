# ROCKET Reproduction — COMP41850: AI for Time Series

Reproduction and analysis of ROCKET (RandOm Convolutional KErnel Transform)
by Dempster et al. (2020) for time series classification.

**Report:** included in this repository as `AI_Time_Series.pdf`  
**Student:** Kalyan Kumar Bhumaiahgari (25234940), UCD

---

## Repository Structure

```
AI_Timeseries_Notebook.ipynb          # Full experiment notebook (A–J + scalability)
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
Open `AI_Timeseries_Notebook.ipynb` in Google Colab and run all cells.
Results are checkpointed automatically to `results/results_in_progress.csv`
after every method-dataset pair, so the run can be interrupted and resumed.

### 3.Regenerate plots
To regenerate plots only without rerunning experiments, 
run the last cell in the notebook. It loads results/results.csv 
and regenerates all five figures.

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

## Design Decisions and What We Learned

### Why Ridge and not a fancier classifier
Ridge regression with generalised cross-validation is mathematically
well-matched to ROCKET's feature regime: 20,000 features against
datasets with as few as 20 training examples. The GCV selects the
regularisation parameter without needing a held-out validation set,
which matters a lot when you only have 20 samples to begin with.
XGBoost and Random Forest make hard axis-aligned splits, so in a
20,000-dimensional space they ignore most of the informative features
in any given tree. We confirmed this empirically — even after adding
depth limits and heavy regularisation, XGBoost was 0.0245 below
baseline and took 110 seconds per dataset on average.

### Why PPP and not true MPV
The honest answer is that aeon's Rocket transformer does not expose
per-timestep feature maps, only the two aggregate statistics (max and
PPV) per kernel. A true Mean Positive Value would require computing
the mean of all activations above zero across the full convolution
output, which is not accessible without modifying the aeon source.
PPP = max * PPV is a composite of the two available statistics, not
an approximation of MPV. We renamed it specifically to avoid making
a false mathematical claim.

### Why PAM underperformed PPV
PAM uses the training-set mean PPV per kernel as its threshold,
making it a relative prevalence measure. PPV uses zero (the bias
threshold) as its anchor. On normalised UCR datasets, the feature
distributions are roughly symmetric around zero, so the zero threshold
and the mean threshold carry similar information. PAM adds no new
signal over PPV in this setting. It might be more useful on
non-normalised data where absolute activation levels vary widely
across datasets.

### Why the all-features combination did not beat the baseline
PPV+max+PPP+PAM is 4k features. Ridge with GCV handles the extra
dimensionality fine. But the reason it does not improve over 2k is
that PPP and PAM are not independent of PPV and max — PPP is literally
a product of them, and PAM is a binarisation of PPV. Ridge can already
learn to weight PPV and max in ways that implicitly capture their
interaction. Adding explicit interaction terms to a linear model only
helps if the interaction is nonlinear, which Ridge cannot model anyway.

### What we would do differently with more time
- Run a Wilcoxon signed-rank test between Method A and each other
  method across datasets, which is what the original paper does.
  We have the data to do this — it just was not done.
- Investigate whether the PPV-max accuracy gap correlates with series
  length. ItalyPowerDemand (length 24) is one of three cases where
  max beats PPV. With 21 datasets we have enough data to test this
  as a correlation, but we ran out of time before the submission.
- Implement a proper adaptive-k calibration using a held-out
  validation split rather than a fixed linear heuristic. The Lightning2
  exception (gap = -0.0295 at k=3000) suggests the linear formula
  breaks down on some small datasets.

### Why the checkpointing system was necessary
Google Colab disconnects sessions after 90 minutes of inactivity and
hard-limits sessions at 12 hours. Running 10 methods across 21
datasets with 10 seeds each produces 2,100 individual experiment runs.
XGBoost alone took 110 seconds per dataset on average, meaning the
full C-method pass took roughly 40 minutes. Without checkpointing,
a single disconnect would have lost hours of computation. The resume
bug (hollow rows being treated as completed) cost about 3 hours of
re-running before it was identified and fixed.
