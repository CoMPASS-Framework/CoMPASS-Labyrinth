# CoMPASS Level-1: Custom HMM for Locomotor Behavioral State Modeling

The **CoMPASS Level-1 module** implements a fully custom **Hidden Markov Model (HMM)** that segments continuous locomotion into short-timescale behavioral states using instantaneous kinematic features.  

---

## Objective

Identify fundamental locomotor motifs that form the building blocks of navigation:

| State | Description | Movement Signature |
|:------|:-------------|:-------------------|
| **State 1** | **Active Surveillance** | Low step length + High turning variability |
| **State 2** | **Automated Ambulation** | High step length + Low turning variability |

These states capture exploratory versus surveillance movement dynamics that precede higher-order decision processes.

---

## Model Structure

Each observation corresponds to a single frame of movement, represented by:
- **`step`** → instantaneous displacement (cm · frame⁻¹)
- **`angle`** → change in heading between consecutive steps (radians)

To represent both **linear** and **circular** motion within one probabilistic framework:
- Step lengths are modeled with a **Gamma distribution** (positive, right-skewed data)
- Turning angles are modeled using either:
  - **von Mises distribution** — directional model  
  - **Gamma distribution of absolute angles** — magnitude model  

This hybrid Gamma–von Mises (or Gamma–Gamma) formulation generalizes standard Gaussian HMMs to capture realistic locomotor statistics.

---

## Parameter Initialization

Initialization is **data-driven and robust to outliers**:

- **Step mean / variance:** median ± interquartile range (IQR)
- **Step SD range:** median absolute deviation (MAD × 1.4826)
- **Angle mean / concentration:** circular mean and resultant length
- **Angles:** wrapped within (−π, π] to preserve directional continuity  

These robust statistics prevent spurious tracking jumps or noise from biasing model initialization.

---

## Optimization and Fitting

Model parameters — start probabilities, transition matrix, and emission parameters — are estimated by **maximizing the total log-likelihood** via a custom **Expectation–Maximization (EM)** routine.

### EM Steps
- **E-step:** posterior state probabilities computed using a numerically stable forward–backward algorithm  
- **M-step:** emission parameters re-optimized with `scipy.optimize.minimize`

### Optimizers Tested
- `BFGS`  
- `L-BFGS-B`  
- `Nelder-Mead`  
- `Powell`

### Convergence Criteria
- `max_iter` = 200  
- `tol` = 1e-4 change in log-likelihood  

Models are compared using **Akaike Information Criterion (AIC)**, and the best-scoring model satisfying behavioral constraints is retained.

---

## Behavioral Constraint Enforcement

State interpretability is standardized post-hoc:
- **State 1 = Low step + High turning** → Active Surveillance  
- **State 2 = High step + Low turning** → Automated Ambulation  

Models failing this constraint are automatically rejected.

---

## Inference and Outputs

- **Viterbi decoding:** most probable hidden-state sequence per frame  
- **Posterior probabilities:** soft assignments for each time point  

**Output columns include:**
- `HMM_State` (1 or 2)  
- `Post_Prob_1`, `Post_Prob_2`  
- Model diagnostics (AIC, log-likelihood, start probabilities, transition matrix)

---

## Key Function Parameters

| Parameter | Description |
|:-----------|:-------------|
| `n_states` | Number of hidden states (default = 2) |
| `n_repetitions` | Repetitions per optimizer for robustness (default = 20) |
| `opt_methods` | Optimizers to test (`BFGS`, `L-BFGS-B`, `Nelder-Mead`, `Powell`) |
| `max_iter` | Maximum EM iterations (default = 200) |
| `use_abs_angle` | `(True, False)` → Gamma(abs(angle)) or von Mises(angle) |
| `stationary_flag` | `"auto"` for automatic detection or manual `True/False` |
| `angle_mean_biased` | Initial directional bias for VM branch (default = (π/2, 0.0)) |
| `use_data_driven_ranges` | Compute parameter ranges via IQR/MAD (default = True) |
| `seed` | Random seed for reproducibility (default = 123) |
| `show_progress` | Display progress bar during fitting (default = True) |

---

## Output Files

Saved using `save_compass_level_1_results()`:

- `model_summary.json` – fitted parameters and diagnostics  
- `data_with_states.csv` – dataset with HMM state labels  
- `model_selection_records.csv` – all model attempts and metrics  
- `fitted_model.joblib` – serialized HMM object for reuse  

---

## Interpretation

The CoMPASS Level-1 model provides a **behaviorally constrained segmentation** of locomotion into interpretable states of exploration and ambulation.  
These decoded sequences and their transition probabilities serve as the **foundational latent variables** for Level-2 goal-directed (BGMM → GMM-HMM) modeling.
