# CoMPASS Level-1 (HMM Behavioral State Modeling)

This module documents the pipeline for **Hidden Markov Model (HMM)**-based classification of behavioral states using step length and turning angle parameters. It is designed to capture latent navigation states in freely behaving animals based on movement features.

---

## Objective

To identify and classify behavioral states into:
- **State 1:** Low step size, high angular variability (indicative of **Active Surveillance**).
- **State 2:** High step size, low angular variability (indicative of **Automated Ambulation**).
  
---

## Features & Workflow

1. **Preprocessing:**
   - Input coordinates in UTM format.

2. **Model Fitting:**
   - Fits multiple candidate HMMs across smoothing levels, angle types (circular vs absolute), and optimizers.
       - Tested a range of optimization methods to ensure convergence across parameter spaces.
   - Initialization parameters drawn from either:
       - Predefined ranges, or
       - Dynamically computed ranges using IQR, MAD, and circular dispersion.
   - Model selection based on:
       - Log-likelihood
       - AIC
       - Behavioral interpretability 

4. **Validation:**
   - Loop through parameter sets and retain only models that satisfy predefined state characteristics.
   - Option to parallelize model runs.

5. **Post-processing:**
   - Visualization of state transitions across space and time.
   - Mapping to decision points and trajectory segments for downstream analyses.
  
