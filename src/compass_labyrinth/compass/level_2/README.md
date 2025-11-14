# CoMPASS Level-2

This package implements a hierarchical probabilistic framework for analyzing behavioral navigation data, specifically designed for multi-session rodent maze experiments involving naturalistic movement and reward-based decision-making. It integrates domain-specific features with robust modeling and interpretable post-hoc analysis tools.

---

## Key Features

- **Data-Stream Initialization** (`datastreams.py`):
  - Scaled Kernel Density Estimates
  - Angular deviation from reference vector
  - Value-weighted reward distance
  - (HMM (Level 1) binary state sequence --> created within Level 1, but output required here)

- **Modeling** (`model.py`):
  - Bayesian Gaussian Mixture Models (BGMM) for initialization
  - GMM-HMM fitting using with custom emission probabilities
  - Covariance regularization
  - CoMPASS architecture: Leave-One-Session-Out (LOSO) & phase-aligned validation, with Early stopping based on log-likelihood trends
  - Log-likelihood and AIC tracking for model selection
  - Visualize CV results based on different parameter combination trends
  - Modular CoMPASS Labeling
    - Assigns 'Reward Oriented' or 'Non-Reward Oriented' to Level 2 states 
    - Dynamically infers behavioral orientation per session
    - Combines Level 1 + Level 2 into:
      - 'Active Surveillance, Reward Oriented'
      - 'Active Surveillance, Non-Reward Oriented'
      - 'Ambulatory, Reward Oriented'
      - 'Ambulatory, Non-Reward Oriented'
    - (Can be adjusted/modified as per user experiment/goals)

- **Plotting** (`plots.py`):
  - Spatial heatmaps
  - KDE overlays
  - CoMPASS Sequence Visualization
  - Timeline plots of CoMPASS states per session

- **Utilities** (`utils.py`):
  - Assign Bout numbers in the dataframe
  - Create 'Phases' based on the Bout numbers
 
- **Config** (`compass_config.py`):
  - Path Initializations
  - Pre-fixed values (not to be changed for the current labyrinth maze setup)


