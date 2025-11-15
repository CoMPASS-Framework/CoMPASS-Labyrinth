# CoMPASS Level-2: Hierarchical HMM for Goal-Directed Navigation State Modeling

The **CoMPASS Level-2** module implements a **Hierarchical Hidden Markov Model**  that integrates locomotor motifs with spatial and task-relevant features to infer long-timescale, goal-directed navigation states.

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
 
---

## Objective

Identify **latent goal-directed navigation strategies**:

| Level-2 State | Interpretation | Feature Signature |
|---------------|----------------|-------------------|
| **Reward-Oriented Navigation** | Direct, structured approach | High KDE on optimal path, low distance-to-reward, aligned heading |
| **Non-Reward Oriented Navigation** | Off-path search | Higher heading variability, off-path KDE, low alignment |

---

## Model Structure

Each observation vector includes:

- `HMM_State` → Level-1 locomotor state  
- `KDE` → spatial occupancy density  
- `VB_Distance` → value-weighted distance to reward  
- `Targeted_Angle_smooth_abs` → alignment to reward direction, smoothed and taking an absolute value 

The Level-2 model consists of:

- **Gaussian Mixture emissions (GMM)**  
- **Hidden Markov transitions (HMM)**  
- **BGMM initialization** to enhance stability  

---

## Early Stopping and Patience Tuning

During inner-CV:

- Log-likelihood improvement is monitored
- Training halts when `no_improve >= patience`
- If `patience == "tune"`:
  - The routine automatically selects the **optimal patience window**
  - Based on average validation log-likelihood

This prevents overfitting and accelerates computation.

---

## Optimization and Fitting

For each hyperparameter configuration:

1. **Fit BGMM** on training data  
2. **Regularize covariances**  
3. **Initialize GMM-HMM** with tiled means/covariances  
4. **Fit HMM** using EM (`model.fit`)  
5. **Score validation data** (`model.score`)  
6. **Compute AIC** on training data  

Best model per phase/session is chosen via:

- Highest **validation log-likelihood**
- Lowest **AIC**

---

## Inference and Outputs

- Apply best model to held-out session per phase
- Predict Level-2 states (`model.predict`)
- Assign decoded strategy labels as `Level_2_States`

Outputs are concatenated across folds and saved to:  `csvs/combined/hhmm_state_file.csv`


---

## Key Function Parameters

| Parameter | Description |
|----------|-------------|
| `phase_options` | Number of temporal bins for phase alignment |
| `ncomp_options` | BGMM mixture components to search |
| `k_options` | Number of Gaussian mixtures per HMM state |
| `reg_options` | Covariance regularization strengths |
| `terminal_values` | Grid nodes defining bout termination |
| `patience` | Patience setting or `"tune"` for auto-selection |
| `patience_candidates` | Patience values to search |
| `features` | Columns used for Level-2 modeling |

---

## Output Files

- **`hhmm_state_file.csv`** — final Level-2 state assignments  
- **Log-likelihood and AIC records** for all candidate models  
- **Per-fold diagnostic summaries**  
- **Best model per CV fold** retained in memory  

---

## Interpretation

CoMPASS Level-2 reconstructs the **latent cognitive organization** of navigation by identifying when mice:

- Shift into **reward-oriented** goal-directed behavior  
- Explore via **non-reward** strategies  
- Transition between strategies across bouts and across learning phases  
- Express stable, repeatable state sequences across sessions  

This hierarchical decoding links **movement**, **navigation strategy**, and **neural dynamics**, enabling an integrated view of behavioral computation.
