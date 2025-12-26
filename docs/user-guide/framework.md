# CoMPASS Framework

CoMPASS (Cognitive Mapping of Planned Actions with State Spaces) is a hierarchical probabilistic framework that integrates local movement dynamics with goal-directed cognitive states to decode latent behavioral states from complex spatial navigation data.

---

## Two-Level Hierarchy

### Level 1: Fine-Grained Motor States

**Goal:** Identify moment-to-moment behavioral states based on instantaneous movement kinematics.

**Modeling Strategy:**

- Features/Data Streams:
    - Step size (instantaneous displacement between frames)
    - Turn angle (change in heading direction)
- Apply a custom Hidden Markov Model (HMM) with specialized emission distributions designed for locomotor data:
    - Step lengths → Gamma distribution (models positive, right-skewed displacement data)
    - Turn angles → von Mises distribution (circular/directional model) OR Gamma distribution of absolute angles (magnitude model)
- Robust parameter initialization using median, IQR, and MAD statistics to handle tracking noise
- EM optimization with multiple algorithms (BFGS, L-BFGS-B, Nelder-Mead, Powell) tested across repetitions
- Behavioral constraints enforced: State 1 must have lower step + higher turn than State 2

**Output:**

- Hidden States decoded:
    - **State 1**: Low step length + High turn angle → **Active Surveillance**
    - **State 2**: High step length + Low turn angle → **Automated Ambulation**
- Per-frame latent state sequence (Viterbi decoding)
- Posterior state probabilities for uncertainty quantification

**Rationale:** The Gamma distribution is appropriate for step lengths because they are strictly positive and often right-skewed. The von Mises distribution is the circular analogue of the Gaussian distribution, making it ideal for directional data like turn angles.

---

### Level 2: Goal-Directed Cognitive States

**Goal:** Identify internal goal states that guide behavior in pursuit of the reward zone, overlaying the motor states with cognitive intent.

**Modeling Strategy:**

- Use the Level 1 state sequence as input
- Combine with reward-contextual features:
    - **Angular deviation** - Sternum-based angular deviation from the reward path
    - **Value-based distance** - Distance to target weighted by value function
    - **KDE-based proximity** - Kernel density estimate quantifying spatial proximity to target zone
- **Bayesian Gaussian Mixture Model (BGMM)** for robust initialization
- **GMM-HMM** (Gaussian Mixture Model - Hidden Markov Model) to capture temporal dependencies and transitions between goal states
- Leave-One-Session-Out (LOSO) cross-validation with phase-aligned training
- Early stopping based on log-likelihood trends (optional patience tuning)
- Model selection using AIC

**Output:**

- Level 2 States decoded (per session):
    - **Reward Oriented** - Goal-directed behavior toward target
    - **Non-Reward Oriented** - Exploratory or non-goal-directed behavior

**Final Hierarchical States:**

The framework combines Level 1 motor states with Level 2 cognitive states to produce **4 interpretable behavioral modes**:

1. **Active Surveillance, Reward Oriented** - Cautious, high-turning exploration while oriented toward reward
2. **Active Surveillance, Non-Reward Oriented** - Cautious, high-turning exploration without reward orientation
3. **Automated Ambulation, Reward Oriented** - Fast, directed movement toward reward
4. **Automated Ambulation, Non-Reward Oriented** - Fast, directed movement in non-rewarded directions

These composite states capture both the motor dynamics (how the animal moves) and the cognitive intent (what the animal is pursuing), providing a rich behavioral segmentation for downstream analysis.

