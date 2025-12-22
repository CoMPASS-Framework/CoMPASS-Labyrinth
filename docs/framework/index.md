# CoMPASS Framework

!!! info "Coming Soon"
    Detailed framework documentation is currently being developed. Please check back soon!

## Overview

CoMPASS (Computational Model for Probabilistic Assessment of Spatial Strategies) is a hierarchical probabilistic framework that integrates local movement dynamics with goal-directed cognitive states to decode latent behavioral states from complex spatial navigation data.

## Two-Level Hierarchy

### Level 1: Fine-Grained Motor States

**Goal:** Identify moment-to-moment behavioral states based on raw or smoothened movement features.

**Modeling Strategy:**
- Apply Hidden Markov Models (HMM) with Gaussian emissions to short-timescale features
- Features/Data Streams:
  - Step size
  - Turn angle
  - (Optionally) Smoothed or Log-Transformed variants

**Output:**
- Hidden States decoded:
  - **State 1**: Low step length + High turn angle → **Surveillance**
  - **State 2**: High step length + Low turn angle → **Ambulation**
- Per-frame latent state sequence

---

### Level 2: Goal-Directed Cognitive States

**Goal:** Identify internal goal states that guide transitions across decision points in the maze, in pursuit of the target zone.

**Modeling Strategy:**
- Use the Level 1 state sequence as input
- Combine with reward-contextual features:
  - Sternum-based angular deviation (from path leading to the reward path)
  - Value-based distance to target
  - KDE-based proximity to target zone
- Fit a Bayesian Gaussian Mixture Model (BGMM) to extract latent states (clusters)
- Feed BGMM outputs into a GMM-HMM to model longer timescale dependencies across time and maze structure

**Output:**
- States decoded:
  - **Oriented** - Goal-directed behavior toward target
  - **Non-Oriented** - Exploratory or non-goal-directed behavior
- Per-frame latent state sequence

---

## Composite States

Combining Level 1 and Level 2 states yields four composite behavioral states:

1. **Ambulatory, Oriented** - Direct goal-directed navigation
2. **Ambulatory, Non-Oriented** - Fast exploration
3. **Active Surveillance, Oriented** - Careful goal-directed navigation
4. **Active Surveillance, Non-Oriented** - Local exploration/assessment

## Key References

For more detailed information, see:

- [Tutorials](../tutorials.md) - Practical examples of running CoMPASS
- [API Reference](../api/index.md) - Complete technical documentation
- Source code in `src/compass_labyrinth/compass/`

## Mathematical Details

!!! note "Under Development"
    Detailed mathematical formulations, model equations, and algorithm descriptions will be added here.

## Validation

!!! note "Under Development"
    Information about model validation, cross-validation strategies, and performance metrics will be added here.
