# CoMPASS Level 2 - Models

Bayesian Gaussian Mixture Models and GMM-HMM for cognitive state inference.

## Overview

This module implements the Level 2 models that identify goal-directed cognitive states:

- **BGMM (Bayesian Gaussian Mixture Model)**: Clusters behavioral features
- **GMM-HMM**: Models temporal dependencies across states
- Cross-validation for model selection
- State sequence decoding

The models identify:

- **Oriented states**: Goal-directed behavior toward target
- **Non-Oriented states**: Exploratory or non-goal-directed behavior

!!! tip "Model Selection"
    Use `run_compass` with cross-validation to automatically select the optimal number of states.

---

## Main Modeling Function

### run_compass

::: compass_labyrinth.compass.level_2.model.run_compass

---

## Model Initialization

### initialize_bgmm

::: compass_labyrinth.compass.level_2.model.initialize_bgmm

### initialize_gmmhmm

::: compass_labyrinth.compass.level_2.model.initialize_gmmhmm

---

## Model Evaluation

### compute_aic

::: compass_labyrinth.compass.level_2.model.compute_aic

### visualize_cv_results

::: compass_labyrinth.compass.level_2.model.visualize_cv_results

### regularize_covariances

::: compass_labyrinth.compass.level_2.model.regularize_covariances

---

## State Analysis

### get_unique_states

::: compass_labyrinth.compass.level_2.model.get_unique_states

### generate_state_color_map

::: compass_labyrinth.compass.level_2.model.generate_state_color_map

### assign_reward_orientation

::: compass_labyrinth.compass.level_2.model.assign_reward_orientation

### assign_hhmm_state

::: compass_labyrinth.compass.level_2.model.assign_hhmm_state

---

## State Sequence Visualization

### plot_state_sequence_for_session

::: compass_labyrinth.compass.level_2.model.plot_state_sequence_for_session

### plot_state_sequences

::: compass_labyrinth.compass.level_2.model.plot_state_sequences

### plot_hhmm_state_sequence

::: compass_labyrinth.compass.level_2.model.plot_hhmm_state_sequence

---

## Related

- [Data Streams](level-2-datastreams.md) - Compute features for Level 2
- [Plotting](level-2-plots.md) - Additional visualization functions
- [Utilities](level-2-utils.md) - Helper functions
