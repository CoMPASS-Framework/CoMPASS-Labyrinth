# CoMPASS Level 1 - HMM Models

Hidden Markov Model implementations for inferring fine-grained motor states.

## Overview

This module implements Gamma-distributed HMM models to identify behavioral states based on:

- Step length distributions (Gamma)
- Turn angle distributions (von Mises)

The model identifies two primary states:

- **Surveillance**: Low step length + High turn angle
- **Ambulation**: High step length + Low turn angle

!!! info "Model Architecture"
    The GammaHMM uses a custom implementation with forward-backward and Viterbi algorithms optimized for behavioral state inference.

---

## Core Model Class

### GammaHMM

::: compass_labyrinth.compass.level_1.momentuPY.GammaHMM
    options:
      show_source: false
      members: true

---

## Model Fitting

### fit_best_hmm

::: compass_labyrinth.compass.level_1.momentuPY.fit_best_hmm

### save_compass_level_1_results

::: compass_labyrinth.compass.level_1.momentuPY.save_compass_level_1_results

---

## Parameter Estimation

### compute_parameter_ranges

::: compass_labyrinth.compass.level_1.momentuPY.compute_parameter_ranges

---

## Inference Algorithms

### forward_backward

::: compass_labyrinth.compass.level_1.momentuPY.forward_backward

### viterbi

::: compass_labyrinth.compass.level_1.momentuPY.viterbi

---

## Probability Distributions

### logpdf_gamma

::: compass_labyrinth.compass.level_1.momentuPY.logpdf_gamma

### logpdf_vonmises

::: compass_labyrinth.compass.level_1.momentuPY.logpdf_vonmises

---

## Utilities

### print_hmm_summary

::: compass_labyrinth.compass.level_1.momentuPY.print_hmm_summary

---

## Related

- [Data Preparation](level-1-prep.md) - Prepare data for HMM fitting
- [Visualization](level-1-viz.md) - Visualize HMM results
