# HMM (Level 1) Post-Analysis

This module contains post-hoc analysis and visualization tools for interpreting behavioral states derived from HMM Level 1 model. The aim is to characterize how **internal locomotor states** vary across **region types**, **node types**, and **bout-level behaviors**.

---

## Core Features

- **Heatmap Representations of HMM State Probabilities** (`grid_heatmap.py`)
  - Grid-wise probability heatmaps showing the proportion of time spent in a specific HMM state across all sessions.
  - Interactive version of the visualization available too.


- **Temporal Evolution of State Probabilities at Nodes** (`temporal_analysis.py`)
  - Tracks **probability of being in a chosen HMM state** over **time bins** per genotype and session-averaged plots.


- **State Distributions by Region and NodeType** (`spatial_analysis.py`)
  - Comparison of proportion of time spent in a state across Maze regions and Node types.
  - Allows genotype level comparisons behavioral states.


- **Bout-Type based State Comparisons** (`bout_analysis.py`)
  - Classifies bouts as **successful** or **unsuccessful** based on target reach.
  - Computes and compares HMM state proportions across these bout types.

  
