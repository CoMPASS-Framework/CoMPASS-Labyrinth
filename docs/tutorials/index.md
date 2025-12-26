# Overview

This section provides step-by-step tutorials for using CoMPASS-Labyrinth to analyze behavioral data from the labyrinth navigation task.

## Available Tutorials

### [00. DLC Grid Processing](00_dlc_grid_processing.md)
Process video data with DeepLabCut, create spatial grids, and annotate trajectories with grid locations for labyrinth navigation analysis.

### [01. Create Project](01_create_project.md)
Initialize a new CoMPASS-Labyrinth project, ingest DeepLabCut results, and preprocess combined session data for downstream analysis.

### [02. Task Performance Analysis](02_task_performance_analysis.md)
Analyze task performance metrics including spatial heatmaps, Shannon entropy, region usage, bout-level success rates, and deviation from optimal paths.

### [03. Simulated Agent Modelling](03_simulated_agent_modelling.md)
Compare animal navigation strategies to simulated agents using chi-square analysis, multi-agent comparisons, and exploration-exploitation modeling.

### [04. CoMPASS Level 1](04_compass_level_1.md)
Fit Hidden Markov Models to infer fine-grained motor states (surveillance vs. ambulation) from step length and turn angle distributions.

### [05. CoMPASS Level 1 Post-Analysis](05_compass_level_1_post_analysis.md)
Perform post-hoc analysis of Level 1 HMM results including bout-level state analysis, spatial mapping, and temporal dynamics visualization.

### [06. CoMPASS Level 2](06_compass_level_2.md)
Apply hierarchical modeling to integrate multiple behavioral and physiological data streams for multi-scale inference of cognitive states.

---

## Prerequisites

- Python environment with CoMPASS-Labyrinth installed
- DeepLabCut pose estimation results (or raw videos for Tutorial 00)
- Project metadata file (Excel/CSV format)
- Basic familiarity with Jupyter notebooks

For installation instructions, see the [Installation Guide](../user-guide/installation.md).
