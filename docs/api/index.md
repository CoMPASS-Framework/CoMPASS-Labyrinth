# API Reference

This section provides complete API documentation for all CoMPASS-Labyrinth modules, automatically generated from docstrings in the source code.

## Overview

The CoMPASS-Labyrinth API is organized into several main modules:

### Core Module
Core utilities for project initialization, data loading, and figure management.

- [**Core Utilities**](core.md) - Project initialization, data loading, configuration

### Behavior Module
Tools for behavioral data preprocessing and analysis.

- [**Preprocessing**](behavior/preprocessing.md) - Data preprocessing utilities
- [**Pose Estimation**](behavior/pose-estimation.md) - DeepLabCut utilities
- [**Performance Metrics**](behavior/performance-metrics.md) - Task performance analysis
- [**Success Metrics**](behavior/success-metrics.md) - Success rate computation
- [**Trajectory Analysis**](behavior/trajectory-analysis.md) - Movement trajectory analysis
- [**Simulated Agents**](behavior/simulated-agents.md) - Agent-based modeling

### CoMPASS Module
Implementation of the two-level hierarchical probabilistic framework.

#### Level 1: Motor State Inference
- [**Data Preparation**](compass/level-1-prep.md) - Prepare data for Level 1 modeling
- [**HMM Models**](compass/level-1-model.md) - Hidden Markov Model classes and fitting
- [**Visualization**](compass/level-1-viz.md) - Level 1 visualization utilities

#### Level 2: Cognitive State Inference
- [**Data Streams**](compass/level-2-datastreams.md) - Feature computation for Level 2
- [**Models**](compass/level-2-model.md) - BGMM and GMM-HMM implementations
- [**Plotting**](compass/level-2-plots.md) - Level 2 visualization functions
- [**Utilities**](compass/level-2-utils.md) - Helper functions for Level 2

### Post-hoc Analysis Module
Tools for analyzing model outputs.

- [**Spatial Analysis**](post-hoc/spatial-analysis.md) - Spatial distribution analysis
- [**Temporal Analysis**](post-hoc/temporal-analysis.md) - Temporal dynamics analysis
- [**Bout Analysis**](post-hoc/bout-analysis.md) - Bout-wise analysis
- [**Grid Heatmaps**](post-hoc/grid-heatmap.md) - Grid-based visualization

## Quick Import Reference

```python
# Core functions
from compass_labyrinth import init_project, load_project

# Behavior analysis
from compass_labyrinth.behavior.preprocessing import preprocessing_utils
from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
    performance_metrics,
    success_metrics,
    trajectory_analysis
)

# CoMPASS Level 1
from compass_labyrinth.compass.level_1 import prep_data, momentu, visualization

# CoMPASS Level 2
from compass_labyrinth.compass.level_2 import datastreams, model, plots, utils

# Post-hoc analysis
from compass_labyrinth.post_hoc_analysis.level_1 import (
    spatial_analysis,
    temporal_analysis,
    bout_analysis,
    grid_heatmap
)
```

## Documentation Conventions

- **Parameters** Function parameters with type hints and descriptions
- **Returns:** Return values with type information
- **Examples:** Usage examples where available
- **Source:** Link to view source code on GitHub

All documentation is auto-generated from numpy-style docstrings in the source code.
