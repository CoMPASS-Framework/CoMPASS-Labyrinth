# Task Performance Analysis

A modular toolkit to analyze spatial and temporal task performance within the complex labyrinth maze, enabling quantification and visualization of behavioral engagement and performance dynamics.


It supports structured analysis of:
- Time-binned regional occupancy
- Target zone usage and exclusion filtering
- Heatmap visualizations of session-wise behavior
- Shannon entropy for behavioral diversity
- Cumulative successful bout tracking over time
- Deviation from Optimal Path and Bout-wise Velocity Profiles

---

##  Features

### - Binning & Regional Analysis (`performance_metrics.py`)
- Compute time-binned occupancy of regions (e.g., Loops, Reward Path, Dead Ends)
- Normalize occupancy by region length and session duration

### - Session Filtering (`performance_metrics.py`)
- Apply thresholds based on:
  - Target zone usage (mean per session)
  - Total number of frames

### - Plotting Utilities (`performance_metrics.py`)
- Stacked region heatmaps per time bin (aligned across sessions)
- Target Usage vs. Frames scatterplots (per session, genotype, sex)

### - Shannon Entropy (`performance_metrics.py`)
- Quantifies behavioral variability across sessions within each bin
- Tracks entropy dynamics over time by genotype

### - Successful Bout Analysis (`success_metrics.py`)
- Track cumulative count of successful bouts per session or over time

### - Trajectory Deviation and Movement Dynamics Across Bouts (`trajectory_analysis.py`)
- Calculates spatial or topological deviation from the optimal path
- Supports visualization of deviation trends across sessions or genotypes
