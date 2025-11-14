# Grid Trajectory and Survival Analysis 

This module provides a two-step analysis pipeline to quantify and compare mouse navigation behavior within a labyrinth maze. It focuses on generating structured bout-level trajectory data and performing survival analysis to evaluate navigational efficiency.

---

## Key Features

### Trajectory Extraction (`generate_grid_trajectories.R`):

- Extracts entry and exit times into the maze for each bout using video frame data  
- Computes the number of grid nodes visited and tracking confidence per bout  
- Generates structured bout-wise trajectory lists per mouse with:
  - Grid node identity  
  - Entry frame and duration in frames  

### Survival Analysis (`generate_hazard_ratios.R`):

- Performs survival modeling to compare genotypes (e.g., WT vs. AppSAA)  
- Evaluates latency (in frames and steps) to reach predefined target nodes  
- Computes hazard ratios and visualizes survival distributions  
