# Resources

This folder contains reusable static files and reference maps required by various modules. These are not raw data files, but configuration-driven or model-specific resources that guide downstream processing and analysis.

---

## Contents

### `Value_Function_perGridCell.csv`
- Defines a precomputed or task-derived value assigned to each grid node in the maze.
- Used within CoMPASS

### `4step_adjacency_matrix.csv`
- Binary matrix that defines valid transitions between neighboring grid nodes in the maze — where only upto 4 connections/steps in 0.2 sec (1 frame) are allowed
- Used within data preprocessing
