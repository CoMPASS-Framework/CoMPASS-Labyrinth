# CHANGELOG.md

# v0.2.0

Highlights:
- Integrated the `movement` I/O interface for loading DeepLabCut pose-estimation data.
- Standardized pose data as xarray datasets in memory and NetCDF files on disk.
- Added utilities to map tracked positions onto labyrinth grid cells and load session datasets from initialized projects.

Changes:
- Updated `init_project` to ingest DLC HDF5 files, attach grid positions, and save per-session NetCDF datasets.
- Standardized analysis DataFrame columns to lowercase snake_case names across preprocessing, behavioral metrics, simulation, CoMPASS levels 1 and 2, and post-hoc analysis.
- Added independent controls for saving combined and per-session preprocessed CSV files.
- Updated tutorials, documentation, and test fixtures for the movement-based project workflow.

Fixes:
- Improved metadata and session filename handling during project initialization and preprocessing.
- Added validation and reporting for missing session files.
- Fixed numeric dtype handling and several pandas warnings across the analysis pipeline.

# v0.1.1

Fixes:
- Added missing PyYAML dependency to `pyproject.toml`


# v0.1.0

Re-organization of the CoMPASS-Labyrinth project:
- Python package organization with `pyproject.toml`
- Source code moved to `src/`
- Initiate project function
- Tests for all main modules
- Coverage metrics
- Improved docstrings
- Improved type annotation
- Black formatting
- Tutorial notebooks showing usage as importable Python package
