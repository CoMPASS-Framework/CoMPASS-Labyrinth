"""Test R environment setup."""

import subprocess
import pytest
from pathlib import Path


def test_r_packages():
    """Test all required R packages are installed."""
    # Get project root (one level up from tests/ directory)
    # This ensures renv is activated from the correct directory
    project_root = Path(__file__).parent.parent

    packages = [
        "dplyr",
        "ggplot2",
        "tidyverse",
        "readxl",
        "survival",
        "here",
        "zoo",
        "circular",
        "remotes",
        "devtools",
        "DHARMa",
        "moveHMM",
        "momentuHMM",
        "nhm",
        "furrr",
        "future",
        "progressr",
    ]

    for pkg in packages:
        result = subprocess.run(
            ["Rscript", "-e", f"renv::activate(); library({pkg})"],
            capture_output=True,
            text=True,
            # cwd=project_root,  # Run from project root to activate renv
        )
        assert result.returncode == 0, f"R package '{pkg}' not installed"
