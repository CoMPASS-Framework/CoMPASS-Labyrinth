"""Test R environment setup."""

import subprocess
import pytest


def test_r_packages():
    """Test all required R packages are installed."""
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
        "progressr"
    ]
    
    for pkg in packages:
        result = subprocess.run(
            ["Rscript", "-e", f"library({pkg})"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"R package '{pkg}' not installed"
