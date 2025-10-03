"""Test R environment setup."""

import subprocess
import pytest


def test_r_packages():
    """Test core R packages are installed."""
    packages = ["dplyr", "ggplot2", "moveHMM", "momentuHMM"]
    
    for pkg in packages:
        result = subprocess.run(
            ["Rscript", "-e", f"library({pkg})"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"R package '{pkg}' not installed"
