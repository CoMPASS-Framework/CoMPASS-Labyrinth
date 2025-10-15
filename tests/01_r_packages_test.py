"""Test R environment setup."""

# tests/R_test.py
import subprocess
import platform
import pytest

def test_r_packages():
    """Test all required R packages are installed under renv."""
    packages = [
        "dplyr", "ggplot2", "tidyverse", "readxl", "survival",
        "here", "zoo", "circular", "remotes", "devtools",
        "DHARMa", "momentuHMM", "furrr", "future", "progressr", "terra"
    ]

    # Single R invocation so renv only activates once (faster, clearer logs)
    pkg_vec = ", ".join(f"'{p}'" for p in packages)
    expr = (
        "source('renv/activate.R'); "
        f"pkgs <- c({pkg_vec}); "
        "missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; "
        "if (length(missing)) {{ "
        "  writeLines(paste('MISSING:', paste(missing, collapse = ', '))); "
        "  quit(status = 1) "
        "}} else {{ "
        "  cat('All R packages available under renv.\\n') "
        "}}"
    )

    result = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Some R packages are missing.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

@pytest.mark.skipif(platform.system().lower() != "windows", reason="terra load-order is Windows-specific")
def test_terra_then_momentuHMM():
    """Ensure terra loads before momentuHMM on Windows."""
    expr = (
        "source('renv/activate.R'); "
        "suppressPackageStartupMessages(library(terra)); "
        "suppressPackageStartupMessages(library(momentuHMM)); "
        "cat('terra and momentuHMM loaded successfully\\n')"
    )
    result = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Failed to load terra -> momentuHMM on Windows.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
