"""Test R environment setup."""

import subprocess
import platform
import os
import pytest

def run_r(expr: str):
    """Helper to run an R one-liner safely."""
    return subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)

@pytest.fixture(scope="session")
def renv_present():
    """Detect whether renv.lock exists; skip package tests if it doesn't."""
    return os.path.exists("renv.lock")

def test_renv_bootstrapped():
    expr = r"""
      if (!requireNamespace('renv', quietly=TRUE)) quit(status=1)
      cat('renv available\n')
    """
    res = run_r(expr)
    assert res.returncode == 0, f"renv not available.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

def test_renv_active_paths():
    expr = r"""
      if (file.exists('renv/activate.R')) source('renv/activate.R')
      cat('LibPaths:\n'); print(.libPaths())
      cat('Working dir:', getwd(), '\n')
    """
    res = run_r(expr)
    assert res.returncode == 0, f"renv activation/paths failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

def test_r_packages(renv_present):
    if not renv_present:
        pytest.skip("No renv.lock in repo; skipping package-availability test.")

    pkgs = [
        "dplyr", "ggplot2", "tidyverse", "readxl", "survival",
        "here", "zoo", "circular",
        "DHARMa","movehmm", "momentuHMM","nhm",
        "furrr", "future", "progressr", "terra"
    ]
    r_vec = ", ".join(f"'{p}'" for p in pkgs)
    expr = rf"""
      if (file.exists('renv/activate.R')) source('renv/activate.R')
      pkgs <- c({r_vec})
      missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
      if (length(missing)) {{
        writeLines(paste('MISSING:', paste(missing, collapse=', ')))
        quit(status = 1)
      }} else {{
        cat('All requested packages are available under renv.\n')
      }}
    """
    res = run_r(expr)
    assert res.returncode == 0, f"Packages missing.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

@pytest.mark.skipif(platform.system().lower() != "windows", reason="terra load order is Windows-specific")
def test_terra_then_momentuHMM(renv_present):
    if not renv_present:
        pytest.skip("No renv.lock; skipping Windows terra/momentuHMM load-order test.")
    expr = r"""
      if (file.exists('renv/activate.R')) source('renv/activate.R')
      suppressPackageStartupMessages(library(terra))
      suppressPackageStartupMessages(library(momentuHMM))
      cat('terra + momentuHMM loaded.\n')
    """
    res = run_r(expr)
    assert res.returncode == 0, f"Failed to load terra->momentuHMM.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
