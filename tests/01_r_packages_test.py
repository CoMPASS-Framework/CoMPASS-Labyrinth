"""Test R environment setup."""

import subprocess
import platform
import pytest

def run_r(expr: str):
    return subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)

def test_r_packages():
    pkgs = [
        "dplyr", "ggplot2", "tidyverse", "readxl", "survival",
        "here", "zoo", "circular",
        "DHARMa", "momentuHMM",
        "furrr", "future", "progressr",
        "terra",
        # Add "moveHMM"/"nhm" ONLY if they are truly needed and in renv.lock
    ]
    expr = r"""
      if (file.exists('renv/activate.R')) {
        source('renv/activate.R')
      }
      pkgs <- c(%s)
      missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
      if (length(missing)) {
        writeLines(paste('MISSING:', paste(missing, collapse = ', ')))
        quit(status = 1)
      } else {
        cat('All required R packages available.\n')
      }
    """ % (", ".join(f"'{p}'" for p in pkgs))
    res = run_r(expr)
    assert res.returncode == 0, f"Packages missing.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

@pytest.mark.skipif(platform.system().lower() != "windows", reason="Windows-specific DLL load order check")
def test_terra_then_momentuHMM():
    expr = r"""
      if (file.exists('renv/activate.R')) {
        source('renv/activate.R')
      }
      suppressPackageStartupMessages(library(terra))
      suppressPackageStartupMessages(library(momentuHMM))
      cat('terra + momentuHMM loaded OK\n')
    """
    res = run_r(expr)
    assert res.returncode == 0, f"Failed to load terra -> momentuHMM.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
