"""Test R environment setup."""

import subprocess
import platform
import pytest

def test_r_packages():
    """Check that required packages are available inside renv."""
    pkgs = [
        "dplyr", "ggplot2", "tidyverse", "readxl", "survival",
        "here", "zoo", "circular", "remotes", "devtools",
        "DHARMa", "momentuHMM", "furrr", "future", "progressr", "terra"
        # Add "moveHMM" / "nhm" here ONLY if they’re in renv.lock and truly needed.
    ]

    expr = r"""
      source('renv/activate.R')
      pkgs <- c(%s)
      missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
      if (length(missing)) {
        writeLines(paste('MISSING:', paste(missing, collapse = ', ')))
        quit(status = 1)
      } else {
        cat('All R packages available under renv.\n')
      }
    """ % (", ".join(f"'{p}'" for p in pkgs))

    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Packages missing.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )

@pytest.mark.skipif(platform.system().lower() != "windows", reason="Windows-only DLL load order check")
def test_terra_then_momentuHMM():
    """On Windows, ensure terra loads before momentuHMM."""
    expr = r"""
      source('renv/activate.R')
      suppressPackageStartupMessages(library(terra))
      suppressPackageStartupMessages(library(momentuHMM))
      cat('terra + momentuHMM loaded.\n')
    """
    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Failed to load terra->momentuHMM.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )

def test_renv_active_paths():
    """Sanity-check renv activation & library paths (useful debug)."""
    expr = r"""
      source('renv/activate.R')
      cat('LibPaths:\n')
      print(.libPaths())
      cat('renv project:', renv::project(), '\n')
    """
    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"renv activation failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )