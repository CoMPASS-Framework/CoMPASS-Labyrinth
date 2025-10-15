"""Test R environment setup."""

import subprocess
import platform
import pytest

def _r_expr_for_pkg_check(pkgs):
    # R snippet: source activate if it exists, then check packages quietly
    quoted = ", ".join(f"'{p}'" for p in pkgs)
    return rf"""
      if (file.exists('renv/activate.R')) source('renv/activate.R')
      pkgs <- c({quoted})
      missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
      if (length(missing)) {{
        writeLines(paste('MISSING:', paste(missing, collapse = ', ')))
        quit(status = 1)
      }} else {{
        cat('All R packages available.\n')
      }}
    """

def test_r_packages():
    pkgs = [
        "dplyr", "ggplot2", "tidyverse", "readxl", "survival",
        "here", "zoo", "circular", "remotes", "devtools",
        "DHARMa", "momentuHMM", "furrr", "future", "progressr", "terra",
        # add "moveHMM" or "nhm" only if they’re actually in renv.lock
    ]
    expr = _r_expr_for_pkg_check(pkgs)
    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert res.returncode == 0, f"Packages missing.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

@pytest.mark.skipif(platform.system().lower() != "windows", reason="Windows-only DLL order check")
def test_terra_then_momentuHMM():
    expr = r"""
      if (file.exists('renv/activate.R')) source('renv/activate.R')
      suppressPackageStartupMessages(library(terra))
      suppressPackageStartupMessages(library(momentuHMM))
      cat('terra + momentuHMM loaded.\n')
    """
    res = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    assert res.returncode == 0, f"Failed to load terra->momentuHMM.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
