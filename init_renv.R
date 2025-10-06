# init_renv.R
# Initialize R environment using renv for CoMPASS-Labyrinth
# Run this script AFTER activating your conda environment

cat("\n=== CoMPASS-Labyrinth R Environment Setup ===\n\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# On Windows, force binaries to avoid compilation errors
if (Sys.info()["sysname"] == "Windows") {
  options(install.packages.compile.from.source = "never")
}

# Check R version
r_version <- getRversion()
cat("R version:", as.character(r_version), "\n")
if (r_version < "4.4.0") {
  warning("R version 4.4.0 or higher is recommended. Current version: ", r_version)
}

# Install renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  cat("\nInstalling renv package manager...\n")
  install.packages("renv")
}

library(renv)

# Initialize renv if not already initialized
if (!file.exists("renv.lock")) {
  cat("\nInitializing renv environment...\n")
  renv::init(bare = TRUE)  # bare = TRUE prevents automatic snapshot
} else {
  cat("\nrenv already initialized. Using existing renv.lock\n")
}

# List of required packages
required_packages <- c(
  # Packages previously managed via conda
  "dplyr",       # Data manipulation
  "ggplot2",     # Visualization
  "tidyverse",   # Data science ecosystem
  "readxl",      # Excel file reading
  "survival",    # Survival analysis
  "here",        # Portable file paths
  "zoo",         # Time series
  "circular",    # Circular statistics
  "remotes",     # Install from GitHub/other sources
  "devtools",    # Development tools
  
  # Packages previously from CRAN
  "DHARMa",      # Model diagnostics
  "moveHMM",     # Movement HMM models
  "momentuHMM",  # Advanced movement HMMs
  "nhm",         # Nested HMM models
  "furrr",       # Parallel purrr
  "future",      # Futures for parallelization
  "progressr"    # Progress bars
)

cat("\nInstalling required R packages...\n")
cat("This may take several minutes on first run.\n\n")

# Install packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing:", pkg, "...\n")
    tryCatch({
      renv::install(pkg)
    }, error = function(e) {
      message(paste("Failed to install:", pkg, "→", e$message))
    })
  } else {
    cat("✓ Already installed:", pkg, "\n")
  }
}

# Create snapshot to lock package versions
cat("\nCreating renv.lock snapshot...\n")
renv::snapshot(prompt = FALSE)

cat("\n=== R Environment Setup Complete! ===\n")
cat("\nKey files created:\n")
cat("  - renv.lock: Package version lockfile\n")
cat("  - .Rprofile: Auto-activates renv\n")
cat("  - renv/: Package library (add to .gitignore)\n")

cat("\nNext time you start R in this directory, renv will automatically activate.\n")
cat("To restore this environment on another machine, run: renv::restore()\n\n")
