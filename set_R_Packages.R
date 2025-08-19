# set_R_Packages.R
# Run AFTER activating your conda env (e.g., compass-labyrinth)

options(repos = c(CRAN = "https://cloud.r-project.org/"))

# On Windows, force binaries to avoid compilation errors
if (Sys.info()["sysname"] == "Windows") {
  options(install.packages.compile.from.source = "never")
}

required_packages <- c(
  "DHARMa",       # Model diagnostics
  "moveHMM",      # Not on conda
  "momentuHMM",   # Not on conda
  "nhm",          # Not on conda
  "here",         # Portable file paths
  "zoo",          # Time series
  "circular",     # Circular statistics
  "furrr",        # Parallel purrr
  "future",       # Futures for parallelization
  "progressr"     # Progress bars
)

cat("Installing R packages...\n")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    tryCatch({
      install.packages(pkg, type = "binary")
    }, error = function(e) {
      message(paste("Failed to install:", pkg, "→", e$message))
    })
  } else {
    message(paste(" Already installed:", pkg))
  }
}

cat("All packages attempted.\n")
