# init_renv.R
# Initialize R environment using renv for CoMPASS-Labyrinth
# Run this script AFTER activating your conda environment

cat("\n=== CoMPASS-Labyrinth R Environment Setup ===\n\n")

options(repos = c(CRAN = "https://cloud.r-project.org/"))

if (Sys.info()["sysname"] == "Windows") {
  options(install.packages.compile.from.source = "never")
}

r_version <- getRversion()
cat("R version:", as.character(r_version), "\n")
if (r_version < "4.4.0") {
  warning("R version 4.4.0 or higher is recommended. Current version: ", r_version)
}

if (!requireNamespace("renv", quietly = TRUE)) {
  cat("\nInstalling renv package manager...\n")
  install.packages("renv")
}
library(renv)

if (!file.exists("renv.lock")) {
  cat("\nInitializing renv environment...\n")
  renv::init(bare = TRUE)
} else {
  cat("\nrenv already initialized. Using existing renv.lock\n")
  renv::activate()
  renv::restore(prompt = FALSE)
}

renv::settings$use.cache(TRUE)
renv::settings$snapshot.type("implicit")

required_packages <- c(
  # doc build
  "rmarkdown", "tinytex", "knitr", "htmltools", "bslib", "sass",
  
  # analysis
  "tidyverse", "here", "zoo", "circular", "progressr", "furrr", "future",
  "dplyr", "ggplot2", "readxl", "survival",
  
  # HMM + spatial
  "momentuHMM", "DHARMa", "nhm", "terra", "sp",
  
  # dev / misc
  "remotes", "devtools",
  
  "coxme", "lmerTest", "msm", "survminer"
)

cat("\nInstalling required R packages...\n\n")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing:", pkg, "...\n")
    tryCatch(renv::install(pkg),
             error = function(e) message("Failed to install ", pkg, " → ", e$message))
  } else {
    cat("✓ Already installed:", pkg, "\n")
  }
}

# ---- TinyTeX (Windows-friendly) ----
ensure_tinytex <- function() {                               # <<< CHANGED
  if (nzchar(Sys.which("pdflatex"))) return(invisible(TRUE))
  if (!requireNamespace("tinytex", quietly = TRUE)) renv::install("tinytex")
  
  message("\nInstalling TinyTeX LaTeX toolchain (one-time)...")
  # Use the standard installer everywhere (install_prebuilt() may not exist)
  tryCatch(
    tinytex::install_tinytex(force = TRUE),
    error = function(e) stop("Failed to install TinyTeX: ", conditionMessage(e))
  )
  
  # Add TinyTeX bin to PATH for this session
  tt <- tryCatch(tinytex::tinytex_root(), error = function(e) NA_character_)
  if (!is.na(tt)) {
    bindir <- if (.Platform$OS.type == "windows")
      file.path(tt, "bin", "windows") else file.path(tt, "bin", "x86_64-linux")
    if (dir.exists(bindir) && !grepl(bindir, Sys.getenv("PATH"), fixed = TRUE)) {
      Sys.setenv(PATH = paste(bindir, Sys.getenv("PATH"), sep = .Platform$path.sep))
    }
    # Make sure tlmgr is on PATH
    try(tinytex::tlmgr_path(), silent = TRUE)
  }
  if (!nzchar(Sys.which("pdflatex"))) {
    message("TinyTeX installed but 'pdflatex' not yet on PATH; a shell restart may be required.")
  }
  invisible(TRUE)
}

if (!nzchar(Sys.which("pdflatex"))) ensure_tinytex()         

# ---- Snapshot (robust on R 4.5) ----
cat("\nCreating renv.lock snapshot...\n")
op <- options(
  renv.config.snapshot.validate = FALSE,                     
  renv.config.install.transactional = FALSE
)
on.exit(options(op), add = TRUE)

# Force a fresh dependency scan and snapshot; retry once if needed
try_snapshot <- function() {
  renv::hydrate(packages = required_packages, prompt = FALSE)
  renv::snapshot(prompt = FALSE, force = TRUE)
}

ok <- TRUE
tryCatch(try_snapshot(), error = function(e) { ok <<- FALSE; message("Snapshot failed once: ", e$message) })
if (!ok) {
  message("Retrying snapshot with a clean dependency rescan...")
  try(renv::dependencies(), silent = TRUE)
  try_snapshot()
}

cat("\n=== R Environment Setup Complete! ===\n")
cat("\nKey files created:\n")
cat("  - renv.lock: Package version lockfile\n")
cat("  - .Rprofile: Auto-activates renv\n")
cat("  - renv/: Package library (add to .gitignore)\n\n")
cat("To restore on another machine: renv::restore()\n\n")
