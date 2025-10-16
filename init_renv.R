# ============================================================
# init_renv.R — CoMPASS-Labyrinth
# Robust R 4.4.3+ environment setup for local + CI
# ============================================================

cat("\n=== CoMPASS-Labyrinth R Environment Setup ===\n\n")

if (getRversion() < "4.4.0") {
  stop("This project requires R >= 4.4.0. Please upgrade R.")
}

repo <- if (tolower(Sys.info()[["sysname"]]) == "windows") {
  "https://cloud.r-project.org/"
} else if (tolower(Sys.info()[["sysname"]]) == "darwin") {
  "https://cloud.r-project.org/"
} else {
  "https://packagemanager.posit.co/cran/__linux__/noble/latest"
}
options(repos = c(CRAN = repo))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
suppressPackageStartupMessages(library(renv))  

renv::activate()

renv::settings$use.cache(TRUE)
renv::settings$snapshot.type("implicit")

if (file.exists("renv.lock")) {
  cat("renv.lock found → restoring...\n")
  renv::restore(prompt = FALSE)
} else {
  cat("No renv.lock found → initializing...\n")
  renv::init(bare = TRUE)
}

# ---- Required packages ----
required_packages <- c(
  # Documentation
  "rmarkdown", "tinytex", "knitr", "htmltools", "bslib", "sass",
  # Analysis
  "tidyverse", "here", "zoo", "circular", "progressr", "furrr", "future",
  "dplyr", "ggplot2", "readxl", "survival",
  # HMM + spatial
  "momentuHMM", "moveHMM", "DHARMa", "nhm", "terra", "sp",
  # Development / misc
  "remotes", "devtools",
  # Modeling extensions
  "coxme", "lmerTest", "msm", "survminer"
)

# ---- Install missing packages ----
missing <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing)) {
  cat("Installing missing packages:\n  - ", paste(missing, collapse = ", "), "\n")
  renv::install(missing)
}

# ---- TinyTeX installer ----
ensure_tinytex <- function() {
  if (nzchar(Sys.which("pdflatex"))) return(invisible(TRUE))
  if (!requireNamespace("tinytex", quietly = TRUE)) renv::install("tinytex")
  message("\nInstalling TinyTeX LaTeX toolchain...")
  tryCatch(
    tinytex::install_tinytex(force = TRUE),
    error = function(e) message("TinyTeX install failed: ", e$message)
  )
  invisible(TRUE)
}
if (!nzchar(Sys.which("pdflatex"))) ensure_tinytex()

# ---- Safe snapshot (no clean arg) ----
cat("\nCreating renv.lock snapshot...\n")
op <- options(
  renv.config.snapshot.validate = FALSE,
  renv.config.install.transactional = FALSE
)
on.exit(options(op), add = TRUE)

try_snapshot <- function() {
  renv::hydrate(packages = required_packages, prompt = FALSE)
  renv::snapshot(prompt = FALSE, force = TRUE)
}

ok <- TRUE
tryCatch(try_snapshot(), error = function(e) {
  ok <<- FALSE
  message("Snapshot failed once: ", e$message)
})
if (!ok) {
  message("Retrying snapshot with dependency rescan...")
  try(renv::dependencies(), silent = TRUE)
  try_snapshot()
}

cat("\n=== R Environment Setup Complete! ===\n")
cat("Key outputs:\n  • renv.lock\n  • .Rprofile\n  • renv/ (library cache)\n")
cat("To restore elsewhere: renv::restore()\n\n")
