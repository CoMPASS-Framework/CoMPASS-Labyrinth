source("renv/activate.R")
# .Rprofile for CoMPASS-Labyrinth
# This file automatically activates the renv environment when R starts in this directory

# Only activate if renv has been initialized
try({
  if (file.exists("renv/activate.R")) source("renv/activate.R")
  if (is.null(getOption("repos")) || identical(getOption("repos"), c(CRAN="@CRAN@"))) {
    options(repos = c(CRAN = "https://cloud.r-project.org/"))
  }
}, silent = TRUE)
