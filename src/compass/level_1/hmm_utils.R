# -----------------------------------------------------------
# HMM Utilities for Behavioral State Inference using momentuHMM
# Author: Shreya Bangera
# -----------------------------------------------------------

library(momentuHMM)
library(zoo)
library(circular)

# -------------------------------
# Parameter Estimation Utilities
# -------------------------------

# Manual estimates of parameters are decided based on -
##  1. domain knowledge 
##  2. observing distributions and summary stats
##  3. objective of study
# ** OPTION TO ALSO USE THE AUTOMATED ESTIMATES BASED ON IQR AND OUTLIER DETECTION

# Data stream distributions are decided based on -
##  1. observing distributions and summary stats

# Both are tested -> use of angle ('vm' or any other circular distribution) or absolute values of angle ('gamma',etc.)


compute_parameter_ranges <- function(df, angle_var, use_vm = FALSE) {
  # Step ranges
  step_iqr <- IQR(df$step, na.rm = TRUE)
  step_median <- median(df$step, na.rm = TRUE)
  step_mean_range <- c(step_median - step_iqr, step_median + step_iqr)
  step_sd_range <- c(0.1, 1.5 * mad(df$step, na.rm = TRUE))
  
  # Angle ranges
  angle_data <- df[[angle_var]]
  angle_median <- median(angle_data, na.rm = TRUE)
  angle_iqr <- IQR(angle_data, na.rm = TRUE)
  angle_mean_range <- c(max(0, angle_median - angle_iqr), angle_median + angle_iqr)
  angle_sd_range <- c(0.1, 1.5 * mad(angle_data, na.rm = TRUE))
  
  # von Mises concentration estimate
  angle_conc_range <- NULL
  if (use_vm) {
    angle_circ <- circular(angle_data, units = "radians", template = "none")
    circ_disp <- var(angle_circ)
    if (!is.na(circ_disp) && circ_disp > 0) {
      angle_conc_range <- c(1, min(1 / circ_disp, 50))  # cap to avoid instability
    } else {
      angle_conc_range <- c(1, 15)  # fallback
    }
  }
  
  return(list(
    step_mean_range = step_mean_range,
    step_sd_range = step_sd_range,
    angle_mean_range = angle_mean_range,
    angle_sd_range = angle_sd_range,
    angle_conc_range = angle_conc_range
  ))
}

# -------------------------------
# Behavioral Objective Function
# -------------------------------

satisfies_objective <- function(model) {
  step_means <- model$mle$step[1:2]
  angle_param <- model$mle$angle
  angle_metric <- if ("concentration" %in% names(angle_param)) {
    -angle_param$concentration
  } else {
    angle_param[1:2]
  }
  
  behavior_ok <- (step_means[1] > step_means[2] && angle_metric[1] < angle_metric[2]) ||
    (step_means[2] > step_means[1] && angle_metric[2] < angle_metric[1])
  
  trans_ok <- all(diag(model$conditions$trMat[[1]]) < 0.99)
  return(behavior_ok && trans_ok)
}

# -------------------------------
# Main HMM Model Fitting Function
# -------------------------------

fit_best_hmm <- function(preproc_df,
                         n_states = 2,
                         n_iter = 15,
                         rolling_k = c(3,5,7),
                         opt_methods = c("BFGS", "L-BFGS-B", "Nelder-Mead", "nlm"),
                         use_abs_angle = c(TRUE, FALSE),
                         use_data_driven_ranges = TRUE,
                         angle_mean_biased = c(pi / 2, 0),
                         stationary_flag = "auto",
                         session_col = "Session",
                         seed = 123) {
  
  set.seed(seed)
  all_tasks <- list()
  
  # Stationarity decision
  if (stationary_flag == "auto") {
    if (!session_col %in% names(preproc_df)) stop("session_col not found in preproc_df")
    lengths <- rle(as.character(preproc_df[[session_col]]))$lengths
    stationary_flag_used <- median(lengths) >= 100
  } else {
    stationary_flag_used <- stationary_flag
  }
  
  # Build task grid
  for (log_step in c(FALSE, TRUE)) {
    for (log_angle in c(FALSE, TRUE)) {
      for (k_step in rolling_k) {
        for (abs_flag in use_abs_angle) {
          for (method in opt_methods) {
            for (i in seq_len(n_iter)) {
              all_tasks[[length(all_tasks) + 1]] <- list(
                k_step = k_step,
                abs_flag = abs_flag,
                method = method,
                iter = i,
                log_step = log_step,
                log_angle = log_angle,
                stationary = stationary_flag_used
              )
            }
          }
        }
      }
    }
  }
  
  p <- progressr::progressor(along = all_tasks)
  
  results <- future_map(all_tasks, function(task) {
    p()
    
    df <- preproc_df
    df <- df %>%
      arrange(ID,session_col, S_no) %>%
      filter(complete.cases(.)) 
    df$x <- as.numeric(df$x)
    df$y <- as.numeric(df$y)
    final_df <- prepData(df, type = "UTM", coordNames = c("x", "y"))
    final_df <- final_df[complete.cases(final_df), ]
    
    final_df$step_smooth <- zoo::rollmean(final_df$step, k = task$k_step, fill = NA, align = "center")
    final_df$angle_smooth <- zoo::rollmean(final_df$angle, k = task$k_step, fill = NA, align = "center")
    final_df <- final_df[complete.cases(final_df[, c("step_smooth", "angle_smooth")]), ]
    
    final_df$step_smooth <- pmin(pmax(final_df$step_smooth, 1e-3), 1500)
    final_df <- final_df[abs(final_df$angle_smooth) <= pi, ]
    
    if (task$log_step) final_df$step_smooth <- log(final_df$step_smooth + 1e-3)
    if (task$log_angle) final_df$angle_smooth <- log(abs(final_df$angle_smooth) + 1e-3)
    final_df <- final_df[is.finite(final_df$step_smooth) & is.finite(final_df$angle_smooth), ]
    
    angle_var <- if (task$abs_flag) "angle_abs_smooth" else "angle_smooth"
    angle_dist <- if (task$abs_flag || task$log_angle) "gamma" else "vm"
    if (task$abs_flag) final_df$angle_abs_smooth <- abs(final_df$angle_smooth)
    
    dist_list <- list(
      step = if (task$log_step) "norm" else "gamma",
      angle = angle_dist
    )
    
    if (use_data_driven_ranges) {
      ranges <- compute_parameter_ranges(final_df, angle_var, use_vm = (angle_dist == "vm"))
      step_mean_range <- ranges$step_mean_range
      step_sd_range <- ranges$step_sd_range
      angle_mean_range <- ranges$angle_mean_range
      angle_sd_range <- ranges$angle_sd_range
      angle_conc_range <- if (!is.null(ranges$angle_conc_range)) ranges$angle_conc_range else c(1, 15)
    } else {
      step_mean_range <- if (task$log_step) log(c(1, 600)) else c(10, 600)
      step_sd_range <- if (task$log_step) c(0.1, 2) else c(0.1, 150)
      angle_mean_range <- if (task$log_angle) log(c(0.01, pi)) else c(0, pi)
      angle_sd_range <- if (task$log_angle) c(0.1, 2) else c(0.1, 1.5)
      angle_conc_range <- c(1, 15)
    }
    
    stepMean0 <- runif(n_states, step_mean_range[1], step_mean_range[2])
    stepSD0 <- runif(n_states, step_sd_range[1], step_sd_range[2])
    
    if (angle_dist == "vm") {
      angleMean0 <- angle_mean_biased
      angleCon0 <- runif(n_states, angle_conc_range[1], angle_conc_range[2])
      anglePar <- c(angleMean0, angleCon0)
    } else {
      angleMean0 <- runif(n_states, angle_mean_range[1], angle_mean_range[2])
      angleSD0 <- runif(n_states, angle_sd_range[1], angle_sd_range[2])
      anglePar <- c(angleMean0, angleSD0)
    }
    
    initPar <- list(step = c(stepMean0, stepSD0), angle = anglePar)
    
    model_try <- tryCatch({
      suppressMessages(
        fitHMM(
          data = final_df,
          nbStates = n_states,
          dist = dist_list,
          Par0 = initPar,
          estAngleMean = list(angle = (angle_dist == "vm")),
          formula = ~1,
          optMethod = task$method,
          stationary = task$stationary
        )
      )
    }, error = function(e) NULL)
    
    gc(verbose = FALSE)
    
    if (!is.null(model_try) && satisfies_objective(model_try)) {
      list(model = model_try, meta = list(
        k = task$k_step, abs = task$abs_flag, method = task$method,
        log_step = task$log_step, log_angle = task$log_angle,
        stationary = task$stationary, AIC = AIC(model_try),
        index_map = as.integer(rownames(final_df))
      ))
    } else {
      NULL
    }
  }, .options = furrr_options(seed = TRUE))
  
  valid_models <- results[!sapply(results, is.null)]
  if (length(valid_models) == 0) stop("No valid models met the objective.")
  
  best_idx <- which.min(sapply(valid_models, function(x) x$meta$AIC))
  best_model <- valid_models[[best_idx]]$model
  best_meta <- valid_models[[best_idx]]$meta
  index_map <- valid_models[[best_idx]]$meta$index_map
  
  # Reorder states: low step, high angle conc → state 1
  step_means <- best_model$mle$step[1, ]
  angle_param <- best_model$mle$angle
  angle_conc <- if ("concentration" %in% names(angle_param)) {
    angle_param$concentration
  } else {
    -angle_param[(n_states + 1):(2 * n_states)]
  }
  
  #ordered <- order(step_means, -angle_conc)
  #best_model <- momentuHMM::reorderStates(best_model, order = ordered)
  
  decoded <- factor(viterbi(best_model))
  preproc_df$'HMM_State' <- NA
  preproc_df$'HMM_State'[index_map] <- decoded
  
  return(list(
    model = best_model,
    summary = best_meta,
    data = preproc_df
  ))
}


# -------------------------------
# Summary Printer
# -------------------------------

print_hmm_summary <- function(model_summary, best_model) {
  cat("\n Best Model Characteristics:\n")
  cat("• Smoothing window (k):", model_summary$k, "\n")
  cat("• Angle type:", ifelse(model_summary$abs_angle, "abs(angle) ~ gamma", "angle ~ von Mises"), "\n")
  cat("• Optimizer:", model_summary$opt_method, "\n")
  cat("• AIC:", round(model_summary$AIC, 2), "\n\n")
  
  cat("• Step Means:", round(best_model$mle$step[1, ], 2), "\n")
  
  if ("concentration" %in% names(best_model$mle$angle)) {
    cat("• Angle Concentrations (von Mises):", round(best_model$mle$angle$concentration, 2), "\n")
  } else {
    cat("• Angle Means:", round(best_model$mle$angle[1:2], 2), "\n")
    cat("• Angle SDs:", round(best_model$mle$angle[3:4], 2), "\n")
  }
  
  cat("• Final state ordering: State 1 = low step + high angle\n\n")
}
