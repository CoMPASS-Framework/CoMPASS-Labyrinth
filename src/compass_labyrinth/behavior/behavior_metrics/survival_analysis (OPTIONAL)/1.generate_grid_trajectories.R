#!/usr/bin/env Rscript

###########################################################################
## Project: Process data from the labyrinth
## Author: Reuben Thomas
##
## Script Goal: Two R objects are created with this script: 
## 1. Generate information on bouts into the maze for each mouse in terms
## entry and exit times in terms of video frame numbers, the number of grid nodes visited 
## in each bout and the median and mean likelihood of the DLC estimates across of location of mouse in
## each grid node 
## 2. List of grid trajectories for each mouse. The data for each mouse is represented
## as a list of bouts made by that mouse, each element of that list is a data frame with columns representing a grid node, entry time
## to the grid node, frame no representing the entry time into that node and duration in frames spent at that node 
##
## Usage : Reads in csv files representing the motion tracked positions of each mouse to put the above R objects
##
###########################################################################


require(dplyr)
require(magrittr)
require(zoo)
require(ggplot2)
require(lmerTest)
require(DHARMa)
require(nhm)
require(msm)
require(survival)
require(coxme)
require(readxl)

likelihood_threshold <- 0.8
likelihood_threshold_low <- 0.2
body_part <- "belly"
window_size <- 7

setwd("~/Dropbox (Gladstone)/Labyrinth Mazes discussions/Tracking Results")

pheno_data <- read.csv("~/Dropbox (Gladstone)/scripts/JP_PH01/230504 Denali cohort3 MICEv2.csv", header = TRUE)
pheno_data %<>% mutate(Genotype = Genotype.1) %>% select(-Genotype.1)

pheno_data_learning <- read_xlsx("~/Dropbox (Gladstone)/scripts/JP_PH01/20230302_AppSAA_Cohort3_Labyrinth_DLC_Info_Sheet_v3.xlsx", sheet = "Learning Trial")

pheno_data_learning %<>% mutate(Session.1 = Session) %>% select(Session.1, `New Mouse ID`)
pheno_data %<>% merge(., pheno_data_learning, all.x = TRUE)
colnames(pheno_data)[1] <- "Session.learning"

pheno_data %<>% select(-Session, -order)

get_mouse_entry_exit_times <- function(session, trial, pheno_data) {
  print(session)
  ##read in session info
  if(trial == "learning") {
    body_part_info <- read.csv(paste0("Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 0, nrows = 1) %>%
      dplyr::select(-1) %>% as.character() 
    
    body_parts <- body_part_info %>% unique()
    
    spatial_info <- read.csv(paste0("Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 1, nrows = 1) %>%
      dplyr::select(-1) %>% as.character() 
    
    
    data <- read.csv(paste0("Session-", session,"withSpeedandLinearKey.csv"), header = FALSE, skip = 3) %>% dplyr::select(-1)
  }else{
    body_part_info <- read.csv(paste0(getwd(), "/Probe Trial/Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 0, nrows = 1) %>%
      dplyr::select(-1) %>% as.character() 
    
    body_parts <- body_part_info %>% unique()
    
    spatial_info <- read.csv(paste0(getwd(), "/Probe Trial/Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 1, nrows = 1) %>%
      dplyr::select(-1) %>% as.character() 
    
    
    data <- read.csv(paste0(getwd(), "/Probe Trial/Session-", session,"withSpeedandLinearKey.csv"), header = FALSE, skip = 3) %>% dplyr::select(-1)
    
  }
  
  ##filter for body part results
  data <- data[, (body_part_info == body_part)]
  colnames(data) <- spatial_info[body_part_info == body_part]
  
  data %<>% mutate(downstream_likelihood = rollmedian(likelihood, window_size, align = "left", fill = NA))
  data %<>% mutate(upstream_likelihood = rollmedian(likelihood, window_size, align = "right", fill = NA))
  data %<>% mutate(diff_likelihood = upstream_likelihood - downstream_likelihood)
  
  ##assign NAs to grid locations with likelihood less than likelihood_threshold_low
  data$`Grid Number`[data$upstream_likelihood < likelihood_threshold_low |
                       data$downstream_likelihood < likelihood_threshold_low] <- NA
  
  
  ##create list of forays into labyrinth annotated by entry time, exit time
  entry_times <- which((is.na(data$`Grid Number`) & 
                          !is.na(lead(data$`Grid Number`)))  ) + 1
  exit_times <- which(is.na(data$`Grid Number`) & 
                        !is.na(lag(data$`Grid Number`)) ) - 1
  
  if(!is.na(data$`Grid Number`[1]))
    entry_times %<>% append(1, .)
  if(!is.na(data$`Grid Number`[nrow(data)]))
    exit_times %<>% append(., nrow(data))
  
  print(length(entry_times))
  print(length(exit_times))
  entry_exit_times <- data.frame(entry_times=NA,
                                 exit_times=NA,
                                 median_likelihood=NA,
                                 mean_likelihood=NA,
                                 ngrids=NA,
                                 entry_time_likelihood=NA,
                                 exit_time_likelihood=NA,
                                 length=NA,
                                 median_velocity=NA,
                                 Session=session,
                                 nentries=0)
  if(length(entry_times) > 0) {
    entry_exit_times <- data.frame(entry_times, exit_times)
    
    print(dim(entry_exit_times))
    median_likelihood <-  vector(mode = "numeric")
    mean_likelihood <-  vector(mode = "numeric")
    ngrids <- vector(mode = "numeric")
    for(f in 1:nrow(entry_exit_times)) {
      median_likelihood[f] <- median(data$likelihood[entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f]])
      mean_likelihood[f] <- mean(data$likelihood[entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f]])
      ngrids[f] <- length(unique(data$`Grid Number`[entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f]]))
    }
    entry_exit_times %<>% mutate(median_likelihood = median_likelihood, mean_likelihood = mean_likelihood, ngrids = ngrids)
    
    
    entry_exit_times %<>% filter(mean_likelihood > likelihood_threshold)
    
    
    ##merge exits in previous entry and entry in next entry if needed
    
    if(nrow(entry_exit_times) > 1) {
      # Specify the distance threshold for merging intervals
      distance_threshold <- 5
      
      # Initialize an empty list to store merged intervals
      merged_intervals <- list()
      
      # Initialize variables to track the current interval
      current_start <- entry_exit_times$entry_times[1]
      current_end <- entry_exit_times$exit_times[1]
      
      # Iterate through the data frame
      for (i in 2:nrow(entry_exit_times)) {
        next_start <- entry_exit_times$entry_times[i]
        next_end <- entry_exit_times$exit_times[i]
        
        # Check if the next interval is within the distance threshold
        if (next_start - current_end <= distance_threshold) {
          # Merge the intervals
          current_end <- next_end
        } else {
          # Add the current interval to the list of merged intervals
          merged_intervals <- c(merged_intervals, list(c(current_start, current_end)))
          
          # Update the current interval
          current_start <- next_start
          current_end <- next_end
        }
      }
      
      # Add the last merged interval to the list
      merged_intervals <- c(merged_intervals, list(c(current_start, current_end)))
      
      # Convert the list of merged intervals to a data frame
      entry_exit_times <- do.call(rbind.data.frame, merged_intervals)
      colnames(entry_exit_times) <- c("entry_times", "exit_times")
      # Rename the columns if needed
      median_likelihood <-  vector(mode = "numeric")
      mean_likelihood <-  vector(mode = "numeric")
      ngrids <- vector(mode = "numeric")
      for(f in 1:nrow(entry_exit_times)) {
        median_likelihood[f] <- median(data$likelihood[entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f]])
        mean_likelihood[f] <- mean(data$likelihood[entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f]])
        ngrids[f] <- length(unique(data$`Grid Number`[entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f]]))
      }
      entry_exit_times %<>% mutate(median_likelihood = median_likelihood, mean_likelihood = mean_likelihood, ngrids = ngrids)
      
    }
    entry_time_likelihood <- vector(mode = "integer")
    exit_time_likelihood <- vector(mode = "integer")
    median_velocity <- vector(mode = "numeric")
    for(f in 1:nrow(entry_exit_times)) {
      temp_data <- data %>% slice(entry_exit_times$entry_times[f]:entry_exit_times$exit_times[f])
      entry_time_likelihood[f] <- entry_exit_times$entry_times[f] - 1 + which(-temp_data$diff_likelihood > likelihood_threshold)[1]
      exit_time_likelihood[f] <- entry_exit_times$entry_times[f] - 1 + which(temp_data$diff_likelihood > likelihood_threshold)[1]
      
      if(is.na(entry_time_likelihood[f]))
        entry_time_likelihood[f] <- entry_exit_times$entry_times[f]
      if(is.na(exit_time_likelihood[f]))
        exit_time_likelihood[f] <- entry_exit_times$exit_times[f]
      
      if(exit_time_likelihood[f] < entry_time_likelihood[f]) {
        exit_time_indices <- entry_exit_times$entry_times[f] - 1 + which(temp_data$diff_likelihood > likelihood_threshold)
        exit_time_indices %<>% intersect(., entry_time_likelihood[f]:entry_exit_times$exit_times[f])
        exit_time_likelihood[f] <- exit_time_indices[1]
      }
      
      #no velocity estimate in probe trial; so will not store this information for both the learning and probe trial
      median_velocity[f] <- NA
    }
    
    entry_exit_times %<>% mutate(entry_time_likelihood = entry_time_likelihood, 
                                 exit_time_likelihood = exit_time_likelihood, 
                                 length = (exit_time_likelihood - entry_time_likelihood + 1),
                                 median_velocity = median_velocity)
    
    entry_exit_times %<>% mutate(Session = rep(session, nrow(.)), nentries=rep(nrow(.), nrow(.)))
    print(dim(entry_exit_times))
    
  }
  return(entry_exit_times)
}

all_mice_entry_exit_times_learning <- lapply(pheno_data$Session.learning, get_mouse_entry_exit_times,"learning", pheno_data)
all_mice_entry_exit_times_learning %<>% do.call("rbind", .)




saveRDS(all_mice_entry_exit_times_learning, paste0("~/Dropbox (Gladstone)/scripts/JP_PH01/all_mice_entry_exit_times_learning_update_bout_defs_include_mouseID_1842.rds"))


##save grid locations for each foray into the labyrinth
get_grid_trajectories <- function(session, trial, pheno_data) {
  print(session)
  grid_trajectories <- list()
  if(trial == "learning") {
    all_mice_entry_exit_times <- all_mice_entry_exit_times_learning
  }else{
    all_mice_entry_exit_times <- all_mice_entry_exit_times_probe
  }
  session_data <- all_mice_entry_exit_times %>%
    dplyr::filter(Session == session & ngrids > 10)
  if(nrow(session_data) > 0) {
    ##read in session info
    
    if(trial == "learning") {
      body_part_info <- read.csv(paste0("Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 0, nrows = 1) %>%
        dplyr::select(-1) %>% as.character() 
      
      body_parts <- body_part_info %>% unique()
      
      spatial_info <- read.csv(paste0("Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 1, nrows = 1) %>%
        dplyr::select(-1) %>% as.character() 
      
      
      data <- read.csv(paste0("Session-", session,"withSpeedandLinearKey.csv"), header = FALSE, skip = 3) %>% dplyr::select(-1)
    }else{
      body_part_info <- read.csv(paste0(getwd(), "/Probe Trial/Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 0, nrows = 1) %>%
        dplyr::select(-1) %>% as.character() 
      
      body_parts <- body_part_info %>% unique()
      
      spatial_info <- read.csv(paste0(getwd(), "/Probe Trial/Session-", session,"withSpeedandLinearKey.csv"), header = TRUE, skip = 1, nrows = 1) %>%
        dplyr::select(-1) %>% as.character() 
      
      
      data <- read.csv(paste0(getwd(), "/Probe Trial/Session-", session,"withSpeedandLinearKey.csv"), header = FALSE, skip = 3) %>% dplyr::select(-1)
      
    }
    
    
    
    
    ##filter for body part results
    data <- data[, (body_part_info == body_part)]
    colnames(data) <- spatial_info[body_part_info == body_part]
    
    data %<>% mutate(downstream_likelihood = rollmedian(likelihood, window_size, align = "left", fill = NA))
    data %<>% mutate(upstream_likelihood = rollmedian(likelihood, window_size, align = "right", fill = NA))
    data %<>% mutate(diff_likelihood = upstream_likelihood - downstream_likelihood)
    
    for(e in 1:nrow(session_data)) {
      temp_data <- data %>% 
        slice(session_data$entry_time_likelihood[e]:session_data$exit_time_likelihood[e]) %>%
        filter(!is.na(`Grid Number`))
      hold_times <- rle(temp_data$`Grid Number`) 
      node_visits <- data.frame(node = hold_times$values, duration = hold_times$lengths)
      node_visits %<>% mutate(entry_time = rep(session_data$entry_time_likelihood[e], nrow(.)))
      grid_trajectories[[e]] <- node_visits
    }
    
  }
  return(grid_trajectories)
}

grid_trajectories_learning <- lapply(pheno_data$Session.learning, get_grid_trajectories,"learning", pheno_data)


saveRDS(grid_trajectories_learning, paste0("~/Dropbox (Gladstone)/scripts/JP_PH01/grid_trajectories_learning_update_bout_defs_include_mouseID_1842.rds"))



