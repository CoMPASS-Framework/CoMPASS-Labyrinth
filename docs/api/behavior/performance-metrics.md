# Performance Metrics

Functions for computing task performance metrics including region usage, entropy measures, and heatmaps.

## Overview

This module provides comprehensive metrics for analyzing behavioral performance in the labyrinth task, including:

- Frame counting and session duration
- Target zone usage analysis
- Region-based heatmaps
- Shannon entropy calculations  
- Statistical testing (ANOVA, mixed models)

---

## Session and Frame Analysis

### compute_frames_per_session

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.compute_frames_per_session

### get_max_session_row_bracket

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.get_max_session_row_bracket

---

## Target Zone Usage

### compute_target_zone_usage

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.compute_target_zone_usage

### summarize_target_usage

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.summarize_target_usage

### exclude_low_performing_sessions

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.exclude_low_performing_sessions

---

## Visualization Functions

### plot_target_usage_vs_frames

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_target_usage_vs_frames

### plot_target_usage_with_exclusions

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_target_usage_with_exclusions

---

## Region Heatmaps

### generate_region_heatmap_pivots

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.generate_region_heatmap_pivots

### subset_pivot_dict_sessions

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.subset_pivot_dict_sessions

### plot_region_heatmaps

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_region_heatmaps

### plot_region_heatmaps_all_genotypes

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_region_heatmaps_all_genotypes

---

## Entropy Analysis

### compute_shannon_entropy_per_bin

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.compute_shannon_entropy_per_bin

### plot_entropy_over_bins

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_entropy_over_bins

---

## Statistical Testing

### run_entropy_anova

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.run_entropy_anova

### run_fdr_pairwise_tests

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.run_fdr_pairwise_tests

### run_mixed_model_per_genotype_pair

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.run_mixed_model_per_genotype_pair

---

## Region Usage Over Time

### compute_region_usage_over_bins

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.compute_region_usage_over_bins

### plot_region_usage_over_bins

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_region_usage_over_bins

### plot_all_regions_usage_over_bins

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.plot_all_regions_usage_over_bins

### run_region_usage_stats_mixedlm

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.run_region_usage_stats_mixedlm

### run_region_usage_stats_fdr

::: compass_labyrinth.behavior.behavior_metrics.task_performance_analysis.performance_metrics.run_region_usage_stats_fdr
