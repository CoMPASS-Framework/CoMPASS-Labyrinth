# Table of Contents

1. [Load Project](#Load-Project)
    1. [Load the HMM results](#Load-the-HMM-results)
2. [Plot 1: Heatmap Representations of HMM States](#Plot-1:-Heatmap-Representations-of-HMM-States)
    1. [Heatmap Representations of all genotypes](#Heatmap-representations-of-all-genotypes)
    2. [Interactive Heatmap version](#Interactive-Heatmap-version)
    3. [Interactive Heatmap version for all genotypes](#Interactive-Heatmap-version-of-all-genotypes)
3. [Plot 2: Probability of Surveillance across Node Types and Regions](#Plot-2:-Probability-of-Surveillance-across-Node-Types-and-Regions)
4. [Plot 3: Probability of States over Times](#Plot-3:-Probability-of-States-over-Time)
5. [Plot 4: Surveillance Probability by Bout Type](#Plot-4:-Surveillance-Probability-by-Bout-Type)

# Load Project

```python
from pathlib import Path
import pandas as pd
from compass_labyrinth import load_project


project_path = Path(".").resolve() / "my_project_2"

# Import config and metadata
config, cohort_metadata = load_project(project_path=project_path)
```

### Load the HMM results

```python
df_hmm = pd.read_csv(project_path / "results" / "compass_level_1" / "data_with_states.csv")
df_hmm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>step</th>
      <th>angle</th>
      <th>x</th>
      <th>y</th>
      <th>Grid Number</th>
      <th>likelihood</th>
      <th>S_no</th>
      <th>Region</th>
      <th>Session</th>
      <th>Genotype</th>
      <th>Sex</th>
      <th>NodeType</th>
      <th>Velocity</th>
      <th>HMM_State</th>
      <th>Post_Prob_1</th>
      <th>Post_Prob_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>3.912921</td>
      <td>-1.149582</td>
      <td>267.526978</td>
      <td>873.733704</td>
      <td>47</td>
      <td>0.986050</td>
      <td>750</td>
      <td>entry_zone</td>
      <td>3</td>
      <td>WT</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>19.564603</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1.981172</td>
      <td>0.620240</td>
      <td>265.571991</td>
      <td>873.412659</td>
      <td>47</td>
      <td>0.953975</td>
      <td>751</td>
      <td>entry_zone</td>
      <td>3</td>
      <td>WT</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>9.905860</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.066163</td>
      <td>-2.518697</td>
      <td>266.325684</td>
      <td>874.166748</td>
      <td>47</td>
      <td>0.958631</td>
      <td>752</td>
      <td>entry_zone</td>
      <td>3</td>
      <td>WT</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>5.330815</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.899351</td>
      <td>2.238799</td>
      <td>265.432495</td>
      <td>874.271851</td>
      <td>47</td>
      <td>0.932229</td>
      <td>753</td>
      <td>entry_zone</td>
      <td>3</td>
      <td>WT</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>4.496755</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0.519677</td>
      <td>1.625876</td>
      <td>265.400269</td>
      <td>873.753174</td>
      <td>47</td>
      <td>0.909978</td>
      <td>754</td>
      <td>entry_zone</td>
      <td>3</td>
      <td>WT</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>2.598385</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>265485</th>
      <td>7</td>
      <td>58.644322</td>
      <td>-0.866077</td>
      <td>228.690262</td>
      <td>814.897949</td>
      <td>34</td>
      <td>0.999385</td>
      <td>225311</td>
      <td>reward_path</td>
      <td>7</td>
      <td>KO</td>
      <td>Female</td>
      <td>Non-Decision (Reward)</td>
      <td>293.221609</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>265486</th>
      <td>7</td>
      <td>67.785952</td>
      <td>0.178698</td>
      <td>292.517883</td>
      <td>837.722717</td>
      <td>46</td>
      <td>0.999992</td>
      <td>225312</td>
      <td>entry_zone</td>
      <td>7</td>
      <td>KO</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>338.929760</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>265487</th>
      <td>7</td>
      <td>24.729589</td>
      <td>1.114421</td>
      <td>295.305054</td>
      <td>862.294739</td>
      <td>47</td>
      <td>0.999458</td>
      <td>225313</td>
      <td>entry_zone</td>
      <td>7</td>
      <td>KO</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>123.647944</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>265488</th>
      <td>7</td>
      <td>14.627917</td>
      <td>0.109074</td>
      <td>295.361694</td>
      <td>876.922546</td>
      <td>47</td>
      <td>0.999976</td>
      <td>225314</td>
      <td>entry_zone</td>
      <td>7</td>
      <td>KO</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>73.139586</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>265489</th>
      <td>7</td>
      <td>24.111518</td>
      <td>0.041934</td>
      <td>294.444183</td>
      <td>901.016602</td>
      <td>47</td>
      <td>0.962276</td>
      <td>225315</td>
      <td>entry_zone</td>
      <td>7</td>
      <td>KO</td>
      <td>Female</td>
      <td>Entry Nodes</td>
      <td>120.557592</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>265490 rows × 17 columns</p>
</div>



### State Representation: Characteristics

- State 1 --> Low Step Length, High Turn Angle --> "Surveillance" state (Red)
- State 2 --> High Step Length, Low Turn Angle --> "Ambulatory" state (Blue)

# Plot 1: Heatmap Representations of HMM States

This workflow visualizes the spatial distribution of HMM state proportions over a grid-mapped maze using heatmaps. It overlays key regions such as decision and target nodes for the specified genotype.

### Recommended Use:
1. Ensure `df_hmm` contains columns: 'Genotype', 'Grid.Number', 'HMM_State', 'x', and 'y'.
2. Set `genotype_name` to the genotype of interest (e.g., 'WT-WT').
3. Set `grid_filename` to the corresponding shapefile for the session.
4. Use `compute_state_proportion()` to calculate per-grid HMM state proportions.
5. Load the grid geometry using `create_grid_geodata()`.
6. Use `map_points_to_grid()` and `sjoin()` to align state estimates with grid polygons.
7. Merge state proportions back into the grid using `merge_state_proportions_to_grid()`.
8. Plot the heatmap using `plot_grid_heatmap()` and highlight decision/target regions.

```python
from compass_labyrinth.post_hoc_analysis.level_1 import (
    compute_state_proportion,
    create_grid_geodata,
    map_points_to_grid,
    merge_state_proportions_to_grid,
    plot_grid_heatmap,
)

genotype_name = 'WT'
grid_filename = "Session-3 grid.shp"

# Step 1: Compute HMM state proportions by grid
state_df = compute_state_proportion(
    df=df_hmm,
    genotype_name=genotype_name,
    hmm_state=2,
)

# Step 2: Load session-specific grid geometry
grid = create_grid_geodata(
    config=config,
    grid_filename=grid_filename,
)

# Step 3: Map mean (x, y) points to grid polygons
pointInPolys = map_points_to_grid(state_df, grid)

# Step 4: Merge proportions with grid polygons
grid_mapped = merge_state_proportions_to_grid(grid, state_df)

# Step 5: Plot heatmap with overlays for Decision and Target zones
ax = plot_grid_heatmap(
    config=config,
    grid=grid_mapped,
    genotype_name=genotype_name, 
    highlight_grids="decision_reward",
    target_grids="target_zone",
)

print('Decision Nodes highlighted in BLACK')
print('Target Nodes highlighted in YELLOW')
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/WT_grid_heatmap.png



    
![png](05_compass_level_1_post_analysis_files/05_compass_level_1_post_analysis_7_1.png)
    


    Decision Nodes highlighted in BLACK
    Target Nodes highlighted in YELLOW


## Heatmap representations of all genotypes

```python
from compass_labyrinth.post_hoc_analysis.level_1 import plot_all_genotype_heatmaps


plot_all_genotype_heatmaps(
    config=config,
    df_hmm=df_hmm,
    grid_filename = grid_filename,
    highlight_grids="decision_reward",
    target_grids="target_zone",
    hmm_state=2,
    cmap='RdBu',
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_grid_heatmap.pdf



    
![png](05_compass_level_1_post_analysis_files/05_compass_level_1_post_analysis_9_1.png)
    


## Interactive Heatmap version

```python
from compass_labyrinth.post_hoc_analysis.level_1 import (
    compute_state_proportion,
    create_grid_geodata,
    get_grid_centroids,
    plot_interactive_heatmap,
    overlay_trajectory_lines_plotly,
)


genotype_name = 'WT'

# 1. Compute state proportions
state_df = compute_state_proportion(df_hmm, genotype_name)

# 2. Load session-specific grid geometry
grid = create_grid_geodata(
    config=config,
    grid_filename=grid_filename,
)

# 3. Merge proportion values to grid
grid_mapped = merge_state_proportions_to_grid(grid, state_df)

# 4. Get grid centroids
grid_centroids = get_grid_centroids(grid_mapped)

# 5. Plot heatmap
fig = plot_interactive_heatmap(
    config=config,
    grid_mapped=grid_mapped,
    genotype_name=genotype_name,
    decision_grids="decision_reward",
    target_grids="target_zone",
    show_fig=False,
    return_fig=True,
)

# 6. Overlay smooth trajectory lines
overlay_trajectory_lines_plotly(
    fig=fig,
    df_hmm=df_hmm,
    genotype_name=genotype_name,
    grid_centroids=grid_centroids,
    top_percent=.1,
)

# 7. Show
fig.show()
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/WT_interactive_grid_heatmap.html




## Interactive Heatmap version of all genotypes

```python
from compass_labyrinth.post_hoc_analysis.level_1 import plot_all_genotype_interactive_heatmaps


plot_all_genotype_interactive_heatmaps(
    config=config,
    df_hmm=df_hmm,
    grid_filename="Session-3 grid.shp",
    hmm_state=2,
    decision_grids="decision_reward",
    target_grids="target_zone",
    top_percent=.1,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_interactive_grid_heatmap.html




# Plot 2: Probability of Surveillance across Node Types and Regions

This workflow computes and visualizes the probability of HMM state occupancy across behavioral Node Types/Regions

### Recommended Use:
1. Ensure `df_hmm` includes 'Genotype', 'Session', 'HMM_State', 'NodeType', and 'Grid.Number'.
2. If analyzing decision complexity, pass lists of `decision_3way` and `decision_4way` grid numbers.
3. Use `compute_state_probability()` to get HMM state proportions per category.
4. Use `plot_state_probability_boxplot()` to visualize across genotypes.

```python
from compass_labyrinth.post_hoc_analysis.level_1 import (
    compute_state_probability,
    plot_state_probability_boxplot,
)


column_of_interest = 'NodeType'
values_displayed = [
    '3-way Decision (Reward)', '4-way Decision (Reward)','Non-Decision (Reward)', 
    'Decision (Non-Reward)', 'Non-Decision (Non-Reward)',
    'Corner (Reward)', 'Corner (Non-Reward)'
]
state = 1

# Step 1: Compute proportions
state_count_df = compute_state_probability(
    df_hmm=df_hmm,
    column_of_interest=column_of_interest,
    values_displayed=values_displayed,
    state=state,
)

# Step 2: Plot boxplot
plot_state_probability_boxplot(
    config=config,
    state_count_df=state_count_df,
    column_of_interest=column_of_interest,
    state=state,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/state_1_probability_by_NodeType.pdf



    
![png](05_compass_level_1_post_analysis_files/05_compass_level_1_post_analysis_15_1.png)
    


```python
from compass_labyrinth.post_hoc_analysis.level_1 import run_pairwise_ttests


# T-tests amongst genotypes per category
ttest_results = run_pairwise_ttests(state_count_df, column_of_interest='NodeType')
print(ttest_results.sort_values("P-value"))
```

                           Group Genotype1 Genotype2    T-stat   P-value
    2        Corner (Non-Reward)        KO        WT  1.054990  0.370853
    6      Non-Decision (Reward)        KO        WT -0.986303  0.406441
    4      Decision (Non-Reward)        KO        WT -1.294013  0.414199
    3            Corner (Reward)        KO        WT  0.730535  0.580503
    0    3-way Decision (Reward)        KO        WT -0.355269  0.768368
    5  Non-Decision (Non-Reward)        KO        WT  0.290684  0.803928
    1    4-way Decision (Reward)        KO        WT  0.098086  0.933951


# Plot 3: Probability of States over Time

This workflow analyzes how the probability of being in a specific HMM state (e.g., State 2) evolves over time at decision vs. non-decision nodes. It computes median probabilities across sessions using sliding time bins and plots the resulting curves to compare decision-related dynamics.

### Recommended Use:
1. Ensure `df_hmm` contains columns: 'Time', 'Session', 'Grid.Number', 'HMM_State', 'NodeType', and 'Genotype'.
2. Define grid numbers corresponding to `Decision_Reward` and `NonDecision_Reward` nodes.
3. Set a time window (`lower_limit` to `upper_limit`) and a `bin_size` (e.g., 2000 frames) to compute time bins.
4. Use `compute_node_state_medians_over_time()` to calculate median state occupancy at decision vs. non-decision nodes.
5. Optionally filter the resulting DataFrame (`Deci_DF`) using a `threshold` to only keep early time bins.
6. Visualize the trajectory of state occupancy using `plot_node_state_median_curve()`.

```python
from compass_labyrinth.post_hoc_analysis.level_1 import (
    get_max_session_row_bracket,
    get_min_session_row_bracket,
    compute_node_state_medians_over_time,
    plot_node_state_median_curve,
)


lower_limit = 0
upper_limit = get_max_session_row_bracket(df_hmm)
threshold =  get_min_session_row_bracket(df_hmm)  # Only show bins where all sessions are present
bin_size = 2000
palette = ['grey', 'black']
figure_ylimit = (0.6, 1.1)

# Step 1: Compute median probability of being in State 1 across time bins
deci_df = compute_node_state_medians_over_time(
    df_hmm=df_hmm,
    state_types=[2],
    lower_lim=lower_limit,
    upper_lim=upper_limit,
    bin_size=bin_size
)

# Step 2: Optional filter to only plot early session bins
deci_df = deci_df.loc[deci_df.Time_Bins < threshold]

# Step 3: Plot time-evolving median probability curves
plot_node_state_median_curve(
    config=config,
    deci_df=deci_df,
    palette=palette,
    figure_ylimit=figure_ylimit,
    fig_title = 'Median Probability of Ambulatory State'
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/temporal_median_state_probability_curve.pdf



    
![png](05_compass_level_1_post_analysis_files/05_compass_level_1_post_analysis_18_1.png)
    


# Plot 4: Surveillance Probability by Bout Type

This workflow evaluates behavioral surveillance patterns at decision nodes across navigational bouts in the maze. It segments the session into bouts, computes  surveillance probability for each bout at decision nodes, and visualizes the average surveillance behavior across successful and unsuccessful bouts.

### Recommended Use:
1. Ensure `df_hmm` contains columns: 'Grid.Number', 'Session', 'Genotype', and maze node annotations.
2. Use `assign_bout_indices()` to segment sessions into bouts based on re-entries into the maze (e.g., delimiter node 47).
3. Run `compute_surveillance_probabilities()` to calculate surveillance at specified decision nodes for each bout.
4. Use `plot_surveillance_by_bout()` to generate a grouped barplot comparing surveillance probability between successful and unsuccessful bouts, including a t-test p-value.

```python
from compass_labyrinth.post_hoc_analysis.level_1 import (
    assign_bout_indices,
    compute_surveillance_probabilities,
    plot_surveillance_by_bout,
)


# Assign Bout Numbers 
# Bout = Entry node 47 --> Other non-entry nodes --> Entry node 47
df_hmm = assign_bout_indices(
    df=df_hmm,
    delimiter_node=47,
)

# Compute surveillance probability at Decision nodes by Bout type
# Successful-> reached the target atleast once/Unsuccessful-> doesn't reached the target
index_df, median_df = compute_surveillance_probabilities(
    df_hmm=df_hmm,
    decision_nodes="decision_reward",
)

# Barplot to depict the above with ttest-ind pvalue
plot_surveillance_by_bout(
    config=config,
    median_df=median_df,
    ylim=0.6,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/surveillance_probability_by_bout.pdf



    
![png](05_compass_level_1_post_analysis_files/05_compass_level_1_post_analysis_20_1.png)
    


```python
from compass_labyrinth.post_hoc_analysis.level_1 import run_within_genotype_mixedlm_with_fdr


# LMM for same genotype comparison across Bout types
df_within = run_within_genotype_mixedlm_with_fdr(index_df)

# Print results
print("Within-Genotype (Successful vs Unsuccessful):")
print(df_within)
```

    Within-Genotype (Successful vs Unsuccessful):
    Empty DataFrame
    Columns: [FDR P-value, Significant (FDR < 0.05)]
    Index: []


```python
from compass_labyrinth.post_hoc_analysis.level_1 import test_across_genotypes_per_bout


# T-test across genotypes under Successful Bouts
df_across_success = test_across_genotypes_per_bout(median_df, bout_type='Successful')

# T-test across genotypes under Unsuccessful Bouts
df_across_unsuccess = test_across_genotypes_per_bout(median_df, bout_type='Unsuccessful')

# Print results
print("\n Across Genotypes (Successful only):")
print(df_across_success)

print("\n Across Genotypes (Unsuccessful only):")
print(df_across_unsuccess)
```

    
     Across Genotypes (Successful only):
    Empty DataFrame
    Columns: []
    Index: []
    
     Across Genotypes (Unsuccessful only):
          Bout Type Genotype 1 Genotype 2    T-stat   P-value
    0  Unsuccessful         KO         WT -0.714032  0.556053

