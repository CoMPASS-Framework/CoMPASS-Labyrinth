# Table of Contents

1. [Load Project](##Load-Project) 
2. [Simulated Agent Modelling](#Simulated-Agent-Modelling)
    1. [Plot 1: Simulated Agent vs Mouse Performance across Time](#Plot-1:-Simulated-Agent-vs-Mouse-Performance-across-Time)
    2. [Plot 2: Relative Performance across Time](#Plot-2:-Relative-Performance-across-Time)
    3. [Plot 3: Avg. Simulated Agent and Mouse Performance across Sessions(/Mice)](#Plot-3:-Avg.-Simulated-Agent-and-Mouse-Performance-across-Sessions)
3. [Chi Square Analysis](#Chi-Square-Analysis)
4. [Simulated Agent, Binary Agent, 3 or 4-way Agent Modelling & Comparison](#Simulated-Agent,-Binary-Agent,-3-or-4-way-Agent-Modelling-&-Comparison)
    1. [Plot 5: All Agents Comparative Performance over time](#Plot-5:-All-Agents-Comparative-Performance-over-time)
    2. [Plot 6: Cumulative Agent Performance](#Plot-6:-Cumulative-Multiple-Agent-Performance)
5. [Exploration-Exploitation (EE) Agent Evaluation](#Exploration-Exploitation-(EE)-Agent-Evaluation)
    1. [Plot 7: Agent Performance Across Varying Exploration Rates](#Plot-7:-Agent-Performance-Across-Varying-Exploration-Rates)

# Load Project

Load project's configuration, metadata and combined data.

```python
from pathlib import Path
import pandas as pd
from compass_labyrinth import load_project


project_path = Path(".").resolve() / "my_project_2"

# Import config and metadata
config, cohort_metadata = load_project(project_path=project_path)

# Import all sessions combined pose-estimation CSV
df_all_csv = pd.read_csv(project_path / "csvs" / "combined" / "Preprocessed_combined_file_exclusions.csv")
```

# Simulated Agent Modelling

This function evaluates the performance of a simulated agent navigating the maze by estimating the proportion of optimal decisions (e.g., choosing the reward path) within fixed-size epochs across multiple simulations and bootstrap samples.

It also trims the simulated agent performance dataframe to the set of common epochs shared across all simulations, ensuring clean aggregation and plotting.

### Recommended Use:
1. Ensure `df_all_csv` includes 'Session', 'Grid Number', and the specified decision node labels.
2. Set `epoch_size` to define the number of frames grouped into each epoch (e.g., 1000).
3. Set `n_bootstrap` to the number of bootstrap resamples per simulation.
4. Set `n_simulations` to the number of independent simulated agents to evaluate.
5. Set `decision_label` to the node type representing decision points (e.g., 'Decision (Reward)').
6. Set `reward_label` to the region considered as the optimal path (e.g., 'Reward Path').

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import evaluate_agent_performance

# Set these values
EPOCH_SIZE = 1000
N_BOOTSTRAP = 10000
N_SIMULATIONS = 100
DECISION_LABEL = "Decision (Reward)"
REWARD_LABEL = "reward_path"
GENOTYPE = "WT"

sim_results = evaluate_agent_performance(
    df=df_all_csv,
    epoch_size=EPOCH_SIZE,
    n_bootstrap=N_BOOTSTRAP,
    n_simulations=N_SIMULATIONS,
    decision_label=DECISION_LABEL,
    reward_label=REWARD_LABEL,
    trim=True,
)
sim_results["WT"]
```

     Max common epoch across all sessions: 26
     Max common epoch across all sessions: 55





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
      <th>Actual Reward Path %</th>
      <th>Simulated Agent Reward Path %</th>
      <th>Actual Reward Path % CI Lower</th>
      <th>Actual Reward Path % CI Upper</th>
      <th>Simulated Agent Reward Path % CI Lower</th>
      <th>Simulated Agent Reward Path % CI Upper</th>
      <th>Relative Performance</th>
      <th>Session</th>
      <th>Epoch Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.923940</td>
      <td>0.748434</td>
      <td>0.885496</td>
      <td>0.961832</td>
      <td>0.732214</td>
      <td>0.764198</td>
      <td>1.234498</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.927493</td>
      <td>0.747362</td>
      <td>0.900763</td>
      <td>0.954198</td>
      <td>0.737023</td>
      <td>0.757521</td>
      <td>1.241023</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.896362</td>
      <td>0.699103</td>
      <td>0.862069</td>
      <td>0.931034</td>
      <td>0.684397</td>
      <td>0.713793</td>
      <td>1.282159</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.917142</td>
      <td>0.730479</td>
      <td>0.888430</td>
      <td>0.946281</td>
      <td>0.718965</td>
      <td>0.742068</td>
      <td>1.255534</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.932080</td>
      <td>0.718258</td>
      <td>0.900524</td>
      <td>0.958115</td>
      <td>0.702984</td>
      <td>0.733353</td>
      <td>1.297696</td>
      <td>3</td>
      <td>5</td>
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
    </tr>
    <tr>
      <th>73</th>
      <td>0.885978</td>
      <td>0.680947</td>
      <td>0.822785</td>
      <td>0.936709</td>
      <td>0.658734</td>
      <td>0.702785</td>
      <td>1.301098</td>
      <td>5</td>
      <td>22</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.916607</td>
      <td>0.710688</td>
      <td>0.882353</td>
      <td>0.946078</td>
      <td>0.697252</td>
      <td>0.723775</td>
      <td>1.289746</td>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.914044</td>
      <td>0.754817</td>
      <td>0.880383</td>
      <td>0.942584</td>
      <td>0.745742</td>
      <td>0.763493</td>
      <td>1.210947</td>
      <td>5</td>
      <td>24</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.897510</td>
      <td>0.670536</td>
      <td>0.860215</td>
      <td>0.930108</td>
      <td>0.654409</td>
      <td>0.686454</td>
      <td>1.338496</td>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.960761</td>
      <td>0.723498</td>
      <td>0.936275</td>
      <td>0.980392</td>
      <td>0.712353</td>
      <td>0.734510</td>
      <td>1.327939</td>
      <td>5</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>78 rows × 9 columns</p>
</div>



## Plot 1: Simulated Agent vs Mouse Performance across Time

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_agent_transition_performance


plot_agent_transition_performance(
    config=config,
    evaluation_results=sim_results,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_sim_agent_mouse_perf.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_6_1.png)
    


## Plot 2: Relative Performance across Time

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_relative_agent_performance


plot_relative_agent_performance(
    config=config,
    evaluation_results=sim_results,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_relative_perf.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_8_1.png)
    


## Plot 3: Avg. Simulated Agent and Mouse Performance across Sessions

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import run_mixedlm_for_all_genotypes


pvals_by_genotype = run_mixedlm_for_all_genotypes(
    config=config,
    evaluation_results=sim_results,
    plot_palette=["purple", "grey"],
)
print(pvals_by_genotype)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/cumulative_sim_agent_mouse_perf.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_10_1.png)
    


    {'WT': np.float64(6.127002099983219e-215), 'KO': np.float64(0.0)}


# Chi Square Analysis

This workflow calculates the chi-square divergence between actual animal performance and a simulated agent’s expected behavior across epochs, and summarizes the result using both rolling and cumulative statistics to track behavioral divergence over time.

### Recommended Use:
1. Ensure `df_sim` contains 'Actual Reward Path %', 'Simulated Agent Reward Path %', 'Epoch Number', and 'Session' columns.
2. Use `compute_chi_square_statistic()` to compute per-epoch chi-square scores comparing actual vs. simulated usage.
3. Use `compute_rolling_chi_square()` to calculate rolling averages over time for trend visualization.
4. Use `compute_cumulative_chi_square()` to track the running average of chi-square divergence over all prior epochs.

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import run_chi_square_analysis


ROLLING_WINDOW = 3

# Compute chi-square stats for each genotype
chisquare_results = run_chi_square_analysis(
    config=config,
    evaluation_results=sim_results,
    rolling_window=ROLLING_WINDOW
)
```

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_chi_square_and_rolling


plot_chi_square_and_rolling(
    config=config,
    chisquare_results=chisquare_results,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_chi_square_rolling.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_13_1.png)
    


```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_rolling_mean


plot_rolling_mean(
    config=config,
    chisquare_results=chisquare_results,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_average_chi_square_rolling.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_14_1.png)
    


```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_cumulative_chi_square


plot_cumulative_chi_square(
    config=config,
    chisquare_results=chisquare_results,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/all_genotypes_cumulative_chi_square.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_15_1.png)
    


# Simulated Agent, Binary Agent, 3 or 4-way Agent Modelling & Comparison

This function simulates multiple agent types (simulated, binary, and 3/4-way), evaluates their decision performance over fixed-size epochs, and compares them to actual animal decisions using bootstrap confidence intervals.

### Recommended Use:
1. Ensure `df_all_csv` includes 'Session', 'NodeType', 'Region', and 'Grid Number'.
2. Use `epoch_size` to define the time resolution (e.g., 1000 frames per epoch).
3. Set `n_simulations` and `n_bootstrap` to define agent variability and confidence bounds.
4. Use `decision_label` and `reward_label` to define relevant transitions (e.g., 'Decision (Reward)', 'Reward Path').
5. Optional: Pass custom 3-way and 4-way decision node sets.

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import evaluate_agent_performance_multi


# Evaluate agent vs. actual decisions
df_all_simulated = evaluate_agent_performance_multi(
    df=df_all_csv,
    epoch_size=EPOCH_SIZE,
    n_bootstrap=N_BOOTSTRAP,
    n_simulations=N_SIMULATIONS,
    decision_label=DECISION_LABEL,
    reward_label=REWARD_LABEL,
    trim=True,
)
df_all_simulated
```

     Max common epoch across all sessions: 26





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
      <th>Actual Reward Path %</th>
      <th>Random Agent Reward Path %</th>
      <th>Binary Agent Reward Path %</th>
      <th>Three/Four Way Agent Reward Path %</th>
      <th>Actual Reward Path % CI Lower</th>
      <th>Actual Reward Path % CI Upper</th>
      <th>Random Agent Reward Path % CI Lower</th>
      <th>Random Agent Reward Path % CI Upper</th>
      <th>Binary Agent Reward Path % CI Lower</th>
      <th>Binary Agent Reward Path % CI Upper</th>
      <th>Three/Four Way Agent Reward Path % CI Lower</th>
      <th>Three/Four Way Agent Reward Path % CI Upper</th>
      <th>Relative Performance (Actual/Random)</th>
      <th>Relative Performance (Actual/Binary)</th>
      <th>Session</th>
      <th>Epoch Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.923615</td>
      <td>0.751548</td>
      <td>0.497364</td>
      <td>0.300797</td>
      <td>0.885496</td>
      <td>0.961832</td>
      <td>0.735573</td>
      <td>0.767176</td>
      <td>0.489084</td>
      <td>0.505420</td>
      <td>0.291756</td>
      <td>0.310076</td>
      <td>1.228951</td>
      <td>1.857021</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.927951</td>
      <td>0.749971</td>
      <td>0.496901</td>
      <td>0.309913</td>
      <td>0.900763</td>
      <td>0.954198</td>
      <td>0.739998</td>
      <td>0.760153</td>
      <td>0.491945</td>
      <td>0.501832</td>
      <td>0.304008</td>
      <td>0.315840</td>
      <td>1.237315</td>
      <td>1.867476</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.896551</td>
      <td>0.692951</td>
      <td>0.496214</td>
      <td>0.317039</td>
      <td>0.862069</td>
      <td>0.926724</td>
      <td>0.678103</td>
      <td>0.707845</td>
      <td>0.491078</td>
      <td>0.501422</td>
      <td>0.310733</td>
      <td>0.323362</td>
      <td>1.293815</td>
      <td>1.806782</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.917495</td>
      <td>0.731893</td>
      <td>0.501693</td>
      <td>0.314694</td>
      <td>0.888430</td>
      <td>0.946281</td>
      <td>0.719669</td>
      <td>0.744132</td>
      <td>0.496446</td>
      <td>0.507025</td>
      <td>0.308719</td>
      <td>0.320702</td>
      <td>1.253591</td>
      <td>1.828797</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.932093</td>
      <td>0.720083</td>
      <td>0.499386</td>
      <td>0.325859</td>
      <td>0.900524</td>
      <td>0.963351</td>
      <td>0.704660</td>
      <td>0.735395</td>
      <td>0.493243</td>
      <td>0.505550</td>
      <td>0.319948</td>
      <td>0.331780</td>
      <td>1.294425</td>
      <td>1.866479</td>
      <td>3</td>
      <td>5</td>
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
    </tr>
    <tr>
      <th>125</th>
      <td>0.947535</td>
      <td>0.684012</td>
      <td>0.497143</td>
      <td>0.313984</td>
      <td>0.926573</td>
      <td>0.968531</td>
      <td>0.674091</td>
      <td>0.694058</td>
      <td>0.492483</td>
      <td>0.501853</td>
      <td>0.308846</td>
      <td>0.318986</td>
      <td>1.385261</td>
      <td>1.905960</td>
      <td>7</td>
      <td>22</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0.907552</td>
      <td>0.683316</td>
      <td>0.497201</td>
      <td>0.323391</td>
      <td>0.870370</td>
      <td>0.944444</td>
      <td>0.669812</td>
      <td>0.696728</td>
      <td>0.490864</td>
      <td>0.503642</td>
      <td>0.315864</td>
      <td>0.330864</td>
      <td>1.328159</td>
      <td>1.825321</td>
      <td>7</td>
      <td>23</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0.946756</td>
      <td>0.717656</td>
      <td>0.492673</td>
      <td>0.319343</td>
      <td>0.920705</td>
      <td>0.969163</td>
      <td>0.705419</td>
      <td>0.729868</td>
      <td>0.487225</td>
      <td>0.498062</td>
      <td>0.313348</td>
      <td>0.325419</td>
      <td>1.319234</td>
      <td>1.921670</td>
      <td>7</td>
      <td>24</td>
    </tr>
    <tr>
      <th>128</th>
      <td>0.940703</td>
      <td>0.697772</td>
      <td>0.496165</td>
      <td>0.310431</td>
      <td>0.911330</td>
      <td>0.965517</td>
      <td>0.685663</td>
      <td>0.709901</td>
      <td>0.490342</td>
      <td>0.501970</td>
      <td>0.303547</td>
      <td>0.317241</td>
      <td>1.348153</td>
      <td>1.895950</td>
      <td>7</td>
      <td>25</td>
    </tr>
    <tr>
      <th>129</th>
      <td>0.919268</td>
      <td>0.688790</td>
      <td>0.495408</td>
      <td>0.332614</td>
      <td>0.879195</td>
      <td>0.953020</td>
      <td>0.674094</td>
      <td>0.703221</td>
      <td>0.488792</td>
      <td>0.501950</td>
      <td>0.325973</td>
      <td>0.339060</td>
      <td>1.334613</td>
      <td>1.855579</td>
      <td>7</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>130 rows × 16 columns</p>
</div>



## Plot 5: All Agents Comparative Performance over time

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_agent_vs_mouse_performance_multi


GENOTYPE = 'WT'

plot_agent_vs_mouse_performance_multi(
    config=config,
    df_metrics=df_all_simulated,
    cohort_metadata=cohort_metadata,
    genotype=GENOTYPE,
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/WT_multiple_agent.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_19_1.png)
    


## Plot 6: Cumulative Multiple Agent Performance

This function compares average reward path transition percentages across sessions for mouse and simulated agents using a boxplot. Useful for visualizing group-wise strategy efficiency.

### Recommended Use:
1. Provide `df_metrics` from evaluate_agent_performance_multi().
2. Provide `cohort_metadata` with 'Session #' and 'Genotype'.
3. Set `genotype` to the group you want to compare.

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_cumulative_agent_comparison_boxplot_multi


GENOTYPE = 'WT'

plot_cumulative_agent_comparison_boxplot_multi(
    config=config,
    df_metrics=df_all_simulated,
    cohort_metadata=cohort_metadata,
    genotype=GENOTYPE,
    figsize=(7, 7),
)
```

    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/WT_cumulative_multiple_agent.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_21_1.png)
    


# Exploration-Exploitation (EE) Agent Evaluation

This module simulates an exploration-exploitation agent with a tunable exploration rate and compares its reward path transition performance to that of real mice over fixed-size epochs.

### Recommended Use:
1. Ensure `df_all_csv` includes 'Session', 'NodeType', 'Region', and 'Grid Number' columns.
2. Set `exploration_rate` to define the agent’s behavior:
      - Low values (e.g., 0.2) bias toward exploitation (optimal paths).
      - High values (e.g., 0.9) bias toward exploration (random paths).
3. Use `segment_size` consistently in both analysis and plotting functions.
4. Use `n_simulations` and `n_bootstrap` to determine statistical confidence.
5. Use `decision_label` and `reward_label` to define what constitutes a decision and a reward-oriented move.

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import run_exploration_agent_analysis_EE
import numpy as np


SEGMENT_SIZE = 1000  # Ensure this is consistent across functions
EXPLORATION_RATE = 0.5  # Tunable parameter for agent behavior

df_agent_perf = run_exploration_agent_analysis_EE(
    df=df_all_csv,
    exploration_rate=EXPLORATION_RATE,
    segment_size=SEGMENT_SIZE,
    n_bootstrap=10000,
    n_simulations=100,
    decision_label='Decision (Reward)',
    reward_label='reward_path',
)
df_agent_perf
```

     Max common epoch across all sessions: 26





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
      <th>Actual Reward Path %</th>
      <th>Agent Reward Path %</th>
      <th>Actual Reward Path % CI Lower</th>
      <th>Actual Reward Path % CI Upper</th>
      <th>Agent Reward Path % CI Lower</th>
      <th>Agent Reward Path % CI Upper</th>
      <th>Relative Performance</th>
      <th>Session</th>
      <th>Epoch Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.863970</td>
      <td>0.855894</td>
      <td>0.826291</td>
      <td>0.901408</td>
      <td>0.847509</td>
      <td>0.864131</td>
      <td>1.009437</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.894246</td>
      <td>0.863607</td>
      <td>0.855556</td>
      <td>0.933333</td>
      <td>0.855331</td>
      <td>0.871722</td>
      <td>1.035478</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.897874</td>
      <td>0.851012</td>
      <td>0.862245</td>
      <td>0.933673</td>
      <td>0.841939</td>
      <td>0.859745</td>
      <td>1.055066</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.884607</td>
      <td>0.847398</td>
      <td>0.843537</td>
      <td>0.925170</td>
      <td>0.837347</td>
      <td>0.857483</td>
      <td>1.043909</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.877539</td>
      <td>0.873403</td>
      <td>0.836257</td>
      <td>0.918129</td>
      <td>0.865322</td>
      <td>0.881345</td>
      <td>1.004736</td>
      <td>6</td>
      <td>5</td>
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
    </tr>
    <tr>
      <th>125</th>
      <td>0.885952</td>
      <td>0.848380</td>
      <td>0.822785</td>
      <td>0.936709</td>
      <td>0.836329</td>
      <td>0.860259</td>
      <td>1.044287</td>
      <td>5</td>
      <td>22</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0.916440</td>
      <td>0.856181</td>
      <td>0.882353</td>
      <td>0.946078</td>
      <td>0.848529</td>
      <td>0.863676</td>
      <td>1.070381</td>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0.914030</td>
      <td>0.876937</td>
      <td>0.880383</td>
      <td>0.942584</td>
      <td>0.871148</td>
      <td>0.882632</td>
      <td>1.042298</td>
      <td>5</td>
      <td>24</td>
    </tr>
    <tr>
      <th>128</th>
      <td>0.897738</td>
      <td>0.837555</td>
      <td>0.860215</td>
      <td>0.935484</td>
      <td>0.829194</td>
      <td>0.845806</td>
      <td>1.071855</td>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <th>129</th>
      <td>0.960766</td>
      <td>0.860857</td>
      <td>0.936275</td>
      <td>0.980392</td>
      <td>0.854363</td>
      <td>0.867255</td>
      <td>1.116057</td>
      <td>5</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>130 rows × 9 columns</p>
</div>



## Plot 7: Agent Performance Across Varying Exploration Rates

Visualize agent vs. mouse performance across exploration rates:

```python
from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_exploration_rate_performance_EE


# Range of exploration rate values
EXPLORATION_RATE_RANGE = np.arange(0.2, 1.0, 0.1)

plot_exploration_rate_performance_EE(
    config=config,
    df_source=df_all_csv,
    exploration_rates=EXPLORATION_RATE_RANGE,
    segment_size=SEGMENT_SIZE,
    trim=True,
)
```

    Exploration rate =  0.2  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.3  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.4  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.5  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.6  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.7  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.8  being processed....
     Max common epoch across all sessions: 26
    Exploration rate =  0.9  being processed....
     Max common epoch across all sessions: 26
     Max common epoch across all sessions: 26
    Figure saved at: /Users/luiztauffer/Github/CoMPASS-Labyrinth/notebooks/my_project_2/figures/ee_agent.pdf



    
![png](03_simulated_agent_modelling_files/03_simulated_agent_modelling_25_1.png)
    

