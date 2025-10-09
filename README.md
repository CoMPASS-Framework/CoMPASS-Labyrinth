# CoMPASS-Labyrinth

CoMPASS-Labyrinth is a unified computational and behavioral framework for analyzing goal-directed navigation in complex, ethologically valid maze environments using hierarchical probabilistic models. This project integrates behavioral modeling with neural data analysis to uncover latent cognitive states and their underlying neural dynamics during complex decision-making tasks.


<p align="center">
  <img src="media/compass_logo.png" alt="CoMPASS Logo" width="220"/>
  &nbsp;&nbsp;&nbsp;
  <a href="media/labyrinth_demo.mp4">
    <img src="media/labyrinth_thumbnail.png" alt="Watch Demo" width="220"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <img src="media/maze_layout.png" alt="Maze Layout" width="220"/>
</p>

<hr style="height:2px; border:none; background:linear-gradient(to right, #ccc, #333, #ccc); width:80%;">

<p align="center">
  <img src="media/compass_framework.png" alt="Framework" width="900"/>
</p>


## Key Features

1. **Naturalistic Maze Framework**: A Novel Labyrinth maze paradigm that elicits spontaneous, intrinsically motivated, and untrained navigation behavior in rodents, closely mimicking real-world foraging.

2. **CoMPASS**: A Hierarchical Probabilistic Framework integrating local movement dynamics with goal-directed cognitive states.

3. **Latent State Inference**: Identification of fine-grained cognitive states that underlie navigation strategies, beyond what is captured by task performance alone.

4. **Neural-Behavioral Integration**: Linking probabilistically inferred behavioral states with neural oscillatory signatures, reflecting how internal cognitive processes manifest in circuit-level dynamics.

5. **Translational Relevance**: Sensitive detection of early cognitive deficits in models of neurodegenerative disease (e.g., App-KI mice), with broader implications for human cognition, learning, and memory.

## Prerequisites

- Python 3.11 or higher
- R version 4.4.0 or lower

## Installation

### Option A: Using Conda

1. **Clone the repository**
   ```bash
   git clone https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth.git
   cd CoMPASS-Labyrinth
   ```

2. **Create Python environment**
   ```bash
   conda env create -f environment.yml
   conda activate compass-labyrinth
   ```
   This automatically installs the package with all dependencies from `pyproject.toml`.

3. **Initialize R environment**
   ```bash
   Rscript init_renv.R
   ```
   This installs all required R packages using renv. First run may take several minutes.

### Option B: Using pip with pyproject.toml

1. **Clone the repository**
   ```bash
   git clone https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth.git
   cd CoMPASS-Labyrinth
   ```

2. **Create Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package with all dependencies**
   ```bash
   pip install -e ".[dev]"  # Includes Jupyter and development tools
   # or
   pip install -e .  # Core dependencies only
   ```

4. **Initialize R environment**
   ```bash
   Rscript init_renv.R
   ```

### Notes on Dependency Management

- **Python packages**: Managed via `pyproject.toml`
- **R packages**: Managed via `renv`
- When you start R in this directory, renv will automatically activate

## Usage

1. See individual README.md files within **CoMPASS-Labyrinth\src** for detailed instructions for descriptions of individual modules and code elements.

2. See **CoMPASS-Labyrinth\notebooks** for examples of running the code and instructions for usage. See `data` for example data, dataset information, and metadata. Example datasets are already preprocessed, so avoid notebooks 01, 02, 08

3. Set paths within the **CoMPASS-Labyrinth\notebooks\init_config.py** and **CoMPASS-Labyrinth\src\compass\level_2\compass_config.py**

4. Run **CoMPASS-Labyrinth\notebooks**. Jupyter is highly recommended for interactive features.

## Citation
If you use this framework, please cite the below manuscript:
