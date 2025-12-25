## Environment Setup

**⚠️ Important: DeepLabCut Environment Required**

This notebook requires DeepLabCut to be installed and should be run in a DeepLabCut conda environment.

### Installation Instructions

If you haven't already installed DeepLabCut, follow these steps:

1. Create a new conda environment with DeepLabCut:
```bash
   conda create -n dlc python=3.9
   conda activate dlc
   pip install deeplabcut[gui]
```

2. Launch Jupyter and select the "Python (DLC)" kernel for this notebook

### Running This Notebook

Before running this notebook:
- Ensure you have activated the DeepLabCut environment: `conda activate dlc`
- Select the correct kernel in Jupyter: Kernel → Change Kernel → Python (DLC)

For more information, see the [DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)

# Install Necessary Packages (Only need to do once)

```python
# Install Necessary Packages
! pip install geopandas
! pip install shapely
! pip install opencv-python

```

```python
# autoreload
%load_ext autoreload
%autoreload 2
```

# Imports

```python
# Importing the toolbox (takes several seconds)
import pandas as pd
import numpy as np
from pathlib import Path
import os
from pylab import *
import os
import geopandas as gpd
import scipy.stats as sp
from datetime import datetime
import sys

from pathlib import Path

# DeepLabCut Import
import deeplabcut
import tensorflow as tf
import keras
os.environ['DLClight'] = 'True'
print(tf.__version__)

# Specify path to dlc_utils file
custom_utils_path = Path(os.getcwd()).parent/'src'/'compass_labyrinth'/'behavior'/'pose_estimation'
sys.path.append(str(custom_utils_path))

import dlc_utils
```

### Specify Paths and Other Information

```python
##### CONFIGURE PATHS FOR YOUR LOCAL SYSTEM #####

# === VIDEO FILE LOCATIONS ===
# NOTE: The following paths are specific to Palop Lab workflow for copying videos from multiple recording computers
# For most users: skip video_path_1 and video_path_2, and only set videofile_path to your video directory

# Original video locations from recording computers (Palop Lab specific - IGNORE FOR MOST CASES)
video_path_1 = r'D:\Gladstone Dropbox\Palop Lab\Patrick\Machine Learning Behavioral Analysis\Labyrinth\Noldus\20251019_AppSAA_DSI_Labyrinth\Media Files'
video_path_2 = ''  # Leave empty if not using multiple computers

# Central video location where all videos are stored for processing
source_data_path = r'D:\Gladstone Dropbox\Palop Lab\Patrick\Machine Learning Behavioral Analysis\Labyrinth\Noldus\20251019_AppSAA_DSI_Labyrinth\Media Files\source_data_test'
videofile_path = source_data_path

# === DEEPLABCUT CONFIGURATION ===
# Path to your trained DLC model config file
dlc_config_path = Path(r'D:\Gladstone Dropbox\Palop Lab\Patrick\Machine Learning Behavioral Analysis\Labyrinth\DeepLabCut Projects\Labyrinth-Nick-2023-03-13\config.yaml')
dlc_scorer = 'DLC_resnet50_LabyrinthMar13shuffle1_1000000'  # Scorer name from your DLC model

# === OUTPUT PATHS ===
# Grid-based files for HMM state heatmap visualizations
grid_path = source_data_path

# Output directory for figures
figure_path = os.path.join(source_data_path, 'figures')

# === METADATA ===
# Excel file containing animal and session metadata
user_metadata_file_path = r'D:\Gladstone Dropbox\Palop Lab\Patrick\Machine Learning Behavioral Analysis\DLC Info Sheets\20250725_LG124KI3_Cohort4_DLC_InfoSheet_v1.xlsx'
trial_type = 'Labyrinth_DSI'  # Sheet/tab name in the metadata file

# === SUMMARY ===
print("=== Path Configuration ===")
print(f"Video Source 1: {video_path_1}")
print(f"Video Source 2: {video_path_2}")
print(f"Central Video Location: {videofile_path}")
print(f"Metadata: {user_metadata_file_path}")
print(f"Figures Output: {figure_path}")
```

```python
python --version
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 python --version


    NameError: name 'python' is not defined


# Import Mouse Metadata

```python
# Add Metadata to metadata directory
mouseinfo = dlc_utils.import_cohort_metadata(user_metadata_file_path, trial_type)

# Validate the metadata
dlc_utils.validate_metadata(mouseinfo)

# Display summary of the metadata
dlc_utils.display_metadata_summary(mouseinfo)

mouseinfo.head()
```

# OPTIONAL: Copy and rename videos from original location to VIDEOFILE_PATH


```python
now = datetime.datetime.now(); print(now)

# 1. Save first frames
copy_results = dlc_utils.copy_and_rename_videos(
            mouseinfo_df=mouseinfo,
            video_paths=[video_path_1], 
            destination_path=videofile_path,
        )
```

# Get DeepLabCut Cropping Bounds

```python
# Save first frames for all videos
now = datetime.datetime.now(); print(datetime.datetime.now())

print("\nSaving first frames for all videos...")
frame_results = dlc_utils.batch_save_first_frames(
    mouseinfo_df=mouseinfo,
    video_directory=videofile_path,
    frames_directory=source_data_path
)
```

```python
# Get the DLC cropping bounds for a all videos
now = datetime.datetime.now(); print(datetime.datetime.now())

coordinates_dict = dlc_utils.batch_get_boundary_and_cropping(
    mouseinfo_df=mouseinfo, 
    frames_directory=source_data_path,
    cropping_directory=source_data_path,
    boundaries_directory=source_data_path,
    reprocess_existing=False # set to True if you'd like to redo boundaries
    )
```

```python
# Get the DLC cropping bounds for a individual video
coordinates_dict = dlc_utils.get_labyrinth_boundary_and_cropping(
    frames_directory=source_data_path,
    cropping_directory=source_data_path,
    boundaries_directory=source_data_path,
    session='Session0007',
    chamber_info=None)
```

# Analyze the Videos with DeepLabCut

```python

# Prepare sessions
sessions_to_analyze, prep_summary = dlc_utils.prepare_dlc_analysis(
    mouseinfo, 
    videofile_path, 
    source_data_path, 
    source_data_path
)

# Print preparation summary
print(f"\nPreparation Summary:")
print(f"Ready for analysis: {prep_summary['ready_for_analysis']}")
print(f"Already analyzed: {prep_summary['skipped_existing']}")
print(f"Missing videos: {prep_summary['missing_video']}")
print(f"Missing coordinates: {prep_summary['missing_coordinates']}")

# Now run DLC analysis in the correct environment
analysis_results = []
for session in sessions_to_analyze:
    session_start = datetime.datetime.now()
    print(f"\nAnalyzing {session['session_name']}...")
    
    deeplabcut.analyze_videos(
        dlc_config_path,
        [session['video_path']],
        shuffle=1,
        videotype=".mp4",
        save_as_csv=True,
        cropping=session['cropping_coords'],
        destfolder=session['results_path'],
    )

```

# Create Grids and save as Grid Files

```python
now = datetime.datetime.now()

print(f"\nCreating grids for {len(mouseinfo)} sessions...")

# Run batch grid creation
grid_results = dlc_utils.batch_create_grids(
    mouseinfo_df=mouseinfo,
    boundaries_directory=source_data_path,
    grid_files_directory=source_data_path,
    cropping_directory=source_data_path,
    num_squares=12
)
```

# Initial Visualizations

### Plot the Scatterplot with Grid overlayed for each trial

```python
now = datetime.datetime.now()

print("\n--- Batch Processing Example ---")
batch_results = dlc_utils.batch_create_grid_scatter_plots(
    mouseinfo_df=mouseinfo,
    dlc_results_directory=source_data_path,
    grid_files_directory=source_data_path,
    figures_directory=figure_path,
    dlc_scorer=dlc_scorer,
    bodypart='sternum',
    likelihood_threshold=0.6, # rough threshold to visualize when mice are in maze
    figure_size=(3, 3),
    show_plots=False  # Don't display plots during batch processing
)
```

### Create Trajectory Plots with Grid Overlaid

```python
now = datetime.datetime.now()

print("\n--- Different Colormaps Example ---")
colormaps = ['viridis']

for colormap in colormaps:
    print(f"\nCreating trajectory plots with {colormap} colormap...")
    batch_results = dlc_utils.batch_create_trajectory_plots(
        mouseinfo_df=mouseinfo,  # Just first 2 sessions
        dlc_results_directory=source_data_path,
        grid_files_directory=source_data_path,
        figures_directory=os.path.join(figure_path, "trajectory_plots"),
        dlc_scorer=dlc_scorer,
        bodypart='sternum',
        likelihood_threshold=0.6,
        colormap=colormap,
        show_plots=False
    )
```

# Create CSVs with Grid Numbers

```python
now = datetime.datetime.now()

results = dlc_utils.batch_append_grid_numbers(
    mouseinfo_df=mouseinfo,
    grid_files_directory=source_data_path,
    dlc_results_directory=source_data_path,
    dlc_scorer=dlc_scorer,
    save_directory=source_data_path
)
```
