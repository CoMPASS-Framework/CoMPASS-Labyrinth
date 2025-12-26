# Create a new project and ingest DLC data

1. [Initiate Project](#Initiate-Project)
2. [Concatenating all DLC results](#Concatenating-all-DLC-results)
3. [Preprocessing](#Preprocessing)
4. [Create Velocity column](#Create-Velocity-column)
5. [Save Preprocessed Combined & Individual csvs](#Save-Preprocessed-Combined-&-Individual-csvs)


# Initiate Project

```python
from pathlib import Path
from compass_labyrinth import init_project


project_path = Path(".").resolve()
source_data_path = "/ethoml_labyrinth/data"
user_metadata_file_path = "/ethoml_labyrinth/data/WT_DSI_Labyrinth_Metadata.xlsx"
trial_type = "Labyrinth_DSI"

config, cohort_metadata = init_project(
    project_name="my_project",
    project_path=project_path,
    source_data_path=source_data_path,
    user_metadata_file_path=user_metadata_file_path,
    trial_type="Labyrinth_DSI",
    file_ext=".csv",
    video_type=".mp4",
    dlc_scorer="DLC_resnet50_LabyrinthMar13shuffle1_1000000",
    experimental_groups=["A", "B", "C", "D"],
)
```

```python
print(config.keys())
config
```

# Concatenating all DLC results

```python
from compass_labyrinth.behavior.preprocessing import compile_mouse_sessions


df_comb = compile_mouse_sessions(
    config=config,
    bp='sternum',
)
df_comb
```

# Preprocessing

```python
from compass_labyrinth.behavior.preprocessing import preprocess_sessions


df_all_csv = preprocess_sessions(df_comb=df_comb)
df_all_csv
```

# Create Velocity column

```python
from compass_labyrinth.behavior.preprocessing import ensure_velocity_column


df_all_csv = ensure_velocity_column(df_all_csv, fps=5) 
df_all_csv
```

# Save Preprocessed Combined & Individual csvs

```python
from compass_labyrinth.behavior.preprocessing import save_preprocessed_to_csv


save_preprocessed_to_csv(
    config=config,
    df=df_all_csv,
)
```
