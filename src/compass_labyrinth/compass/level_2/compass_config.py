from pathlib import Path
import os

# =============================================================================
# FILE PATHS/INITIALIZATIONS
# =============================================================================

BASE_PATH = Path(os.getcwd()).parent.parent / "TEST_COMPASS"  # TODO: CHANGE THIS TO YOUR BASE PATH LOCATION

# Metadata file paths
METADATA_PATH = os.path.join(BASE_PATH, "data", "metadata")  # Don't edit
METADATA_FILE = os.path.join(METADATA_PATH, "20241028_WT_DSI_Labyrinth_DLC_InfoSheet_v1.xlsx")  # TODO: EDIT

# Trial configuration
TRIAL_TYPE = "Labyrinth_DSI"  # TODO: EDIT
FILE_EXT = ".csv"
VIDEO_TYPE = ".mp4"

# Specify color of plot in graph
PALETTE = ["grey"]

# Specify Value map path # In the Resources folder, the file can be found
VALUE_MAP_PATH = Path(os.getcwd()).parent.parent / "CoMPASS-Labyrinth" / "resources" / "Value_Function_perGridCell.csv"
