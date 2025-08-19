import os
import matplotlib.pyplot as plt

from init_config import BASE_PATH

def create_organized_directory_structure(base_path):
    """
    Create an organized directory structure for the DeepLabCut project.
    
    Parameters:
    -----------
    base_path : str or Path
        Base project directory
        
    Returns:
    --------
    dict
        Dictionary of all directory paths
    """
    from pathlib import Path
    
    base_path = Path(base_path)
    
    # Define organized directory structure
    DIRS = {
        # Videos folder - original videos and frames
        'videos': base_path / 'videos',
        'videos_original': base_path / 'videos' / 'original_videos',
        'frames': base_path / 'videos' / 'frames',
        
        # Data folder - analysis inputs and outputs
        'data': base_path / 'data',
        'dlc_results': base_path / 'data' / 'dlc_results',
        'dlc_cropping': base_path / 'data' / 'dlc_cropping_bounds',
        'grid_files': base_path / 'data' / 'grid_files',
        'grid_boundaries': base_path / 'data' / 'grid_boundaries',
        'metadata': base_path / 'data' / 'metadata',
        'eeg_edfs': base_path / 'data' / 'processed_eeg_edfs',
        
        # Figures folder - all plots and visualizations
        'figures': base_path / 'figures',
        
        # CSV's folder
        'csvs': base_path / 'csvs',
        'csvs_individual': base_path / 'csvs'/ 'individual',
        'csvs_combined': base_path / 'csvs' / 'combined',

        # Results folders
        'results': base_path / 'results' ,
        'results_task_performance': base_path / 'results' / 'task_performance',
        'results_simulation_agent': base_path / 'results' / 'simulation_agent',
        'results_compass_level_1': base_path / 'results' / 'compass_level_1',
        'results_compass_level_2': base_path / 'results' / 'compass_level_2',
        'results_ephys_compass': base_path / 'results' / 'ephys_compass'
    }
    
    # Create all directories
    print("Creating organized directory structure...")
    for dir_name, dir_path in DIRS.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"{dir_name}: {dir_path}")
    
    return DIRS

def save_figure(fig_name: str, subdir: str = 'results/task_performance', dpi: int = 300, ext: str = 'pdf'):
    """
    Save the current matplotlib figure to a standardized results folder.

    Parameters
    ----------
    fig_name : str
        Name of the figure file, e.g., 'Shannons_entropy' or 'Bout_Success'.
        Extension is automatically appended as defined by `ext`.
    subdir : str
        Subfolder path under BASE_PATH to save the figure.
    dpi : int
        Resolution of saved figure.
    ext : str
        File extension, e.g., 'pdf', 'png', etc.
    """
    os.makedirs(os.path.join(BASE_PATH, subdir), exist_ok=True)
    save_path = os.path.join(BASE_PATH, subdir, f"{fig_name}.{ext}")
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    print(f"Saved: {save_path}")

