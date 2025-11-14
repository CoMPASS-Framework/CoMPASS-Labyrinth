"""
Labyrinth_DLC_Utils.py

Utility functions for DeepLabCut preprocessing and analysis.
Contains functions for metadata handling, video processing, and analysis.

Author: Patrick Honma & Shreya Bangera
Lab: Palop Lab
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, time
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from shapely.geometry import Polygon, Point
import geopandas as gpd
from matplotlib.collections import LineCollection
from typing import Union, Optional

def import_cohort_metadata(
    metadata_path: Union[str, Path],
    trial_sheet_name: Optional[str]=None,
) -> pd.DataFrame:
    """
    Import and process trial metadata from Excel file.

    Parameters:
    -----------
    metadata_path : str or Path
        Path to the Excel file containing trial information
    trial_sheet_name : str or None
        Name of the sheet/tab containing the trial data, needed for multi-sheet files

    Returns:
    --------
    pd.DataFrame
        Cleaned metadata dataframe
    """
    try:
        # Load the Excel sheet
        metadata_path = Path(metadata_path)
        if metadata_path.suffix in [".xlsx", ".xls"]:
            mouseinfo = pd.read_excel(metadata_path, sheet_name=trial_sheet_name)
        elif metadata_path.suffix == ".csv":
            mouseinfo = pd.read_csv(metadata_path)
        print(f"Initial rows loaded: {len(mouseinfo)}")

        # Remove rows with missing Session numbers or where it's 0
        mouseinfo = mouseinfo[~mouseinfo["Session #"].isna() & (mouseinfo["Session #"] != 0)]

        # Special processing for Probe Trial data
        if trial_sheet_name == "Probe Trial":
            if "Cropping Bounds" in mouseinfo.columns:
                mouseinfo["Cropping Bounds"] = mouseinfo["Cropping Bounds"].apply(
                    lambda x: [int(num.strip()) for num in x.split(",")] if pd.notna(x) else None
                )
                print("Processed cropping bounds for Probe Trial")

        # Stringify timestamps
        mouseinfo = mouseinfo.applymap(
            lambda x: (
                x.isoformat()
                if isinstance(x, (pd.Timestamp, datetime, date))
                else x.strftime("%H:%M:%S") if isinstance(x, time) else x
            )
        )

        # Exclude trials marked for exclusion
        if "Exclude Trial" in mouseinfo.columns:
            excluded_trials = mouseinfo["Exclude Trial"] == "yes"
            excluded_count = excluded_trials.sum()
            mouseinfo = mouseinfo.loc[~excluded_trials].reset_index(drop=True)
            print(f"Excluded {excluded_count} trials marked for exclusion")

        print(f"Final dataset: {len(mouseinfo)} trials")
        return mouseinfo

    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    except ValueError:
        raise ValueError(
            f"Sheet '{trial_sheet_name}' not found in Excel file",
            f"Available sheets: {pd.ExcelFile(metadata_path).sheet_names}",
        )
    except Exception:
        raise Exception(f"Error loading metadata from {metadata_path}")


def validate_metadata(df: pd.DataFrame) -> bool:
    """
    Validate the loaded metadata for required columns and data quality.

    Parameters:
    -----------
    df : pd.DataFrame
        Metadata dataframe to validate

    Returns:
    --------
    bool
        True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: No metadata loaded")
        return False

    # Required columns for analysis
    required_columns = ["Session #"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False

    # Check for duplicate sessions
    duplicates = df.duplicated(subset=["Session #"], keep=False)
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate Session # entries")
        print("Duplicate sessions:")
        print(df[duplicates][["Session #"]])

    return True


def display_metadata_summary(df: pd.DataFrame) -> None:
    """Display summary information about the loaded metadata."""
    if df is None or df.empty:
        return

    print("\n" + "=" * 50)
    print("METADATA SUMMARY")
    print("=" * 50)
    print(f"Total trials: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Show session number range
    if "Session #" in df.columns:
        print(f"Session # range: {df['Session #'].min()} - {df['Session #'].max()}")

    # Show group distribution if available
    if "Group" in df.columns:
        group_counts = df["Group"].value_counts()
        print(f"Group distribution:")
        for group, count in group_counts.items():
            print(f"  {group}: {count} trials")

    # Show any missing data
    missing_data = df.isnull().sum()
    if missing_data.any():
        print(f"Missing data:")
        for col, count in missing_data[missing_data > 0].items():
            # Ignore it if it's 'Exclude Trial' column
            if col == "Exclude Trial":
                continue
            print(f"  {col}: {count} missing values")

    print("=" * 50)


def save_first_frame(
    video_path: Union[str, Path],
    frames_dir: Union[str, Path],
) -> None:
    """
    Saves the first frame of a video to the specified destination path.

    Parameters:
    -----------
    video_path : str or Path
        Path to the input video file.
    frames_dir : str or Path
        Directory where the first frame image will be saved.

    Returns:
    --------
    None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False

    # Read the first frame
    success, frame = cap.read()

    if success:
        # Save the frame as an image
        cv2.imwrite(frames_dir / f"{video_path.stem}.png", frame)
    else:
        print("Error: Could not read the first frame.")
        cap.release()
        return False

    # Release the video capture object
    cap.release()


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
        "videos": base_path / "videos",
        "videos_original": base_path / "videos" / "original_videos",
        "frames": base_path / "videos" / "frames",
        # Data folder - analysis inputs and outputs
        "data": base_path / "data",
        "dlc_results": base_path / "data" / "dlc_results",
        "dlc_cropping": base_path / "data" / "dlc_cropping_bounds",
        "grid_files": base_path / "data" / "grid_files",
        "grid_boundaries": base_path / "data" / "grid_boundaries",
        "metadata": base_path / "data" / "metadata",
        # Figures folder - all plots and visualizations
        "figures": base_path / "figures",
        # CSV's folder
        "csvs": base_path / "csvs",
        "csvs_individual": base_path / "csvs" / "individual",
        "csvs_combined": base_path / "csvs" / "combined",
        # Results folders
        "results": base_path / "results",
        "results_task_performance": base_path / "results" / "task_performance",
        "results_simulation_agent": base_path / "results" / "simulation_agent",
        "results_compass_level_1": base_path / "results" / "compass_level_1",
        "results_compass_level_2": base_path / "results" / "compass_level_2",
        "results_ephys_compass": base_path / "results" / "ephys_compass",
    }

    # Create all directories
    print("Creating organized directory structure...")
    for dir_name, dir_path in DIRS.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"{dir_name}: {dir_path}")

    return DIRS


def copy_and_rename_videos(mouseinfo_df, video_paths, destination_path):
    """
    Copy videos from source paths and rename them according to session information.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session and Noldus trial information
    video_paths : list
        List of source video paths (handles multiple computers)
    destination_path : Path
        Destination directory for renamed videos (videos/original_videos)

    Returns:
    --------
    dict
        Summary of copy operations
    """
    destination_path = Path(destination_path)

    # Create destination directory if it doesn't exist
    if not destination_path.exists():
        print("Destination path doesn't exist. Creating folder for original videos...")
        destination_path.mkdir(parents=True, exist_ok=True)

    # Filter out empty video paths
    valid_video_paths = [path for path in video_paths if path and Path(path).exists()]

    if not valid_video_paths:
        print("Error: No valid video source paths found!")
        return None

    print(f"Source video paths: {valid_video_paths}")
    print(f"Destination path: {destination_path}")

    # Track copy operations
    copy_summary = {
        "total_sessions": len(mouseinfo_df),
        "already_exists": 0,
        "successfully_copied": 0,
        "failed_copies": 0,
        "failed_files": [],
    }

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        print("------------------------")
        session_name = f"Session{int(row['Session #']):04d}"
        destination_file = destination_path / f"{session_name}.mp4"

        print(f"Processing {session_name}...")

        # Check if video already exists
        if destination_file.exists():
            print(f"{session_name}.mp4 already exists!")
            copy_summary["already_exists"] += 1
            continue

        # Get Noldus trial information
        noldus_trial = int(row["Noldus Trial"])

        # Handle Noldus filename formatting (different spacing for single vs double digits)
        if noldus_trial <= 9:
            noldus_filename = f"Trial     {noldus_trial}.mp4"  # More spaces for single digits
        else:
            noldus_filename = f"Trial    {noldus_trial}.mp4"  # Fewer spaces for double digits

        print(f"Looking for: {noldus_filename}")

        # Determine which computer/video path to use
        computer_number = int(row["Computer"]) if "Computer" in row and pd.notna(row["Computer"]) else 1

        # Select the appropriate video path based on computer number
        if computer_number == 1 and len(valid_video_paths) >= 1:
            selected_video_path = valid_video_paths[0]  # VIDEO_PATH_1
        elif computer_number == 2 and len(valid_video_paths) >= 2:
            selected_video_path = valid_video_paths[1]  # VIDEO_PATH_2
        elif len(valid_video_paths) == 1:
            # Fallback: if only one path available, use it regardless of computer number
            selected_video_path = valid_video_paths[0]
            print(f"Warning: Computer {computer_number} specified but only one video path available")
        else:
            print(f"Error: Computer {computer_number} specified but corresponding video path not available")
            copy_summary["failed_copies"] += 1
            copy_summary["failed_files"].append(
                {
                    "session": session_name,
                    "noldus_file": noldus_filename,
                    "computer": computer_number,
                    "error": f"Video path for computer {computer_number} not available",
                }
            )
            continue

        # Build source file path
        source_file = Path(selected_video_path) / noldus_filename
        print(f"Computer {computer_number} -> Using path: {selected_video_path}")
        print(f"Looking for: {source_file}")

        # Check if source file exists and copy it
        if source_file.exists():
            try:
                # Copy and rename the file
                shutil.copy2(source_file, destination_file)
                print(f"Successfully copied {session_name}.mp4 from Computer {computer_number}")
                copy_summary["successfully_copied"] += 1
            except Exception as e:
                print(f"Error copying {noldus_filename}: {e}")
                copy_summary["failed_copies"] += 1
                copy_summary["failed_files"].append(
                    {
                        "session": session_name,
                        "noldus_file": noldus_filename,
                        "computer": computer_number,
                        "error": str(e),
                    }
                )
        else:
            print(f"Warning: {noldus_filename} not found at {source_file}")
            copy_summary["failed_copies"] += 1
            copy_summary["failed_files"].append(
                {
                    "session": session_name,
                    "noldus_file": noldus_filename,
                    "computer": computer_number,
                    "error": "File not found at specified path",
                }
            )

    return copy_summary


def batch_save_first_frames(mouseinfo_df, video_directory, frames_directory):
    """
    Save the first frame of all videos as JPEG images.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    video_directory : str or Path
        Directory containing the videos (videos/original_videos)
    frames_directory : str or Path
        Directory to save frames (videos/frames)

    Returns:
    --------
    dict
        Summary of frame saving operations
    """
    video_directory = Path(video_directory)
    frames_directory = Path(frames_directory)

    # Ensure frames directory exists
    frames_directory.mkdir(parents=True, exist_ok=True)

    # Track operations
    frame_summary = {
        "total_sessions": len(mouseinfo_df),
        "frames_saved": 0,
        "already_exists": 0,
        "failed_saves": 0,
        "failed_sessions": [],
        "saved_sessions": [],
    }

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{int(row['Session #']):04d}"

        # Check if video exists
        video_path = video_directory / f"{session_name}.mp4"
        if not video_path.exists():
            print(f"  Video not found: {video_path}")
            frame_summary["failed_saves"] += 1
            frame_summary["failed_sessions"].append(session_name)
            continue

        # Check if frame already exists
        frame_image_path = frames_directory / f"{session_name}Frame1.jpg"
        if frame_image_path.exists():
            frame_summary["already_exists"] += 1
            continue

        # Capture video and save first frame
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()

        if ret:
            # Save the frame as JPEG
            cv2.imwrite(str(frame_image_path), frame)

            frame_summary["frames_saved"] += 1
            frame_summary["saved_sessions"].append(session_name)
        else:
            print(f"  Failed to read video: {session_name}")
            frame_summary["failed_saves"] += 1
            frame_summary["failed_sessions"].append(session_name)

        # Release video capture
        cap.release()

    return frame_summary


def get_labyrinth_boundary_and_cropping(
    frames_directory, cropping_directory, boundaries_directory, session, chamber_info=None
):
    """
    Get labyrinth boundary coordinates (4 corners) and automatically derive DLC cropping bounds.
    Click 4 corners in order: top-left, bottom-left, bottom-right, top-right.

    Parameters:
    -----------
    frames_directory : str or Path
        Directory containing frame images (videos/frames)
    cropping_directory : str or Path
        Directory to save cropping coordinates (data/dlc_cropping_bounds)
    boundaries_directory : str or Path
        Directory to save boundary points (data/grid_boundaries)
    session : str
        Session name (e.g., 'Session-1')
    chamber_info : str, optional
        Chamber information to display

    Returns:
    --------
    tuple
        (boundary_points, cropping_coords) where:
        - boundary_points: np.array of 4 corner coordinates
        - cropping_coords: (X1, X2, Y1, Y2) tuple
    """

    frames_path = Path(frames_directory)
    cropping_path = Path(cropping_directory)
    boundaries_path = Path(boundaries_directory)

    # Ensure directories exist
    cropping_path.mkdir(parents=True, exist_ok=True)
    boundaries_path.mkdir(parents=True, exist_ok=True)

    posList = []
    corner_names = ["Top-Left", "Bottom-Left", "Bottom-Right", "Top-Right"]

    def click_event(event, x, y, flags, params):
        # Left mouse click to select corners
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(posList) < 4:  # Only accept 4 points
                posList.append((x, y))
                corner_index = len(posList) - 1
                corner_name = corner_names[corner_index]

                print(f"{corner_name} corner: ({x}, {y})")

                # Draw point on image with different colors for each corner
                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # Green, Red, Blue, Yellow
                color = colors[corner_index]

                cv2.circle(img_display, (x, y), 8, color, -1)
                cv2.putText(
                    img_display,
                    f"{corner_index + 1}: {corner_name}",
                    (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Draw lines connecting points
                if len(posList) > 1:
                    cv2.line(img_display, posList[-2], posList[-1], color, 2)

                # Close the polygon when we have 4 points
                if len(posList) == 4:
                    cv2.line(img_display, posList[-1], posList[0], colors[3], 2)

                    # Calculate and display cropping bounds
                    x_coords = [p[0] for p in posList]
                    y_coords = [p[1] for p in posList]

                    X1, X2 = min(x_coords), max(x_coords)
                    Y1, Y2 = min(y_coords), max(y_coords)

                    # Draw cropping rectangle
                    cv2.rectangle(img_display, (X1, Y1), (X2, Y2), (255, 255, 255), 2)
                    cv2.putText(
                        img_display,
                        f"Crop: {X2-X1}x{Y2-Y1}",
                        (X1, Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                    print(f"4 corners selected. Cropping bounds: X1={X1}, X2={X2}, Y1={Y1}, Y2={Y2}")
                    print("Press 'q' to confirm, 'r' to reset")
                else:
                    print(f"Click {corner_names[len(posList)]} corner next...")

                cv2.imshow("Labyrinth Boundary Selection", img_display)

        # Right mouse click to show pixel values
        elif event == cv2.EVENT_RBUTTONDOWN:
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv2.putText(img_display, f"RGB: {r},{g},{b}", (x + 10, y + 10), font, 0.4, (255, 255, 0), 1)
            cv2.imshow("Labyrinth Boundary Selection", img_display)
            print(f"Pixel at ({x}, {y}): RGB({r}, {g}, {b})")

    # Load the saved frame
    frame_path = frames_path / f"{session}Frame1.jpg"
    if not frame_path.exists():
        print(f"Error: Frame not found at {frame_path}")
        print("Run batch_save_first_frames() first to create the frame.")
        return None, None

    img = cv2.imread(str(frame_path), 1)
    img_display = img.copy()

    print(f"\nLabyrinth Boundary Selection for {session}")
    if chamber_info:
        print(f"Chamber: {chamber_info}")
    print(f"Image size: {img.shape[1]} x {img.shape[0]} (W x H)")

    print("\nInstructions:")
    print("1. Click on the 4 corners in this order:")
    print("   - Top-Left corner")
    print("   - Bottom-Left corner")
    print("   - Bottom-Right corner")
    print("   - Top-Right corner")
    print("2. Right-click to see pixel RGB values (optional)")
    print("3. Press 'q' to confirm selection after clicking 4 corners")
    print("4. Press 'r' to reset and select again")
    print("5. Press 'c' to cancel")

    cv2.startWindowThread()
    cv2.namedWindow("Labyrinth Boundary Selection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labyrinth Boundary Selection", min(1200, img.shape[1]), min(800, img.shape[0]))
    cv2.imshow("Labyrinth Boundary Selection", img_display)
    cv2.setMouseCallback("Labyrinth Boundary Selection", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") and len(posList) >= 4:
            # Confirm selection
            break
        elif key == ord("r"):
            # Reset selection
            posList.clear()
            img_display = img.copy()
            cv2.imshow("Labyrinth Boundary Selection", img_display)
            print("Selection reset. Click the 4 corners again in order.")
        elif key == ord("c"):
            # Cancel
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None, None
        elif key == 27:  # ESC key
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None, None

        # Check if window is closed
        if cv2.getWindowProperty("Labyrinth Boundary Selection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    if len(posList) >= 4:
        # Convert to numpy array
        boundary_points = np.array(posList[:4])

        # Calculate cropping coordinates from the 4 corners
        x_coords = [p[0] for p in boundary_points]
        y_coords = [p[1] for p in boundary_points]

        X1, X2 = min(x_coords), max(x_coords)
        Y1, Y2 = min(y_coords), max(y_coords)

        cropping_coords = (X1, X2, Y1, Y2)

        # Save boundary points
        boundary_file = boundaries_path / f"{session}_Boundary_Points.npy"
        np.save(str(boundary_file), boundary_points)

        # Save cropping coordinates
        coord_data = {
            "session": session,
            "X1": X1,
            "X2": X2,
            "Y1": Y1,
            "Y2": Y2,
            "width": X2 - X1,
            "height": Y2 - Y1,
            "boundary_points": boundary_points.tolist(),
            "derived_from_boundary": True,
        }

        coord_file = cropping_path / f"{session}_DLC_Cropping_Bounds.npy"
        np.save(str(coord_file), coord_data)

        print(f"Derived cropping bounds: X1={X1}, X2={X2}, Y1={Y1}, Y2={Y2}")
        print(f"Cropping size: {X2-X1} x {Y2-Y1} pixels")
        print(f"Boundary points saved to: {boundary_file}")
        print(f"Cropping coordinates saved to: {coord_file}")

        return boundary_points, cropping_coords
    else:
        print("Insufficient points selected.")
        return None, None


def batch_get_boundary_and_cropping(mouseinfo_df, frames_directory, cropping_directory, boundaries_directory, reprocess_existing=False):
    """
    Get boundary points and cropping coordinates for multiple sessions.
    Automatically skips sessions that already have both files unless reprocess_existing=True.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    frames_directory : str or Path
        Directory containing frame images
    cropping_directory : str or Path
        Directory to save cropping coordinates
    boundaries_directory : str or Path
        Directory to save boundary points
    reprocess_existing : bool, optional
        If True, reprocess sessions even if they already have boundary/cropping files.
        If False (default), skip sessions that already have both files.

    Returns:
    --------
    dict
        Dictionary with results for each session
    """
    from pathlib import Path
    
    cropping_path = Path(cropping_directory)
    boundaries_path = Path(boundaries_directory)
    
    results_dict = {
        "boundary_points": {}, 
        "cropping_coords": {}, 
        "successful_sessions": [], 
        "failed_sessions": [],
        "skipped_sessions": []  # Sessions that already had both files
    }

    print(f"Getting boundary points and cropping coordinates for {len(mouseinfo_df)} sessions...")
    print("Press 'c' to skip a session, or ESC to stop completely.")
    
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"

        # Check if files already exist
        boundary_file = boundaries_path / f"{session_name}_Boundary_Points.npy"
        cropping_file = cropping_path / f"{session_name}_DLC_Cropping_Bounds.npy"
        
        boundary_exists = boundary_file.exists()
        cropping_exists = cropping_file.exists()

        # Skip if both exist and not reprocessing
        if boundary_exists and cropping_exists and not reprocess_existing:
            print(f"✓ {session_name} already has boundary and cropping data - skipping")
            results_dict["skipped_sessions"].append(session_name)
            continue

        print(f"\n{'='*60}")
        print(f"Processing {session_name} ({index+1}/{len(mouseinfo_df)})")

        # Display chamber information if available
        chamber_info = None
        if "Noldus Chamber" in row and pd.notna(row["Noldus Chamber"]):
            chamber_info = row["Noldus Chamber"]
            print(f"Chamber: {chamber_info}")

        # Show what's missing or if reprocessing
        if reprocess_existing and boundary_exists and cropping_exists:
            print("Status: Reprocessing existing data")
        elif not boundary_exists and not cropping_exists:
            print("Status: Missing both boundary and cropping")
        elif not boundary_exists:
            print("Status: Missing boundary points")
        elif not cropping_exists:
            print("Status: Missing cropping coordinates")

        print(f"{'='*60}")

        # Get boundary points and cropping coordinates
        boundary_points, cropping_coords = get_labyrinth_boundary_and_cropping(
            frames_directory=frames_directory,
            cropping_directory=cropping_directory,
            boundaries_directory=boundaries_directory,
            session=session_name,
            chamber_info=chamber_info,
        )

        if boundary_points is not None and cropping_coords is not None:
            results_dict["boundary_points"][session_name] = boundary_points
            results_dict["cropping_coords"][session_name] = cropping_coords
            results_dict["successful_sessions"].append(session_name)
            print(f"✓ Boundary and cropping data saved for {session_name}")
        else:
            results_dict["failed_sessions"].append(session_name)
            print(f"✗ Skipped {session_name}")

            # Ask if user wants to continue
            continue_choice = input("Continue with next session? (y/n): ").strip().lower()
            if continue_choice == "n":
                break

    # Print summary
    print(f"\n{'='*60}")
    print("BOUNDARY AND CROPPING SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sessions: {len(mouseinfo_df)}")
    print(f"Already complete (skipped): {len(results_dict['skipped_sessions'])}")
    print(f"Newly processed: {len(results_dict['successful_sessions'])}")
    print(f"Failed/skipped: {len(results_dict['failed_sessions'])}")

    if results_dict["skipped_sessions"]:
        print(f"\nAlready had data: {results_dict['skipped_sessions']}")

    if results_dict["successful_sessions"]:
        print(f"\nNewly processed: {results_dict['successful_sessions']}")

    if results_dict["failed_sessions"]:
        print(f"\nFailed sessions: {results_dict['failed_sessions']}")

    return results_dict

def check_preprocessing_status(source_data_path, video_type=".mp4"):
    """
    Check preprocessing status and provide guidance on next steps.
    
    This function identifies which preprocessing steps are complete and guides users
    through the appropriate workflow based on available files.
    
    Use Cases:
    ----------
    1. Videos + Grids → Ready for init_project() (DLC data already in grids)
    2. Videos + DLC + Grids → Ready for init_project()
    3. Grids only (withGrids.csv files) → Ready for init_project()
    4. Videos only → Switch to DLC env, run 00_dlc_grid_processing.ipynb (full pipeline)
    5. Videos + DLC (no grids) → Run run_grid_preprocessing() in 01_create_project.ipynb
    6. DLC only (no videos/grids) → Get videos, then run run_grid_preprocessing() in 01_create_project.ipynb
    
    Parameters:
    -----------
    source_data_path : str or Path
        Directory containing preprocessing outputs and/or videos
    video_type : str
        Video file extension (default: ".mp4")
        
    Returns:
    --------
    dict
        Status dictionary with keys: 'ready', 'next_step', 'details'
        
    Raises:
    -------
    FileNotFoundError
        If the workflow cannot proceed
    """
    from pathlib import Path
    
    source_data_path = Path(source_data_path)
    
    print("\n" + "="*70)
    print("CHECKING PREPROCESSING STATUS")
    print("="*70)
    
    # Check for videos
    video_files = list(source_data_path.glob(f"*{video_type}"))
    has_videos = len(video_files) > 0
    
    # Check for DLC output files
    dlc_h5_files = list(source_data_path.glob("*DLC*.h5"))
    dlc_csv_files = list(source_data_path.glob("*DLC*.csv"))
    has_dlc = len(dlc_h5_files) > 0 or len(dlc_csv_files) > 0
    
    # Check for grid files - support multiple naming conventions
    grid_files = list(source_data_path.glob("*_withGrids.csv"))
    grid_files_alt = list(source_data_path.glob("*withGrids.csv"))
    # Combine and deduplicate
    all_grid_files = list(set(grid_files + grid_files_alt))
    has_grids = len(all_grid_files) > 0
    
    # Print what was found
    print(f"\nFound in {source_data_path}:")
    print(f"  Videos ({video_type}): {len(video_files)} files")
    print(f"  DLC outputs (.h5/.csv): {len(dlc_h5_files) + len(dlc_csv_files)} files")
    print(f"  Grid files (*withGrids.csv): {len(all_grid_files)} files")
    print()
    
    # Determine which case we're in and provide guidance
    
    # CASE 1, 2, 3: Ready for init_project if we have grid files
    if has_grids:
        # Note: We don't validate other files if only grid files exist
        # because the grid files contain all necessary pose data
        if has_videos or has_dlc:
            # Validate that all required files exist for each session
            _validate_complete_preprocessing(source_data_path, all_grid_files, require_dlc=False)
        
        print("="*70)
        print("✓ STATUS: READY FOR INIT_PROJECT")
        print("="*70)
        
        # Provide appropriate status message
        if has_videos and has_dlc:
            print("Found: Videos + DLC outputs + Grid preprocessing")
        elif has_videos:
            print("Found: Videos + Grid preprocessing")
            print("Note: DLC .h5/.csv files not found, but pose data is in grid files")
        elif has_dlc:
            print("Found: DLC outputs + Grid preprocessing (videos not required)")
        else:
            print("Found: Grid preprocessing files (*withGrids.csv)")
            print("Note: Videos and DLC files not found, but pose data is in grid files")
        
        print("\nNEXT STEP: Initialize your project")
        print("\n  from compass_labyrinth import init_project")
        print(f"\n  init_project(")
        print(f"      source_data_path=r'{source_data_path}',")
        print(f"      user_metadata_file_path='path/to/metadata.xlsx',")
        print(f"      trial_type='Labyrinth_DSI'")
        print(f"  )")
        print("="*70)
        
        return {
            'ready': True,
            'next_step': 'init_project',
            'details': {
                'has_videos': has_videos,
                'has_dlc': has_dlc,
                'has_grids': has_grids,
                'num_sessions': len(all_grid_files)
            }
        }
    
    # CASE 4: Videos only - need to run full DLC pipeline
    elif has_videos and not has_dlc and not has_grids:
        print("="*70)
        print("STATUS: VIDEOS FOUND - DLC PROCESSING NEEDED")
        print("="*70)
        print(f"Found {len(video_files)} video files")
        print("\nNEXT STEPS:")
        print("\n  1. Switch to DLC environment:")
        print("       conda activate DEEPLABCUT")
        print("\n  2. Open and run: 00_dlc_grid_processing.ipynb")
        print("\n     This will:")
        print("       - Run DeepLabCut pose estimation on your videos")
        print("       - Create first frame images")
        print("       - Interactively define maze boundaries")
        print("       - Generate cropping coordinates")
        print("       - Create grid files with spatial information")
        print("\n  3. Switch back to compass environment:")
        print("       conda activate compass_labyrinth")
        print("\n  4. Return to this notebook (01_create_project.ipynb)")
        print("\n  5. Run init_project()")
        print("="*70)
        
        return {
            'ready': False,
            'next_step': 'run_dlc_and_grids',
            'details': {
                'has_videos': has_videos,
                'has_dlc': has_dlc,
                'has_grids': has_grids,
                'num_videos': len(video_files)
            }
        }
    
    # CASE 5: Videos + DLC but no grids
    elif has_videos and has_dlc and not has_grids:
        print("="*70)
        print("STATUS: DLC COMPLETE - GRID PREPROCESSING NEEDED")
        print("="*70)
        print(f"Found {len(video_files)} videos and DLC outputs")
        print("\nNEXT STEP: Run grid preprocessing in this notebook (01_create_project.ipynb)")
        print("\n  from compass_labyrinth.behavior.pose_estimation.dlc_utils import run_grid_preprocessing")
        print(f"\n  run_grid_preprocessing(")
        print(f"      source_data_path=source_data_path,")
        print(f"      user_metadata_file_path=user_metadata_file_path,")
        print(f"      trial_type=trial_type,")
        print(f"      video_type='{video_type}',")
        print(f"      reprocess_existing=False")
        print(f"  )")
        print("\n  This will:")
        print("    - Extract first frames from videos")
        print("    - Interactively define maze boundaries")
        print("    - Generate cropping coordinates")
        print("    - Create grid files with spatial information")
        print("\n  Then run init_project()")
        print("="*70)
        
        return {
            'ready': False,
            'next_step': 'run_grids_only',
            'details': {
                'has_videos': has_videos,
                'has_dlc': has_dlc,
                'has_grids': has_grids,
                'num_videos': len(video_files),
                'num_dlc_files': len(dlc_h5_files) + len(dlc_csv_files)
            }
        }
    
    # CASE 6: DLC only (no videos, no grids)
    elif has_dlc and not has_videos and not has_grids:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: VIDEOS REQUIRED FOR GRID PREPROCESSING\n"
            f"{'='*70}\n"
            f"Found DLC outputs but no videos in:\n"
            f"  {source_data_path}\n\n"
            f"Grid preprocessing requires videos to:\n"
            f"  - Extract first frame for boundary definition\n"
            f"  - Define cropping coordinates\n"
            f"  - Create spatial grid mappings\n\n"
            f"SOLUTION:\n\n"
            f"  1. Locate your original video files\n\n"
            f"     DLC files found (find matching videos):\n"
        )
        
        # List the DLC files to help identify videos
        for dlc_file in (dlc_h5_files + dlc_csv_files)[:5]:  # Show first 5
            error_msg += f"       - {dlc_file.name}\n"
        if len(dlc_h5_files) + len(dlc_csv_files) > 5:
            error_msg += f"       ... and {len(dlc_h5_files) + len(dlc_csv_files) - 5} more\n"
        
        error_msg += (
            f"\n  2. Copy videos to: {source_data_path}\n"
            f"\n  3. Run grid preprocessing in this notebook (01_create_project.ipynb):\n"
            f"\n       from compass_labyrinth.behavior.pose_estimation.dlc_utils import run_grid_preprocessing"
            f"\n"
            f"\n       run_grid_preprocessing("
            f"\n           source_data_path=r'{source_data_path}',"
            f"\n           user_metadata_file_path='path/to/metadata.xlsx',"
            f"\n           trial_type='Labyrinth_DSI',"
            f"\n           video_type='{video_type}',"
            f"\n           reprocess_existing=False"
            f"\n       )"
            f"\n\n  4. Run init_project()"
            f"\n{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)
    
    # Edge case: Nothing found
    else:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: NO PREPROCESSING FILES FOUND\n"
            f"{'='*70}\n"
            f"No videos, DLC outputs, or grid files found in:\n"
            f"  {source_data_path}\n\n"
            f"SOLUTION:\n\n"
            f"  1. Place your video files in: {source_data_path}\n"
            f"\n  2. Switch to DLC environment: conda activate DEEPLABCUT\n"
            f"\n  3. Run: 00_dlc_grid_processing.ipynb\n"
            f"\n  4. Switch back: conda activate compass_labyrinth\n"
            f"\n  5. Return to this notebook (01_create_project.ipynb)\n"
            f"\n  6. Run init_project()\n"
            f"{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)


def _validate_complete_preprocessing(source_data_path, grid_files, require_dlc=True):
    """
    Helper function to validate that all required preprocessing files exist.
    
    Parameters:
    -----------
    source_data_path : Path
        Directory containing preprocessing outputs
    grid_files : list
        List of Path objects for *withGrids.csv files
    require_dlc : bool
        Whether to require DLC .h5/.csv files (default: True)
        Set to False if pose data is already in grid files
        
    Raises:
    -------
    FileNotFoundError
        If any required files are missing
    """
    from pathlib import Path
    
    source_data_path = Path(source_data_path)
    
    # Extract session names from grid files
    session_names = []
    for grid_file in grid_files:
        # Handle both "SessionXXXX_withGrids.csv" and "Session-XXXXwithGrids.csv" formats
        session_name = grid_file.stem.replace("_withGrids", "").replace("withGrids", "")
        session_names.append(session_name)
    
    # Track missing files per session
    missing_sessions = {
        "boundary": [],
        "cropping": [],
        "dlc": []
    }
    
    # Check each session for required files
    for session_name in session_names:
        # Try different naming conventions for boundary and cropping files
        boundary_file_1 = source_data_path / f"{session_name}_Boundary_Points.npy"
        boundary_file_2 = source_data_path / f"{session_name} Boundary_Points.npy"
        
        cropping_file_1 = source_data_path / f"{session_name}_DLC_Cropping_Bounds.npy"
        cropping_file_2 = source_data_path / f"{session_name} DLC_Cropping_Bounds.npy"
        
        has_boundary = boundary_file_1.exists() or boundary_file_2.exists()
        has_cropping = cropping_file_1.exists() or cropping_file_2.exists()
        
        if not has_boundary:
            missing_sessions["boundary"].append(session_name)
        if not has_cropping:
            missing_sessions["cropping"].append(session_name)
        
        # Only check for DLC files if required
        if require_dlc:
            # Check for DLC file (either .h5 or .csv with DLC in name)
            dlc_h5 = list(source_data_path.glob(f"{session_name}*DLC*.h5"))
            dlc_csv = list(source_data_path.glob(f"{session_name}*DLC*.csv"))
            has_dlc_file = len(dlc_h5) > 0 or len(dlc_csv) > 0
            
            if not has_dlc_file:
                missing_sessions["dlc"].append(session_name)
    
    # Build error message if any files are missing
    missing_outputs = []
    if require_dlc and missing_sessions["dlc"]:
        missing_outputs.append(f"  - DLC outputs missing for: {', '.join(missing_sessions['dlc'])}")
    if missing_sessions["boundary"]:
        missing_outputs.append(f"  - Boundary files missing for: {', '.join(missing_sessions['boundary'])}")
    if missing_sessions["cropping"]:
        missing_outputs.append(f"  - Cropping files missing for: {', '.join(missing_sessions['cropping'])}")
    
    if missing_outputs:
        all_missing = set(
            (missing_sessions['dlc'] if require_dlc else []) + 
            missing_sessions['boundary'] + 
            missing_sessions['cropping']
        )
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: INCOMPLETE PREPROCESSING\n"
            f"{'='*70}\n"
            f"Missing required files for {len(all_missing)} session(s):\n\n"
            + "\n".join(missing_outputs) + "\n\n"
            f"SOLUTION: Run grid preprocessing in this notebook (01_create_project.ipynb):\n"
            f"\n  from compass_labyrinth.behavior.pose_estimation.dlc_utils import run_grid_preprocessing"
            f"\n"
            f"\n  run_grid_preprocessing("
            f"\n      source_data_path=r'{source_data_path}',"
            f"\n      user_metadata_file_path='path/to/metadata.xlsx',"
            f"\n      trial_type='Labyrinth_DSI',"
            f"\n      reprocess_existing=False"
            f"\n  )"
            f"\n{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)
    
    print(f"✓ Validated all required files for {len(session_names)} sessions")
    
def prepare_dlc_analysis(mouseinfo_df, video_directory, cropping_directory, results_directory):
    """
    Prepare videos for DeepLabCut analysis by checking files and loading cropping bounds.
    
    Returns:
    --------
    list of dict
        List of sessions ready for analysis with all necessary paths and parameters
    """
    from pathlib import Path
    import numpy as np
    
    # Ensure directories exist
    results_path = Path(results_directory)
    results_path.mkdir(parents=True, exist_ok=True)
    
    sessions_to_analyze = []
    summary = {
        "total_sessions": len(mouseinfo_df),
        "ready_for_analysis": 0,
        "skipped_existing": 0,
        "missing_video": 0,
        "missing_coordinates": 0,
        "failed_sessions": []
    }
    
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"
        video_name = f"{session_name}.mp4"
        video_path = Path(video_directory) / video_name
        
        # Check if video exists
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            summary["missing_video"] += 1
            summary["failed_sessions"].append(session_name)
            continue
            
        # Check if analysis already exists
        existing_csv = list(results_path.glob(f"{session_name}DLC_*.csv"))
        if existing_csv:
            print(f"Analysis already exists for {session_name}, skipping...")
            summary["skipped_existing"] += 1
            continue
            
        # Load cropping bounds
        coord_file = Path(cropping_directory) / f"{session_name}_DLC_Cropping_Bounds.npy"
        if not coord_file.exists():
            print(f"Error: No saved cropping coordinates for {session_name}")
            summary["missing_coordinates"] += 1
            summary["failed_sessions"].append(session_name)
            continue
            
        try:
            coord_data = np.load(coord_file, allow_pickle=True).item()
            cropping_coords = (coord_data["X1"], coord_data["X2"], 
                             coord_data["Y1"], coord_data["Y2"])
            
            sessions_to_analyze.append({
                "session_name": session_name,
                "video_path": str(video_path),
                "cropping_coords": cropping_coords,
                "results_path": str(results_path)
            })
            summary["ready_for_analysis"] += 1
            print(f"{session_name}: Ready for analysis with bounds {cropping_coords}")
            
        except Exception as e:
            print(f"Error loading coordinates for {session_name}: {e}")
            summary["failed_sessions"].append(session_name)
            
    return sessions_to_analyze, summary

def get_grid_coordinates(posList, num_squares, grid_files_directory, session, cropping_coords=None):
    """
    Create a grid from boundary coordinates and save as shapefile.
    Adjusts coordinates to cropped frame if cropping_coords provided.

    Parameters:
    -----------
    posList : np.array
        Array of 4 boundary coordinates (in original frame coordinates)
    num_squares : int
        Number of squares per side (e.g., 12 for 12x12 grid)
    grid_files_directory : str or Path
        Directory to save grid files (data/grid_files)
    session : str
        Session name
    cropping_coords : tuple, optional
        (X1, X2, Y1, Y2) cropping coordinates to adjust grid to cropped frame

    Returns:
    --------
    gpd.GeoDataFrame
        Grid as geopandas dataframe
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    import numpy as np
    from pathlib import Path

    grid_files_path = Path(grid_files_directory)
    grid_files_path.mkdir(parents=True, exist_ok=True)

    # Get the coordinates of the 4 boundary points
    border = np.array(posList[:4])

    # Adjust boundary points to cropped coordinate system if cropping coords provided
    if cropping_coords is not None:
        X1, X2, Y1, Y2 = cropping_coords

        # Subtract the crop offset from boundary points
        adjusted_border = border.copy()
        adjusted_border[:, 0] = border[:, 0] - X1  # Adjust X coordinates
        adjusted_border[:, 1] = border[:, 1] - Y1  # Adjust Y coordinates
        border = adjusted_border

    # Create a polygon using these 4 coordinates
    grid_polygon = Polygon(border)

    # Define grid boundaries
    xmin, ymin, xmax, ymax = grid_polygon.bounds

    # Determine the size of each square
    width = (xmax - xmin) / num_squares
    height = (ymax - ymin) / num_squares

    rows = int(np.ceil((ymax - ymin) / height))
    cols = int(np.ceil((xmax - xmin) / width))

    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymin
    YbottomOrigin = ymin + height

    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom = YbottomOrigin
        for j in range(rows):
            polygons.append(
                Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])
            )
            Ytop = Ytop + height
            Ybottom = Ybottom + height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width

    grid = gpd.GeoDataFrame({"geometry": polygons})

    # Save the square grid
    grid_shp_path = grid_files_path / f"{session}_grid.shp"

    grid.to_file(str(grid_shp_path))

    return grid


def batch_create_grids(mouseinfo_df, boundaries_directory, grid_files_directory, cropping_directory, num_squares=12):
    """
    Create grids for multiple sessions using saved boundary coordinates.
    Adjusts grid coordinates to match cropped frame coordinate system.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    boundaries_directory : str or Path
        Directory containing boundary point files (data/grid_boundaries)
    grid_files_directory : str or Path
        Directory to save grid files (data/grid_files)
    cropping_directory : str or Path
        Directory containing cropping coordinate files (data/dlc_cropping_bounds)
    num_squares : int, optional
        Number of squares per side (default: 12)

    Returns:
    --------
    dict
        Summary of grid creation operations
    """
    import numpy as np
    from pathlib import Path

    start_time = datetime.now()
    print(f"Batch grid creation started: {start_time}")
    print(f"Grid size: {num_squares} x {num_squares}")

    boundaries_path = Path(boundaries_directory)
    grid_files_path = Path(grid_files_directory)
    cropping_path = Path(cropping_directory)

    # Ensure directories exist
    grid_files_path.mkdir(parents=True, exist_ok=True)

    grid_summary = {
        "total_sessions": len(mouseinfo_df),
        "grids_created": 0,
        "already_exists": 0,
        "failed_creation": 0,
        "no_boundaries": 0,
        "no_cropping": 0,
        "created_sessions": [],
        "failed_sessions": [],
    }

    print(f"Creating grids for {len(mouseinfo_df)} sessions...")
    print(f"Boundary points directory: {boundaries_path}")
    print(f"Cropping coordinates directory: {cropping_path}")
    print(f"Grid files directory: {grid_files_path}")
    print("Grid coordinates will be adjusted to cropped frame coordinate system")

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"

        # Get chamber info if available
        if "Noldus Chamber" in row and pd.notna(row["Noldus Chamber"]):
            chamber_info = row["Noldus Chamber"]

        # Check if grid already exists
        grid_file = grid_files_path / f"{session_name}_grid.shp"
        if grid_file.exists():
            grid_summary["already_exists"] += 1
            continue

        # Load saved boundary points
        boundary_file = boundaries_path / f"{session_name}_Boundary_Points.npy"
        if not boundary_file.exists():
            print(f"Error: No boundary points found for {session_name}")
            print(f"  - Missing file: {boundary_file}")
            print(f"  - Run get_labyrinth_boundary_and_cropping() first")
            grid_summary["no_boundaries"] += 1
            grid_summary["failed_sessions"].append(session_name)
            continue

        # Load saved cropping coordinates
        cropping_file = cropping_path / f"{session_name}_DLC_Cropping_Bounds.npy"
        if not cropping_file.exists():
            print(f"Error: No cropping coordinates found for {session_name}")
            print(f"  - Missing file: {cropping_file}")
            print(f"  - Run get_labyrinth_boundary_and_cropping() first")
            grid_summary["no_cropping"] += 1
            grid_summary["failed_sessions"].append(session_name)
            continue

        try:
            # Load boundary points
            boundary_points = np.load(str(boundary_file))

            # Load cropping coordinates
            cropping_data = np.load(str(cropping_file), allow_pickle=True).item()
            cropping_coords = (cropping_data["X1"], cropping_data["X2"], cropping_data["Y1"], cropping_data["Y2"])

            # Create grid with coordinate adjustment
            grid = get_grid_coordinates(
                posList=boundary_points,
                num_squares=num_squares,
                grid_files_directory=grid_files_directory,
                session=session_name,
                cropping_coords=cropping_coords,
            )

            grid_summary["grids_created"] += 1
            grid_summary["created_sessions"].append(session_name)
            print(f"✓ Grid created for {session_name}")

        except Exception as e:
            print(f"Error creating grid for {session_name}: {e}")
            grid_summary["failed_creation"] += 1
            grid_summary["failed_sessions"].append(session_name)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    grid_summary["duration"] = duration

    print_grid_summary(grid_summary)

    return grid_summary


def print_grid_summary(summary):
    """Print a summary of the grid creation operations."""
    print("\n" + "=" * 60)
    print("GRID CREATION SUMMARY")
    print("=" * 60)
    print(f'Total sessions processed: {summary["total_sessions"]}')
    print(f'Grids created: {summary["grids_created"]}')
    print(f'Already existed: {summary["already_exists"]}')
    print(f'Failed creation: {summary["failed_creation"]}')
    print(f'No boundary points: {summary["no_boundaries"]}')
    print(f'No cropping coordinates: {summary.get("no_cropping", 0)}')
    print(f'Duration: {summary.get("duration", "Unknown")}')

    if summary["created_sessions"]:
        print(f"\nSuccessfully created grids:")
        for session in summary["created_sessions"]:
            print(f"  - {session}")

    if summary["failed_sessions"]:
        print(f"\nFailed sessions:")
        for session in summary["failed_sessions"]:
            print(f"  - {session}")

    print("=" * 60)

    missing_files = summary["no_boundaries"] + summary.get("no_cropping", 0)
    if missing_files > 0:
        print(f"\nNote: {missing_files} sessions need boundary points and/or cropping coordinates.")
        print("Run get_labyrinth_boundary_and_cropping() for these sessions first.")


def create_grid_scatter_plot(
    session,
    dlc_results_directory,
    grid_files_directory,
    dlc_scorer,
    bodypart="sternum",
    likelihood_threshold=0.6,
    figure_size=(3, 3),
    save_plot=True,
    figures_directory=None,
):
    """
    Create a scatter plot of DLC tracking points overlaid on the grid for a single session.

    Parameters:
    -----------
    session : str
        Session name (e.g., 'Session-1')
    dlc_results_directory : str or Path
        Directory containing DLC results (data/dlc_results)
    grid_files_directory : str or Path
        Directory containing grid files (data/grid_files)
    dlc_scorer : str
        DLC scorer name
    bodypart : str, optional
        Bodypart to plot (default: 'sternum')
    likelihood_threshold : float, optional
        Minimum likelihood threshold for points (default: 0.6)
    figure_size : tuple, optional
        Figure size (default: (3, 3))
    save_plot : bool, optional
        Whether to save the plot (default: True)
    figures_directory : str or Path, optional
        Directory to save figures (required if save_plot=True)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """

    dlc_results_path = Path(dlc_results_directory)
    grid_files_path = Path(grid_files_directory)


    # Read the Grid File
    grid_file = grid_files_path / f"{session}_grid.shp"
    if not grid_file.exists():
        print(f"Error: Grid file not found: {grid_file}")
        return None

    try:
        grid = gpd.read_file(str(grid_file))
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return None

    # Read the DLC Results
    dlc_file = dlc_results_path / f"{session}{dlc_scorer}.h5"
    if not dlc_file.exists():
        print(f"Error: DLC results not found: {dlc_file}")
        return None

    try:
        df = pd.read_hdf(str(dlc_file))
        print(f"  Loaded DLC data: {len(df)} frames")
    except Exception as e:
        print(f"Error reading DLC file: {e}")
        return None

    # Check if bodypart exists in the data
    if bodypart not in df[dlc_scorer].columns:
        available_bodyparts = df[dlc_scorer].columns.get_level_values(0).unique()
        print(f"Error: Bodypart '{bodypart}' not found in DLC data")
        print(f"Available bodyparts: {list(available_bodyparts)}")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=figure_size)

    # Plot the grid boundary
    grid.boundary.plot(ax=ax, color="black", linewidth=0.5)

    # Color grid # 84 green
    grid.loc[grid["FID"] == 84, "geometry"].plot(ax=ax, color="green", alpha=0.5)

    # Filter points by likelihood threshold
    likelihood_mask = df[dlc_scorer][bodypart]["likelihood"].values > likelihood_threshold
    x_coords = df[dlc_scorer][bodypart]["x"].values[likelihood_mask]
    y_coords = df[dlc_scorer][bodypart]["y"].values[likelihood_mask]

    # Plot the scatter points
    ax.plot(x_coords, y_coords, ".", color="blue", alpha=0.1)

    # Flip the y-axis to match video coordinates
    ax.invert_yaxis()

    # Set title and labels
    ax.set_title(f"{session} - {bodypart.title()} Tracking\n" f"(Likelihood > {likelihood_threshold})", fontsize=10)
    ax.set_xlabel("X coordinate (pixels)")
    ax.set_ylabel("Y coordinate (pixels)")

    # Remove unnecessary whitespace
    plt.tight_layout()

    # Add statistics to the plot
    total_points = len(df)
    valid_points = np.sum(likelihood_mask)
    ax.text(
        0.02,
        0.98,
        f"Points: {valid_points}/{total_points} ({valid_points/total_points*100:.1f}%)",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Save the figure if requested
    if save_plot and figures_directory:
        figures_path = Path(figures_directory)
        figures_path.mkdir(parents=True, exist_ok=True)

        plot_filename = f"{session}_{bodypart}_scatter_plot.png"
        save_path = figures_path / plot_filename

        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")

    return fig


def batch_create_grid_scatter_plots(
    mouseinfo_df,
    dlc_results_directory,
    grid_files_directory,
    figures_directory,
    dlc_scorer,
    bodypart="sternum",
    likelihood_threshold=0.6,
    figure_size=(3, 3),
    show_plots=False,
):
    """
    Create grid scatter plots for multiple sessions.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    dlc_results_directory : str or Path
        Directory containing DLC results
    grid_files_directory : str or Path
        Directory containing grid files
    figures_directory : str or Path
        Directory to save figures
    dlc_scorer : str
        DLC scorer name
    bodypart : str, optional
        Bodypart to plot (default: 'sternum')
    likelihood_threshold : float, optional
        Minimum likelihood threshold (default: 0.6)
    figure_size : tuple, optional
        Figure size (default: (3, 3))
    show_plots : bool, optional
        Whether to display plots (default: False for batch processing)

    Returns:
    --------
    dict
        Summary of plot creation operations
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    start_time = datetime.now()
    print(f"Batch creating grid scatter plots started: {start_time}")

    # Create figures subdirectory
    scatter_plots_dir = Path(figures_directory) / "scatter_plots"
    scatter_plots_dir.mkdir(parents=True, exist_ok=True)

    # Track operations
    plot_summary = {
        "total_sessions": len(mouseinfo_df),
        "plots_created": 0,
        "failed_plots": 0,
        "successful_sessions": [],
        "failed_sessions": [],
    }

    print(f"Creating scatter plots for {len(mouseinfo_df)} sessions...")
    print(f"Bodypart: {bodypart}")
    print(f"Likelihood threshold: {likelihood_threshold}")
    print(f"Saving plots to: {scatter_plots_dir}")

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"

        print("-----------------------------")

        try:
            # Create the plot
            fig = create_grid_scatter_plot(
                session=session_name,
                dlc_results_directory=dlc_results_directory,
                grid_files_directory=grid_files_directory,
                dlc_scorer=dlc_scorer,
                bodypart=bodypart,
                likelihood_threshold=likelihood_threshold,
                figure_size=figure_size,
                save_plot=True,
                figures_directory=scatter_plots_dir,
            )

            if fig is not None:
                plot_summary["plots_created"] += 1
                plot_summary["successful_sessions"].append(session_name)

                # Show plot if requested
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)  # Close to save memory

                print(f"✓ Plot created for {session_name}")
            else:
                plot_summary["failed_plots"] += 1
                plot_summary["failed_sessions"].append(session_name)
                print(f"✗ Failed to create plot for {session_name}")

        except Exception as e:
            print(f"Error creating plot for {session_name}: {e}")
            plot_summary["failed_plots"] += 1
            plot_summary["failed_sessions"].append(session_name)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("SCATTER PLOT CREATION SUMMARY")
    print("=" * 60)
    print(f'Total sessions: {plot_summary["total_sessions"]}')
    print(f'Plots created: {plot_summary["plots_created"]}')
    print(f'Failed plots: {plot_summary["failed_plots"]}')
    print(f"Duration: {duration}")
    print(f"Plots saved to: {scatter_plots_dir}")

    if plot_summary["failed_sessions"]:
        print(f'\nFailed sessions: {plot_summary["failed_sessions"]}')

    print("=" * 60)

    return plot_summary


def create_trajectory_plot(
    session,
    dlc_results_directory,
    grid_files_directory,
    dlc_scorer,
    bodypart="sternum",
    likelihood_threshold=0.6,
    figure_size=(4, 4),
    colormap="viridis",
    save_plot=True,
    figures_directory=None,
):
    """
    Create a trajectory plot showing the path of movement with color-coded time progression.

    Parameters:
    -----------
    session : str
        Session name (e.g., 'Session-1')
    dlc_results_directory : str or Path
        Directory containing DLC results
    grid_files_directory : str or Path
        Directory containing grid files
    dlc_scorer : str
        DLC scorer name
    bodypart : str, optional
        Bodypart to plot (default: 'sternum')
    likelihood_threshold : float, optional
        Minimum likelihood threshold (default: 0.6)
    figure_size : tuple, optional
        Figure size (default: (4, 4))
    colormap : str, optional
        Colormap for trajectory (default: 'viridis')
    save_plot : bool, optional
        Whether to save the plot (default: True)
    figures_directory : str or Path, optional
        Directory to save figures

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """

    dlc_results_path = Path(dlc_results_directory)
    grid_files_path = Path(grid_files_directory)

    print(f"Creating trajectory plot for {session}...")

    # Read the Grid File
    grid_file = grid_files_path / f"{session}_grid.shp"
    if not grid_file.exists():
        print(f"Error: Grid file not found: {grid_file}")
        return None

    try:
        grid = gpd.read_file(str(grid_file))
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return None

    # Read the DLC Results
    dlc_file = dlc_results_path / f"{session}{dlc_scorer}.h5"
    if not dlc_file.exists():
        print(f"Error: DLC results not found: {dlc_file}")
        return None

    try:
        df = pd.read_hdf(str(dlc_file))
    except Exception as e:
        print(f"Error reading DLC file: {e}")
        return None

    # Check if bodypart exists in the data
    if bodypart not in df[dlc_scorer].columns:
        available_bodyparts = df[dlc_scorer].columns.get_level_values(0).unique()
        print(f"Error: Bodypart '{bodypart}' not found in DLC data")
        print(f"Available bodyparts: {list(available_bodyparts)}")
        return None

    # Filter points by likelihood threshold
    likelihood_mask = df[dlc_scorer][bodypart]["likelihood"].values > likelihood_threshold
    x_cut = df[dlc_scorer][bodypart]["x"].values[likelihood_mask]
    y_cut = df[dlc_scorer][bodypart]["y"].values[likelihood_mask]

    if len(x_cut) < 2:
        print(f"Error: Not enough valid points for trajectory (only {len(x_cut)} points)")
        return None

    # Create time parameter for color coding
    t = np.linspace(0, 1, x_cut.shape[0])

    # Create line segments for trajectory
    points = np.array([x_cut, y_cut]).transpose().reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the plot
    fig, ax = plt.subplots(figsize=figure_size)

    # Create line collection for trajectory
    lc = LineCollection(segs, cmap=plt.get_cmap(colormap), linewidths=1)
    lc.set_array(t)  # Color segments by time

    # Add trajectory to plot
    lines = ax.add_collection(lc)

    # Add scatter points
    ax.scatter(points[:, :, 0], points[:, :, 1])

    # Plot grid boundary
    grid.boundary.plot(ax=ax, color="red")

    # Flip y-axis to match video coordinates
    ax.invert_yaxis()

    # Set title
    ax.set_title(f"{session} Trajectory Plot")

    plt.tight_layout()

    # Save the figure if requested
    if save_plot and figures_directory:
        figures_path = Path(figures_directory)
        figures_path.mkdir(parents=True, exist_ok=True)

        plot_filename = f"{session}_{bodypart}_trajectory_plot.png"
        save_path = figures_path / plot_filename

        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")

    return fig


def batch_create_trajectory_plots(
    mouseinfo_df,
    dlc_results_directory,
    grid_files_directory,
    figures_directory,
    dlc_scorer,
    bodypart="sternum",
    likelihood_threshold=0.6,
    figure_size=(4, 4),
    colormap="viridis",
    show_plots=False,
):
    """
    Create trajectory plots for multiple sessions.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    dlc_results_directory : str or Path
        Directory containing DLC results
    grid_files_directory : str or Path
        Directory containing grid files
    figures_directory : str or Path
        Directory to save figures
    dlc_scorer : str
        DLC scorer name
    bodypart : str, optional
        Bodypart to plot (default: 'sternum')
    likelihood_threshold : float, optional
        Minimum likelihood threshold (default: 0.6)
    figure_size : tuple, optional
        Figure size (default: (4, 4))
    colormap : str, optional
        Colormap for trajectory (default: 'viridis')
    show_plots : bool, optional
        Whether to display plots (default: False)

    Returns:
    --------
    dict
        Summary of plot creation operations
    """

    # Create figures subdirectory
    trajectory_plots_dir = Path(figures_directory) / "trajectory_plots"
    trajectory_plots_dir.mkdir(parents=True, exist_ok=True)

    # Track operations
    plot_summary = {
        "total_sessions": len(mouseinfo_df),
        "plots_created": 0,
        "failed_plots": 0,
        "successful_sessions": [],
        "failed_sessions": [],
    }

    print(f"Creating trajectory plots for {len(mouseinfo_df)} sessions...")
    print(f"Bodypart: {bodypart}")
    print(f"Likelihood threshold: {likelihood_threshold}")
    print(f"Colormap: {colormap}")
    print(f"Saving plots to: {trajectory_plots_dir}")

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"

        print("-----------------------------")

        try:
            # Create the plot
            fig = create_trajectory_plot(
                session=session_name,
                dlc_results_directory=dlc_results_directory,
                grid_files_directory=grid_files_directory,
                dlc_scorer=dlc_scorer,
                bodypart=bodypart,
                likelihood_threshold=likelihood_threshold,
                figure_size=figure_size,
                colormap=colormap,
                save_plot=True,
                figures_directory=trajectory_plots_dir,
            )

            if fig is not None:
                plot_summary["plots_created"] += 1
                plot_summary["successful_sessions"].append(session_name)

                # Show plot if requested
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)  # Close to save memory

                print(f"✓ Trajectory plot created for {session_name}")
            else:
                plot_summary["failed_plots"] += 1
                plot_summary["failed_sessions"].append(session_name)
                print(f"✗ Failed to create trajectory plot for {session_name}")

        except Exception as e:
            print(f"Error creating trajectory plot for {session_name}: {e}")
            plot_summary["failed_plots"] += 1
            plot_summary["failed_sessions"].append(session_name)

    print("\n" + "=" * 60)
    print("TRAJECTORY PLOT CREATION SUMMARY")
    print("=" * 60)
    print(f'Total sessions: {plot_summary["total_sessions"]}')
    print(f'Plots created: {plot_summary["plots_created"]}')
    print(f'Failed plots: {plot_summary["failed_plots"]}')
    print(f"Plots saved to: {trajectory_plots_dir}")

    if plot_summary["failed_sessions"]:
        print(f'\nFailed sessions: {plot_summary["failed_sessions"]}')

    print("=" * 60)

    return plot_summary


def append_grid_numbers_to_csv(
    session,
    dlc_results_directory,
    grid_files_directory,
    dlc_scorer,
    bodyparts=["nose", "belly", "sternum", "leftflank", "rightflank", "tailbase"],
    save_directory=None,
):
    """
    Append grid numbers to DLC CSV results for a single session.

    Parameters:
    -----------
    session : str
        Session name (e.g., 'Session-1')
    dlc_results_directory : str or Path
        Directory containing DLC CSV results
    grid_files_directory : str or Path
        Directory containing grid files
    dlc_scorer : str
        DLC scorer name
    bodyparts : list, optional
        List of bodyparts to process
    save_directory : str or Path, optional
        Directory to save annotated CSV (defaults to dlc_results_directory)

    Returns:
    --------
    pd.DataFrame or None
        Annotated dataframe if successful, None if failed
    """

    dlc_results_path = Path(dlc_results_directory)
    grid_files_path = Path(grid_files_directory)

    if save_directory is None:
        save_directory = dlc_results_directory
    save_path = Path(save_directory)

    print(f"Appending grids to {session} CSV...")

    # Load the DLC Results CSV
    csv_file = dlc_results_path / f"{session}{dlc_scorer}.csv"
    if not csv_file.exists():
        print(f"Error: DLC CSV not found: {csv_file}")
        return None

    try:
        df = pd.read_csv(str(csv_file), header=[0, 1, 2], index_col=0)
    except Exception as e:
        print(f"Error reading DLC CSV: {e}")
        return None

    # Load the Grid
    grid_file = grid_files_path / f"{session}_grid.shp"
    if not grid_file.exists():
        print(f"Error: Grid file not found: {grid_file}")
        return None

    try:
        grid = gpd.read_file(str(grid_file))
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return None

    # Check which bodyparts exist in the data
    available_bodyparts = df[dlc_scorer].columns.get_level_values(0).unique()
    valid_bodyparts = [bp for bp in bodyparts if bp in available_bodyparts]

    if not valid_bodyparts:
        print(f"Error: None of the specified bodyparts found in data")
        return None

    # Process each bodypart
    for bp in valid_bodyparts:
        try:
            # Convert x and y coordinates to geometry points
            x_coords = df[dlc_scorer][bp]["x"].values
            y_coords = df[dlc_scorer][bp]["y"].values

            # Create points, handling NaN values
            points = []
            for x, y in zip(x_coords, y_coords):
                if pd.notna(x) and pd.notna(y):
                    points.append(Point(x, y))
                else:
                    points.append(None)

            # Create GeoDataFrame
            pnt_gpd = gpd.GeoDataFrame(geometry=points, index=np.arange(len(points)), crs=grid.crs)

            # Find which polygon each point is in
            pointInPolys = gpd.tools.sjoin(pnt_gpd, grid, predicate="within", how="left")

            # Add grid numbers to dataframe
            # Use 'FID' column from grid or create sequential numbering if FID doesn't exist
            if "FID" in pointInPolys.columns:
                grid_numbers = pointInPolys["FID"].values
            else:
                # Use index as grid number if FID column doesn't exist
                grid_numbers = pointInPolys["index_right"].values

            # Add grid number column to the original dataframe
            df[dlc_scorer, bp, "Grid Number"] = grid_numbers

        except Exception as e:
            print(f"Error processing {bp}: {e}")
            continue

    # Sort the dataframe columns
    df = df.sort_index(axis=1)

    # Save the annotated CSV
    save_path.mkdir(parents=True, exist_ok=True)
    output_file = save_path / f"{session}_withGrids.csv"

    try:
        df.to_csv(str(output_file))
        print(f"Saved to: {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None

    return df


def batch_append_grid_numbers(
    mouseinfo_df,
    dlc_results_directory,
    grid_files_directory,
    dlc_scorer,
    bodyparts=["nose", "belly", "sternum", "leftflank", "rightflank", "tailbase"],
    save_directory=None,
):
    """
    Append grid numbers to DLC CSV results for multiple sessions.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    dlc_results_directory : str or Path
        Directory containing DLC CSV results
    grid_files_directory : str or Path
        Directory containing grid files
    dlc_scorer : str
        DLC scorer name
    bodyparts : list, optional
        List of bodyparts to process
    save_directory : str or Path, optional
        Directory to save annotated CSVs (defaults to dlc_results_directory)

    Returns:
    --------
    dict
        Summary of grid annotation operations
    """
    from pathlib import Path

    start_time = datetime.now()
    print(f"Batch grid annotation started: {start_time}")

    if save_directory is None:
        save_directory = dlc_results_directory

    # Track operations
    annotation_summary = {
        "total_sessions": len(mouseinfo_df),
        "successfully_annotated": 0,
        "failed_annotation": 0,
        "successful_sessions": [],
        "failed_sessions": [],
    }

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"
        grid_numbers_file = save_directory / f"{session_name}_withGrids.csv"
        
        if grid_numbers_file.exists():
            continue
        try:
            # Append grid numbers for this session
            annotated_df = append_grid_numbers_to_csv(
                session=session_name,
                dlc_results_directory=dlc_results_directory,
                grid_files_directory=grid_files_directory,
                dlc_scorer=dlc_scorer,
                bodyparts=bodyparts,
                save_directory=save_directory,
            )

            if annotated_df is not None:
                annotation_summary["successfully_annotated"] += 1
                annotation_summary["successful_sessions"].append(session_name)
                print(f"Grid annotation completed for {session_name}")
            else:
                annotation_summary["failed_annotation"] += 1
                annotation_summary["failed_sessions"].append(session_name)
                print(f"Failed to annotate {session_name}")

        except Exception as e:
            print(f"Error annotating {session_name}: {e}")
            annotation_summary["failed_annotation"] += 1
            annotation_summary["failed_sessions"].append(session_name)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("GRID ANNOTATION SUMMARY")
    print("=" * 60)
    print(f'Total sessions processed: {annotation_summary["total_sessions"]}')
    print(f'Failed annotation: {annotation_summary["failed_annotation"]}')

    if annotation_summary["failed_sessions"]:
        print(f'\nFailed sessions: {annotation_summary["failed_sessions"]}')

    print("=" * 60)

    return annotation_summary

def run_grid_preprocessing(
    source_data_path: Union[str, Path],
    user_metadata_file_path: Union[str, Path],
    trial_type: str = "Labyrinth_DSI",
    video_type: str = ".mp4",
    dlc_scorer: str = "DLC_resnet50_LabyrinthMar13shuffle1_1000000",
    num_squares: int = 12,
    reprocess_existing: bool = False
):
    """
    Run the grid preprocessing pipeline to prepare spatial data for CoMPASS analysis.
    
    This creates:
    - First frame images from videos
    - Boundary points for the maze
    - Cropping coordinates
    - Grid files with spatial information
    - DLC results with grid numbers appended (_withGrids.csv)
    
    Note: This assumes DLC pose estimation has already been run on the videos.
    Automatically skips steps that are already complete unless reprocess_existing=True.
    Only processes sessions that have video or DLC files in source_data_path.
    
    Parameters:
    -----------
    source_data_path : Path | str
        Directory containing video files and DLC outputs
    user_metadata_file_path : Path | str
        Path to metadata Excel file
    trial_type : str
        Sheet name in the metadata file
    video_type : str
        Video file extension (default: ".mp4")
    dlc_scorer : str
        DLC scorer name for identifying pose estimation files
    num_squares : int
        Number of grid squares per side (default: 12)
    reprocess_existing : bool
        If True, reprocess sessions that already have outputs (default: False)
        
    Returns:
    --------
    dict
        Summary of preprocessing results
    """
    source_data_path = Path(source_data_path).resolve()
    
    if not source_data_path.exists():
        raise ValueError(f"Source data path does not exist: {source_data_path}")
    
    print("="*70)
    print("COMPASS-LABYRINTH GRID PREPROCESSING")
    print("="*70)
    
    # Load metadata
    print("\nLoading metadata...")
    mouseinfo_full = import_cohort_metadata(
        metadata_path=user_metadata_file_path,
        trial_sheet_name=trial_type
    )
    validate_metadata(mouseinfo_full)
    print(f"✓ Loaded {len(mouseinfo_full)} sessions from metadata")
    
    # Filter metadata to only include sessions with files in source_data_path
    print("\nFiltering to sessions in source_data_path...")
    sessions_in_directory = []
    
    for index, row in mouseinfo_full.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"
        
        # Check if this session has a video or DLC file in the directory
        video_file = source_data_path / f"{session_name}{video_type}"
        dlc_h5_files = list(source_data_path.glob(f"{session_name}*DLC*.h5"))
        dlc_csv_files = list(source_data_path.glob(f"{session_name}*DLC*.csv"))
        
        has_video = video_file.exists()
        has_dlc = len(dlc_h5_files) > 0 or len(dlc_csv_files) > 0
        
        if has_video or has_dlc:
            sessions_in_directory.append(index)
    
    if len(sessions_in_directory) == 0:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: No matching sessions found\n"
            f"{'='*70}\n"
            f"No videos or DLC files found in:\n"
            f"  {source_data_path}\n\n"
            f"Metadata contains {len(mouseinfo_full)} sessions, but none match files in the directory.\n\n"
            f"Expected file pattern: SessionXXXX{video_type} or SessionXXXX*DLC*.h5/.csv\n"
            f"{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)
    
    # Create filtered metadata dataframe
    mouseinfo = mouseinfo_full.loc[sessions_in_directory].copy()
    
    print(f"✓ Found {len(mouseinfo)} sessions in directory (filtered from {len(mouseinfo_full)} total)\n")
    
    if len(mouseinfo) < len(mouseinfo_full):
        print("Sessions to process:")
        for index, row in mouseinfo.iterrows():
            session_num = int(row["Session #"])
            print(f"  - Session{session_num:04d}")
        print()
    
    # Check what's already done
    frames_needed = []
    boundaries_needed = []
    cropping_needed = []
    grids_needed = []
    grid_numbers_needed = []
    
    for index, row in mouseinfo.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"
        
        frame_file = source_data_path / f"{session_name}Frame1.jpg"
        boundary_file = source_data_path / f"{session_name}_Boundary_Points.npy"
        cropping_file = source_data_path / f"{session_name}_DLC_Cropping_Bounds.npy"
        grid_file = source_data_path / f"{session_name}_grid.shp"
        grid_numbers_file = source_data_path / f"{session_name}_withGrids.csv"
        
        if not frame_file.exists() or reprocess_existing:
            frames_needed.append(session_name)
        if not boundary_file.exists() or reprocess_existing:
            boundaries_needed.append(session_name)
        if not cropping_file.exists() or reprocess_existing:
            cropping_needed.append(session_name)
        if not grid_file.exists() or reprocess_existing:
            grids_needed.append(session_name)
        if not grid_numbers_file.exists() or reprocess_existing:
            grid_numbers_needed.append(session_name)
    
    # Check if everything is already done
    if not any([frames_needed, boundaries_needed, cropping_needed, grids_needed, grid_numbers_needed]):
        print("✓ All preprocessing outputs already exist!")
        if not reprocess_existing:
            print("  Use reprocess_existing=True to regenerate outputs.")
        print("="*70)
        return {"status": "complete", "sessions_processed": len(mouseinfo)}
    
    # Summary of what's needed
    print("Status check:")
    print(f"  Frames:       {len(mouseinfo) - len(frames_needed)}/{len(mouseinfo)} complete")
    print(f"  Boundaries:   {len(mouseinfo) - len(boundaries_needed)}/{len(mouseinfo)} complete")
    print(f"  Cropping:     {len(mouseinfo) - len(cropping_needed)}/{len(mouseinfo)} complete")
    print(f"  Grids:        {len(mouseinfo) - len(grids_needed)}/{len(mouseinfo)} complete")
    print(f"  Grid Numbers: {len(mouseinfo) - len(grid_numbers_needed)}/{len(mouseinfo)} complete\n")
    
    results = {}
    
    # Step 1: Save first frames (if needed)
    if frames_needed:
        print(f"Extracting first frames ({len(frames_needed)} needed)...")
        frame_results = batch_save_first_frames(
            mouseinfo_df=mouseinfo,
            video_directory=source_data_path,
            frames_directory=source_data_path
        )
        results["frames"] = frame_results
    
    # Step 2: Get boundaries and cropping (if needed)
    if boundaries_needed or cropping_needed:
        sessions_needing_selection = list(set(boundaries_needed + cropping_needed))
        print(f"Getting boundary points and cropping ({len(sessions_needing_selection)} needed)...")
        print("(Interactive - select maze boundaries for each session)\n")
        coordinates_dict = batch_get_boundary_and_cropping(
            mouseinfo_df=mouseinfo, 
            frames_directory=source_data_path,
            cropping_directory=source_data_path,
            boundaries_directory=source_data_path,
            reprocess_existing=reprocess_existing
        )
        results["coordinates"] = coordinates_dict
    
    # Step 3: Create grids (if needed)
    if grids_needed:
        print(f"Creating spatial grids ({len(grids_needed)} needed)...")
        grid_results = batch_create_grids(
            mouseinfo_df=mouseinfo,
            boundaries_directory=source_data_path,
            grid_files_directory=source_data_path,
            cropping_directory=source_data_path,
            num_squares=num_squares
        )
        results["grids"] = grid_results
    
    # Step 4: Append grid numbers to DLC results (if needed)
    if grid_numbers_needed:
        print(f"Appending grid numbers to DLC results ({len(grid_numbers_needed)} needed)...")
        grid_number_results = batch_append_grid_numbers(
            mouseinfo_df=mouseinfo,
            grid_files_directory=source_data_path,
            dlc_results_directory=source_data_path,
            dlc_scorer=dlc_scorer,
            save_directory=source_data_path
        )
        results["grid_numbers"] = grid_number_results
    
    print("="*70)
    print("✓ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(mouseinfo)} sessions")
    print("You can now run init_project()")
    
    results["sessions_processed"] = len(mouseinfo)
    results["sessions_total_in_metadata"] = len(mouseinfo_full)
    
    return results