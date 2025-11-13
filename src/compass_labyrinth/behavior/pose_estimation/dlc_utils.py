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

def check_input_file_status(source_data_path, video_type=".mp4"):
    """
    Check if required pose estimation and grid outputs exist for all video files in the directory.
    
    This validates that spatial grid files, boundary points, and cropping coordinates
    have been created for each video session.
    
    Parameters:
    -----------
    source_data_path : str or Path
        Directory that should contain videos and preprocessing outputs
    video_type : str
        Video file extension (default: ".mp4")
        
    Raises:
    -------
    FileNotFoundError
        If any required grid preprocessing files are missing for sessions with videos
    """
    from pathlib import Path
    
    source_data_path = Path(source_data_path)
    
    print("\nValidating grid preprocessing outputs...")
    
    # Find all video files to determine which sessions exist
    video_files = list(source_data_path.glob(f"*{video_type}"))
    
    if len(video_files) == 0:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: No video files found in:\n"
            f"  {source_data_path}\n"
            f"{'='*70}\n"
            f"Looking for files matching pattern: *{video_type}\n\n"
            f"Make sure your video files are in the source_data_path.\n"
            f"{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)
    
    # Extract session names from video files
    session_names = []
    for video_file in video_files:
        # Remove the video extension to get session name
        session_name = video_file.stem
        session_names.append(session_name)
    
    print(f"Found {len(session_names)} video files")
    
    # Track missing files per session
    missing_sessions = {
        "grid": [],
        "boundary": [],
        "cropping": []
    }
    
    # Check each session for required preprocessing files
    for session_name in session_names:
        grid_file = source_data_path / f"{session_name}_withGrids.csv"
        boundary_file = source_data_path / f"{session_name}_Boundary_Points.npy"
        cropping_file = source_data_path / f"{session_name}_DLC_Cropping_Bounds.npy"
        
        if not grid_file.exists():
            missing_sessions["grid"].append(session_name)
        if not boundary_file.exists():
            missing_sessions["boundary"].append(session_name)
        if not cropping_file.exists():
            missing_sessions["cropping"].append(session_name)
    
    # Build error message if any files are missing
    missing_outputs = []
    if missing_sessions["grid"]:
        missing_outputs.append(f"- Grid files missing for: {', '.join(missing_sessions['grid'])}")
    if missing_sessions["boundary"]:
        missing_outputs.append(f"- Boundary files missing for: {', '.join(missing_sessions['boundary'])}")
    if missing_sessions["cropping"]:
        missing_outputs.append(f"- Cropping files missing for: {', '.join(missing_sessions['cropping'])}")
    
    if missing_outputs:
        all_missing = set(missing_sessions['grid'] + missing_sessions['boundary'] + missing_sessions['cropping'])
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Missing required grid preprocessing outputs in:\n"
            f"  {source_data_path}\n"
            f"{'='*70}\n"
            f"Missing files for {len(all_missing)} session(s):\n"
            + "\n".join(missing_outputs) + "\n\n"
            f"SOLUTION: Run the grid preprocessing pipeline:\n\n"
            f"  from compass_labyrinth import run_grid_preprocessing\n\n"
            f"  run_grid_preprocessing(\n"
            f"      source_data_path=r'{source_data_path}',\n"
            f"      user_metadata_file_path='path/to/metadata.xlsx',\n"
            f"      trial_type='Labyrinth_DSI'\n"
            f"  )\n\n"
            f"This will create:\n"
            f"  - First frame images for each video\n"
            f"  - Boundary points for the maze (interactive)\n"
            f"  - Cropping coordinates\n"
            f"  - Grid files with spatial information\n\n"
            f"Then re-run init_project()\n"
            f"{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)
    
    # If we get here, all files exist for all sessions
    print(f"✓ Found grid files for all {len(session_names)} sessions")
    print(f"✓ Found boundary files for all {len(session_names)} sessions")
    print(f"✓ Found cropping files for all {len(session_names)} sessions")
    print("✓ All required pose estimation and grid outputs found!")

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

def analyze_videos_with_DLC(mouseinfo_df, config_path, video_directory, cropping_directory, results_directory):
    """
    Analyze videos using DeepLabCut with results saved directly to organized directory.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame with session info
    config_path : str or Path
        Path to DeepLabCut config.yaml file
    video_directory : str or Path
        Directory containing videos (videos/original_videos)
    cropping_directory : str or Path
        Directory containing cropping coordinate files (data/dlc_cropping_bounds)
    results_directory : str or Path
        Directory where DLC results will be saved (data/dlc_results)

    Returns:
    --------
    dict
        Summary of analysis operations
    """
    try:
        import deeplabcut
    except ImportError:
        print("Error: DeepLabCut not available")
        return None

    start_time = datetime.now()
    print(f"DeepLabCut analysis started: {start_time}")

    # Ensure results directory exists
    results_path = Path(results_directory)
    results_path.mkdir(parents=True, exist_ok=True)

    # Analysis summary
    analysis_summary = {
        "total_sessions": len(mouseinfo_df),
        "successfully_analyzed": 0,
        "failed_analysis": 0,
        "no_coordinates": 0,
        "skipped_existing": 0,
        "failed_sessions": [],
        "analysis_times": [],
    }

    print(f"Video directory: {video_directory}")
    print(f"Cropping directory: {cropping_directory}")
    print(f"Results directory: {results_path}")

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        session_start_time = datetime.now()
        print("-----------------------------")

        # Get session info
        session_num = int(row["Session #"])
        session_name = f"Session{session_num:04d}"
        video_name = f"{session_name}.mp4"
        video_path = Path(video_directory) / video_name

        print(f"Analyzing {session_name}...")
        print(f"Video: {video_name}")

        # Check if video exists
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            analysis_summary["failed_analysis"] += 1
            analysis_summary["failed_sessions"].append(session_name)
            continue

        # Load saved cropping bounds
        coord_file = Path(cropping_directory) / f"{session_name}_DLC_Cropping_Bounds.npy"

        if not coord_file.exists():
            print(f"Error: No saved cropping coordinates found for {session_name}")
            print(f"  - Missing file: {coord_file}")
            analysis_summary["no_coordinates"] += 1
            analysis_summary["failed_sessions"].append(session_name)
            continue

        try:
            coord_data = np.load(coord_file, allow_pickle=True).item()
            X1, X2, Y1, Y2 = coord_data["X1"], coord_data["X2"], coord_data["Y1"], coord_data["Y2"]
            cropping_coords = (X1, X2, Y1, Y2)
            print(f"Using saved cropping bounds: {cropping_coords}")
        except Exception as e:
            print(f"Error loading coordinates for {session_name}: {e}")
            analysis_summary["failed_analysis"] += 1
            analysis_summary["failed_sessions"].append(session_name)
            continue

        # Check if analysis already exists
        existing_csv = list(results_path.glob(f"{session_name}DLC_*.csv"))

        if existing_csv:
            print(f"Analysis already exists for {session_name}, skipping...")
            analysis_summary["skipped_existing"] += 1
            continue

        # Run DeepLabCut analysis with destfolder parameter
        # Ref: https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#i-analyze-new-videos
        try:
            print(f"Running DeepLabCut analysis...")

            deeplabcut.analyze_videos(
                config_path,
                [str(video_path)],
                shuffle=1,
                videotype=".mp4",
                save_as_csv=True,
                cropping=cropping_coords,
                destfolder=str(results_path),  # Results saved directly to organized directory
            )

            session_end_time = datetime.now()
            session_duration = session_end_time - session_start_time

            print(f"Successfully analyzed {session_name} in {session_duration}")
            analysis_summary["successfully_analyzed"] += 1
            analysis_summary["analysis_times"].append({"session": session_name, "duration": session_duration})

        except Exception as e:
            print(f"Error analyzing {session_name}: {e}")
            analysis_summary["failed_analysis"] += 1
            analysis_summary["failed_sessions"].append(session_name)

    # Print summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    analysis_summary["total_duration"] = total_duration

    return analysis_summary


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
