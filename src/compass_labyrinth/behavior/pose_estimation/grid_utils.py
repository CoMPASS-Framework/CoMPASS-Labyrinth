"""
Functions for Labyrinth Maze Analyses

Extract Maze Coordinates and create square grids

Created by Patrick Honma 4/6/2023
Edited by Patrick Honma 7/5/2023

"""

import numpy as np
from shapely.geometry import Point, Polygon, box
import shapely.ops as so
import geopandas as gpd
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def save_first_frame(video_path, session):
    """
    Read in the first frame of the video, save it as a JPEG image, and display the frame.
    """
    # Capture video
    cap = cv2.VideoCapture(os.path.join(video_path, session + ".mp4"))

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as a JPEG image
        frame_image_path = os.path.join(video_path, session + "Frame1.jpg")
        cv2.imwrite(frame_image_path, frame)

        # Convert the frame from BGR (OpenCV format) to RGB (matplotlib format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using matplotlib
        plt.imshow(frame_rgb)
        plt.title(f"First frame of {session}")
        plt.axis("off")  # Hide axes for better visualization
        plt.show()

        # Release the video capture object
        cap.release()
    else:
        print(f"Failed to read the video file: {session}.mp4")

    # Return ret and frame
    return ret, frame


# def get_labyrinth_boundary(video_path, session, X1, X2, Y1, Y2):

#         posList = []

#         def click_event(event, x, y, flags, params):
#             # checking for left mouse clicks
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 # displaying the coordinates
#                 # on the Shell
#                 posList.append((x,y))
#                 print(x, ' ', y)
#                 # displaying the coordinates
#                 # on the image window
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(img, str(x) + ',' +
#                             str(y), (x,y), font,
#                             1, (255, 0, 0), 2)
#                 cv2.imshow('image', img)
#             # checking for right mouse clicks
#             if event==cv2.EVENT_RBUTTONDOWN:
#                 # displaying the coordinates
#                 # on the Shell
#                 print(x, ' ', y)
#                 # displaying the coordinates
#                 # on the image window
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 b = img[y, x, 0]
#                 g = img[y, x, 1]
#                 r = img[y, x, 2]
#                 cv2.putText(img, str(b) + ',' +
#                             str(g) + ',' + str(r),
#                             (x,y), font, 1,
#                             (255, 255, 0), 2)
#                 cv2.imshow('image', img)


#         img = cv2.imread(os.path.join(video_path, session+'Frame1.jpg'), 1)
#         img = img[Y1:Y2, X1:X2] # cropping bounds in format [Y1:Y2, X1:X2]
#         img = img

#         cv2.startWindowThread()

#         # if cropped, specify the cropping bounds like below
#         cv2.imshow("image", img)
#         # setting mouse handler for the image
#         # and calling the click_event() function
#         #img = np.array(img)
#         cv2.setMouseCallback('image', click_event)
#         # wait for a key to be pressed to exit
#         cv2.waitKey(0)
#         # close the window
#         cv2.destroyAllWindows()

#         posList = np.array(posList)
#         np.save(os.path.join(video_path, session + ' PosList.npy'), posList)

#         return posList

# def get_grid_coordinates(posList, num_squares, video_path, session):
#     # Get the coordinates of the 4 coordinates
#     border = np.array(posList[:4])

#     # Create a polygon using these 4 coordinates
#     grid_polygon = Polygon(border)

#     # Define grid boundaries
#     xmin, ymin, xmax, ymax = grid_polygon.bounds

#     # determine the size of each square
#     width = (grid_polygon.bounds[2] - grid_polygon.bounds[0]) / num_squares
#     height = (grid_polygon.bounds[3] - grid_polygon.bounds[1]) / num_squares
#     center = (grid_polygon.bounds[2] - grid_polygon.bounds[0]), (grid_polygon.bounds[3] - grid_polygon.bounds[1])

#     rows = int(np.ceil((ymax-ymin) /  height))
#     cols = int(np.ceil((xmax-xmin) / width))

#     XleftOrigin = xmin
#     XrightOrigin = xmin + width
#     YtopOrigin = ymin
#     YbottomOrigin = ymin + height

#     polygons = []
#     for i in range(cols):
#         Ytop = YtopOrigin
#         Ybottom =YbottomOrigin
#         for j in range(rows):
#             polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
#             Ytop = Ytop + height
#             Ybottom = Ybottom + height
#         XleftOrigin = XleftOrigin + width
#         XrightOrigin = XrightOrigin + width

#     grid = gpd.GeoDataFrame({'geometry':polygons})

#     # Save the square grid
#     grid.to_file(os.path.join(video_path, session+ " grid.shp"))
#     grid.to_excel(os.path.join(video_path, session+ " grid.xlsx"))

#     print('Saved Grid for ' + session + ' ....')

#     return grid


def get_labyrinth_boundary(video_path, session, cropping_coords, chamber_info=None):
    """
    Get labyrinth boundary coordinates by clicking on the cropped video frame.
    Click 4 corners of the maze boundary.

    Parameters:
    -----------
    video_path : str or Path
        Directory containing the video and frame image
    session : str
        Session name (e.g., 'Session-1')
    cropping_coords : tuple
        (X1, X2, Y1, Y2) cropping coordinates
    chamber_info : str, optional
        Chamber information to display

    Returns:
    --------
    np.array
        Array of 4 boundary coordinates
    """

    X1, X2, Y1, Y2 = cropping_coords
    posList = []

    def click_event(event, x, y, flags, params):
        # Left mouse click to select boundary corners
        if event == cv2.EVENT_LBUTTONDOWN:
            posList.append((x, y))
            print(f"Boundary point {len(posList)}: ({x}, {y})")

            # Draw point on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # Different colors for each corner
            color = colors[len(posList) - 1] if len(posList) <= 4 else (255, 255, 255)

            cv2.circle(img_display, (x, y), 5, color, -1)
            cv2.putText(img_display, f"Corner {len(posList)}: {x},{y}", (x + 10, y - 10), font, 0.5, color, 2)

            # Draw lines connecting points
            if len(posList) > 1:
                cv2.line(img_display, posList[-2], posList[-1], color, 2)

            # Close the polygon when we have 4 points
            if len(posList) == 4:
                cv2.line(img_display, posList[-1], posList[0], color, 2)
                print("4 boundary corners selected. Press 'q' to confirm or 'r' to reset")

            cv2.imshow("Select Labyrinth Boundary", img_display)

        # Right mouse click to show pixel values
        elif event == cv2.EVENT_RBUTTONDOWN:
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv2.putText(img_display, f"RGB: {r},{g},{b}", (x + 10, y + 10), font, 0.4, (255, 255, 0), 1)
            cv2.imshow("Select Labyrinth Boundary", img_display)
            print(f"Pixel at ({x}, {y}): RGB({r}, {g}, {b})")

    # Load and crop the frame
    frame_path = os.path.join(video_path, session + "Frame1.jpg")
    if not os.path.exists(frame_path):
        print(f"Error: Frame not found at {frame_path}")
        return None

    img = cv2.imread(frame_path, 1)
    img = img[Y1:Y2, X1:X2]  # Apply cropping
    img_display = img.copy()

    print(f"\nLabyrinth Boundary Selection for {session}")
    if chamber_info:
        print(f"Chamber: {chamber_info}")
    print(f"Cropped image size: {img.shape[1]} x {img.shape[0]} (W x H)")

    print("\nInstructions:")
    print("1. Click on the 4 corners of the labyrinth boundary (in order)")
    print("2. Right-click to see pixel RGB values (optional)")
    print("3. Press 'q' to confirm selection after clicking 4 corners")
    print("4. Press 'r' to reset and select again")
    print("5. Press 'c' to cancel")

    cv2.startWindowThread()
    cv2.namedWindow("Select Labyrinth Boundary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Labyrinth Boundary", min(1200, img.shape[1]), min(800, img.shape[0]))
    cv2.imshow("Select Labyrinth Boundary", img_display)
    cv2.setMouseCallback("Select Labyrinth Boundary", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") and len(posList) >= 4:
            # Confirm selection
            break
        elif key == ord("r"):
            # Reset selection
            posList.clear()
            img_display = img.copy()
            cv2.imshow("Select Labyrinth Boundary", img_display)
            print("Selection reset. Click 4 boundary corners again.")
        elif key == ord("c"):
            # Cancel
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None
        elif key == 27:  # ESC key
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None

        # Check if window is closed
        if cv2.getWindowProperty("Select Labyrinth Boundary", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    if len(posList) >= 4:
        # Save the boundary coordinates
        posList_array = np.array(posList[:4])  # Take only first 4 points
        boundary_file = os.path.join(video_path, session + "_Boundary_Points.npy")
        np.save(boundary_file, posList_array)

        print(f"\nBoundary coordinates for {session}:")
        for i, (x, y) in enumerate(posList_array):
            print(f"  Corner {i+1}: ({x}, {y})")
        print(f"Coordinates saved to: {boundary_file}")

        return posList_array
    else:
        print("Insufficient boundary points selected.")
        return None


def get_grid_coordinates(posList, num_squares, video_path, session):
    """
    Create a grid from boundary coordinates and save as shapefile.

    Parameters:
    -----------
    posList : np.array
        Array of 4 boundary coordinates
    num_squares : int
        Number of squares per side (e.g., 12 for 12x12 grid)
    video_path : str or Path
        Directory to save grid files
    session : str
        Session name

    Returns:
    --------
    gpd.GeoDataFrame
        Grid as geopandas dataframe
    """

    # Get the coordinates of the 4 boundary points
    border = np.array(posList[:4])

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
    grid_shp_path = os.path.join(video_path, session + "_grid.shp")
    grid_xlsx_path = os.path.join(video_path, session + "_grid.xlsx")

    grid.to_file(grid_shp_path)
    grid.to_excel(grid_xlsx_path)

    print(f"Saved Grid for {session}")
    print(f"  - Shapefile: {grid_shp_path}")
    print(f"  - Excel: {grid_xlsx_path}")
    print(f"  - Grid size: {num_squares}x{num_squares} ({len(polygons)} total squares)")

    return grid


def batch_create_grids(mouseinfo_df, video_directory, num_squares=12):
    """
    Create grids for multiple sessions using saved cropping coordinates.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    video_directory : str or Path
        Directory containing videos and coordinate files
    num_squares : int, optional
        Number of squares per side (default: 12)

    Returns:
    --------
    dict
        Summary of grid creation operations
    """

    print(f"Grid size: {num_squares} x {num_squares}")

    grid_summary = {
        "total_sessions": len(mouseinfo_df),
        "grids_created": 0,
        "already_exists": 0,
        "failed_creation": 0,
        "no_coordinates": 0,
        "created_sessions": [],
        "failed_sessions": [],
    }

    # Process each session
    for index, row in mouseinfo_df.iterrows():
        print("-----------------------------")

        session_num = int(row["Session #"])
        session_name = f"Session-{session_num}"

        print(f"Processing {session_name} ({index+1}/{len(mouseinfo_df)})...")

        # Get chamber info if available
        chamber_info = None
        if "Noldus Chamber" in row and pd.notna(row["Noldus Chamber"]):
            chamber_info = row["Noldus Chamber"]
            print(f"Chamber: {chamber_info}")

        # Check if grid already exists
        grid_file = os.path.join(video_directory, session_name + "_grid.shp")
        if os.path.exists(grid_file):
            print(f"{session_name} grid already exists!")
            grid_summary["already_exists"] += 1
            continue

        # Load saved cropping coordinates
        coord_file = os.path.join(video_directory, session_name + "_DLC_Cropping_Bounds.npy")
        if not os.path.exists(coord_file):
            print(f"Error: No cropping coordinates found for {session_name}")
            print(f"  - Missing file: {coord_file}")
            print(f"  - Run get_dlc_cropping_bounds() first")
            grid_summary["no_coordinates"] += 1
            grid_summary["failed_sessions"].append(session_name)
            continue

        try:
            # Load cropping coordinates
            coord_data = np.load(coord_file, allow_pickle=True).item()
            cropping_coords = (coord_data["X1"], coord_data["X2"], coord_data["Y1"], coord_data["Y2"])
            print(f"Using cropping coordinates: {cropping_coords}")

            # Get boundary coordinates
            boundary_points = get_labyrinth_boundary(video_directory, session_name, cropping_coords, chamber_info)

            if boundary_points is not None:
                # Create grid
                grid = get_grid_coordinates(boundary_points, num_squares, video_directory, session_name)

                grid_summary["grids_created"] += 1
                grid_summary["created_sessions"].append(session_name)
                print(f"✓ Grid created for {session_name}")
            else:
                print(f"✗ Boundary selection cancelled for {session_name}")
                grid_summary["failed_creation"] += 1
                grid_summary["failed_sessions"].append(session_name)

                # Ask if user wants to continue
                continue_choice = input("Continue with next session? (y/n): ").strip().lower()
                if continue_choice == "n":
                    break

        except Exception as e:
            print(f"Error creating grid for {session_name}: {e}")
            grid_summary["failed_creation"] += 1
            grid_summary["failed_sessions"].append(session_name)

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
    print(f'No coordinates: {summary["no_coordinates"]}')
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


def check_grid_status(mouseinfo_df, video_directory):
    """
    Check which sessions already have grid files.

    Parameters:
    -----------
    mouseinfo_df : pd.DataFrame
        DataFrame containing session information
    video_directory : str or Path
        Directory to check for grid files

    Returns:
    --------
    dict
        Status of grid files for each session
    """
    import os

    grid_status = {}
    existing_grids = []
    missing_grids = []

    for index, row in mouseinfo_df.iterrows():
        session_num = int(row["Session #"])
        session_name = f"Session-{session_num}"

        grid_shp = os.path.join(video_directory, session_name + "_grid.shp")
        grid_xlsx = os.path.join(video_directory, session_name + "_grid.xlsx")

        grid_status[session_name] = {
            "shp_exists": os.path.exists(grid_shp),
            "xlsx_exists": os.path.exists(grid_xlsx),
            "has_grid": os.path.exists(grid_shp),  # Shapefile is the primary grid file
        }

        if os.path.exists(grid_shp):
            existing_grids.append(session_name)
        else:
            missing_grids.append(session_name)

    print(f"Grid status: {len(existing_grids)}/{len(mouseinfo_df)} sessions have grids")

    if missing_grids:
        print(f"Sessions missing grids: {missing_grids}")
    else:
        print("✓ All sessions have grids!")

    return grid_status
