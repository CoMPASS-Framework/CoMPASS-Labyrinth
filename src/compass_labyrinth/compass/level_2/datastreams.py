import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata


# ============================================================
# KDE Computation
# ============================================================
def loso_kde_cv(df: pd.DataFrame, smoothing_factors: list) -> float:
    sessions = shuffle(df["Session"].unique(), random_state=42)
    loo = LeaveOneOut()
    cv_results = []

    for sigma in smoothing_factors:
        print("Testing for sigma = ", sigma)
        fold_scores = []

        for train_idx, test_idx in loo.split(sessions):
            train_sessions = sessions[train_idx]
            test_sessions = sessions[test_idx]

            train_df = df[df["Session"].isin(train_sessions)]
            test_df = df[df["Session"].isin(test_sessions)]

            if len(train_df) < 5 or len(test_df) < 5:
                continue

            train_xy = train_df[["x", "y"]].to_numpy().T
            test_xy = test_df[["x", "y"]].to_numpy().T

            kde = gaussian_kde(train_xy, bw_method=lambda s: s.scotts_factor() * sigma)
            eps = 1e-10
            loglik = np.sum(np.log(kde(test_xy) + eps))
            fold_scores.append(loglik)

        if fold_scores:
            avg_score = np.mean(fold_scores)
            cv_results.append((sigma, avg_score))

    if not cv_results:
        raise ValueError("LOSO CV failed: insufficient data in some sessions.")

    best_sigma = max(cv_results, key=lambda x: x[1])[0]
    return best_sigma


def compute_kde_scaled(df, best_sigma, kde_col="KDE"):
    df = df.copy()
    df[kde_col] = np.nan
    scaler = MinMaxScaler((0, 3))
    print("Best Sigma = ", best_sigma)

    for session in df["Session"].unique():
        print(f"Computing KDE for Session {session}...")
        sess_df = df[df["Session"] == session]

        if len(sess_df) < 5:
            continue

        xy = sess_df[["x", "y"]].to_numpy().T
        kde = gaussian_kde(xy, bw_method=lambda s: s.scotts_factor() * best_sigma)
        kde_vals = kde(xy).reshape(-1, 1)

        kde_scaled = scaler.fit_transform(kde_vals) if kde_vals.max() > kde_vals.min() else np.zeros_like(kde_vals)
        df.loc[sess_df.index, kde_col] = kde_scaled.flatten()

    return df


def plot_kde_per_session(df: pd.DataFrame, best_sigma: float, kde_col: str = "KDE") -> None:
    for session in df["Session"].unique():
        sess_df = df[df["Session"] == session]
        if len(sess_df) < 5:
            continue

        x_vals, y_vals = sess_df["x"], sess_df["y"]
        kde_vals = sess_df[kde_col].to_numpy()

        x_grid = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, 150)
        y_grid = np.linspace(y_vals.min() - 0.5, y_vals.max() + 0.5, 150)
        X, Y = np.meshgrid(x_grid, y_grid)

        grid_kde = griddata((x_vals, y_vals), kde_vals, (X, Y), method="linear")
        grid_kde = np.nan_to_num(grid_kde)

        scaled = MinMaxScaler((0, 3)).fit_transform(grid_kde.reshape(-1, 1)).reshape(grid_kde.shape)

        plt.figure(figsize=(8, 7))
        plt.imshow(
            scaled,
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            origin="lower",
            cmap="viridis",
            interpolation="bicubic",
        )
        plt.scatter(x_vals, y_vals, c="white", s=6, edgecolor="black", linewidth=0.2, alpha=0.7)
        plt.title(f"KDE Map – Session {session} (σ = {best_sigma})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Normalized KDE [0–3]")
        plt.tight_layout()
        plt.show()


# ============================================================
# Targeted Angular Deviation
# ============================================================
# ------------------ Mapping Utilities ------------------ #
def map_category(value, category_dict):
    for key, item in category_dict.items():
        if isinstance(item, list) and value in item:
            return key
        elif value == item:
            return key
    return "Unknown"


# ------------------ Reference Axis ------------------ #
def get_reference_vector(ref_axis):
    directions = {"pos_x": (1, 0), "neg_x": (-1, 0), "pos_y": (0, 1), "neg_y": (0, -1)}
    return directions.get(ref_axis, (0, 0))


# ------------------ Angular Deviation ------------------ #
def calculate_deviation(row):
    if np.isnan(row["dx"]) or np.isnan(row["dy"]):
        return np.nan

    vector = np.array([row["dx"], row["dy"]])
    ref_vec = np.array(get_reference_vector(row["Reference_axis"]))

    if np.linalg.norm(vector) == 0 or np.linalg.norm(ref_vec) == 0:
        return np.nan

    cos_theta = np.dot(vector, ref_vec) / (np.linalg.norm(vector) * np.linalg.norm(ref_vec))
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    return -angle if np.cross(vector, ref_vec) < 0 else angle


# ------------------ Main Processing ------------------ #
def compute_smoothed_angle_deviation(df, rolling_window):
    li_sess = []
    for sess in df["Session"].unique():
        df_sess = df[df["Session"] == sess].copy().reset_index(drop=True)
        df_sess["dx"] = df_sess["x"].diff()
        df_sess["dy"] = df_sess["y"].diff()
        df_sess["dx_smooth"] = df_sess["dx"].rolling(rolling_window, center=True, min_periods=1).mean()
        df_sess["dy_smooth"] = df_sess["dy"].rolling(rolling_window, center=True, min_periods=1).mean()
        df_sess["Targeted_Angle_smooth"] = df_sess.apply(calculate_deviation, axis=1)
        li_sess.append(df_sess)

    df_result = pd.concat(li_sess).dropna().reset_index(drop=True)
    df_result["Targeted_Angle_smooth_abs"] = np.abs(df_result["Targeted_Angle_smooth"])
    return df_result


def assign_reference_info(df, close_ref_dict, xy_mapping):
    df = df.copy()
    df["Closest_Node"] = df["Grid Number"].apply(lambda x: map_category(x, close_ref_dict))
    df["x_mean"] = df.groupby("Closest_Node")["x"].transform("mean")
    df["y_mean"] = df.groupby("Closest_Node")["y"].transform("mean")
    df["Reference_axis"] = df["Grid Number"].apply(lambda x: map_category(x, xy_mapping))
    return df


def compute_unsmoothed_angle_deviation(df: pd.DataFrame) -> pd.DataFrame:
    li_dev_angle = []
    for sess in df["Session"].unique():
        df_sess = df[df["Session"] == sess].copy().reset_index(drop=True)
        df_sess["dx"] = df_sess["x"].diff()
        df_sess["dy"] = df_sess["y"].diff()
        df_sess["Targeted_Angle"] = df_sess.apply(calculate_deviation, axis=1)
        li_dev_angle.append(df_sess)

    df_dev = pd.concat(li_dev_angle).dropna().reset_index(drop=True)
    # df_dev['Targeted_Angle_abs'] = np.abs(df_dev['Targeted_Angle'])
    return df_dev


# ============================================================
# Value-Based Distance
# ============================================================
def compute_center_coordinates(df: pd.DataFrame, grid_nodes: list) -> tuple:
    """
    Compute the mean (x, y) position of specified grid nodes.
    """
    x_mean = df[df["Grid Number"].isin(grid_nodes)]["x"].mean()
    y_mean = df[df["Grid Number"].isin(grid_nodes)]["y"].mean()
    return x_mean, y_mean


def compute_euclidean_distance(df, center_x, center_y, out_col="Targeted_Distance"):
    """
    Compute Euclidean distance from each (x, y) to a center (x_mean, y_mean).
    """
    df[out_col] = np.sqrt((df["x"] - center_x) ** 2 + (df["y"] - center_y) ** 2)
    return df


def merge_value_map(df, value_map_path):
    value_map = pd.read_csv(value_map_path)
    print("Value map columns:", value_map.columns)  # Debug

    # Normalize "Value" column name
    colmap = {c: "Value" for c in value_map.columns if c.lower().startswith("value")}
    value_map = value_map.rename(columns=colmap)

    # Force Grid Number to int in both
    df["Grid Number"] = df["Grid Number"].astype(int)
    value_map["Grid Number"] = value_map["Grid Number"].astype(int)

    df = pd.merge(df, value_map, on="Grid Number", how="left")

    if "Value" not in df.columns:
        raise KeyError(f"'Value' column missing after merge. Columns: {df.columns.tolist()}")

    return df


def compute_weighted_and_normalized_distance(
    df: pd.DataFrame,
    distance_col: str = "Targeted_Distance",
    value_col: str = "Value",
    out_col: str = "VB_Distance",
    norm_range: tuple = (0, 3),
) -> pd.DataFrame:
    """
    Compute weighted distance and normalize it using MinMaxScaler.
    """
    df[out_col] = df[distance_col] * df[value_col]
    scaler = MinMaxScaler(feature_range=norm_range)
    df[out_col] = scaler.fit_transform(df[[out_col]])
    return df


def compute_value_distance(
    df: pd.DataFrame,
    value_map_path: str,
    center_grids: list = [84, 85],
) -> pd.DataFrame:
    """
    Full pipeline for value-based distance computation:
    1. Compute center of target grid(s)
    2. Compute Euclidean distance to center
    3. Merge with value map
    4. Compute weighted and normalized distances
    """
    x_mean, y_mean = compute_center_coordinates(df, center_grids)
    df = compute_euclidean_distance(df, x_mean, y_mean)
    df = merge_value_map(df, value_map_path)
    df = compute_weighted_and_normalized_distance(df)
    return df
