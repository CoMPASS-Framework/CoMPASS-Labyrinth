from typing import Literal
import numpy as np
import pandas as pd


# -----------------------
# Helpers
# -----------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters for scalar or vector numpy arrays."""
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _euclid(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _turn_angle(p0, p1, p2):
    """
    Signed turning angle at p1 from segment p0->p1 to p1->p2.
    Returns angle in radians in (-pi, pi].
    """
    v1 = p1 - p0
    v2 = p2 - p1
    # handle degenerate
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return np.nan
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    v1u = v1 / n1
    v2u = v2 / n2
    dot = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
    ang = np.arccos(dot)
    # sign via 2D cross product z-component
    cross_z = v1u[0] * v2u[1] - v1u[1] * v2u[0]
    return np.sign(cross_z) * ang


def _dist_angle(prev_xy, cur_xy, target_xy, coord_type):
    """Return (dist, bearing angle at current segment toward target) analog to distAngle in R."""
    # distance from cur_xy to target
    if coord_type == "LL":
        dist = _haversine_m(prev_xy[1], prev_xy[0], target_xy[1], target_xy[0])  # lon=x, lat=y
    else:
        dist = _euclid(prev_xy[0], prev_xy[1], target_xy[0], target_xy[1])
    # angle from (prev->cur) to (cur->target)
    return dist, _turn_angle(np.array(prev_xy), np.array(cur_xy), np.array(target_xy))


# -----------------------
# Main function
# -----------------------
def prep_data(
    data: pd.DataFrame,
    type: Literal["UTM", "LL"] = "UTM",
    coordNames: tuple[str, str] = ("x", "y"),
    covNames: list[str] | None = None,
    centers: np.ndarray | None = None,
    centroids: dict | None = None,
    angleCovs: list[str] | None = None,
    altCoordNames: str | None = None,
) -> pd.DataFrame:
    """
    Python port of prepData.default (core functionality).
    - Computes step and angle per ID using coordinates.
    - Retains covariates and optionally adds center/centroid distance/angle features.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data.
    type : Literal["UTM", "LL"]
        Coordinate type, either "UTM" or "LL". Default is "UTM".
    coordNames : tuple[str, str]
        Names of the coordinate columns (x, y). Default is ("x", "y").
    covNames : list[str] | None
        List of covariate column names to retain. Default is None.
    centers : np.ndarray | None
        Optional (K,2) array of center coordinates to compute distance/angle to. Default is None.
    centroids : dict | None
        Dictionary mapping centroid names to their coordinates. Default is None.
    angleCovs : list[str] | None
        List of covariate names to include as angle covariates. Default is None.
    altCoordNames : str | None
        Alternative base name for output coordinate columns. Default is None.

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with step, angle, covariates, and optional center/centroid features.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if any(dim == 0 for dim in data.shape):
        raise ValueError("data is empty")
    if len(coordNames) != 2:
        raise ValueError("coordNames must be length 2")

    xcol, ycol = coordNames
    if xcol not in data.columns or ycol not in data.columns:
        raise ValueError("coordNames not found in data")

    # Sort data by ID, Session, and S_no, and remove rows with missing values
    data["ID"] = data["Session"].astype("category")
    data = data.sort_values(by=["ID", "Session", "S_no"]).dropna().reset_index(drop=True)

    # ID handling
    if "ID" in data.columns:
        ID = data["ID"].astype(str)
    else:
        ID = pd.Series(["Animal1"] * len(data), name="ID")
        data = data.copy()
        data["ID"] = ID

    # Validate contiguity per ID (assumes data already grouped/contiguous per ID like in R)
    # If not contiguous, won't reorder automatically; user should sort beforehand.

    # Compute step and angle
    df = data.copy()

    out_x, out_y = ("x", "y")
    if altCoordNames:
        out_x, out_y = f"{altCoordNames}.x", f"{altCoordNames}.y"

    df[out_x] = df[xcol]
    df[out_y] = df[ycol]

    df["step"] = np.nan
    df["angle"] = np.nan

    # Covariates
    covNames = [] if covNames is None else list(dict.fromkeys(covNames))
    angleCovs = [] if angleCovs is None else list(dict.fromkeys(angleCovs))
    cov_all = list(dict.fromkeys(covNames + angleCovs))
    for c in cov_all:
        if c not in df.columns:
            raise ValueError(f"covariate '{c}' not found in data")

    # Forward fill covariates per ID
    if cov_all:
        df[cov_all] = (
            df.groupby("ID", sort=False)[cov_all].apply(lambda g: g.ffill().bfill()).reset_index(level=0, drop=True)
        )

    # Step & angle per ID
    coord_type = "LL" if type.upper() == "LL" else "UTM"

    for _id, g in df.groupby("ID", sort=False):
        idx = g.index
        x = g[xcol].to_numpy()
        y = g[ycol].to_numpy()

        # --- step (same as before) ---
        if coord_type == "LL":
            step = np.full(len(x), np.nan)
            step[1:] = _haversine_m(y[:-1], x[:-1], y[1:], x[1:])
        else:
            step = np.full(len(x), np.nan)
            step[1:] = _euclid(x[:-1], y[:-1], x[1:], y[1:])
        df.loc[idx, "step"] = step

        # --- angle: compute at k (1..n-2), store at k+1 to match R's "shifted down" alignment ---
        angle = np.full(len(x), np.nan)
        for k in range(1, len(x) - 1):
            ang_k = _turn_angle(np.array([x[k - 1], y[k - 1]]), np.array([x[k], y[k]]), np.array([x[k + 1], y[k + 1]]))
            angle[k + 1] = ang_k  # <- shift down by one row (R-style)
        df.loc[idx, "angle"] = angle

    # Centers: fixed Kx2; add .dist and .angle
    if centers is not None:
        centers = np.asarray(centers)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers must be a (K,2) matrix")
        for j in range(centers.shape[0]):
            base = f"center{j+1}"
            dist_col = f"{base}.dist"
            ang_col = f"{base}.angle"
            df[dist_col] = np.nan
            df[ang_col] = np.nan
        # compute per row with previous point (like R: distance/angle uses prev->cur and cur->center)
        for _id, g in df.groupby("ID", sort=False):
            idx = g.index
            x = g[xcol].to_numpy()
            y = g[ycol].to_numpy()
            for j in range(centers.shape[0]):
                dist_vals = np.full(len(g), np.nan)
                ang_vals = np.full(len(g), np.nan)
                for k in range(1, len(g)):
                    d, a = _dist_angle((x[k - 1], y[k - 1]), (x[k], y[k]), (centers[j, 0], centers[j, 1]), coord_type)
                    dist_vals[k] = d
                    ang_vals[k] = a
                df.loc[idx, f"center{j+1}.dist"] = dist_vals
                df.loc[idx, f"center{j+1}.angle"] = ang_vals

    # Centroids: dict[name] -> DataFrame with columns ["x","y", time_col]; time_col must exist in df
    if centroids is not None:
        if not isinstance(centroids, dict):
            raise ValueError("centroids must be a dict of name -> DataFrame([x,y,time]))")
        for name, cdf in centroids.items():
            if not isinstance(cdf, pd.DataFrame) or not set(["x", "y"]).issubset(cdf.columns):
                raise ValueError(f"centroid '{name}' must be a DataFrame with columns ['x','y', time_col]")
            # find the third column as time
            time_cols = [c for c in cdf.columns if c not in ("x", "y")]
            if len(time_cols) != 1:
                raise ValueError(f"centroid '{name}' must have exactly one time column")
            tcol = time_cols[0]
            if tcol not in df.columns:
                raise ValueError(f"time column '{tcol}' for centroid '{name}' not found in data")

            # merge centroid xy onto df by time, then compute dist/angle
            cdf_use = cdf.rename(columns={"x": f"__{name}_x", "y": f"__{name}_y"})
            df = df.merge(cdf_use, how="left", left_on=tcol, right_on=tcol)

            dcol = f"{name}.dist"
            acol = f"{name}.angle"
            df[dcol] = np.nan
            df[acol] = np.nan

            for _id, g in df.groupby("ID", sort=False):
                idx = g.index
                x = g[xcol].to_numpy()
                y = g[ycol].to_numpy()
                cx = g[f"__{name}_x"].to_numpy()
                cy = g[f"__{name}_y"].to_numpy()
                dist_vals = np.full(len(g), np.nan)
                ang_vals = np.full(len(g), np.nan)
                for k in range(1, len(g)):
                    if np.isnan(cx[k]) or np.isnan(cy[k]):
                        continue
                    d, a = _dist_angle((x[k - 1], y[k - 1]), (x[k], y[k]), (cx[k], cy[k]), coord_type)
                    dist_vals[k] = d
                    ang_vals[k] = a
                df.loc[idx, dcol] = dist_vals
                df.loc[idx, acol] = ang_vals

            # drop the merged helper columns
            df.drop(columns=[f"__{name}_x", f"__{name}_y"], inplace=True)

    # Arrange final column order similar to R: ID, step, angle, covariates, coords (+ any added center/centroid cols)
    base_cols = ["ID", "step", "angle"]
    keep_covs = cov_all
    extra_cols = [c for c in df.columns if c.endswith(".dist") or c.endswith(".angle")]
    coord_cols = [out_x, out_y]

    ordered = base_cols + keep_covs + extra_cols + coord_cols
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder]

    # cast ID to category like the R factor
    df["ID"] = df["ID"].astype("category")

    df.attrs["coords"] = coord_cols

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    return df
