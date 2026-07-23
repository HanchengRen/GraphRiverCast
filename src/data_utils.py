"""Data loading and processing utilities for GraphRiverCast inference.

Handles construction of static features, edge attributes, normalization,
and pre-transforms needed to prepare input data for the model.

Supported data formats:
    NetCDF4 (.nc) — single file containing all variables (recommended)
    Legacy NPZ    — dynamic_var.npz + static_var.npz + edge_index.npy
"""

import os
import numpy as np
from datetime import date


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

DYNAMIC_VARIABLES = ['outflw', 'rivdph', 'storage', 'runoff']

STATIC_VARIABLES = ['ctmare', 'elevtn', 'rivhgt', 'rivman', 'grdare', 'nxtdst',
                    'rivlen', 'rivwth_gwdlr', 'uparea', 'width', 'fldhgt']

EDGE_ATTR_INDICES = dict(
    idx_elevtn=1, idx_rivhgt=2, idx_rivman=3, idx_nxtdst=5, idx_uparea=8, idx_width=7,
)


# ═══════════════════════════════════════════════════════════════════
#  Pre-transforms
# ═══════════════════════════════════════════════════════════════════

def sign_log1p(x):
    """Zero-parameter nonlinear compression: sign(x) * log1p(|x|).

    Compresses long-tailed distributions while preserving sign.
    Approximates identity for small values, log for large values.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.sign(x) * np.log1p(np.abs(x))
    return out.astype(np.float32)


def sign_log1p_inverse(y):
    """Inverse of sign_log1p: sign(y) * expm1(|y|)."""
    y = np.asarray(y, dtype=np.float64)
    out = np.sign(y) * np.expm1(np.abs(y))
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Static features
# ═══════════════════════════════════════════════════════════════════

def build_static_features(static_npz, dynamic_npz, dynamic_variables=None,
                          standard_wise='node_wise'):
    """Build the 28-dim static feature matrix [N, 28] (unnormalized).

    Concatenation order:
        10 scalar geomorphic variables + 8 dynamic statistics (mean/std
        interleaved for each of 4 dynamic variables) + 10-dim fldhgt.

    Also returns per-node normalization statistics for dynamic variables,
    computed from the full time series.

    Args:
        static_npz:  np.load result of static_var.npz
        dynamic_npz: np.load result of dynamic_var.npz
        dynamic_variables: list of variable names (default: DYNAMIC_VARIABLES)
        standard_wise: 'node_wise' or 'global_wise' normalization

    Returns:
        static_dataset:  [N, 28] float32, unnormalized
        meanstd_dynamic: {'mean': [N, V], 'std': [N, V]} for z-score
    """
    if dynamic_variables is None:
        dynamic_variables = DYNAMIC_VARIABLES

    dyn_means = []
    dyn_stds = []
    for var in dynamic_variables:
        arr = dynamic_npz[var]  # [T, N]
        dyn_means.append(arr.mean(axis=0, keepdims=True).T)  # [N, 1]
        dyn_stds.append(arr.std(axis=0, keepdims=True).T)

    mean_nv = np.concatenate(dyn_means, axis=-1).astype(np.float32)  # [N, V]
    std_nv = np.concatenate(dyn_stds, axis=-1).astype(np.float32)
    std_nv[std_nv == 0] = 1e-8

    if standard_wise == 'global_wise':
        N = mean_nv.shape[0]
        global_means = [dynamic_npz[v].mean() for v in dynamic_variables]
        global_stds = [dynamic_npz[v].std() for v in dynamic_variables]
        global_mean = np.array(global_means, dtype=np.float32)
        global_std = np.array(global_stds, dtype=np.float32)
        global_std[global_std == 0] = 1e-8
        meanstd_dynamic = {
            'mean': np.tile(global_mean, (N, 1)),
            'std': np.tile(global_std, (N, 1)),
        }
    elif standard_wise == 'node_wise':
        meanstd_dynamic = {'mean': mean_nv, 'std': std_nv}
    else:
        raise ValueError(f"Unknown standard_wise: {standard_wise}")

    dyn_stats_interleaved = []
    for m, s in zip(dyn_means, dyn_stds):
        dyn_stats_interleaved.extend([m, s])

    static_arrays = [arr[:, np.newaxis] if arr.ndim == 1 else arr
                     for arr in [static_npz[key] for key in STATIC_VARIABLES]]
    _GRDARE_IDX = STATIC_VARIABLES.index('grdare')
    static_arrays[_GRDARE_IDX] = static_arrays[_GRDARE_IDX] / 1e6

    static_arrays = static_arrays[:-1] + dyn_stats_interleaved + [static_arrays[-1]]
    static_dataset = np.concatenate(static_arrays, axis=-1).astype(np.float32)
    return static_dataset, meanstd_dynamic


# ═══════════════════════════════════════════════════════════════════
#  Dynamic variables
# ═══════════════════════════════════════════════════════════════════

def stack_dynamic_variables(dynamic_npz, dynamic_variables=None):
    """Stack dynamic variables into [T, N, V] array.

    Args:
        dynamic_npz: np.load result of dynamic_var.npz
        dynamic_variables: list of variable names (default: DYNAMIC_VARIABLES)

    Returns:
        [T, N, V] float32 array
    """
    if dynamic_variables is None:
        dynamic_variables = DYNAMIC_VARIABLES
    return np.stack(
        [dynamic_npz[key] for key in dynamic_variables], axis=-1
    ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════════════════════

def normalize_static(static_dataset):
    """Z-score normalize the static feature matrix.

    Args:
        static_dataset: [N, D] unnormalized

    Returns:
        static_normed:  [N, D] normalized
        meanstd_static: {'mean': [D], 'std': [D]}
    """
    meanstd_static = {
        'mean': np.mean(static_dataset, axis=0),
        'std': np.std(static_dataset, axis=0),
    }
    meanstd_static['std'][meanstd_static['std'] == 0] = 1e-8
    static_normed = (static_dataset - meanstd_static['mean']) / meanstd_static['std']
    return static_normed, meanstd_static


# ═══════════════════════════════════════════════════════════════════
#  Edge attributes
# ═══════════════════════════════════════════════════════════════════

def compute_edge_attrs(static_data, edge_index,
                       idx_elevtn=1, idx_rivhgt=2, idx_rivman=3,
                       idx_nxtdst=5, idx_uparea=8, idx_width=7,
                       normalize=True):
    """Compute 8-dim physical edge attributes from raw static variables.

    Edge attributes encode hydraulic-geometric properties:
        [0] slope       — ground slope (permille)
        [1] bed_slope   — riverbed slope (permille)
        [2] elev_diff   — ground elevation difference (m)
        [3] nxtdst      — inter-node distance (m)
        [4] rivman      — Manning roughness coefficient
        [5] area_ratio  — upstream catchment area ratio
        [6] width_ratio — channel width ratio
        [7] rivhgt_ratio — channel depth ratio

    Args:
        static_data: [N, D] unnormalized static feature matrix
        edge_index:  [2, E] directed edges
        idx_*:       column indices in static_data
        normalize:   whether to z-score normalize (default True)

    Returns:
        pos_edge_attr: [E, 8] float32 (forward: upstream → downstream)
        neg_edge_attr: [E, 8] float32 (reverse: downstream → upstream)
    """
    src, dst = edge_index[0], edge_index[1]

    elev_src = static_data[src, idx_elevtn]
    elev_dst = static_data[dst, idx_elevtn]
    rivhgt_src = static_data[src, idx_rivhgt]
    rivhgt_dst = static_data[dst, idx_rivhgt]
    nxtdst_src = static_data[src, idx_nxtdst]
    area_src = static_data[src, idx_uparea]
    area_dst = static_data[dst, idx_uparea]
    width_src = static_data[src, idx_width]
    width_dst = static_data[dst, idx_width]
    rivman_src = static_data[src, idx_rivman]

    bed_elev_src = elev_src - rivhgt_src
    bed_elev_dst = elev_dst - rivhgt_dst
    _SLOPE_SCALE = 1000

    pos_edge_attr = np.stack([
        (elev_src - elev_dst) / nxtdst_src * _SLOPE_SCALE,
        (bed_elev_src - bed_elev_dst) / nxtdst_src * _SLOPE_SCALE,
        elev_src - elev_dst,
        nxtdst_src,
        rivman_src,
        area_dst / area_src,
        width_dst / width_src,
        rivhgt_dst / rivhgt_src,
    ], axis=-1).astype(np.float32)

    neg_edge_attr = np.stack([
        (elev_dst - elev_src) / nxtdst_src * _SLOPE_SCALE,
        (bed_elev_dst - bed_elev_src) / nxtdst_src * _SLOPE_SCALE,
        elev_dst - elev_src,
        nxtdst_src,
        rivman_src,
        area_src / area_dst,
        width_src / width_dst,
        rivhgt_src / rivhgt_dst,
    ], axis=-1).astype(np.float32)

    if normalize:
        all_attrs = np.concatenate([pos_edge_attr, neg_edge_attr], axis=0)
        ea_mean = all_attrs.mean(axis=0)
        ea_std = all_attrs.std(axis=0)
        ea_std[ea_std == 0] = 1e-8
        pos_edge_attr = (pos_edge_attr - ea_mean) / ea_std
        neg_edge_attr = (neg_edge_attr - ea_mean) / ea_std

    return pos_edge_attr, neg_edge_attr


# ═══════════════════════════════════════════════════════════════════
#  Date utilities
# ═══════════════════════════════════════════════════════════════════

def days_index_2000(year, month, day):
    """Convert a date to the number of days since 2000-01-01."""
    return (date(int(year), int(month), int(day)) - date(2000, 1, 1)).days


def date_to_index(date_str, time_start='2000-01-01'):
    """Convert a date string to array index relative to time_start."""
    target = date.fromisoformat(date_str)
    origin = date.fromisoformat(time_start)
    idx = (target - origin).days
    if idx < 0:
        raise ValueError(f"Target date {date_str} is before data start {time_start}")
    return idx


# ═══════════════════════════════════════════════════════════════════
#  NetCDF4 data loading
# ═══════════════════════════════════════════════════════════════════

def load_from_netcdf(nc_path):
    """Load all model inputs from a single NetCDF4 file.

    Returns:
        dynamic_dict:  {var_name: [T, N] array}
        static_dict:   {var_name: [N] or [N, D] array}
        edge_index:    [2, E] int64 array
        time_start:    date string or None
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path, engine="netcdf4")

    dynamic_dict = {key: ds[key].values for key in DYNAMIC_VARIABLES if key in ds}
    static_keys = [k for k in STATIC_VARIABLES if k in ds and k != 'fldhgt']
    static_dict = {key: ds[key].values for key in static_keys}
    if 'fldhgt' in ds:
        static_dict['fldhgt'] = ds['fldhgt'].values

    edge_index = ds['edge_index'].values.astype(np.int64)
    time_start = ds.attrs.get('time_start', None)
    ds.close()
    return dynamic_dict, static_dict, edge_index, time_start


def load_data_auto(data_dir):
    """Auto-detect data format (NetCDF4 or legacy NPZ) and load.

    Looks for *_sim.nc or *.nc files first; falls back to npz.

    Returns:
        dynamic_dict, static_dict, edge_index, time_start
    """
    import glob as glob_mod
    nc_files = glob_mod.glob(os.path.join(data_dir, "*_sim.nc"))
    if not nc_files:
        nc_files = glob_mod.glob(os.path.join(data_dir, "*.nc"))

    if nc_files:
        return load_from_netcdf(nc_files[0])

    dynamic_npz = np.load(os.path.join(data_dir, "dynamic_var.npz"))
    static_npz = np.load(os.path.join(data_dir, "static_var.npz"))
    edge_index = np.load(os.path.join(data_dir, "edge_index.npy"))
    dynamic_dict = {k: dynamic_npz[k] for k in DYNAMIC_VARIABLES if k in dynamic_npz}
    static_dict = {k: static_npz[k] for k in STATIC_VARIABLES if k in static_npz}
    return dynamic_dict, static_dict, edge_index, '2000-01-01'
