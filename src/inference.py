"""GraphRiverCast inference script.

Runs a single forward pass with a pre-trained checkpoint to predict
river discharge (Q), water depth (H), and channel storage (S).

Supports both NetCDF4 (.nc) and legacy NumPy (.npz) input formats.
Output is saved as NetCDF4 (predictions.nc).

Usage:
    python -m src.inference \\
        --checkpoint checkpoints/pretrain/GRC_ColdStart.ckpt \\
        --data-dir ./data/global \\
        --start-date 2015-01-01 \\
        --history 365 --future 365 \\
        --device cuda --output-dir ./output
"""

import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import torch

from src.data_utils import (
    DYNAMIC_VARIABLES,
    EDGE_ATTR_INDICES,
    build_static_features,
    compute_edge_attrs,
    days_index_2000,
    load_data_auto,
    normalize_static,
    sign_log1p,
    stack_dynamic_variables,
)
from src.model import GraphRiverCast


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GraphRiverCast inference: predict river hydrodynamics"
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .ckpt file")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing dynamic_var.npz, static_var.npz, edge_index.npy")
    p.add_argument("--output-dir", type=str, default="./output",
                   help="Directory to save predictions (default: ./output)")

    p.add_argument("--start-date", type=str, default="2015-01-01",
                   help="Start date for history window (YYYY-MM-DD)")
    p.add_argument("--history", type=int, default=365,
                   help="History window length in days (default: 365)")
    p.add_argument("--future", type=int, default=365,
                   help="Future prediction window in days (default: 365)")

    p.add_argument("--device", default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device (default: cuda)")
    p.add_argument("--pre-transform", default="none",
                   choices=["none", "sign_log1p"],
                   help="Pre-transform for static/edge features (default: none)")
    p.add_argument("--dynamic-std", default="node_wise",
                   choices=["node_wise", "global_wise"],
                   help="Dynamic variable standardization (must match training)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═══════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path, device):
    """Load model configuration and weights from a Lightning checkpoint.

    Returns:
        cfg:   model config dict
        state: cleaned state_dict (without 'net.' prefix)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "hyper_parameters" not in checkpoint:
        raise KeyError("Checkpoint missing 'hyper_parameters'")

    hp = checkpoint["hyper_parameters"]["model_arch"]
    cfg = None
    for key in ["GraphRiverCast_V2", "GRC_V2"]:
        if key in hp:
            cfg = hp[key].get("cfg")
            break
    if cfg is None:
        raise KeyError(f"No supported model found in checkpoint. Keys: {list(hp.keys())}")

    cfg.setdefault("spatial_num_layer", 2)
    cfg.setdefault("edge_feat_dim", 9)

    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {}
    for k, v in raw_state.items():
        if k.startswith("net."):
            state[k[4:]] = v
        elif not k.startswith("criterion"):
            state[k] = v

    return cfg, state


# ═══════════════════════════════════════════════════════════════════
#  Data preparation
# ═══════════════════════════════════════════════════════════════════

def prepare_data(data_dir, start_date, history, future,
                 pre_transform='none', standard_wise='node_wise'):
    """Load and prepare input data for inference.

    Args:
        data_dir:      directory with dynamic_var.npz, static_var.npz, edge_index.npy
        start_date:    start of history window (YYYY-MM-DD)
        history:       history window length in days
        future:        prediction window length in days
        pre_transform: 'none' or 'sign_log1p'
        standard_wise: 'node_wise' or 'global_wise'

    Returns:
        dict with all tensors needed for model forward pass
    """
    dynamic_dict, static_dict, edge_index, time_start = load_data_auto(data_dir)
    if time_start is None:
        time_start = '2000-01-01'

    static_raw, meanstd_dynamic = build_static_features(
        static_dict, dynamic_dict, DYNAMIC_VARIABLES, standard_wise)
    pos_edge_attr, neg_edge_attr = compute_edge_attrs(
        static_raw, edge_index, **EDGE_ATTR_INDICES)

    if pre_transform == 'sign_log1p':
        static_raw = sign_log1p(static_raw)
        pos_tf = sign_log1p(pos_edge_attr)
        neg_tf = sign_log1p(neg_edge_attr)
        all_tf = np.concatenate([pos_tf, neg_tf], axis=0)
        ea_mean = all_tf.mean(axis=0)
        ea_std = all_tf.std(axis=0)
        ea_std[ea_std == 0] = 1e-8
        pos_edge_attr = (pos_tf - ea_mean) / ea_std
        neg_edge_attr = (neg_tf - ea_mean) / ea_std

    static_norm, meanstd_static = normalize_static(static_raw)

    dynamic_all = stack_dynamic_variables(dynamic_dict, DYNAMIC_VARIABLES)

    seq_len = history + future
    y, m, d = map(int, start_date.split("-"))
    pred_s = days_index_2000(y, m, d)
    pred_e = pred_s + seq_len - 1
    predictset = dynamic_all[pred_s: pred_e + 1]

    d_mean = meanstd_dynamic['mean']
    d_std = meanstd_dynamic['std']
    predict_norm = (predictset - d_mean) / d_std

    river_ch = 3
    hist_arr = predict_norm[:history]
    fut_arr = predict_norm[history:]

    end_date = (dt.date(y, m, d) + dt.timedelta(days=seq_len - 1)).strftime("%Y-%m-%d")

    return {
        'river_hist': hist_arr[..., :river_ch],
        'river_fut': fut_arr[..., :river_ch],
        'runoff_hist': hist_arr[..., -1:],
        'runoff_fut': fut_arr[..., -1:],
        'static_var': static_norm,
        'edge_index': edge_index,
        'pos_edge_attr': pos_edge_attr,
        'neg_edge_attr': neg_edge_attr,
        'meanstd_dynamic': meanstd_dynamic,
        'meta': {
            'start_date': start_date,
            'end_date': end_date,
            'history': history,
            'future': future,
            'num_nodes': static_norm.shape[0],
            'num_edges': edge_index.shape[1],
        }
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"[GraphRiverCast] Device: {device}")
    print(f"[GraphRiverCast] Checkpoint: {args.checkpoint}")
    print(f"[GraphRiverCast] Data dir: {args.data_dir}")

    # ── Load model ──
    t0 = time.perf_counter()
    cfg, state = load_checkpoint(args.checkpoint, device)
    task = {
        "type": "predict",
        "paths": {"data_dir": args.data_dir},
        "window": {"predict": {"history": args.history, "future": args.future}},
    }
    model = GraphRiverCast(cfg, task)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[GraphRiverCast] Model loaded: {n_params:,} parameters")

    # ── Prepare data ──
    data = prepare_data(
        data_dir=args.data_dir,
        start_date=args.start_date,
        history=args.history,
        future=args.future,
        pre_transform=args.pre_transform,
        standard_wise=args.dynamic_std,
    )
    t_data = time.perf_counter() - t0
    print(f"[GraphRiverCast] Data prepared: {data['meta']['num_nodes']:,} nodes, "
          f"{data['meta']['num_edges']:,} edges ({t_data:.1f}s)")

    # ── Build model input ──
    def to_tensor(x):
        return torch.from_numpy(x).float().unsqueeze(0).to(device)

    inputs = {
        "river_hist": to_tensor(data['river_hist']),
        "runoff_hist": to_tensor(data['runoff_hist']),
        "runoff_fut": to_tensor(data['runoff_fut']),
        "static_var": to_tensor(data['static_var']),
        "edge_index": torch.from_numpy(data['edge_index']).long().to(device),
        "pos_edge_attr": to_tensor(data['pos_edge_attr']),
        "neg_edge_attr": to_tensor(data['neg_edge_attr']),
    }

    # ── Inference ──
    t_infer = time.perf_counter()
    with torch.no_grad():
        output = model(inputs)
    pred_norm = output["river_fut_hat"].squeeze(0).cpu().numpy()  # [T_fut, N, 3]
    t_infer = time.perf_counter() - t_infer

    # ── Denormalize ──
    d_mean = data['meanstd_dynamic']['mean'][:, :3]  # [N, 3]
    d_std = data['meanstd_dynamic']['std'][:, :3]
    pred_phys = pred_norm * d_std + d_mean

    gt_norm = data['river_fut']
    gt_phys = gt_norm * d_std + d_mean

    # ── Save ──
    out_path = os.path.join(args.output_dir, "predictions.nc")
    try:
        import xarray as xr
        T_f, N_f, _ = pred_phys.shape
        ds_out = xr.Dataset({
            'discharge': (['time', 'reach'], pred_phys[:, :, 0]),
            'water_depth': (['time', 'reach'], pred_phys[:, :, 1]),
            'storage': (['time', 'reach'], pred_phys[:, :, 2]),
            'discharge_truth': (['time', 'reach'], gt_phys[:, :, 0]),
            'water_depth_truth': (['time', 'reach'], gt_phys[:, :, 1]),
            'storage_truth': (['time', 'reach'], gt_phys[:, :, 2]),
        })
        ds_out.attrs['units'] = 'discharge: m3/s, water_depth: m, storage: m3'
        ds_out.attrs['model'] = 'GraphRiverCast'
        ds_out.to_netcdf(out_path, engine='netcdf4')
    except ImportError:
        out_path = os.path.join(args.output_dir, "predictions.npz")
        np.savez_compressed(
            out_path,
            predictions=pred_phys,
            ground_truth=gt_phys,
            variable_names=np.array(['discharge_m3s', 'water_depth_m', 'storage_m3']),
        )

    meta_path = os.path.join(args.output_dir, "inference_meta.json")
    meta = {
        **data['meta'],
        'checkpoint': os.path.basename(args.checkpoint),
        'num_parameters': n_params,
        'inference_time_s': round(t_infer, 2),
        'data_prep_time_s': round(t_data, 2),
        'device': str(device),
        'output_shape': list(pred_phys.shape),
        'variables': {
            'channel_0': 'discharge (m³/s)',
            'channel_1': 'water depth (m)',
            'channel_2': 'channel storage (m³)',
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[GraphRiverCast] Inference complete: {pred_phys.shape} "
          f"({t_infer:.1f}s on {device})")
    print(f"[GraphRiverCast] Predictions saved to {out_path}")
    print(f"[GraphRiverCast] Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
