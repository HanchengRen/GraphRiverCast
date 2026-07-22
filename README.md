<div align="center">

# GraphRiverCast

[![arXiv](https://img.shields.io/badge/arXiv-2602.22293-b31b1b.svg)](https://arxiv.org/abs/2602.22293)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG 2.5+](https://img.shields.io/badge/PyG-2.5%2B-7B3FA0.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Topology enables learning-based prediction of global river hydrodynamics**

[Hancheng Ren](mailto:)<sup>1,2</sup>,
[Gang Zhao](mailto:zhao.g.eb91@m.isct.ac.jp)<sup>1,3,†</sup>,
[Shuo Wang](mailto:)<sup>4</sup>,
[Louise Slater](mailto:)<sup>2</sup>,
[Dai Yamazaki](mailto:)<sup>5</sup>,
Shu Liu<sup>6</sup>,
Jingfang Fan<sup>4</sup>,
Shibo Cui<sup>7</sup>,
Ziming Yu<sup>8</sup>,
Shengyu Kang<sup>9</sup>,
Depeng Zuo<sup>1</sup>,
Dingzhi Peng<sup>1</sup>,
Zongxue Xu<sup>1</sup>,
[Bo Pang](mailto:pb@bnu.edu.cn)<sup>1,†</sup>

<sup>1</sup>Beijing Normal University &nbsp;
<sup>2</sup>University of Oxford &nbsp;
<sup>3</sup>Institute of Science Tokyo &nbsp;
<sup>4</sup>Beijing Normal University (Systems Science) &nbsp;
<sup>5</sup>University of Tokyo &nbsp;
<sup>6</sup>China Institute of Water Resources and Hydropower Research &nbsp;
<sup>7</sup>Tsinghua University &nbsp;
<sup>8</sup>Beijing Normal University (AI) &nbsp;
<sup>9</sup>Wuhan University

<sup>†</sup> Corresponding authors

arXiv preprint, 2026

---

</div>

<p align="center">
  <img src="figures/figure1.png" width="100%" alt="GraphRiverCast Framework">
</p>
<p align="center"><b>Figure 1.</b> GraphRiverCast enables state-free, zero-shot prediction of global river hydrodynamics. <b>a,</b> The global river network (127,581 reaches, ~4.4 million km). <b>b,</b> Dynamical rationale: rivers are strongly dissipative — initial-state influence decays, making state-free (ColdStart) prediction tractable. Architecture: feature encoder + graph encoder (signed bidirectional GCN) + temporal encoder (LSTM). <b>c,</b> Two-stage training: pretrain on CaMa-Flood simulations → finetune on sparse GRDC gauges → zero-shot global inference.</p>

---

## Highlights

- **State-free prediction (ColdStart)** — reconstructs discharge, water depth and channel storage across 127,581 river reaches worldwide without any historical river states
- **Single unified global model** — one model for the entire planet, not a patchwork of basin-specific models
- **~97K parameters** — lightweight enough to run inference on a consumer GPU
- **ColdStart NSE 0.936** at 13-day lead, sustaining skill out to 700+ days (100× beyond training horizon)
- **Topology-dominant** — encoder ablation shows graph topology contributes 50% of ColdStart skill vs 12% for temporal memory
- **Pretrain → finetune paradigm** — pretrained on physics-based CaMa-Flood simulations, finetuned with 1,996 sparse GRDC gauge stations

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Global Performance](#-global-performance)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Inference](#-inference)
- [Checkpoints](#-checkpoints)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/HanchengRen/GraphRiverCast.git
cd GraphRiverCast

# 2. Install
conda env create -f environment.yml
conda activate graphrivercast

# 3. Run inference (prepare your data first — see Data Preparation)
python -m src.inference \
    --checkpoint checkpoints/GRC_ColdStart.ckpt \
    --data-dir ./data/global \
    --start-date 2015-01-01 \
    --history 365 --future 365
```

---

## 🌊 Overview

Rivers are central to the global water cycle, yet ~60% of basins lack long-term records needed to initialise standard autoregressive models. GraphRiverCast exploits the **strongly dissipative** nature of rivers — unlike the chaotic atmosphere, river initial-condition perturbations decay rather than amplify — enabling **state-free prediction** from runoff forcing and network topology alone.

### ColdStart vs HotStart

| Regime | Initial State | Input Data | Use Case |
|:---|:---|:---|:---|
| **ColdStart** | Zero (no historical states) | Runoff forcing + topology + geomorphic features | Ungauged reaches, long-horizon prediction |
| **HotStart** | From simulation/observation | All of the above + historical river states | Short-term forecasting in gauged reaches |

> **Key finding:** At 13-day lead time, ColdStart (median NSE = 0.936) matches HotStart (0.932). Beyond day 13, ColdStart sustains skill while HotStart degrades to ~0.4 by day 30 due to autoregressive error compounding. ColdStart maintains NSE ≈ 0.93 out to 700 days — a 100× extrapolation beyond the 7-day training horizon.

---

## 🧬 Model Architecture

GraphRiverCast is a spatio-temporal graph neural network with three complementary encoders:

```
Input (per reach, per timestep)
  │
  ├── Runoff forcing [1]       ─┐
  ├── River state (Q, H, S) [3] ├──► Feature Encoder ──► Embedding [H]
  └── Static features [28]    ─┘           │
                                           ▼
                              ┌── Feature Mixing (SiLU + MLP) ──┐
                              │                                  │
                              ▼                                  │
                     Signed GCN Encoder                          │
                    ┌────────────────────┐                       │
                    │ Upstream → Downstream (pos path)           │
                    │ Downstream → Upstream (neg path)           │
                    │ Edge-attribute gating (9-dim)              │
                    │ Per-node fusion gate                       │
                    └────────────────────┘                       │
                              │                                  │
                              ▼                                  │
                     Temporal Encoder                            │
                    ┌────────────────────┐                       │
                    │ LSTMCell + RMSNorm                         │
                    │ h_t and c_t normalization                  │
                    └────────────────────┘                       │
                              │                                  │
                              ▼                                  │
                     Residual Readout ◄──────────────────────────┘
                              │
                              ▼
              Output: ΔQ, ΔH, ΔS (added to current state)
```

### Key Components

| Component | Description |
|:---|:---|
| **EdgeAttrConv** | Message passing with edge-attribute gating: `msg = Linear(x_j) × σ(EdgeMLP(edge_attr))` |
| **SignedGCN Encoder** | Dual-path graph convolution — independent weights for upstream→downstream and downstream→upstream directions |
| **9-dim Edge Attributes** | 8 static hydraulic-geometric features (slope, bed slope, elevation diff, distance, Manning n, area/width/depth ratios) + 1 dynamic water-depth gradient |
| **Fusion Gate** | Per-node learned gate blending upstream and downstream representations: `x = x + g·h_pos + (1−g)·h_neg` |
| **LSTMCell** | Single-step temporal encoder with RMSNorm on both hidden and cell states |
| **Residual Readout** | Predicts state change Δ(Q, H, S), added to current state for autoregressive rollout |

### Model Specifications

| Parameter | Value |
|:---|:---|
| Hidden dimension | 64 |
| Feature mixing dimension | 128 |
| GCN layers | 2 |
| Total parameters | 96,643 |
| Input features | 32 (1 runoff + 3 river state + 28 static) |
| Output channels | 3 (Q, H, S) |
| Edge attribute dimension | 9 (8 static + 1 dynamic) |

---

## 🌍 Global Performance

<p align="center">
  <img src="figures/figure2.png" width="100%" alt="Global Performance">
</p>
<p align="center"><b>Figure 2.</b> Global performance of ColdStart and HotStart regimes at 13-day lead time. Both regimes achieve comparable skill across the global river network, with ColdStart slightly outperforming HotStart in humid climates where runoff is continuous.</p>

### Pre-training Performance (CaMa-Flood validation)

| Regime | Median Combined NSE | Q NSE | H NSE | S NSE |
|:---|:---:|:---:|:---:|:---:|
| ColdStart (13-day lead) | **0.936** | 0.93 | 0.95 | 0.94 |
| HotStart (13-day lead) | 0.932 | 0.93 | 0.94 | 0.93 |

### Encoder Ablation: What drives prediction?

| Encoder | ColdStart contribution | HotStart contribution |
|:---|:---:|:---:|
| **Graph (topology)** | **50%** | 22% |
| Feature | 38% | 19% |
| Temporal | 12% | **59%** |

> ColdStart shifts the model from temporal autoregression to **topology-governed routing** — the graph encoder becomes the dominant contributor when historical states are removed.

---

## 💻 Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate graphrivercast
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch Geometric (`torch_geometric`) and `torch_scatter` may require platform-specific installation. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Option 3: Setup Scripts

```bash
# Linux / macOS
bash scripts/setup_unix.sh

# Windows
scripts\setup_windows.bat
```

### Supported Platforms

| Platform | GPU | Status |
|:---|:---|:---:|
| Linux (Ubuntu 20.04+) | NVIDIA CUDA 11.8+ | ✅ Fully tested |
| Linux | CPU-only | ✅ |
| Windows 10/11 | NVIDIA CUDA 11.8+ | ✅ |
| macOS (Apple Silicon) | MPS | ⚠️ CPU fallback |
| macOS (Intel) | CPU-only | ✅ |

---

## 📊 Data Preparation

GraphRiverCast requires three input files placed in a data directory:

### Required Files

<table>
<tr><th>File</th><th>Format</th><th>Description</th></tr>
<tr>
  <td><code>dynamic_var.npz</code></td>
  <td>NumPy compressed</td>
  <td>Time-varying river variables and forcing</td>
</tr>
<tr>
  <td><code>static_var.npz</code></td>
  <td>NumPy compressed</td>
  <td>Geomorphic and channel geometry features</td>
</tr>
<tr>
  <td><code>edge_index.npy</code></td>
  <td>NumPy array</td>
  <td>River network topology (directed graph)</td>
</tr>
</table>

### Dynamic Variables (`dynamic_var.npz`)

Each variable is a `float64` array of shape `[T, N]` where `T` = number of daily timesteps and `N` = number of river reaches.

| Key | Variable | Unit | Description |
|:---|:---|:---|:---|
| `outflw` | Discharge | m³/s | River discharge at each reach |
| `rivdph` | Water depth | m | Water depth in the main channel |
| `storage` | Channel storage | m³ | Total water volume stored in the reach |
| `runoff` | Runoff | m³/s | Lateral runoff forcing into each reach |

### Static Variables (`static_var.npz`)

Each variable is an array of shape `[N]` or `[N, D]`.

| Key | Variable | Unit | Shape |
|:---|:---|:---|:---|
| `ctmare` | Catchment area | m² | [N] |
| `elevtn` | Ground elevation | m | [N] |
| `rivhgt` | Channel depth (bankfull) | m | [N] |
| `rivman` | Manning roughness | - | [N] |
| `grdare` | Grid area | m² | [N] |
| `nxtdst` | Distance to next downstream | m | [N] |
| `rivlen` | Channel length | m | [N] |
| `rivwth_gwdlr` | Channel width | m | [N] |
| `uparea` | Upstream drainage area | m² | [N] |
| `width` | Bankfull width | m | [N] |
| `fldhgt` | Floodplain height profile | m | [N, 10] |

### Edge Index (`edge_index.npy`)

- Shape: `[2, E]` where `E` = number of edges
- Row 0: source node indices (upstream)
- Row 1: target node indices (downstream)
- Directed edges following natural flow direction

### Data Sources

| Data | Source | Reference |
|:---|:---|:---|
| River network & geomorphic features | MERIT Hydro / CaMa-Flood | [Yamazaki et al., 2019](https://doi.org/10.1029/2019WR024873); [Yamazaki et al., 2011](https://doi.org/10.1029/2010WR009726) |
| Runoff forcing | GRADES | [Lin et al., 2019](https://doi.org/10.1029/2019WR025287) |
| Gauge observations (fine-tuning) | GRDC | [grdc.bafg.de](https://grdc.bafg.de) |
| River simulations (pre-training) | CaMa-Flood v4 | [Yamazaki et al., 2011](https://doi.org/10.1029/2010WR009726) |

### Creating Data Files

```python
import numpy as np

# Example: create dynamic_var.npz
# T = number of daily timesteps, N = number of river reaches
T, N = 7305, 127581  # e.g., 20 years of daily data

np.savez("data/global/dynamic_var.npz",
    outflw=np.zeros((T, N), dtype=np.float64),   # discharge [m³/s]
    rivdph=np.zeros((T, N), dtype=np.float64),    # water depth [m]
    storage=np.zeros((T, N), dtype=np.float64),   # storage [m³]
    runoff=np.zeros((T, N), dtype=np.float64),    # runoff forcing [m³/s]
)

# Example: create edge_index.npy
E = 127580  # number of edges
edge_index = np.zeros((2, E), dtype=np.int64)
np.save("data/global/edge_index.npy", edge_index)
```

---

## 🔮 Inference

### Basic Usage

```bash
python -m src.inference \
    --checkpoint checkpoints/GRC_ColdStart.ckpt \
    --data-dir ./data/global \
    --start-date 2015-01-01 \
    --history 365 --future 365 \
    --output-dir ./output
```

### ColdStart vs HotStart

```bash
# ColdStart: state-free prediction (recommended for most use cases)
python -m src.inference \
    --checkpoint checkpoints/GRC_ColdStart.ckpt \
    --data-dir ./data/global \
    --start-date 2015-01-01 \
    --history 365 --future 365

# HotStart: autoregressive prediction with initial states
python -m src.inference \
    --checkpoint checkpoints/GRC_HotStart.ckpt \
    --data-dir ./data/global \
    --start-date 2015-01-01 \
    --history 365 --future 365
```

### Command-Line Arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--checkpoint` | *(required)* | Path to `.ckpt` file |
| `--data-dir` | *(required)* | Directory with input data files |
| `--output-dir` | `./output` | Directory for prediction outputs |
| `--start-date` | `2015-01-01` | Start of history window (YYYY-MM-DD) |
| `--history` | `365` | History window in days |
| `--future` | `365` | Prediction window in days |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--pre-transform` | `none` | `none` or `sign_log1p` |
| `--dynamic-std` | `node_wise` | `node_wise` or `global_wise` |

### Output Format

Predictions are saved as `predictions.npz` with:

| Key | Shape | Description |
|:---|:---|:---|
| `predictions` | [T_future, N, 3] | Predicted Q, H, S in physical units |
| `ground_truth` | [T_future, N, 3] | Ground truth Q, H, S (if available) |
| `variable_names` | [3] | `['discharge_m3s', 'water_depth_m', 'storage_m3']` |

Metadata is saved as `inference_meta.json` with timing, configuration, and output shape.

### GPU Memory Requirements

| Network Size | Approx. VRAM | Device |
|:---|:---|:---|
| ~5,000 reaches | ~2 GB | Any GPU |
| ~50,000 reaches | ~8 GB | RTX 3080+ |
| 127,581 reaches (full global) | ~24 GB | RTX 3090 / A100 |

> **Tip:** For full global inference on GPUs with limited memory, use `--device cpu` (slower but no VRAM constraint).

---

## 📦 Checkpoints

Pre-trained model weights are included in the `checkpoints/` directory:

| Checkpoint | Regime | Median NSE | Size | Description |
|:---|:---|:---:|:---:|:---|
| `GRC_ColdStart.ckpt` | ColdStart | 0.936 | 1.7 MB | State-free prediction (recommended) |
| `GRC_HotStart.ckpt` | HotStart | 0.932 | 1.7 MB | Autoregressive prediction |

Both checkpoints were pre-trained on CaMa-Flood v4 simulations (2000–2019) across 127,581 reaches of the global river network.

---

## 📝 Citation

If you use GraphRiverCast in your research, please cite:

```bibtex
@article{ren2026topology,
  title   = {Topology enables learning-based prediction of global river hydrodynamics},
  author  = {Ren, Hancheng and Zhao, Gang and Wang, Shuo and Slater, Louise and
             Yamazaki, Dai and Liu, Shu and Fan, Jingfang and Cui, Shibo and
             Yu, Ziming and Kang, Shengyu and Zuo, Depeng and Peng, Dingzhi and
             Xu, Zongxue and Pang, Bo},
  journal = {arXiv preprint arXiv:2602.22293},
  year    = {2026},
  doi     = {},
  url     = {https://arxiv.org/abs/2602.22293}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

*GraphRiverCast — Learning the global river network as a unified predictable system*

</div>
