# Data Acquisition Guide

GraphRiverCast accepts input data in **NetCDF4** format (recommended) or legacy NumPy archives. This guide describes how to obtain source data and prepare it for inference.

> **Note:** Raw data are not included in this repository due to size and license restrictions.

---

## 1. River Network & Geomorphic Features — CaMa-Flood / MERIT Hydro

**Source:** CaMa-Flood v4 model package (includes MERIT Hydro river maps)

**URL:** http://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/

**References:**
- Yamazaki, D. et al. A physically based description of floodplain inundation dynamics in a global river routing model. *Water Resour. Res.* **47**, W04501 (2011). https://doi.org/10.1029/2010WR009726
- Yamazaki, D. et al. MERIT Hydro: A high-resolution global hydrography map based on latest topography dataset. *Water Resour. Res.* **55**, 5053–5073 (2019). https://doi.org/10.1029/2019WR024873

**What to download:**
- CaMa-Flood v4 model package (registration required)
- River network parameters at 15-arc-minute (~25 km) resolution
- Pre-computed CaMa-Flood simulation outputs (or run your own)

**Geomorphic variables → `static` group in NetCDF:**

| CaMa-Flood Variable | NetCDF Key | Description | Unit |
|:---|:---|:---|:---|
| `ctmare` | `ctmare` | Catchment area | m² |
| `elevtn` | `elevtn` | Ground elevation | m |
| `rivhgt` | `rivhgt` | Channel bankfull depth | m |
| `rivman` | `rivman` | Manning roughness coefficient | – |
| `grdare` | `grdare` | Grid cell area | m² |
| `nxtdst` | `nxtdst` | Distance to next downstream reach | m |
| `rivlen` | `rivlen` | Channel length | m |
| `rivwth` | `rivwth_gwdlr` | Channel width (GWD-LR) | m |
| `uparea` | `uparea` | Upstream drainage area | m² |
| `width` | `width` | Bankfull width | m |
| `fldhgt` | `fldhgt` | Floodplain height profile (10 bins) | m |

**Simulation outputs → `dynamic` group in NetCDF:**

| CaMa-Flood Output | NetCDF Key | Description | Unit |
|:---|:---|:---|:---|
| `outflw` | `outflw` | River discharge | m³/s |
| `rivdph` | `rivdph` | Water depth in main channel | m |
| `storage` | `storage` | Channel water storage | m³ |

**River topology → `edge_index` in NetCDF:**

The directed edge list is derived from CaMa-Flood's downstream pointer (`nextxy`). Each reach flowing into another produces one directed edge (upstream → downstream).

### Steps

1. Register and download CaMa-Flood v4 from the URL above
2. Extract the 15-arc-minute river map parameters
3. Vectorize the gridded river network into a graph of $N$ reaches
4. Run CaMa-Flood simulations (or use pre-computed outputs) to obtain daily `outflw`, `rivdph`, `storage`
5. Extract the directed adjacency from `nextxy`

---

## 2. Runoff Forcing — GRADES

**Source:** Global Reach-Level A Priori Discharge Estimation System (GRADES)

**Reference:**
- Lin, P. et al. Global reconstruction of naturalized river flows at 2.94 million reaches. *Water Resour. Res.* **55**, 6499–6516 (2019). https://doi.org/10.1029/2019WR025287

**URL:** Data can be requested from the authors; see the publication for access details.

**What to obtain:**
- Daily lateral runoff at each CaMa-Flood river reach
- This serves as the external forcing input to GraphRiverCast

| GRADES Variable | NetCDF Key | Description | Unit |
|:---|:---|:---|:---|
| Lateral runoff | `runoff` | Runoff forcing into each reach | m³/s |

### Steps

1. Obtain GRADES runoff data following the instructions in Lin et al. (2019)
2. Regrid/aggregate to the CaMa-Flood 15-arc-minute river network
3. Include as the `runoff` variable in the NetCDF file

---

## 3. Gauge Observations — GRDC (for fine-tuning only)

**Source:** Global Runoff Data Centre (GRDC)

**URL:** https://grdc.bafg.de/

> GRDC data are access-restricted and must be requested from the German Federal Institute of Hydrology (BfG) under the GRDC data policy.

**What to obtain:**
- Daily discharge observations from GRDC gauge stations
- Used for Stage 2 fine-tuning only (not required for pre-trained inference)

### Steps

1. Apply for data access at https://grdc.bafg.de/
2. Download daily discharge for the stations of interest
3. Match GRDC gauge locations to CaMa-Flood river reaches
4. Quality-control the records (remove stations with <5 years of data, excessive gaps, or known regulation)

---

## 4. Output Data Format

### NetCDF4 (recommended)

Combine all variables into a single NetCDF4 file:

```python
import xarray as xr
import numpy as np

T, N = 7305, 127581  # 20 years daily, global network
E = 127580

ds = xr.Dataset({
    # Dynamic variables [T, N]
    'outflw':  (['time', 'reach'], np.zeros((T, N), dtype='float64')),
    'rivdph':  (['time', 'reach'], np.zeros((T, N), dtype='float64')),
    'storage': (['time', 'reach'], np.zeros((T, N), dtype='float64')),
    'runoff':  (['time', 'reach'], np.zeros((T, N), dtype='float64')),
    # Static variables [N] or [N, D]
    'ctmare':  (['reach'], np.zeros(N, dtype='float64')),
    'elevtn':  (['reach'], np.zeros(N, dtype='float64')),
    'rivhgt':  (['reach'], np.zeros(N, dtype='float64')),
    'rivman':  (['reach'], np.zeros(N, dtype='float64')),
    'grdare':  (['reach'], np.zeros(N, dtype='float64')),
    'nxtdst':  (['reach'], np.zeros(N, dtype='float64')),
    'rivlen':  (['reach'], np.zeros(N, dtype='float64')),
    'rivwth_gwdlr': (['reach'], np.zeros(N, dtype='float64')),
    'uparea':  (['reach'], np.zeros(N, dtype='float64')),
    'width':   (['reach'], np.zeros(N, dtype='float64')),
    'fldhgt':  (['reach', 'fldhgt_bin'], np.zeros((N, 10), dtype='float64')),
    # Graph [2, E]
    'edge_index': (['direction', 'edge'], np.zeros((2, E), dtype='int64')),
})
ds.attrs['time_start'] = '2000-01-01'
ds.to_netcdf('data/global/simulation.nc', engine='netcdf4')
```

### Legacy NumPy format

```
data/global/
├── dynamic_var.npz    # Keys: outflw, rivdph, storage, runoff — each [T, N]
├── static_var.npz     # Keys: ctmare, elevtn, rivhgt, ... — each [N] or [N,D]
└── edge_index.npy     # Shape: [2, E]
```

The inference script auto-detects the format.

---

## 5. Additional Resources

| Resource | URL |
|:---|:---|
| CaMa-Flood model | http://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/ |
| MERIT Hydro | http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ |
| GRADES (Lin et al.) | https://doi.org/10.1029/2019WR025287 |
| GRDC gauge data | https://grdc.bafg.de/ |
| LamaH-CE dataset | https://doi.org/10.5281/zenodo.4525244 |
| GloFAS-ERA5 reanalysis | https://www.globalfloods.eu |

For questions about data preparation, please open an issue on this repository.
