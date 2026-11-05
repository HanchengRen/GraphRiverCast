# GraphRiverCast model card

GraphRiverCast learns river-system hydrodynamics across connected river networks.
It is a topology-informed river foundation model that jointly predicts daily
discharge, water depth and channel storage, with global modelling at **0.25°**
resolution and demonstrated adaptation to a finer regional river network.

| Item | Description |
| --- | --- |
| Model | GraphRiverCast (GRC) |
| Model type | Recurrent spatiotemporal neural operator over a signed river graph |
| Primary task | Multivariate river-state prediction and reconstruction |
| Outputs | Discharge Q (m³/s), water depth H (m), channel storage S (m³) |
| Time resolution | Daily |
| Global spatial representation | Connected river reaches at 0.25° resolution |
| Prediction regimes | Initial-state-free ColdStart; history-conditioned HotStart |
| Implementation | Python, PyTorch and PyTorch Geometric |
| Documentation updated | 5 November 2026 |

## Intended uses

The model supports scientific investigation of global river dynamics, prediction
at reaches without recent river-state observations, and adaptation of a globally
learned hydrodynamic representation to regional networks with sparse gauges.
The supplied models also support comparisons of initialisation strategies,
network connectivity and simulation-based pre-training.

For forecast applications, runoff forcing must be supplied over the prediction
window. For reconstruction, forcing may come from an existing historical record.
ColdStart removes the requirement for historical target river states; it retains
the requirements for runoff forcing, river topology and compatible static
features. Historical observations used during fine-tuning are distinct from
observations supplied as inputs during inference.

## Architecture

GraphRiverCast is a topology-informed river foundation model formulated as a
recurrent spatiotemporal neural operator over connected river networks. Its
residual architecture integrates three complementary encoders for local feature
fusion, hydraulic spatial propagation and temporal memory. The feature encoder
combines geomorphic attributes, runoff forcing and the evolving river state into
a shared latent representation. The graph encoder uses a **dual-branch, signed
bidirectional message-passing architecture**, with independently parameterised
downstream and upstream branches. **Edge-wise, channel-specific hydraulic gates**
modulate information transfer using static hydraulic geometry and dynamic
water-depth gradients, which are recomputed from the predicted state at every
time step. A **reach-specific adaptive fusion gate** then balances the two
directional representations, allowing the model to represent both upstream and
downstream hydraulic influences. The temporal encoder maintains reach-wise hidden
and cell states through LSTM recurrence, coupling hydrological memory with
spatial propagation. Residual connections and RMS normalisation support stable
autoregressive rollout. A shared increment-based readout jointly updates
discharge, water depth and channel storage across the network. The same operator
supports both initial-state-free ColdStart prediction and history-conditioned
HotStart prediction.

The principal architecture uses a latent width of 64, two message-passing layers
in each directional branch, a feature-mixing width of 128, and 96,643 trainable
parameters. Each directed connection carries eight static hydraulic-geometric
attributes and one dynamic water-depth-gradient attribute. Model architecture
and training details are described in the manuscript Methods and Supplementary
Notes S5–S7; inference pseudocode is in Supplementary Note S12.

## Inputs and outputs

| Input group | Contents |
| --- | --- |
| Runoff forcing | Daily lateral runoff aligned to the model's river reaches |
| River network | Directed hydraulic connections, represented in both orientations with signed attributes |
| Static features | Reach geometry, floodplain profile and per-reach hydrological scale descriptors |
| River state | Internally propagated predictions; reference history is additionally supplied in HotStart |

ColdStart begins from zero in the standardised state space and evolves through
a forcing-driven spin-up. HotStart uses supplied reference states in its history
window. Over the forward prediction window, both regimes advance using their own
previous predictions without consuming observations step by step.

Outputs are organised by time and river reach. The main inference entry point
writes the three physical variables to NetCDF, together with inference metadata.
Input preparation must preserve feature order, units, graph orientation and the
preprocessing associated with the chosen checkpoint. See the
[data guide](data/DATA_GUIDE.md) and [inference instructions](README.md#-inference).

## Training data and procedure

### Global pre-training

Pre-training learns a multivariate global reference field from CaMa-Flood v4
simulations driven by GRADES runoff, using river topology and geomorphic
information derived from MERIT Hydro and the CaMa-Flood parameter set. The study
uses daily data covering 2000–2019. Its global temporal benchmark partitions
2000–2015 for training, 2016–2017 for validation, and 2018–2019 for testing.

The manuscript training protocol uses a 21-day history and a 7-day forward
rollout, Adam optimisation with an initial learning rate of 10⁻³ and cosine
annealing, and validation-based checkpoint selection. The pre-training objective
combines a spatially and temporally weighted multivariate error term with a
mass-conservation regulariser. Supplementary Tables 4 and 5 document the
pre-training configuration and computational resources.

### Adaptation to gauge observations

Global fine-tuning uses daily discharge observations from 2,090 GRDC gauges
retained after spatial matching and quality screening. Of these, 1,996 have
computable scores during the 2018–2019 temporal evaluation period. Fine-tuning
updates all model parameters using a masked Huber discharge loss. Water depth
and storage adapt through the shared representation; the mass-conservation
regulariser used during pre-training is disabled in this stage. Supplementary
Table 6 documents the adaptation configurations.

Regional adaptation is studied on a 0.1° Danube river network using 325 matched
LamaH-CE gauges. The temporal partitions are 2000–2009 for training, 2010–2013
for validation and 2014–2017 for testing. Nested supervision subsets span 10–90%
of matched gauges. The accompanying [LamaH-CE demo](examples/lamah_demo/)
provides the stored subset memberships and per-gauge metrics from three training
runs at each supervision level.

## Evaluation and reported performance

The study evaluates complementary capabilities:

- Multivariate reconstruction of the CaMa-Flood reference field, including
  lead-time-dependent ColdStart and HotStart comparisons and long autoregressive
  rollouts.
- Real-observation discharge prediction over the global temporal test period,
  comparing GRC, a data-driven LSTM benchmark, CaMa-Flood and GloFAS-ERA5.
- Spatial transfer using five-fold gauge holdouts, with the same fold assignments
  across compared models.
- Adaptation from the global 0.25° network to the finer 0.1° Danube network under
  varying gauge-supervision densities.
- Encoder ablations, retrained architecture ablations, integrated-gradient
  attribution and feature-permutation sensitivity analysis.

For the manuscript's 2018–2019 discharge benchmark at 1,996 gauges, the reported
across-gauge medians are:

| Metric | Fine-tuned GraphRiverCast |
| --- | ---: |
| Nash–Sutcliffe efficiency (NSE) | 0.458 |
| Kling–Gupta efficiency (KGE) | 0.525 |
| Min–max normalised RMSE | 0.11 |
| Percent bias (PBIAS) | +2.8% |

The LSTM benchmark has a reported median NSE of 0.367 under the same temporal
comparison. These are manuscript results for the specified data and protocol;
simulation-reference scores and gauge-observation scores describe different
evaluation targets.

The 700-day experiment evaluates internally generated error accumulation in
pseudo-hindcasts with supplied future runoff forcing. It demonstrates stability
under those forcing conditions; the useful horizon of an operational forecast
also depends on the predictability and accuracy of its forcing.

## Scope, biases and limitations

**Forcing and reference-model dependence.** Simulation pre-training provides a
globally coherent target and can carry systematic errors from CaMa-Flood and its
runoff inputs. Gauge fine-tuning helps align discharge with observations.
Forcing uncertainty remains material, particularly in arid and strongly seasonal
regions, and should be considered when applying the model to new forcing products.

**Uneven observational coverage.** GRDC gauges provide sparse and geographically
uneven supervision, with stronger representation in monitored regions and larger
rivers. Network connectivity supports information transfer between connected
reaches. Completely isolated subnetworks without observational anchors rely more
heavily on the representation learned during pre-training.

**Human interventions.** Dam and reservoir operating decisions are not explicitly
represented as dynamic controls. Predictions in strongly regulated systems depend
on how those effects are reflected in the available forcing, reference fields
and observations.

**Variable-specific evidence.** GRDC fine-tuning and its real-observation benchmark
directly supervise and evaluate discharge. Water depth and storage are jointly
predicted and assessed against the simulation reference and through their
cross-variable response to fine-tuning. Their evidence should be interpreted
according to those respective targets.

**Resolution and deployment.** The demonstrated settings are daily global
prediction at 0.25° and regional adaptation at 0.1°. Further resolutions, forcing
regimes and applications require evaluation in their intended setting. Coupling
to operational weather forecasts, observational assimilation and uncertainty-aware
prediction are directions identified in the manuscript discussion.

## Access and reproducibility

- [Model source and usage](README.md) and [model checkpoints](checkpoints/) are
  supplied in this repository. Checkpoints are grouped into global pre-training,
  GRDC adaptation and ablation experiments. Select weights for the intended
  architecture and prediction regime.
- The [LamaH-CE supervision-density demo](examples/lamah_demo/README.md) runs with
  `python examples/lamah_demo/demo.py`. It reproduces score aggregation from the
  supplied archived metrics; optional plotting instructions are included.
- The reviewer software archive also supplies an eight-reach synthetic inference
  example with expected numerical outputs. This checks model loading and forward
  prediction; paper accuracy results use the research datasets and evaluation
  protocols described above.
- Full research inputs are obtained through their original providers. The small
  examples contain synthetic inputs or study-derived memberships and scores.
  GRDC observation time series are not redistributed.

Data access routes include [CaMa-Flood and MERIT Hydro](http://hydro.iis.u-tokyo.ac.jp/~yamadai/),
[GRADES](https://doi.org/10.1029/2019WR025287), [GRDC](https://grdc.bafg.de/),
[LamaH-CE](https://doi.org/10.5281/zenodo.4525244), and
[GloFAS](https://www.globalfloods.eu). GRDC observations require a request to BfG
under the GRDC data policy; other sources retain their own access conditions.
The [web demonstration](https://hong.ringsaturn.me/) displays river prediction
products.

## Computational resources

The manuscript reports training on eight NVIDIA RTX 3090 GPUs (24 GB each) with
distributed data parallelism. Reported training wall-clock times are approximately
31.1 hours for ColdStart and 26.7 hours for HotStart. These measurements describe
the research setup.

The reviewer archive's small inference example has been checked on Windows with
Python 3.12.14, PyTorch 2.6.0+cpu, PyTorch Geometric 2.6.1 and NumPy 2.2.6. Its
documented runtime is approximately 11 seconds including imports and checkpoint
loading, with about 0.1 seconds for model inference. GPU hardware is not needed
for that example or the archived-metric LamaH demo.

## Licence, authorship and contact

The model is developed by Hancheng Ren and the GraphRiverCast study authors.
Source code, supplied model weights and documentation are distributed under the
repository [CC BY-NC-ND 4.0 licence](LICENSE). Third-party software and datasets
retain their respective licences and access conditions.

The associated study is *Learning global river system hydrodynamics with a
topology-informed foundation model*. See the repository
[citation entry](README.md#-citation) for the author list and bibliographic record.
Corresponding authors are Gang Zhao and Bo Pang. Questions about model use can
be raised through [repository issues](https://github.com/HanchengRen/GraphRiverCast/issues)
or directed to [Hancheng Ren](mailto:hancheng.ren@mail.bnu.edu.cn).
