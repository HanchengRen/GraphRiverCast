# LamaH-CE supervision-density demo

This small example reproduces the score aggregation behind the LamaH-CE
supervision-density curves in Figure 5c. It uses the archived gauge subsets and
per-gauge NSE/KGE scores for three model configurations and three training seeds.
It runs on a CPU without model checkpoints or raw discharge records.

## Run

From the repository root, using Python 3.9 or later:

```bash
python examples/lamah_demo/demo.py
```

The command checks the gauge subsets and writes two CSV files under
`examples/lamah_demo/output/`:

- `per_seed_station_medians.csv`: each run's median score across gauges.
- `supervision_summary.csv`: the median, minimum, maximum and half-range of the
  three run-level medians for each model, supervision level and metric.

To also draw the curves:

```bash
python -m pip install -r examples/lamah_demo/requirements.txt
python examples/lamah_demo/demo.py --plot
```

The figure is written to `output/lamah_supervision_demo.png`. Use `--output-dir`
to choose another destination. Both commands work from any working directory
when the script is addressed by its absolute path.

## Included data

| File | Contents |
| --- | --- |
| `station_subsets.csv` | The 325 matched gauges, represented by their zero-based node indices in the study's 1,409-reach Danube network, and membership of each supervision subset. |
| `metrics/*_per_station.csv` | Archived per-gauge NSE/KGE scores, supervision flags and individual-run scores for training seeds 27, 42 and 123. |
| `experiment.json` | Supervision levels, gauge counts, training seeds and temporal partitions. |

The five nested supervision levels contain 32, 97, 162, 227 and 292 gauges
(10%, 30%, 50%, 70% and 90% of the 325 matched gauges). The published membership
table is taken directly from the stored experiment indices and agrees with the
supervision flags in all 15 metric files. These node indices refer to the
study-specific network, rather than to LamaH catchment identifiers.

The model configurations are:

- `finetune`: globally pre-trained GRC fine-tuned on LamaH-CE.
- `scratch`: GRC trained from scratch on LamaH-CE.
- `lstm_finetune`: globally pre-trained LSTM fine-tuned on LamaH-CE.

The three training seeds identify repeated training runs. Each run uses the same
stored station subset at a given supervision level. The demo loads these exact
memberships rather than regenerating them from a new random draw.

## Aggregation

For each training seed, the script computes the median NSE or KGE across all
325 evaluated gauges, omitting missing metric values. It then takes the median
of the three run-level medians as the plotted centre. The shaded band is their
minimum-to-maximum range; the summary CSV also gives `(maximum - minimum) / 2`.
The supplied `NSE` and `KGE` columns are station-level summaries across runs;
the demo uses the individual `*_s27`, `*_s42` and `*_s123` columns to reproduce
the figure's order of aggregation.

## Data source and scope

Observations originate from
[LamaH-CE](https://doi.org/10.5281/zenodo.4525244), described by
[Klingler et al. (2021)](https://doi.org/10.5194/essd-13-4529-2021).
The files in this example contain study-generated station memberships and
evaluation metrics. Raw observation time series, runoff forcing and CaMa-Flood
simulation arrays are obtained separately from their original providers, as
described in the [data guide](../../data/DATA_GUIDE.md).

This demo reproduces the archived score aggregation and supervision-density
curves. Running model training or computing these metrics anew from predicted
and observed time series requires the full research inputs and training setup.
The example is covered by the repository [licence](../../LICENSE).
