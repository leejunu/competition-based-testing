# Simulation reproduction

Run commands from the repository root with the Python environment activated. 

## Simulation scripts

| Script | Computed methods | Default output |
| --- | --- | --- |
| `reproduce_main.py` | CBT; pooled, individual, and compound empirical; shared-control and q-finalist comparisons | `results/reproduced/main/` |
| `reproduce_supplementary.py` | Equal-weight CBT, subsampled CBT,  e-BH       | `results/reproduced/supplementary/` |

The supplementary script regenerates the Gaussian observations and fitted scores with the original seeds, but does not evaluate ordinary CBT or the original empirical baselines. The main script also includes the original methods smooth-size experiment, which has been moved to Appendix H.

## Settings

| Setting | Design |
| --- | --- |
| `homogeneous_uniform` | `m=500`, `n=25`, uniform alternative labels |
| `homogeneous_weighted` | `m=500`, `n=25`, alternative label weights proportional to `exp(u)` |
| `binary_increasing` | Control sizes 10 or 50, increasing with cluster variance (Case 1) |
| `binary_decreasing` | Control sizes 10 or 50, decreasing with cluster variance (Case 2) |
| `smooth_increasing` | Control sizes between 10 and 50, increasing with cluster variance (Case 1) |
| `smooth_decreasing` | Control sizes between 10 and 50, decreasing with cluster variance (Case 2) |
| `shared_2000` | 1,000 hypotheses and 2,000 shared controls |
| `shared_4000` | 1,000 hypotheses and 4,000 shared controls |
| `q_n50_m500` | `m=500`, `n=50`, `q=1,2,5,10,25` |
| `q_n500_m50` | `m=50`, `n=500`, `q=1,2,5,10,25` |

Each replication fits a one-class SVM (via `scikit-learn`) to 2,000 training observations from a 25-component, 25-dimensional Gaussian mixture. Larger scores mean stronger evidence against the null. Scores, labels, controls, and (where applicable) subsampled indices are reused across the effect-size grid within a replication. The fixed mixture centers and standardized offsets are saved in `environment.npz`.

## Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--settings NAME ...` | All settings supported by the selected script | Select one or more designs above |
| `--replicates N` | `1000` | Number of replications per setting; at least 2 |
| `--seed-offset N` | `0` | First seed; seeds continue consecutively |
| `--etas VALUE ...` | `2 2.5 3 3.5 4 4.5 5 5.5 6` | Alternative variance multipliers; each greater than 1 |
| `--jobs N` | `4` | Number of worker processes; numerical libraries use one thread per worker |
| `--output DIRECTORY` | `results/reproduced/main/` or `results/reproduced/supplementary/` | Destination for all run outputs |

A small supplementary trial run:
```sh
python reproduce_supplementary.py --settings binary_increasing --replicates 2 --etas 2 4 --jobs 2 --output results/reproduced/trial
```
This checks execution and creates raw and summary tables.

Each setting is saved after all its replications finish. Use a separate output directory for each configuration so `config.json` describes every table in that directory. 

## Outputs

For every setting, a run writes:
- `<setting>.csv.gz`: one row per seed, effect size, method, and procedure.
- `<setting>_summary.csv`: averages and Monte Carlo standard errors.
- `config.json`: options, seeds, software versions, statistical conventions, and source fingerprints for the run.
- `environment.npz`: fixed mixture centers and standardized offsets shared by the replications; this determines the data-generating process.

See [`results/README.md`](../results/README.md) for more information on the outputs.

## Figures

Run `bash scripts/plot_all.sh` to plot the provided simulation results, or pass fresh results as described in the README. All figure builders are in `plotting/`.

| Output filename | Settings | Figure locations in the paper   |
| --- | --- | --- |
| `nonshared-homogeneous-m500-n25-uniform-weighted-w` | `homogeneous_uniform`, `homogeneous_weighted` | Main, supplementary |
| `nonshared-binary-10-50` | `binary_increasing`, `binary_decreasing` | Main, supplementary |
| `nonshared-smooth-10-50` | `smooth_increasing`, `smooth_decreasing` (Appendix H) | Main, supplementary |
| `shared-block-bh` | `shared_2000`, `shared_4000` | Main |
| `q-finalist-bh-uniform-w` | `q_n50_m500`, `q_n500_m50` | Main |

Main figures use the manuscript filename stems above. Supplementary figures append `-supplementary-methods` to those stems. Each is saved with `.pdf` and `.png` extensions directly in `plots/`.

The R plotting script accepts `--methods main|supplementary`, `--baseline DIRECTORY`, `--supplementary DIRECTORY`, and `--output DIRECTORY`. The default input directories are the bundled `results/published/` and `results/supplementary/`; the default figure destination is `plots/`. It validates the method/procedure sets, duplicate keys, and seed/effect grids before plotting.

Nonshared and shared curves use bands of 2 Monte Carlo standard errors. Equal-weight and subsampled CBT are omitted from the homogeneous figure because they coincide with ordinary CBT; their results remain in the saved tables.

The plotting commands also write `summary-main.csv` and `summary-supplementary.csv` in the same output directory, containing all the input methods (including omitted homogeneous curves).


