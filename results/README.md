# Saved simulation and application results

These tables are included so the simulation and application figures can be plotted without rerunning the experiments. From the repository root, run `bash scripts/plot_all.sh`.

| Directory | Contents |
| --- | --- |
| `published/` | Ten compressed tables containing the original methods used in the main text and the Appendix H smooth-size figure |
| `supplementary/` | Six compressed tables containing freshly computed equal-weight CBT, subsampled CBT, and CBT (e-BH), plus summaries and run configuration |
| `applications/` | UTOPIA metrics for scales 1, 2, and 4, and CRISPR unrestricted/global-null metrics |
| `reproduced/` | Default destination for new user runs; created when needed and excluded from version control |

Each saved design uses seeds 0–999 and effect sizes `2, 2.5, ..., 6`. The supplementary set contains 270,000 rows over 6,000 replications. It does not contain recomputed main-text methods. Figure builders combine the two sets after checking their seed/effect grids.

## Simulation tables

`<setting>.csv.gz` contains one row per `(setting, seed, eta, method, procedure)`:

| Column | Meaning |
| --- | --- |
| `setting` | Design identifier; see the [setting table](../docs/simulations.md#settings) |
| `seed` | Replication seed |
| `eta` | Alternative variance multiplier |
| `method` | Inference method; the raw key `CBT e-values` is displayed as `CBT (e-BH)` in figures |
| `procedure` | `BH`, `Bonferroni`, or `eBH` |
| `discoveries` | Total rejections |
| `false` | Rejections of null hypotheses |
| `true` | Rejections of alternative hypotheses |
| `fdp` | `false / max(discoveries, 1)` |
| `power` | `true / number_of_alternatives` |
| `fwer` | Indicator that at least one null hypothesis was rejected |

Load a raw table directly with pandas or base R:

```python
import pandas as pd
results = pd.read_csv("results/supplementary/binary_increasing.csv.gz")
```

```r
results <- read.csv("results/supplementary/binary_increasing.csv.gz")
```

`<setting>_summary.csv` reports `replicates` and the means and standard errors of `fdp`, `power`, `fwer`, and `discoveries`, grouped by setting, effect size, method, and procedure. The R plotters independently aggregate the raw tables and save `summary-main.csv` and `summary-supplementary.csv` alongside the figures in `plots/`.

## Application tables

The application plotter reads `applications/utopia/` and `applications/crispr/` by default:

- `utopia/scale{1,2,4}_curves.csv`: 500 rows per scale, covering five methods at alpha = 0.005, 0.010, ..., 0.5. Columns contain the scale, method, alpha, rejection count (`discoveries`), true/false rejection counts, FDP, and hypothesis count.
- `crispr/unrestricted_curves.csv` and `crispr/global_null_curves.csv`: 160 rows each, covering four methods at alpha = 0.005, 0.010, ..., 0.2. Columns contain the task, method, alpha, hypothesis and truth-label counts, rejection counts, labeled FDP (`kFDR`), and global-null FDP where applicable.

These are aggregate metric tables, not individual observations, model checkpoints, or public upstream code. UTOPIA metrics and the unrestricted CRISPR comparison come from the saved manuscript tables; three global-null curves also come from the saved table. The global-null within-cell BH curve was obtained by aggregating the already-saved Dempster p-values. No training, scoring, or matching was rerun.

`applications/manifest.json` records source paths and checksums, column/label adaptations, and the bundled table checksums. Source paths identify the research archive and are not required to plot the bundled tables. For fresh application results, pass the analysis output directory to `applications/plot_applications.R --input`.

## Provenance

`published/manifest.json` records each original source path, its SHA-256 fingerprint, the bundled table's fingerprint, and the original experiment configuration. The source paths describe the research archive; reproducing plots does not require that archive.

`supplementary/config.json` records the supplementary run options, environment versions, fixed seeds, and execution/packaging source fingerprints at the time the run was assembled. `supplementary/environment.npz` stores its fixed Gaussian-mixture centers and offsets. Later documentation, CLI, and plotting changes do not imply that this saved run was repeated.

`verification.json` records checks of the bundled results. `plotting-verification.json` records numerical and rendering checks. Raw application inputs and model checkpoints are not bundled; see the application-specific acquisition instructions in `applications/`.
