# Compound p-values for competition-based testing

This repository implements the methods and reproduces the results in the following paper:

J. Lee, R. F. Barber, R. J. Samworth, Z. Ren. "Compound p-values for competition-based testing." ([arXiv](https://arxiv.org/abs/2610.20411))

We provide two options:
1. Plot our pre-computed numerical results.
2. Compute the simulation results from scratch through our code.
The underlying methods are available separately in [`cbt.py`](cbt.py). The application code includes our UTOPIA model training, scoring, and CBT analyses.

To begin, clone this repository and enter its root directory.  
```sh
git clone https://github.com/leejunu/competition-based-testing.git
cd competition-based-testing
```

## Dependencies and requirements

We assume usage of a laptop or desktop with Python and R installed and a Unix-like OS (e.g. MacOS).

### For plotting (Options 1 and 2)

All figures are plotted in R. Install the three required packages:
```sh
Rscript --vanilla -e 'install.packages(c("ggplot2", "gtable", "scales"), repos="https://cloud.r-project.org")'
```
The supplied figures were generated with R 4.4.2, ggplot2 4.0.3, gtable 0.3.6, and scales 1.4.0. 

### For simulations (Option 2)

Use Python 3.11 or newer and install the provided dependencies in a virtual environment:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
These experiments do not require access to a high-performance cluster. For all simulation code, please use `--jobs` to select an appropriate number of worker processes for your machine.  

## Option 1: Plot pre-computed results

The simulation results are included in this repository: [`results/published/`](results/published/) contains the original manuscript methods, and [`results/supplementary/`](results/supplementary/) contains additional results corresponding to the equal-weight CBT, subsampled CBT, and e-BH methods. Pre-computed UTOPIA and CRISPR results are also included in [`results/applications/`](results/applications/). Python is **not required** for this option.

From the repository root, run:

```sh
bash scripts/plot_all.sh
```

This reads the saved tables and produces all simulation and application figures as PDF and PNG files. All figures are written directly to `plots/`. See the [results guide](results/README.md) for file contents and the [figure map](docs/simulations.md#figures) for the corresponding experiments.

## Option 2: Compute simulation results from scratch

After setting up Python, run either or both of the following scripts. Fresh results are written under `results/reproduced/`, separately from the bundled pre-computed numerical results.

### Simulations for CBT and the baseline methods

Run (and plot) the empirical simulations that compare our CBT methodology (and its variants) against baseline methods in the settings of Section 5 in the paper with the following commands.
```sh
python reproduce_main.py --jobs 8
Rscript --vanilla plot_results.R --methods main --baseline results/reproduced/main
```
The above does not run the additional methods compared in the supplement of the paper (Section H).


### Equal-weight CBT, subsampled CBT, and competition-based testing compound e-values

Recompute (and plot) the experimental results using the equal-weight CBT, subsampled CBT, and e-BH methods with the following:
```sh
python reproduce_supplementary.py --jobs 8
Rscript --vanilla plot_results.R --methods supplementary --supplementary results/reproduced/supplementary
```
Once you have run both simulation scripts, redraw the simulation figures from those fresh results and the application figures from the bundled tables:
```sh
bash scripts/plot_all.sh results/reproduced/main results/reproduced/supplementary
```

Both code workflows use 1,000 replications. The [simulation guide](docs/simulations.md) documents every setting, seed, output, and command-line option, including a small trial run. Each script also supports `--help`.

## Underlying methods

[`cbt.py`](cbt.py) implements CBT p-values (including equal weighting, control group subsampling, and q-finalist variations), compound e-values, baseline p-value methods, BH, and e-BH. It depends only on NumPy and can be used without running the simulation engine. See the [method API and examples](docs/methods.md).

## Real-data applications

To regenerate the application figures from the pre-computed numerical results in [`results/applications/`](results/applications/), run the commands below. These require only R and the plotting packages; no data download, model training, or Python installation is needed.
```sh
Rscript --vanilla applications/plot_applications.R --application utopia
Rscript --vanilla applications/plot_applications.R --application crispr
```
Figures are written to `plots/` and are not included in the repository. 

To recompute the analyses, install the additional dependencies:
```sh
python -m pip install -r applications/requirements.txt
```

Follow the application-specific instructions to set up the data analysis and reproduce the results:
- [UTOPIA](applications/GET_UTOPIA.txt)
- [CRISPR](applications/GET_CRISPR.txt)
These text files give instructions on how to obtain the public datasets for the two applications. We also provide our application code in `applications/`, with the required helper functions in `applications/support/` and figure plotting code in `plotting/`.
Note that UTOPIA trains a neural network to produce the test and control scores; depending on hardware and library versions, the model may be fit differently.

## Repository layout

| Location | Contents |
| --- | --- |
| [`cbt.py`](cbt.py) | Reusable inference methods |
| [`simulate.py`](simulate.py) | Data generation, scoring, and simulation execution |
| [`reproduce_main.py`](reproduce_main.py), [`reproduce_supplementary.py`](reproduce_supplementary.py) | Separate simulation scripts |
| [`scripts/plot_all.sh`](scripts/plot_all.sh), [`plot_results.R`](plot_results.R) | Commands to plot saved results |
| [`plotting/`](plotting/) | All R figure builders and shared styling |
| [`results/`](results/README.md) | Pre-computed simulation results and application metrics |
| `plots/` | Locally generated figures and plotting summaries; excluded from version control |
| [`applications/`](applications/) | Real-data workflows and their instructions |
| [`docs/`](docs/) | Simulation details and method API |
