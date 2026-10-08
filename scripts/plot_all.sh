#!/usr/bin/env bash
# Redraw simulation and application figures from saved tables; only R is required.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: bash scripts/plot_all.sh [BASELINE_DIR [SUPPLEMENTARY_DIR [OUTPUT_DIR]]]

Defaults: results/published, results/supplementary, plots
Relative paths are resolved from the repository root, regardless of the caller's directory.
Writes five main figures (including Appendix H) and three supplementary figures,
plus UTOPIA and both CRISPR figures, each as PDF and PNG in OUTPUT_DIR.
Also writes summary-main.csv and summary-supplementary.csv.
Application inputs are the bundled tables in results/applications/; to plot fresh
application outputs, use applications/plot_applications.R with --input.

Examples:
  bash scripts/plot_all.sh
  bash scripts/plot_all.sh results/reproduced/main results/reproduced/supplementary

Requires R with ggplot2, gtable, and scales; see README.md for installation.
USAGE
  exit 0
fi
if (( $# > 3 )); then
  echo "Expected at most three arguments; use --help." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
baseline="${1:-results/published}"
supplementary="${2:-results/supplementary}"
output="${3:-plots}"

# The same baseline tables are used for both sets of figures.
Rscript --vanilla plot_results.R --methods main --baseline "$baseline" --output "$output"
Rscript --vanilla plot_results.R --methods supplementary --baseline "$baseline" \
  --supplementary "$supplementary" --output "$output"
Rscript --vanilla applications/plot_applications.R --application utopia --output "$output"
Rscript --vanilla applications/plot_applications.R --application crispr --output "$output"
