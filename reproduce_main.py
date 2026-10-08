"""Reproduce the main simulation methods and original Appendix H curves.

Run from the repository root:
    python reproduce_main.py --jobs 8

Defaults: all ten settings, 1,000 seeds, eta = 2, 2.5, ..., 6.
Outputs go to results/reproduced/main/. This script does not compute the
equal-weight, subsampled, or e-BH comparisons. See docs/simulations.md for
the setting names and options; use --help for command-line usage.
"""

from simulate import main

if __name__ == "__main__":
    main("main")
