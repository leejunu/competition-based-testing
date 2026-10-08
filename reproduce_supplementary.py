"""Recompute equal-weight CBT, subsampled CBT, and CBT (e-BH).

Run from the repository root:
    python reproduce_supplementary.py --jobs 8

Defaults: six nonshared settings, 1,000 seeds, eta = 2, 2.5, ..., 6.
Outputs go to results/reproduced/supplementary/. Original main-text and
Appendix H methods are not evaluated. Plot against the bundled baseline
results or a separate main reproduction; see docs/simulations.md.
"""

from simulate import main

if __name__ == "__main__":
    main("supplementary")
