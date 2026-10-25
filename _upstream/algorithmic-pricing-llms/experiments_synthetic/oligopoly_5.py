# experiments_synthetic/oligopoly_5.py
"""
5-firm oligopoly: most competitive setting in the thesis.

Prices expected closest to Nash, yet still above it. 1,500 calls per 300-round run.

Usage (defaults = one prompt, alpha 1, one seed, 300 rounds; see --help):
    python experiments_synthetic/oligopoly_5.py --rounds 30      # smoke test
    python experiments_synthetic/oligopoly_5.py                  # positive control
    python experiments_synthetic/oligopoly_5.py --memory 10      # one-variable ablation
    python experiments_synthetic/oligopoly_5.py --no-notes
    python experiments_synthetic/oligopoly_5.py --shuffle-names
    python experiments_synthetic/oligopoly_5.py --temperature 1.0 --seeds 1 2 3

Output: experiments_synthetic/experiments_runs/<n>_agents/<timestamp>_<name>/
    metadata.json (incl. end-of-run summary), results.json, environment_history.parquet,
    results_plot.svg, log.txt
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from common import main_from_cli  # noqa: E402

N_FIRMS = 5

if __name__ == "__main__":
    main_from_cli(N_FIRMS, "5-firm oligopoly: most competitive setting in the thesis.")
