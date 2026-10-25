# experiments_synthetic/monopoly.py
"""
Monopoly baseline: a single agent should converge to the monopoly price.

Sanity check for the provider/model you picked before spending on multi-firm runs.

Usage (defaults = one prompt, alpha 1, one seed, 300 rounds; see --help):
    python experiments_synthetic/monopoly.py --rounds 30      # smoke test
    python experiments_synthetic/monopoly.py                  # positive control
    python experiments_synthetic/monopoly.py --memory 10      # one-variable ablation
    python experiments_synthetic/monopoly.py --no-notes
    python experiments_synthetic/monopoly.py --shuffle-names
    python experiments_synthetic/monopoly.py --temperature 1.0 --seeds 1 2 3

Output: experiments_synthetic/experiments_runs/<n>_agents/<timestamp>_<name>/
    metadata.json (incl. end-of-run summary), results.json, environment_history.parquet,
    results_plot.svg, log.txt
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from common import main_from_cli  # noqa: E402

N_FIRMS = 1

if __name__ == "__main__":
    main_from_cli(
        N_FIRMS, "Monopoly baseline: a single agent should converge to the monopoly price."
    )
