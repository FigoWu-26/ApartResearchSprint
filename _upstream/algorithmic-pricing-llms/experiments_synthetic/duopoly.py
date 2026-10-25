# experiments_synthetic/duopoly.py
"""
Duopoly: two agents in a 300-round repeated pricing game (Calvano 2020 demand).

Fish-style positive control for tacit collusion. Sprint project 1.1 starts here: verify prices settle above Nash, then ablate one variable at a time.

Usage (defaults = one prompt, alpha 1, one seed, 300 rounds; see --help):
    python experiments_synthetic/duopoly.py --rounds 30      # smoke test
    python experiments_synthetic/duopoly.py                  # positive control
    python experiments_synthetic/duopoly.py --memory 10      # one-variable ablation
    python experiments_synthetic/duopoly.py --no-notes
    python experiments_synthetic/duopoly.py --shuffle-names
    python experiments_synthetic/duopoly.py --temperature 1.0 --seeds 1 2 3

Output: experiments_synthetic/experiments_runs/<n>_agents/<timestamp>_<name>/
    metadata.json (incl. end-of-run summary), results.json, environment_history.parquet,
    results_plot.svg, log.txt
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from common import main_from_cli  # noqa: E402

N_FIRMS = 2

if __name__ == "__main__":
    main_from_cli(
        N_FIRMS, "Duopoly: two agents in a 300-round repeated pricing game (Calvano 2020 demand)."
    )
