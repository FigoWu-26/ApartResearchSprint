"""Capital + futures extension of the Agrawal et al. (2025) double auction.

The legacy experiment in ``src/continuous_double_auction`` is not modified. This package adds a
separate engine with the same spot convention (one lot per agent per round, end-of-round matching of
the best bid against the best ask at the midpoint, unmatched orders persist), plus cash accounting,
a cash-settled futures contract with margin, and configurable initial capital.

All money is held in integer cents so ledgers reconcile exactly.
"""
