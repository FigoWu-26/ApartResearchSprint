"""Named random streams derived from the run seed.

Each consumer draws from its own stream, so a condition that makes more draws in one place
(for example futures quoting) does not shift the draws anywhere else. Paired runs that share a
seed therefore share agent names, initial orders, processing order and the external reference.
"""

import hashlib
import random


def stream(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def round_stream(seed: int, name: str, round_number: int) -> random.Random:
    """A fresh stream per round, so a round's draws do not depend on how many earlier draws were made."""
    return stream(seed, f"{name}:round{round_number}")
