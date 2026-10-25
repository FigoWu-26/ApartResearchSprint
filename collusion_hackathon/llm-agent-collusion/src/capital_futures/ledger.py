"""Machine-readable run output: one JSONL file per ledger, plus JSON summaries."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

LEDGERS = ("actions", "orders", "trades", "positions", "accounts", "external_flows", "variation_margin",
           "margin", "messages", "regulator", "rounds", "llm_calls", "observations", "clearing")


class Recorder:
    """Writes ledgers to ``out_dir`` when given, and always keeps them in memory for tests/analysis."""

    def __init__(self, out_dir: Optional[Path] = None):
        self.out_dir = Path(out_dir) if out_dir else None
        self.rows: dict[str, list[dict]] = {name: [] for name in LEDGERS}
        self._files = {}
        self._lock = threading.Lock()
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if any(self.out_dir.glob("*.json*")):
                raise FileExistsError(f"refusing to overwrite run directory: {self.out_dir}")
            self._files = {name: open(self.out_dir / f"{name}.jsonl", "w") for name in LEDGERS}

    def write(self, ledger: str, record: dict) -> None:
        with self._lock:  # LLM agents record their calls from worker threads
            self.rows[ledger].append(record)
            if self._files:
                self._files[ledger].write(json.dumps(record, default=str) + "\n")
                self._files[ledger].flush()

    def write_json(self, name: str, obj) -> None:
        if self.out_dir:
            (self.out_dir / name).write_text(json.dumps(obj, indent=2, default=str))

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files = {}


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
