#!/usr/bin/env python3
"""
Combine three LLM result files (pick, guess, coordinate) — each containing multiple
variations per question — into a single JSON array in the same shape as your human
data:

[
  {
    "id": "TA1",
    "choices": [...],
    "pick": [...],
    "guess": [...],
    "coordinate": [...]
  },
  ...
]

Each input file is expected to contain a JSON array (or JSONL) of records like:
{
  "idx": "TA1",
  "variation-idx": "0|1|2",
  "prompt": "... list of options and their score: X: 10, Y: 10, ...",
  "responses": { "<answer>x</answer>": 13, "<answer>y</answer>": 7 }
}

We:
  • Sum the response counts across ALL variations (variation-idx 0,1,2).
  • Parse the prompt to discover option labels and establish a canonical order per question.
  • Build one unified "choices" list (first-seen order across the three files) per id.
  • Output counts for each case (pick/guess/coordinate) aligned to that choices order.
"""

import argparse
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Any

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

def load_json_or_jsonl(path: str) -> List[dict]:
    """Load a JSON array file OR a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # Try regular JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # If someone wrapped the list under a key
            for v in data.values():
                if isinstance(v, list):
                    return v
        raise ValueError("Unsupported JSON structure at top level.")
    except json.JSONDecodeError:
        # Fallback: JSON Lines
        out = []
        with open(path, "r", encoding="utf-8") as f2:
            for line in f2:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

def extract_label_from_answer(s: str) -> str:
    """Return the text between <answer>...</answer> if present, else s."""
    m = ANSWER_TAG_RE.search(s or "")
    return (m.group(1) if m else s).strip()

def norm_key(s: str) -> str:
    """Normalization key for matching labels across variations/cases."""
    return extract_label_from_answer(s).strip().lower()

PROMPT_LIST_RE = re.compile(
    r"list of options.*?:\s*(.+?)\.($|\s)",
    flags=re.IGNORECASE | re.DOTALL
)

def parse_prompt_options(prompt: str) -> List[str]:
    """
    Extract ordered option labels from the prompt section like:
      "... list of options and their score: A: 10, B: 10, C: 10."
    Returns a list of clean labels: ["A","B","C"].
    If parsing fails, returns [].
    """
    if not prompt:
        return []
    m = PROMPT_LIST_RE.search(prompt)
    if not m:
        return []
    tail = m.group(1)
    # Split on commas, then split each chunk on ":" and take the label part.
    opts = []
    for chunk in tail.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # typical piece: "grijs: 10"
        label = chunk.split(":")[0].strip()
        if label:
            opts.append(label)
    return opts

def union_in_order(base_list: List[str], to_add: Iterable[str]) -> None:
    """Append items from to_add into base_list if not already present (case-sensitive)."""
    seen = set(base_list)
    for x in to_add:
        if x not in seen:
            base_list.append(x)
            seen.add(x)

def aggregate_file(path: str) -> Tuple[Dict[str, Counter], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Aggregate one case file (pick/guess/coordinate) across all variations.

    Returns:
      counts_by_id: {id -> Counter({normalized_label: total_count})}
      display_order_by_id: {id -> [display_label1, display_label2, ...]}  (first-seen order from prompts)
      display_map_by_id: {id -> {normalized_label -> display_label}}       (first-seen display form)
    """
    data = load_json_or_jsonl(path)
    counts_by_id: Dict[str, Counter] = defaultdict(Counter)
    display_order_by_id: Dict[str, List[str]] = defaultdict(list)
    display_map_by_id: Dict[str, Dict[str, str]] = defaultdict(dict)

    for rec in data:
        qid = str(rec.get("idx", "")).strip()
        if not qid:
            continue

        prompt = rec.get("prompt", "")
        resp = rec.get("responses", {}) or {}

        # 1) Parse prompt options to establish display order and map
        options = parse_prompt_options(prompt)
        # Build/update mapping for this question
        for disp in options:
            nk = norm_key(disp)
            # If not seen yet for this id, record first-seen display form
            if nk not in display_map_by_id[qid]:
                display_map_by_id[qid][nk] = disp
            # And ensure this display label is present in the canonical order list
            # (use the stored display form for the normalized key)
            union_in_order(display_order_by_id[qid], [display_map_by_id[qid][nk]])

        # 2) Aggregate response counts (sum across variations)
        for raw_ans, c in resp.items():
            nk = norm_key(raw_ans)
            try:
                cnt = int(c)
            except Exception:
                continue
            counts_by_id[qid][nk] += cnt

            # If we saw an answer not listed in the prompt (rare), ensure it appears in display lists.
            if nk not in display_map_by_id[qid]:
                # Use the raw answer (without tags) as display, keeping its original case
                disp = extract_label_from_answer(raw_ans)
                display_map_by_id[qid][nk] = disp
                union_in_order(display_order_by_id[qid], [disp])

    return counts_by_id, display_order_by_id, display_map_by_id

def merge_three_cases(pick_path: str, guess_path: str, coord_path: str) -> List[dict]:
    """
    Merge three files into the target output structure.
    The "choices" order for each id is the union (in first-seen order) of the prompt-derived
    choices across the three files (pick → guess → coordinate precedence).
    Counts are aligned to that order for each case.
    """
    p_counts, p_order, p_disp = aggregate_file(pick_path)
    g_counts, g_order, g_disp = aggregate_file(guess_path)
    c_counts, c_order, c_disp = aggregate_file(coord_path)

    # Build the union of IDs across the three cases
    all_ids = set(p_counts.keys()) | set(g_counts.keys()) | set(c_counts.keys()) \
              | set(p_order.keys())  | set(g_order.keys())  | set(c_order.keys())

    out = []

    # Helper: build canonical display order per id from pick→guess→coord
    def canonical_order_for_id(qid: str) -> Tuple[List[str], Dict[str, str]]:
        # Start with pick, then extend with guess, then coordinate
        display_map: Dict[str, str] = {}
        order: List[str] = []

        # Merge a case's display map & order into the canonical
        def merge_case(case_order: List[str], case_map: Dict[str, str]):
            # case_order contains display labels; we need to re-derive NKs in the same way
            for disp in case_order:
                nk = norm_key(disp)
                if nk not in display_map:
                    display_map[nk] = case_map.get(nk, disp)
                    union_in_order(order, [display_map[nk]])

        merge_case(p_order.get(qid, []), p_disp.get(qid, {}))
        merge_case(g_order.get(qid, []), g_disp.get(qid, {}))
        merge_case(c_order.get(qid, []), c_disp.get(qid, {}))

        # Also ensure that any NK present in counts but missing from order is appended
        for nk in p_counts.get(qid, {}):
            if nk not in display_map:
                display_map[nk] = p_disp.get(qid, {}).get(nk, nk)
                union_in_order(order, [display_map[nk]])
        for nk in g_counts.get(qid, {}):
            if nk not in display_map:
                display_map[nk] = g_disp.get(qid, {}).get(nk, nk)
                union_in_order(order, [display_map[nk]])
        for nk in c_counts.get(qid, {}):
            if nk not in display_map:
                display_map[nk] = c_disp.get(qid, {}).get(nk, nk)
                union_in_order(order, [display_map[nk]])

        return order, display_map

    # Natural-ish sort for ids like TA1..TA14 / TN1..TN14
    def id_sort_key(s: str):
        m = re.match(r"([A-Za-z]+)(\d+)$", s)
        if m:
            return (m.group(1), int(m.group(2)))
        return (s, 0)

    for qid in sorted(all_ids, key=id_sort_key):
        choices_display, disp_map = canonical_order_for_id(qid)

        # Build arrays in the canonical order
        def counts_to_array(counter: Counter) -> List[int]:
            arr = []
            for disp in choices_display:
                nk = norm_key(disp)
                arr.append(int(counter.get(nk, 0)))
            return arr

        pick_arr = counts_to_array(p_counts.get(qid, Counter()))
        guess_arr = counts_to_array(g_counts.get(qid, Counter()))
        coord_arr = counts_to_array(c_counts.get(qid, Counter()))

        out.append({
            "id": qid,
            "choices": choices_display,
            "pick": pick_arr,
            "guess": guess_arr,
            "coordinate": coord_arr,
        })

    return out

def main():
    
    pick_file = "./results/openai/gpt-oss-120b_high/nottingham_problem-pick.jsonl"
    guess_file = "./results/openai/gpt-oss-120b_high/nottingham_problem-guess.jsonl"
    coordinate_file = "./results/openai/gpt-oss-120b_high/nottingham_problem-coordinate.jsonl"
    out = "nottingham_merged_gpt_counts.json"
    merged = merge_three_cases(pick_file, guess_file, coordinate_file)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(merged)} items to {out}")

if __name__ == "__main__":
    main()
