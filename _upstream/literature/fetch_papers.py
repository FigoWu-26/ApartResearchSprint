#!/usr/bin/env python3
"""Download the open-access PDFs listed in papers.csv into literature/pdfs/.

The PDFs are not stored in this repo (most are not licensed for redistribution);
this script fetches them from their original sources instead.

    python3 literature/fetch_papers.py                 # everything with a direct PDF link
    python3 literature/fetch_papers.py --sprint-only   # only papers cited on the sprint page
    python3 literature/fetch_papers.py --project 1.3   # only papers tagged with sprint project 1.3
    python3 literature/fetch_papers.py --list          # show what would be downloaded, download nothing

Only the Python standard library is needed. Files that already exist are skipped, so the
script can be re-run. Entries without a direct PDF link (blog posts, paywalled papers, some
web pages) are listed at the end with their URL to open in a browser.
"""
import argparse, csv, re, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "papers.csv"
OUT = HERE / "pdfs"
UA = "Mozilla/5.0 (collusion-sprint literature fetcher; +https://apartresearch.com)"


def slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sprint-only", action="store_true", help="only papers cited on the sprint page")
    ap.add_argument("--project", help="only papers tagged with this sprint project, e.g. 1.3")
    ap.add_argument("--list", action="store_true", help="list matching entries without downloading")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between downloads (be polite to arXiv)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    if args.sprint_only:
        rows = [r for r in rows if r["on_sprint_page"] == "yes"]
    if args.project:
        rows = [r for r in rows if args.project in [p.strip() for p in r["sprint_projects"].split(";")]]

    direct = [r for r in rows if r["pdf_url"]]
    manual = [r for r in rows if not r["pdf_url"]]
    print(f"{len(rows)} entries match: {len(direct)} with a direct PDF link, {len(manual)} to open in a browser.")
    if args.list:
        for r in direct:
            print(f"  {r['id']:>4}  {r['citation']}: {r['pdf_url']}")
        for r in manual:
            print(f"  {r['id']:>4}  {r['citation']} (browser): {r['url'] or 'no link'}")
        return

    OUT.mkdir(exist_ok=True)
    failed = []
    for i, r in enumerate(direct, 1):
        dest = OUT / f"{r['id']}_{slug(r['citation'])}_{slug(r['title'], 50)}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            req = urllib.request.Request(r["pdf_url"], headers={"User-Agent": UA})
            data = urllib.request.urlopen(req, timeout=60).read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(data)
            print(f"[{i}/{len(direct)}] {r['citation']}")
        except Exception as e:  # noqa: BLE001 - report and carry on
            failed.append((r, str(e)[:80]))
            print(f"[{i}/{len(direct)}] FAILED {r['citation']}: {e}")
        time.sleep(args.delay)

    if failed or manual:
        print("\nOpen these in a browser:")
        for r, why in failed:
            print(f"  {r['citation']} ({why}): {r['url'] or r['pdf_url']}")
        for r in manual:
            print(f"  {r['citation']}: {r['url'] or 'no link (see notes in papers.csv)'}")
    print(f"\nPDFs are in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
