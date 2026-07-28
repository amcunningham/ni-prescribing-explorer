#!/usr/bin/env python3
"""
Download the raw source data for the NI GP Prescribing Explorer.

Pulls every monthly GP prescribing CSV and every quarterly practice
reference file from OpenDataNI into data/prescribing/ and
data/practice_list_sizes/. Both directories are gitignored — the raw CSVs
are several GB and only the derived parquets belong in the repo.

Files already present are skipped, so re-running this each month fetches
only what is new. Run it, then run build_parquets.py.

    python fetch_opendatani.py
    python fetch_opendatani.py --dry-run     # list what would be fetched

Data: OpenDataNI, Open Government Licence.
"""

import argparse
import os
import re
import sys
import time

import requests

CKAN_PACKAGE_SHOW = "https://admin.opendatani.gov.uk/api/3/action/package_show"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

DATASETS = [
    {
        "id": "gp-prescribing-data",
        "subdir": "prescribing",
        "prefix": "gp-prescribing-northern-ireland",
    },
    {
        "id": "gp-practice-list-sizes",
        "subdir": "practice_list_sizes",
        "prefix": "gp-practice-reference",
    },
]


def parse_month_year(text):
    """Pull (year, month) out of a resource name.

    Resource names are inconsistent across the years — 'GP Prescribing
    Northern Ireland, May 2026', 'GP Prescribing, Northern Ireland -
    December 2018', 'GP Practice Reference File – July 2023', some with a
    trailing '.csv'. Matching on the month name and a 4-digit year handles
    all of them without a rule per format.
    """
    low = text.lower()
    year_match = re.search(r"(20\d{2})", low)
    if not year_match:
        return None
    for name, num in MONTHS.items():
        if name in low:
            return int(year_match.group(1)), num
    return None


def target_filename(prefix, year, month):
    """Stable, sortable name: sorting the directory gives chronological order."""
    return f"{prefix}-{year}-{month:02d}.csv"


def list_resources(dataset_id, session):
    resp = session.get(CKAN_PACKAGE_SHOW, params={"id": dataset_id}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN returned success=false for {dataset_id}")
    return payload["result"]["resources"]


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f}{unit}" if unit == "B" else f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024


def download(url, dest, session):
    tmp = dest + ".part"
    with session.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    os.replace(tmp, dest)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be downloaded, fetch nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download files that are already present")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "ni-prescribing-explorer/1.0"

    grand_total = 0
    for spec in DATASETS:
        out_dir = os.path.join(DATA_DIR, spec["subdir"])
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {spec['id']} → data/{spec['subdir']}/")

        try:
            resources = list_resources(spec["id"], session)
        except Exception as exc:
            print(f"  ! could not read the dataset listing: {exc}")
            print("    (check your connection, then re-run — nothing was changed)")
            return 1

        wanted, skipped_unparsed = [], []
        for res in resources:
            if (res.get("format") or "").strip().lower() not in ("csv", "csv."):
                continue
            parsed = parse_month_year(res.get("name") or "")
            if not parsed:
                skipped_unparsed.append(res.get("name"))
                continue
            year, month = parsed
            wanted.append((year, month, target_filename(spec["prefix"], year, month),
                           res.get("url")))

        wanted.sort()
        if skipped_unparsed:
            print(f"  note: {len(skipped_unparsed)} resource(s) had no recognisable "
                  f"month/year and were skipped: {skipped_unparsed[:3]}")

        to_fetch = [w for w in wanted
                    if args.force or not os.path.exists(os.path.join(out_dir, w[2]))]
        have = len(wanted) - len(to_fetch)
        print(f"  {len(wanted)} monthly files listed "
              f"({wanted[0][0]}-{wanted[0][1]:02d} to {wanted[-1][0]}-{wanted[-1][1]:02d}), "
              f"{have} already present, {len(to_fetch)} to fetch")

        if args.dry_run:
            for _, _, fname, _ in to_fetch:
                print(f"    would fetch {fname}")
            continue

        for i, (year, month, fname, url) in enumerate(to_fetch, 1):
            dest = os.path.join(out_dir, fname)
            print(f"  [{i}/{len(to_fetch)}] {fname} ... ", end="", flush=True)
            for attempt in (1, 2, 3):
                try:
                    size = download(url, dest, session)
                    grand_total += size
                    print(human(size))
                    break
                except Exception as exc:
                    if attempt == 3:
                        print(f"FAILED ({exc})")
                        print("    re-run the script to retry just this file")
                    else:
                        time.sleep(3 * attempt)

    if not args.dry_run:
        print(f"\nDone — {human(grand_total)} downloaded this run.")
        print("Next: python build_parquets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
