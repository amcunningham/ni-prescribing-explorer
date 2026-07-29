#!/usr/bin/env python3
"""
Rebuild every derived parquet the NI GP Prescribing Explorer reads, from the
raw OpenDataNI CSVs fetched by fetch_opendatani.py.

    python fetch_opendatani.py      # first — downloads the CSVs
    python build_parquets.py        # then — rebuilds data/*.parquet
    python build_parquets.py --validate    # rebuild to a scratch dir and
                                           # diff against the current parquets

Rebuilt here:
    practices.parquet
    prescribing.parquet                       (snapshot, latest 3 whole months)
    prescribing_practice_monthly.parquet
    prescribing_lcg_monthly.parquet
    standardised_rates_practice.parquet
    standardised_rates_lcg.parquet
    therapeutic_area_ni_monthly.parquet
    therapeutic_area_practice_monthly.parquet

Left alone, because their inputs are not in this repo:
    starpu_denominators_practice.parquet, starpu_denominators_lcg.parquet
        Built by build_starpu_ni_weights.py from the NI STAR-PU weights CSV
        and the BSO practice_demographics parquet. Neither file is in the
        repo, so the existing denominators are read and reused as-is. Years
        they do not cover get null STAR-PU rates, exactly as 2013 does today.
    qof.parquet, prevalence.parquet
        Sourced from the QOF publications, not from prescribing data.

Deprivation (Ward_Dep_Rank, DepQuintile) and Federation are likewise carried
forward from the existing practices.parquet by practice number — they come
from NIMDM and the federation map, neither of which is in the repo.

Data: OpenDataNI, Open Government Licence.
"""

import argparse
import glob
import os
import re
import shutil
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PRESCRIBING_CSV_DIR = os.path.join(DATA_DIR, "prescribing")
LIST_SIZE_CSV_DIR = os.path.join(DATA_DIR, "practice_list_sizes")

SNAPSHOT_MONTHS = 3          # how many recent whole months the snapshot averages

LCG_TO_TRUST = {
    "Belfast": "Belfast HSC Trust",
    "South Eastern": "South Eastern HSC Trust",
    "Northern": "Northern HSC Trust",
    "Southern": "Southern HSC Trust",
    "Western": "Western HSC Trust",
}

# Therapeutic areas, kept deliberately identical to THERAPEUTIC_AREAS in
# app.py. The names here are the short ones the app maps onto via
# TA_APP_TO_PARQUET, so changing either side alone will silently break the
# lookup — change both together.
THERAPEUTIC_AREAS = {
    # Anchored at the start on purpose: an unanchored negative lookahead
    # matches 'Nystatin' from the second character onward, where 'nystatin'
    # is no longer ahead.
    "Statins": r"^(?=.*statin)(?!.*nystatin).*",
    "Ezetimibe": r"ezetimibe",
    "UTI antibiotics": (r"nitrofurantoin|trimethoprim|pivmecillinam|fosfomycin|"
                        r"cefalexin|ciprofloxacin|co-amoxiclav|amoxicillin"),
    "Antidepressants": (r"sertraline|citalopram|fluoxetine|mirtazapine|venlafaxine|"
                        r"duloxetine|paroxetine|escitalopram|trazodone"),
    "Gabapentinoids": r"gabapentin|pregabalin",
    "Opioids": (r"morphine|codeine|tramadol|oxycodone|fentanyl|buprenorphine|"
                r"dihydrocodeine|co-codamol|co-dydramol|tapentadol|pethidine"),
    "PPIs": r"omeprazole|lansoprazole|pantoprazole|esomeprazole|rabeprazole",
    "Diabetes (non-insulin)": (r"metformin|gliclazide|sitagliptin|empagliflozin|"
                               r"dapagliflozin|canagliflozin|semaglutide|liraglutide|"
                               r"dulaglutide|pioglitazone|alogliptin|linagliptin|"
                               r"saxagliptin"),
    "SGLT2 inhibitors": r"dapagliflozin|empagliflozin|canagliflozin|ertugliflozin",
    "GLP-1 agonists": r"semaglutide|liraglutide|dulaglutide|exenatide|lixisenatide",
    "DPP-4 inhibitors": r"sitagliptin|linagliptin|saxagliptin|alogliptin|vildagliptin",
    "Anticoagulants": r"warfarin|apixaban|rivaroxaban|edoxaban|dabigatran",
    "Antihypertensives": (r"ramipril|lisinopril|perindopril|enalapril|captopril|"
                          r"losartan|candesartan|irbesartan|valsartan|olmesartan|"
                          r"telmisartan|amlodipine|felodipine|nifedipine|"
                          r"lercanidipine|bendroflumethiazide|indapamide|"
                          r"chlortalidone|doxazosin"),
    "HRT": (r"estradiol|oestrogen|progesterone|utrogestan|tibolone|norethisterone|"
            r"dydrogesterone|medroxyprogesterone|conjugated oestrogens"),
    "Losartan": r"losartan",
    "Ramipril": r"ramipril",
}


# ── column resolution ────────────────────────────────────────────────────
# The CSVs are not consistent across thirteen years: capitalisation varies,
# some have a '(£)' suffix, some a stray space. Resolve by intent rather than
# by literal name, and fail loudly listing what was actually found.

def resolve(df, label, *candidates, required=True):
    """Find the first column matching any candidate substring."""
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        for low, original in lowered.items():
            if low == cand:
                return original
    for cand in candidates:
        for low, original in lowered.items():
            if cand in low:
                return original
    if required:
        raise KeyError(
            f"could not find the {label} column. Looked for {candidates}; "
            f"the file has: {list(df.columns)}"
        )
    return None


def read_prescribing_csv(path):
    """Read one monthly prescribing CSV, normalised to a known schema."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    cols = {
        "practice": resolve(df, "practice", "practice"),
        "year": resolve(df, "year", "year"),
        "month": resolve(df, "month", "month"),
        "vtm_nm": resolve(df, "drug name", "vtm_nm"),
        "total_items": resolve(df, "items", "total items"),
        "total_cost": resolve(df, "actual cost", "actual cost"),
        "gross_cost": resolve(df, "gross cost", "gross cost", required=False),
        "total_quantity": resolve(df, "quantity", "total quantity"),
        "bnf_chapter": resolve(df, "BNF chapter", "bnf chapter"),
    }
    keep = {v: k for k, v in cols.items() if v is not None}
    df = df[list(keep)].rename(columns=keep)
    if "gross_cost" not in df.columns:
        df["gross_cost"] = pd.NA

    for col in ("practice", "year", "month", "total_items", "total_cost",
                "gross_cost", "total_quantity", "bnf_chapter"):
        # Older files write thousands separators, and some carry a stray
        # currency symbol, so a column can arrive as text. Test with
        # is_numeric_dtype rather than `dtype == object`: under pandas 3.0
        # text columns report dtype 'str', so the object test would skip the
        # cleaning and to_numeric would silently turn every value into NaN.
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col].astype(str)
                              .str.replace(",", "", regex=False)
                              .str.replace("£", "", regex=False)
                              .str.strip())
        before = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lost = before - df[col].notna().sum()
        if lost > 0.01 * max(1, before):
            print(f"      ! {os.path.basename(path)}: {lost:,} of {before:,} "
                  f"values in '{col}' would not parse as numbers — check this "
                  f"file's format before trusting the totals")

    df["vtm_nm"] = df["vtm_nm"].astype(str).str.strip()
    return df.dropna(subset=["practice", "year", "month"])


def read_practice_reference(path):
    """Read one quarterly practice reference file."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols = {
        "PracNo": resolve(df, "practice number", "practice no", "practice number",
                          "pracno", "practice"),
        "PracticeName": resolve(df, "practice name", "practice name", "practicename"),
        "Postcode": resolve(df, "postcode", "postcode"),
        "LCG": resolve(df, "LCG", "lcg", "local commissioning group"),
        "RegisteredPatients": resolve(df, "registered patients", "registered patients",
                                      "registered_patients", "list size"),
        "Address1": resolve(df, "address", "address 1", "address1", "address",
                            required=False),
    }
    keep = {v: k for k, v in cols.items() if v is not None}
    out = df[list(keep)].rename(columns=keep)
    out["PracNo"] = pd.to_numeric(out["PracNo"], errors="coerce").astype("Int64")
    out["RegisteredPatients"] = pd.to_numeric(out["RegisteredPatients"],
                                              errors="coerce")
    for text_col in ("PracticeName", "LCG", "Postcode", "Address1"):
        if text_col in out.columns:
            out[text_col] = out[text_col].astype(str).str.strip()
    out["LCG"] = out["LCG"].str.replace(r"\s+", " ", regex=True).str.strip()
    return out.dropna(subset=["PracNo"])


def classify(vtm_series):
    """Return {area: boolean mask} for the therapeutic areas."""
    return {
        area: vtm_series.str.contains(pattern, case=False, na=False, regex=True)
        for area, pattern in THERAPEUTIC_AREAS.items()
    }


# ── build steps ──────────────────────────────────────────────────────────

def build_practices(out_dir, existing_dir):
    ref_files = sorted(glob.glob(os.path.join(LIST_SIZE_CSV_DIR, "*.csv")))
    if not ref_files:
        raise SystemExit(f"No practice reference CSVs in {LIST_SIZE_CSV_DIR} — "
                         "run fetch_opendatani.py first")
    latest = ref_files[-1]
    print(f"  practice reference: {os.path.basename(latest)}")
    practices = read_practice_reference(latest)
    practices["Trust"] = practices["LCG"].map(LCG_TO_TRUST).fillna("Unknown")

    # Deprivation and federation are not in any OpenDataNI file this script
    # fetches, so carry them forward from the parquet already in the repo.
    prev_path = os.path.join(existing_dir, "practices.parquet")
    if os.path.exists(prev_path):
        prev = pd.read_parquet(prev_path)
        carry = [c for c in ("Ward_Dep_Rank", "DepQuintile", "Federation")
                 if c in prev.columns]
        prev_slim = prev[["PracNo"] + carry].copy()
        prev_slim["PracNo"] = pd.to_numeric(prev_slim["PracNo"],
                                            errors="coerce").astype("Int64")
        practices = practices.merge(prev_slim, on="PracNo", how="left")
        missing = practices["DepQuintile"].isna().sum() if "DepQuintile" in practices else 0
        if missing:
            print(f"  ! {missing} practice(s) have no deprivation/federation carried "
                  f"forward — they are new since the last build and will be blank "
                  f"in the deprivation views")
    else:
        for col in ("Ward_Dep_Rank", "DepQuintile", "Federation"):
            practices[col] = pd.NA

    order = ["PracNo", "PracticeName", "Postcode", "LCG", "Trust",
             "RegisteredPatients", "Ward_Dep_Rank", "DepQuintile",
             "Address1", "Federation"]
    practices = practices.reindex(columns=[c for c in order if c in practices.columns])
    practices.to_parquet(os.path.join(out_dir, "practices.parquet"), index=False)
    print(f"  practices.parquet: {len(practices)} practices")
    return practices


def build_lcg_map(ref_files):
    """practice → LCG for every practice that has EVER appeared, not just current ones.

    The LCG series runs back to 2013, so it has to include practices that have
    since closed or merged. Mapping from the latest reference file alone silently
    drops them — about 8% of 2013 items, tapering to nothing by 2025, which bends
    the historical trend upward. Reading every quarterly file and keeping each
    practice's most recent LCG fixes that.
    """
    frames = []
    for path in ref_files:
        d = read_practice_reference(path)[["PracNo", "LCG"]].copy()
        d["src"] = os.path.basename(path)
        frames.append(d)
    allm = pd.concat(frames, ignore_index=True)
    allm["LCG"] = allm["LCG"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    latest = (allm.sort_values("src").groupby("PracNo", as_index=False).last()
                  [["PracNo", "LCG"]]
                  .rename(columns={"PracNo": "practice", "LCG": "lcg"}))
    latest["practice"] = latest["practice"].astype("Int64")
    return latest


def scan_prescribing(csv_files, snapshot_periods):
    """One pass over every monthly CSV, accumulating each aggregate we need."""
    monthly, ta_rows, snapshot = [], [], []

    for i, path in enumerate(csv_files, 1):
        if i % 20 == 0 or i == len(csv_files):
            print(f"    {i}/{len(csv_files)} files")
        df = read_prescribing_csv(path)

        # 'VTM_NM' is '-' for items with no mapped virtual therapeutic moiety —
        # dressings, appliances, unclassified products. In the RAW data these
        # rows carry genuine per-row quantities, so they are kept everywhere
        # except therapeutic-area matching, where they can never match a drug
        # name anyway. (The old prescribing.parquet had the practice-month
        # total broadcast onto every one of these rows, which inflated the
        # quantity metric roughly thirtyfold. Aggregating per row fixes it.)
        named = df[df["vtm_nm"] != "-"]

        monthly.append(
            df.groupby(["practice", "year", "month", "bnf_chapter"], dropna=False)
              .agg(total_items=("total_items", "sum"),
                   total_cost=("total_cost", "sum"),
                   gross_cost=("gross_cost", "sum"),
                   total_quantity=("total_quantity", "sum"))
              .reset_index()
        )

        masks = classify(named["vtm_nm"])
        for area, mask in masks.items():
            sub = named[mask]
            if sub.empty:
                continue
            agg = (sub.groupby(["practice", "year", "month"])
                      .agg(total_items=("total_items", "sum"),
                           total_cost=("total_cost", "sum"),
                           total_quantity=("total_quantity", "sum"))
                      .reset_index())
            agg["therapeutic_area"] = area
            ta_rows.append(agg)

        if snapshot_periods:
            in_snap = df[
                pd.MultiIndex.from_arrays(
                    [df["year"].astype(int), df["month"].astype(int)]
                ).isin(snapshot_periods)
            ]
            if not in_snap.empty:
                snapshot.append(
                    in_snap.groupby(["practice", "year", "month", "vtm_nm"])
                           .agg(TotalItems=("total_items", "sum"),
                                ActualCost=("total_cost", "sum"),
                                TotalQuantity=("total_quantity", "sum"))
                           .reset_index()
                )

    return monthly, ta_rows, snapshot


def periods_available(csv_files):
    """(year, month) pairs implied by the filenames, newest last."""
    found = []
    for path in csv_files:
        m = re.search(r"(20\d{2})-(\d{2})\.csv$", os.path.basename(path))
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="build into data/_rebuild/ and diff against the current "
                         "parquets instead of overwriting them")
    args = ap.parse_args()

    out_dir = os.path.join(DATA_DIR, "_rebuild") if args.validate else DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(PRESCRIBING_CSV_DIR, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No prescribing CSVs in {PRESCRIBING_CSV_DIR} — "
                         "run fetch_opendatani.py first")
    periods = periods_available(csv_files)
    print(f"Found {len(csv_files)} monthly prescribing CSVs "
          f"({periods[0][0]}-{periods[0][1]:02d} to {periods[-1][0]}-{periods[-1][1]:02d})")

    snapshot_periods = set(periods[-SNAPSHOT_MONTHS:])
    print(f"Snapshot will average: "
          f"{', '.join(f'{y}-{m:02d}' for y, m in sorted(snapshot_periods))}")

    print("\nPractices")
    practices = build_practices(out_dir, DATA_DIR)
    ref_files = sorted(glob.glob(os.path.join(LIST_SIZE_CSV_DIR, "*.csv")))
    prac_lcg = build_lcg_map(ref_files)
    print(f"  LCG map covers {len(prac_lcg)} practices "
          f"(from all {len(ref_files)} quarterly reference files)")

    print("\nScanning prescribing CSVs")
    monthly, ta_rows, snapshot = scan_prescribing(csv_files, snapshot_periods)

    # ── monthly practice × chapter ──────────────────────────────────────
    print("\nAggregating")
    practice_monthly = (pd.concat(monthly, ignore_index=True)
                          .groupby(["practice", "year", "month", "bnf_chapter"],
                                   dropna=False)
                          .sum().reset_index()
                          .sort_values(["practice", "year", "month", "bnf_chapter"]))
    practice_monthly.to_parquet(
        os.path.join(out_dir, "prescribing_practice_monthly.parquet"), index=False)
    print(f"  prescribing_practice_monthly.parquet: {len(practice_monthly):,} rows")

    # ── monthly LCG × chapter ───────────────────────────────────────────
    pm = practice_monthly.copy()
    pm["practice"] = pm["practice"].astype("Int64")
    merged_lcg = pm.merge(prac_lcg, on="practice", how="left")
    unmapped = merged_lcg[merged_lcg["lcg"].isna()]
    if len(unmapped):
        share = unmapped["total_items"].sum() / max(1, merged_lcg["total_items"].sum())
        print(f"  ! {unmapped['practice'].nunique()} practice(s) never appear in any "
              f"reference file and are excluded from the LCG series "
              f"({unmapped['total_items'].sum():,.0f} items, {share:.3%} of the total) "
              f"— they closed before the first reference file (July 2015)")
    lcg_monthly = (merged_lcg.dropna(subset=["lcg"])
                     .groupby(["lcg", "year", "month", "bnf_chapter"], dropna=False)
                     .agg(total_items=("total_items", "sum"),
                          total_cost=("total_cost", "sum"),
                          gross_cost=("gross_cost", "sum"),
                          total_quantity=("total_quantity", "sum"))
                     .reset_index()
                     .sort_values(["lcg", "year", "month", "bnf_chapter"]))
    lcg_monthly.to_parquet(
        os.path.join(out_dir, "prescribing_lcg_monthly.parquet"), index=False)
    print(f"  prescribing_lcg_monthly.parquet: {len(lcg_monthly):,} rows")

    # ── therapeutic areas ───────────────────────────────────────────────
    ta_practice = (pd.concat(ta_rows, ignore_index=True)
                     .groupby(["practice", "year", "month", "therapeutic_area"])
                     .sum().reset_index())
    ta_practice = ta_practice[["practice", "year", "month", "therapeutic_area",
                               "total_items", "total_cost", "total_quantity"]]
    ta_practice.to_parquet(
        os.path.join(out_dir, "therapeutic_area_practice_monthly.parquet"), index=False)
    print(f"  therapeutic_area_practice_monthly.parquet: {len(ta_practice):,} rows")

    ta_ni = (ta_practice.groupby(["year", "month", "therapeutic_area"])
                        .agg(total_items=("total_items", "sum"),
                             total_cost=("total_cost", "sum"),
                             total_quantity=("total_quantity", "sum"))
                        .reset_index())
    ta_ni.to_parquet(
        os.path.join(out_dir, "therapeutic_area_ni_monthly.parquet"), index=False)
    print(f"  therapeutic_area_ni_monthly.parquet: {len(ta_ni):,} rows")

    # ── STAR-PU standardised rates ──────────────────────────────────────
    starpu_path = os.path.join(DATA_DIR, "starpu_denominators_practice.parquet")
    if os.path.exists(starpu_path):
        starpu = pd.read_parquet(starpu_path)
        starpu["practice"] = starpu["practice"].astype("Int64")
        covered = sorted(starpu["year"].unique())
        rates = pm.merge(starpu, on=["year", "practice", "bnf_chapter"], how="left")
        uncovered = sorted(set(rates["year"].dropna().astype(int)) - set(covered))
        if uncovered:
            print(f"  ! no STAR-PU denominators for {uncovered} — those years get "
                  f"null STAR-PU rates (rebuild them with build_starpu_ni_weights.py "
                  f"once the demographics file is available)")
    else:
        rates = pm.copy()
        rates["starpu"] = pd.NA
        rates["total_population"] = pd.NA
        print("  ! starpu_denominators_practice.parquet missing — STAR-PU rates "
              "will be entirely null")

    rates["cost_per_starpu"] = rates["total_cost"] / rates["starpu"]
    rates["items_per_starpu"] = rates["total_items"] / rates["starpu"]
    rates["items_per_capita"] = rates["total_items"] / rates["total_population"]
    rates["cost_per_capita"] = rates["total_cost"] / rates["total_population"]
    rates["quantity_per_capita"] = rates["total_quantity"] / rates["total_population"]
    rates = rates[["practice", "year", "month", "bnf_chapter", "total_items",
                   "total_cost", "gross_cost", "total_quantity", "starpu",
                   "total_population", "cost_per_starpu", "items_per_starpu",
                   "items_per_capita", "cost_per_capita", "quantity_per_capita"]]
    rates.to_parquet(
        os.path.join(out_dir, "standardised_rates_practice.parquet"), index=False)
    print(f"  standardised_rates_practice.parquet: {len(rates):,} rows")

    starpu_lcg_path = os.path.join(DATA_DIR, "starpu_denominators_lcg.parquet")
    lcg_rates = lcg_monthly.copy()
    if os.path.exists(starpu_lcg_path):
        sp_lcg = pd.read_parquet(starpu_lcg_path)
        lcg_rates = lcg_rates.merge(sp_lcg, on=["year", "lcg", "bnf_chapter"],
                                    how="left")
    else:
        lcg_rates["starpu"] = pd.NA
    lcg_rates["cost_per_starpu"] = lcg_rates["total_cost"] / lcg_rates["starpu"]
    lcg_rates["items_per_starpu"] = lcg_rates["total_items"] / lcg_rates["starpu"]
    lcg_rates["quantity_per_starpu"] = lcg_rates["total_quantity"] / lcg_rates["starpu"]
    lcg_rates = lcg_rates[["lcg", "year", "month", "bnf_chapter", "total_items",
                           "total_cost", "gross_cost", "total_quantity", "starpu",
                           "cost_per_starpu", "items_per_starpu",
                           "quantity_per_starpu"]]
    lcg_rates.to_parquet(
        os.path.join(out_dir, "standardised_rates_lcg.parquet"), index=False)
    print(f"  standardised_rates_lcg.parquet: {len(lcg_rates):,} rows")

    # ── snapshot ────────────────────────────────────────────────────────
    snap = (pd.concat(snapshot, ignore_index=True)
              .groupby(["practice", "year", "month", "vtm_nm"]).sum().reset_index())
    snap = snap.rename(columns={"practice": "Practice", "year": "Year",
                                "month": "Month", "vtm_nm": "VTM_NM"})
    snap["Practice_num"] = pd.to_numeric(snap["Practice"], errors="coerce").astype("Int64")
    prac_attrs = practices.rename(columns={"PracNo": "Practice_num"})
    snap = snap.merge(
        prac_attrs[["Practice_num", "PracticeName", "LCG", "Trust",
                    "RegisteredPatients", "Ward_Dep_Rank", "DepQuintile"]],
        on="Practice_num", how="left")
    snap["Practice"] = snap["Practice_num"].astype(str)
    snap = snap[["Practice", "PracticeName", "LCG", "Trust", "RegisteredPatients",
                 "VTM_NM", "TotalItems", "ActualCost", "Month", "Year",
                 "Ward_Dep_Rank", "DepQuintile", "TotalQuantity"]]
    snap.to_parquet(os.path.join(out_dir, "prescribing.parquet"), index=False)
    print(f"  prescribing.parquet: {len(snap):,} rows, "
          f"{snap['Practice'].nunique()} practices")

    # ── build_info.json ─────────────────────────────────────────────────
    # app.py reads this for its period captions, so they stay correct after a
    # rebuild instead of being hardcoded and quietly drifting.
    import json
    import calendar as _cal
    _lab = lambda v: f"{_cal.month_name[v % 100]} {v // 100}"
    _p = sorted({y * 100 + m for y, m in
                 zip(practice_monthly["year"].astype(int),
                     practice_monthly["month"].astype(int))})
    m = re.search(r"(20\d{2})-(\d{2})\.csv$", os.path.basename(ref_files[-1]))
    ref_label = f"{_cal.month_name[int(m.group(2))]} {m.group(1)}" if m else "latest available"
    try:
        _dem = pd.read_parquet(starpu_path, columns=["year"])
        dem_label = f"{int(_dem['year'].min())}\u2013{int(_dem['year'].max())}"
    except Exception:
        dem_label = ""
    info = {
        "prescribing_period": f"{_lab(_p[0])} \u2013 {_lab(_p[-1])}",
        "prescribing_months": len(_p),
        "list_size_reference": ref_label,
        "demographics_period": dem_label,
    }
    with open(os.path.join(out_dir, "build_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(f"  build_info.json: {info['prescribing_period']} "
          f"({info['prescribing_months']} months), "
          f"list sizes {info['list_size_reference']}")

    if args.validate:
        print("\nValidation — rebuilt vs current, on overlapping periods")
        compare(out_dir, DATA_DIR)
        print(f"\nRebuilt files are in {out_dir} — nothing in data/ was overwritten.")
    else:
        print("\nDone. Run the app locally to check it before committing.")
    return 0


def compare(new_dir, old_dir):
    """Diff the rebuilt parquets against the current ones where they overlap."""
    checks = [
        ("prescribing_practice_monthly.parquet",
         ["practice", "year", "month", "bnf_chapter"],
         ["total_items", "total_cost", "total_quantity"]),
        ("therapeutic_area_practice_monthly.parquet",
         ["practice", "year", "month", "therapeutic_area"],
         ["total_items", "total_cost", "total_quantity"]),
        ("prescribing_lcg_monthly.parquet",
         ["lcg", "year", "month", "bnf_chapter"],
         ["total_items", "total_cost", "total_quantity"]),
    ]
    for fname, keys, values in checks:
        new_path, old_path = os.path.join(new_dir, fname), os.path.join(old_dir, fname)
        if not (os.path.exists(new_path) and os.path.exists(old_path)):
            print(f"  {fname}: skipped (one side missing)")
            continue
        new = pd.read_parquet(new_path)
        old = pd.read_parquet(old_path)
        # Align key dtypes across the two sides: numeric keys may be int on
        # one side and float on the other, text keys may differ in padding.
        # Do NOT run text keys through to_numeric — under pandas 3.0 string
        # columns no longer report dtype 'object', and coercing them turns
        # every key into NaN, which merge then happily joins to everything.
        for frame in (new, old):
            for k in keys:
                if pd.api.types.is_numeric_dtype(frame[k]):
                    frame[k] = pd.to_numeric(frame[k], errors="coerce").astype("float64")
                else:
                    frame[k] = frame[k].astype(str).str.strip()
        merged = old.merge(new, on=keys, how="inner", suffixes=("_old", "_new"))
        if merged.empty:
            print(f"  {fname}: no overlapping rows to compare")
            continue
        print(f"  {fname}: {len(merged):,} overlapping rows "
              f"(old {len(old):,}, new {len(new):,})")
        for v in values:
            a, b = merged[f"{v}_old"], merged[f"{v}_new"]
            diff = (a - b).abs()
            worst = diff.max()
            mismatched = int((diff > 0.01).sum())
            flag = "ok" if mismatched == 0 else f"{mismatched:,} rows differ"
            print(f"      {v:<16} max abs diff {worst:,.4f}   {flag}")


if __name__ == "__main__":
    sys.exit(main())
