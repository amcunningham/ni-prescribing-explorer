#!/usr/bin/env python3
"""
Build STAR-PU denominator parquets using NI-specific weights.

Replaces the previous English STAR-PU weights with NI-derived weights
from the 2023 Prescribing Formula Review (PFR2024-02), collapsed to
the 4 BSO age bands (<18, 18-44, 45-64, 65+).

Inputs:
  - ni_starpu_2023_collapsed_bso_bands.csv   (NI weights by age/sex/chapter)
  - NI/output/practice_demographics.parquet  (BSO practice populations by age/sex/year)
  - data/practices.parquet                   (practice → LCG mapping)

Outputs:
  - data/starpu_denominators_practice.parquet
  - data/starpu_denominators_lcg.parquet
"""

import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

WEIGHTS_CSV = os.path.join(
    SCRIPT_DIR, "..", "prescribing", "ni_starpu_2023_collapsed_bso_bands.csv"
)
DEMOGRAPHICS_PARQUET = os.path.join(
    SCRIPT_DIR, "..", "prescribing", "NI", "output", "practice_demographics.parquet"
)
PRACTICES_PARQUET = os.path.join(DATA_DIR, "practices.parquet")

OUT_PRACTICE = os.path.join(DATA_DIR, "starpu_denominators_practice.parquet")
OUT_LCG = os.path.join(DATA_DIR, "starpu_denominators_lcg.parquet")

# ── Chapter number mapping ───────────────────────────────────────────────────
# NI weights CSV uses "BNF1", "BNF2", etc.; parquets use integer chapter numbers
CHAPTER_MAP = {
    "BNF1": 1, "BNF2": 2, "BNF3": 3, "BNF4": 4, "BNF5": 5,
    "BNF6": 6, "BNF7": 7, "BNF9": 9, "BNF10": 10, "BNF13": 13,
}


def main():
    # ── Load NI weights ──────────────────────────────────────────────────────
    weights = pd.read_csv(WEIGHTS_CSV)
    weights["bnf_chapter"] = weights["chapter"].map(CHAPTER_MAP)
    weights["sex"] = weights["sex"].str.capitalize()  # "female" → "Female"
    weights.rename(columns={"bso_age_band": "age_band", "ni_starpu_weight": "weight"}, inplace=True)
    weights = weights[["age_band", "sex", "bnf_chapter", "weight"]]
    print(f"Loaded {len(weights)} NI weight rows across {weights['bnf_chapter'].nunique()} chapters")

    # ── Load practice demographics ───────────────────────────────────────────
    demog = pd.read_parquet(DEMOGRAPHICS_PARQUET)
    demog = demog[["year", "practice", "sex", "age_band", "population"]]
    print(f"Loaded demographics: {len(demog)} rows, years {demog['year'].min()}-{demog['year'].max()}")

    # ── Compute practice-level STAR-PU ───────────────────────────────────────
    # Cross-join demographics with all 10 chapters, then merge weights
    chapters = weights["bnf_chapter"].unique()
    demog_expanded = pd.concat(
        [demog.assign(bnf_chapter=ch) for ch in chapters],
        ignore_index=True,
    )

    merged = demog_expanded.merge(
        weights,
        on=["age_band", "sex", "bnf_chapter"],
        how="left",
    )

    # STAR-PU = sum of (population × weight) for each practice/year/chapter
    merged["weighted_pop"] = merged["population"] * merged["weight"]

    practice_starpu = (
        merged.groupby(["year", "practice", "bnf_chapter"])
        .agg(starpu=("weighted_pop", "sum"), total_population=("population", "sum"))
        .reset_index()
    )

    # Sort for tidy output
    practice_starpu.sort_values(["year", "practice", "bnf_chapter"], inplace=True)
    practice_starpu.reset_index(drop=True, inplace=True)

    print(f"Practice STAR-PU: {len(practice_starpu)} rows, "
          f"{practice_starpu['practice'].nunique()} practices")

    # ── Save practice parquet ────────────────────────────────────────────────
    practice_starpu.to_parquet(OUT_PRACTICE, index=False)
    print(f"Saved {OUT_PRACTICE}")

    # ── Compute LCG-level STAR-PU ────────────────────────────────────────────
    practices = pd.read_parquet(PRACTICES_PARQUET)
    prac_lcg = practices[["PracNo", "LCG"]].copy()
    prac_lcg["practice"] = pd.to_numeric(prac_lcg["PracNo"], errors="coerce").astype("Int64")
    prac_lcg["LCG"] = prac_lcg["LCG"].str.strip()
    prac_lcg = prac_lcg[["practice", "LCG"]].dropna()

    lcg_data = practice_starpu.copy()
    lcg_data["practice"] = lcg_data["practice"].astype("Int64")
    lcg_data = lcg_data.merge(prac_lcg, on="practice", how="left")

    # Some older practices may not be in the current practices list — drop them
    lcg_data = lcg_data.dropna(subset=["LCG"])

    lcg_starpu = (
        lcg_data.groupby(["year", "LCG", "bnf_chapter"])
        .agg(starpu=("starpu", "sum"), total_population=("total_population", "sum"))
        .reset_index()
    )
    lcg_starpu.rename(columns={"LCG": "lcg"}, inplace=True)
    lcg_starpu.sort_values(["year", "lcg", "bnf_chapter"], inplace=True)
    lcg_starpu.reset_index(drop=True, inplace=True)

    print(f"LCG STAR-PU: {len(lcg_starpu)} rows, LCGs: {sorted(lcg_starpu['lcg'].unique())}")

    # ── Save LCG parquet ─────────────────────────────────────────────────────
    lcg_starpu.to_parquet(OUT_LCG, index=False)
    print(f"Saved {OUT_LCG}")

    # ── Summary comparison ───────────────────────────────────────────────────
    print("\n── Sample: Practice 1, 2025 ──")
    sample = practice_starpu[
        (practice_starpu["practice"] == 1) & (practice_starpu["year"] == 2025)
    ]
    for _, row in sample.iterrows():
        print(f"  Ch {int(row['bnf_chapter']):2d}: STAR-PU = {row['starpu']:,.1f}  (pop = {int(row['total_population'])})")


if __name__ == "__main__":
    main()
