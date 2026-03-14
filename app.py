"""
NI GP Prescribing Explorer
==========================
A Streamlit dashboard for exploring GP prescribing variation
across practices, LCGs and HSC Trust areas in Northern Ireland.

Data is automatically downloaded from OpenDataNI (Open Government Licence)
on first run if not already present locally.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import requests
import time

# ── page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Northern Ireland GP Prescribing Explorer",
    page_icon="💊",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
CACHE_FILE = os.path.join(APP_DIR, ".cache", "merged.pkl")
CACHE_PRACTICES = os.path.join(APP_DIR, ".cache", "practices.pkl")
PARQUET_PRESCRIBING = os.path.join(DATA_DIR, "prescribing.parquet")
PARQUET_PRACTICES = os.path.join(DATA_DIR, "practices.parquet")
PARQUET_QOF = os.path.join(DATA_DIR, "qof.parquet")
PARQUET_PREVALENCE = os.path.join(DATA_DIR, "prevalence.parquet")

# ── OpenDataNI CKAN API ────────────────────────────────────────────────
CKAN_API = "https://admin.opendatani.gov.uk/api/3/action"
PRESCRIBING_DATASET = "gp-prescribing-data"
PRACTICE_LIST_DATASET = "gp-practice-list-sizes"

# LCG → Trust mapping
LCG_TO_TRUST = {
    "Belfast": "Belfast HSC Trust",
    "South Eastern": "South Eastern HSC Trust",
    "Northern": "Northern HSC Trust",
    "Southern": "Southern HSC Trust",
    "Western": "Western HSC Trust",
}

# ── therapeutic areas ───────────────────────────────────────────────────
THERAPEUTIC_AREAS = {
    "All prescribing": {"filter": None, "description": "All items prescribed"},
    "Statins": {
        "filter": lambda df: df[df["VTM_NM"].str.contains("statin", case=False, na=False)
                                & ~df["VTM_NM"].str.contains("nystatin", case=False, na=False)],
        "description": "HMG-CoA reductase inhibitors for cardiovascular risk reduction (NICE CG181)",
    },
    "Ezetimibe": {
        "filter": lambda df: df[df["VTM_NM"].str.contains("ezetimibe", case=False, na=False)],
        "description": "Cholesterol absorption inhibitor, add-on to statin therapy (NICE CG181)",
    },
    "UTI antibiotics": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "nitrofurantoin|trimethoprim|pivmecillinam|fosfomycin|cefalexin|ciprofloxacin|co-amoxiclav|amoxicillin",
            case=False, na=False
        )],
        "description": "Antibiotics commonly used for urinary tract infections (NICE NG109)",
    },
    "Antidepressants (excl. TCAs)": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "sertraline|citalopram|fluoxetine|mirtazapine|venlafaxine|duloxetine|paroxetine|escitalopram|trazodone",
            case=False, na=False
        )],
        "description": "SSRIs, SNRIs and mirtazapine — excludes TCAs which are widely prescribed for pain/migraine (NICE CG90)",
    },
    "Gabapentinoids": {
        "filter": lambda df: df[df["VTM_NM"].str.contains("gabapentin|pregabalin", case=False, na=False)],
        "description": "Gabapentin and pregabalin – controlled drugs since April 2019 (NICE NG193)",
    },
    "Opioids": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "morphine|codeine|tramadol|oxycodone|fentanyl|buprenorphine|dihydrocodeine|co-codamol|co-dydramol|tapentadol|pethidine",
            case=False, na=False
        )],
        "description": "Opioid analgesics — excludes methadone (mostly opioid substitution therapy) (NICE NG193)",
    },
    "PPIs": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "omeprazole|lansoprazole|pantoprazole|esomeprazole|rabeprazole",
            case=False, na=False
        )],
        "description": "Proton pump inhibitors – review for step-down/deprescribing",
    },
    "Diabetes (non-insulin)": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "metformin|gliclazide|sitagliptin|empagliflozin|dapagliflozin|canagliflozin|semaglutide|liraglutide|dulaglutide|pioglitazone|alogliptin|linagliptin|saxagliptin",
            case=False, na=False
        )],
        "description": "Non-insulin glucose-lowering agents (NICE NG28)",
    },
    "SGLT2 inhibitors": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "dapagliflozin|empagliflozin|canagliflozin|ertugliflozin",
            case=False, na=False
        )],
        "description": "Sodium-glucose co-transporter 2 inhibitors – CV and renal benefits beyond glycaemic control (NICE NG28, TA775, TA877)",
    },
    "GLP-1 receptor agonists": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "semaglutide|liraglutide|dulaglutide|exenatide|lixisenatide",
            case=False, na=False
        )],
        "description": "GLP-1 RAs – CV benefit, weight management; supply constraints apply (NICE NG28)",
    },
    "DPP-4 inhibitors": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "sitagliptin|linagliptin|saxagliptin|alogliptin|vildagliptin",
            case=False, na=False
        )],
        "description": "Dipeptidyl peptidase-4 inhibitors – consider stepping down to SGLT2i or GLP-1 RA where appropriate (NICE NG28)",
    },
    "Anticoagulants (oral)": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "warfarin|apixaban|rivaroxaban|edoxaban|dabigatran",
            case=False, na=False
        )],
        "description": "Oral anticoagulants – DOACs and warfarin for AF, VTE and stroke prevention (NICE CG180, NG196)",
    },
    "Antihypertensives": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "ramipril|lisinopril|perindopril|enalapril|captopril"
            "|losartan|candesartan|irbesartan|valsartan|olmesartan|telmisartan"
            "|amlodipine|felodipine|nifedipine|lercanidipine"
            "|bendroflumethiazide|indapamide|chlortalidone"
            "|doxazosin",
            case=False, na=False
        )],
        "description": "ACEi, ARBs, CCBs, thiazide-like diuretics and alpha-blockers for hypertension (NICE NG136)",
    },
    "HRT": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "estradiol|oestrogen|progesterone|utrogestan|tibolone|norethisterone"
            "|dydrogesterone|medroxyprogesterone|conjugated oestrogens",
            case=False, na=False
        )],
        "description": "Hormone replacement therapy — includes estrogens, progestogens and combination preparations (NICE NG23)",
    },
}

# ── BNF chapter names ────────────────────────────────────────────────
BNF_CHAPTERS = {
    0: "Other / unclassified",
    1: "Gastro-intestinal system",
    2: "Cardiovascular system",
    3: "Respiratory system",
    4: "Central nervous system",
    5: "Infections",
    6: "Endocrine system",
    7: "Obstetrics, gynaecology & UTI",
    8: "Malignant disease & immunosuppression",
    9: "Nutrition & blood",
    10: "Musculoskeletal & joint diseases",
    11: "Eye",
    12: "Ear, nose & oropharynx",
    13: "Skin",
    14: "Immunological products & vaccines",
    15: "Anaesthesia",
    18: "Preparations used in diagnosis",
    19: "Other drugs & preparations",
    20: "Dressings",
    21: "Appliances",
    22: "Incontinence appliances",
    23: "Stoma appliances",
    99: "Not classified",
}

# Chapters that have STAR-PU weightings
STARPU_CHAPTERS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13]

# Parquet paths for time-series data
PARQUET_TS_PRACTICE = os.path.join(DATA_DIR, "standardised_rates_practice.parquet")
PARQUET_TS_LCG = os.path.join(DATA_DIR, "standardised_rates_lcg.parquet")
PARQUET_TA_NI = os.path.join(DATA_DIR, "therapeutic_area_ni_monthly.parquet")
PARQUET_TA_PRACTICE = os.path.join(DATA_DIR, "therapeutic_area_practice_monthly.parquet")
PARQUET_STARPU_PRACTICE = os.path.join(DATA_DIR, "starpu_denominators_practice.parquet")
PARQUET_PRESC_PRACTICE = os.path.join(DATA_DIR, "prescribing_practice_monthly.parquet")
PARQUET_PRESC_LCG = os.path.join(DATA_DIR, "prescribing_lcg_monthly.parquet")


# ════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD & LOADING
# ════════════════════════════════════════════════════════════════════════

def ckan_get(action, params=None):
    """Call the OpenDataNI CKAN API."""
    url = f"{CKAN_API}/{action}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            if body.get("success"):
                return body["result"]
        except Exception:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None


def download_csv(url):
    """Download a CSV from a URL, return as bytes."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None


def download_data_from_opendatani(latest_n=3, progress_bar=None):
    """
    Download the latest N months of prescribing data and the most recent
    practice list from OpenDataNI.
    Returns (prescribing_frames, practice_df) or raises on failure.
    """
    # ── practice list ───────────────────────────────────────────────────
    if progress_bar:
        progress_bar.progress(0.05, "Fetching practice list metadata…")

    result = ckan_get("package_show", {"id": PRACTICE_LIST_DATASET})
    if not result:
        raise ConnectionError("Could not connect to OpenDataNI API")

    csv_resources = [r for r in result["resources"] if r["format"].upper() == "CSV"]
    csv_resources.sort(key=lambda r: r.get("created", ""), reverse=True)

    if not csv_resources:
        raise FileNotFoundError("No practice list CSV found on OpenDataNI")

    if progress_bar:
        progress_bar.progress(0.10, "Downloading practice list…")

    prac_bytes = download_csv(csv_resources[0]["url"])
    if not prac_bytes:
        raise ConnectionError("Failed to download practice list")

    practice_df = pd.read_csv(io.BytesIO(prac_bytes), dtype={"PracNo": str})

    # Normalise columns
    col_map = {}
    for c in practice_df.columns:
        cl = c.strip().lower()
        if "pracno" in cl or "prac_no" in cl or cl == "practice":
            col_map[c] = "PracNo"
        elif "practicename" in cl or "practice_name" in cl or "practice name" in cl:
            col_map[c] = "PracticeName"
        elif cl == "postcode":
            col_map[c] = "Postcode"
        elif cl == "lcg":
            col_map[c] = "LCG"
        elif "registered" in cl and "patient" in cl:
            col_map[c] = "RegisteredPatients"
    practice_df = practice_df.rename(columns=col_map)
    practice_df["PracNo"] = practice_df["PracNo"].astype(str).str.strip()
    if "LCG" in practice_df.columns:
        practice_df["LCG"] = practice_df["LCG"].str.strip()
        practice_df["Trust"] = practice_df["LCG"].map(LCG_TO_TRUST).fillna("Unknown")
    if "RegisteredPatients" in practice_df.columns:
        practice_df["RegisteredPatients"] = pd.to_numeric(practice_df["RegisteredPatients"], errors="coerce")
    keep = [c for c in ["PracNo", "PracticeName", "Postcode", "LCG", "Trust", "RegisteredPatients"]
            if c in practice_df.columns]
    practice_df = practice_df[keep].copy()

    # ── prescribing data ────────────────────────────────────────────────
    if progress_bar:
        progress_bar.progress(0.20, "Fetching prescribing data metadata…")

    result = ckan_get("package_show", {"id": PRESCRIBING_DATASET})
    csv_resources = [r for r in result["resources"] if r["format"].upper() == "CSV"]
    csv_resources.sort(key=lambda r: r.get("created", ""), reverse=True)
    csv_resources = csv_resources[:latest_n]

    frames = []
    for i, r in enumerate(csv_resources):
        name = r.get("name", "") or r["url"].split("/")[-1]
        frac = 0.25 + (0.65 * (i / len(csv_resources)))
        if progress_bar:
            progress_bar.progress(frac, f"Downloading {name}…")
        raw = download_csv(r["url"])
        if raw:
            df = pd.read_csv(io.BytesIO(raw), dtype={"Practice": str},
                             low_memory=False, encoding="latin-1")
            frames.append(df)

    if not frames:
        raise ConnectionError("Failed to download prescribing data")

    prescribing = pd.concat(frames, ignore_index=True)

    # Normalise prescribing columns
    col_map = {}
    for c in prescribing.columns:
        cl = c.strip().lower().replace("()", "").replace("£", "").strip()
        if cl == "practice": col_map[c] = "Practice"
        elif cl == "year": col_map[c] = "Year"
        elif cl == "month": col_map[c] = "Month"
        elif cl == "vtm_nm": col_map[c] = "VTM_NM"
        elif cl == "vmp_nm": col_map[c] = "VMP_NM"
        elif cl == "amp_nm": col_map[c] = "AMP_NM"
        elif cl == "presentation": col_map[c] = "Presentation"
        elif cl == "strength": col_map[c] = "Strength"
        elif "total items" in cl: col_map[c] = "TotalItems"
        elif "total quantity" in cl: col_map[c] = "TotalQuantity"
        elif "gross" in cl and "cost" in cl: col_map[c] = "GrossCost"
        elif "actual" in cl and "cost" in cl: col_map[c] = "ActualCost"
        elif cl == "bnf code": col_map[c] = "BNFCode"
        elif cl == "bnf chapter": col_map[c] = "BNFChapter"
        elif cl == "bnf section": col_map[c] = "BNFSection"
        elif cl == "bnf paragraph": col_map[c] = "BNFParagraph"
        elif "bnf sub" in cl and "paragraph" in cl: col_map[c] = "BNFSubParagraph"
    prescribing = prescribing.rename(columns=col_map)
    prescribing["Practice"] = prescribing["Practice"].astype(str).str.strip()
    for col in ["TotalItems", "TotalQuantity", "GrossCost", "ActualCost"]:
        if col in prescribing.columns:
            prescribing[col] = pd.to_numeric(prescribing[col], errors="coerce")

    # ── merge ───────────────────────────────────────────────────────────
    if progress_bar:
        progress_bar.progress(0.92, "Merging data…")

    merged = prescribing.merge(practice_df, left_on="Practice", right_on="PracNo", how="left")

    return merged, practice_df


def load_local_data():
    """Try to load data from local CSV files (data/ directory)."""
    import glob
    presc_pattern = os.path.join(DATA_DIR, "prescribing", "*.csv")
    prac_pattern = os.path.join(DATA_DIR, "practice_list_sizes", "*.csv")
    presc_files = sorted(glob.glob(presc_pattern))
    prac_files = sorted(glob.glob(prac_pattern))

    if not presc_files or not prac_files:
        return None, None

    # Load practice list (most recent)
    practice_df = pd.read_csv(prac_files[-1], dtype={"PracNo": str})
    col_map = {}
    for c in practice_df.columns:
        cl = c.strip().lower()
        if "pracno" in cl or "prac_no" in cl or cl == "practice": col_map[c] = "PracNo"
        elif "practicename" in cl or "practice_name" in cl or "practice name" in cl: col_map[c] = "PracticeName"
        elif cl == "postcode": col_map[c] = "Postcode"
        elif cl == "lcg": col_map[c] = "LCG"
        elif "registered" in cl and "patient" in cl: col_map[c] = "RegisteredPatients"
    practice_df = practice_df.rename(columns=col_map)
    practice_df["PracNo"] = practice_df["PracNo"].astype(str).str.strip()
    if "LCG" in practice_df.columns:
        practice_df["LCG"] = practice_df["LCG"].str.strip()
        practice_df["Trust"] = practice_df["LCG"].map(LCG_TO_TRUST).fillna("Unknown")
    if "RegisteredPatients" in practice_df.columns:
        practice_df["RegisteredPatients"] = pd.to_numeric(practice_df["RegisteredPatients"], errors="coerce")
    keep = [c for c in ["PracNo", "PracticeName", "Postcode", "LCG", "Trust", "RegisteredPatients"]
            if c in practice_df.columns]
    practice_df = practice_df[keep].copy()

    # Load prescribing
    frames = []
    for f in presc_files:
        df = pd.read_csv(f, dtype={"Practice": str}, low_memory=False, encoding="latin-1")
        frames.append(df)
    prescribing = pd.concat(frames, ignore_index=True)

    col_map = {}
    for c in prescribing.columns:
        cl = c.strip().lower().replace("()", "").replace("£", "").strip()
        if cl == "practice": col_map[c] = "Practice"
        elif cl == "vtm_nm": col_map[c] = "VTM_NM"
        elif "total items" in cl: col_map[c] = "TotalItems"
        elif "actual" in cl and "cost" in cl: col_map[c] = "ActualCost"
        elif cl == "bnf paragraph": col_map[c] = "BNFParagraph"
    prescribing = prescribing.rename(columns=col_map)
    prescribing["Practice"] = prescribing["Practice"].astype(str).str.strip()
    for col in ["TotalItems", "ActualCost"]:
        if col in prescribing.columns:
            prescribing[col] = pd.to_numeric(prescribing[col], errors="coerce")

    merged = prescribing.merge(practice_df, left_on="Practice", right_on="PracNo", how="left")
    return merged, practice_df


@st.cache_data(show_spinner="Loading data…")
def load_data():
    """
    Load data with this priority:
    1. Pickle cache (.cache/ directory)
    2. Bundled parquet files (data/ directory)
    3. Local CSV files (data/ directory)
    4. Download from OpenDataNI
    """
    # 1. Try pickle cache
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_PRACTICES):
        merged = pd.read_pickle(CACHE_FILE)
        practices = pd.read_pickle(CACHE_PRACTICES)
        return merged, practices

    # 2. Try bundled parquet files
    if os.path.exists(PARQUET_PRESCRIBING) and os.path.exists(PARQUET_PRACTICES):
        practices = pd.read_parquet(PARQUET_PRACTICES)
        prescribing = pd.read_parquet(PARQUET_PRESCRIBING)
        merged = prescribing  # Already merged when parquet was created
        # Cache as pickle for faster subsequent loads
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        merged.to_pickle(CACHE_FILE)
        practices.to_pickle(CACHE_PRACTICES)
        return merged, practices

    # 3. Try local CSVs
    merged, practices = load_local_data()
    if merged is not None:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        merged.to_pickle(CACHE_FILE)
        practices.to_pickle(CACHE_PRACTICES)
        return merged, practices

    # 4. Download from OpenDataNI
    return None, None  # Signal that download is needed


def per_cap(merged, area_filter):
    """Compute per-capita items by practice for a therapeutic area.

    Uses only months where all (or nearly all) practices reported,
    then divides by the number of complete months to give a monthly average.
    """
    if area_filter is None:
        df = merged
    else:
        df = area_filter(merged)

    # Identify complete months (those with a reasonable number of practices)
    if "Month" in df.columns:
        practices_per_month = df.groupby("Month")["Practice"].nunique()
        max_practices = practices_per_month.max()
        # Keep months where at least 80% of practices reported
        complete_months = practices_per_month[practices_per_month >= max_practices * 0.8].index
        if len(complete_months) > 0:
            df = df[df["Month"].isin(complete_months)]
        n_months = len(complete_months) if len(complete_months) > 0 else 1
    else:
        n_months = 1

    group_cols = ["Practice", "PracticeName", "LCG", "Trust", "RegisteredPatients"]
    if "DepQuintile" in df.columns:
        group_cols.append("DepQuintile")
    if "Ward_Dep_Rank" in df.columns:
        group_cols.append("Ward_Dep_Rank")
    agg = (
        df.groupby(group_cols)
        .agg(TotalItems=("TotalItems", "sum"), TotalCost=("ActualCost", "sum"))
        .reset_index()
    )
    # Monthly average per capita
    agg["TotalItems"] = agg["TotalItems"] / n_months
    agg["TotalCost"] = agg["TotalCost"] / n_months
    agg["ItemsPerCapita"] = agg["TotalItems"] / agg["RegisteredPatients"]
    agg["CostPerCapita"] = agg["TotalCost"] / agg["RegisteredPatients"]
    return agg.sort_values("ItemsPerCapita", ascending=False)


@st.cache_data(show_spinner=False)
def per_cap_by_name(_merged, area_name):
    """Cached per-capita computation keyed by therapeutic area name.
    Avoids recomputing on every tab render."""
    ta = THERAPEUTIC_AREAS[area_name]
    return per_cap(_merged, ta["filter"])


@st.cache_data(show_spinner=False)
def load_qof():
    """Load QOF clinical achievement data if available.
    Normalises practice codes (Z00001 → 1) to match prescribing data."""
    if os.path.exists(PARQUET_QOF):
        qof = pd.read_parquet(PARQUET_QOF)
        # QOF uses Z00001 format; prescribing uses plain numbers
        qof["Practice"] = qof["Practice"].str.replace(r"^Z0*", "", regex=True)
        return qof
    return None


@st.cache_data(show_spinner=False)
def load_prevalence():
    """Load disease prevalence data (QOF register / practice list size).
    Practice codes already normalised to plain numbers."""
    if os.path.exists(PARQUET_PREVALENCE):
        return pd.read_parquet(PARQUET_PREVALENCE)
    return None


@st.cache_data(show_spinner="Loading time-series data…")
def load_timeseries_lcg():
    """Load LCG-level monthly standardised rates (all 154 months)."""
    if os.path.exists(PARQUET_TS_LCG):
        df = pd.read_parquet(PARQUET_TS_LCG)
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        df["chapter_name"] = df["bnf_chapter"].map(BNF_CHAPTERS).fillna("Unknown")
        return df
    return None


@st.cache_data(show_spinner="Loading practice time-series data…")
def load_timeseries_practice():
    """Load practice-level monthly standardised rates (all 154 months)."""
    if os.path.exists(PARQUET_TS_PRACTICE):
        df = pd.read_parquet(PARQUET_TS_PRACTICE)
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        df["chapter_name"] = df["bnf_chapter"].map(BNF_CHAPTERS).fillna("Unknown")
        return df
    return None


@st.cache_data(show_spinner="Loading therapeutic area time-series…")
def load_ta_ni():
    """Load NI-level monthly therapeutic area time series."""
    if os.path.exists(PARQUET_TA_NI):
        df = pd.read_parquet(PARQUET_TA_NI)
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        return df
    return None


@st.cache_data(show_spinner="Loading therapeutic area practice data…")
def load_ta_practice():
    """Load practice-level monthly therapeutic area time series."""
    if os.path.exists(PARQUET_TA_PRACTICE):
        df = pd.read_parquet(PARQUET_TA_PRACTICE)
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        return df
    return None


@st.cache_data(show_spinner="Loading STAR-PU denominators…")
def load_starpu_practice():
    """Load practice-level STAR-PU denominators by year and chapter."""
    if os.path.exists(PARQUET_STARPU_PRACTICE):
        return pd.read_parquet(PARQUET_STARPU_PRACTICE)
    return None


# Map therapeutic areas to their parent BNF chapter (for STAR-PU lookups)
TA_TO_CHAPTER = {
    "Statins": 2, "Ezetimibe": 2, "Anticoagulants": 2, "Antihypertensives": 2,
    "PPIs": 1, "UTI antibiotics": 5,
    "Antidepressants": 4, "Gabapentinoids": 4, "Opioids": 4,
    "Diabetes (non-insulin)": 6, "SGLT2 inhibitors": 6, "GLP-1 agonists": 6,
    "DPP-4 inhibitors": 6, "HRT": 6,
}


# ════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════

def caterpillar_chart(pc, highlight_practices=None, title="", metric="ItemsPerCapita"):
    """NI-wide caterpillar (ranking) plot with optional highlighted practices."""
    pc_sorted = pc.sort_values(metric).reset_index(drop=True)
    pc_sorted["rank"] = range(1, len(pc_sorted) + 1)
    ni_mean = pc_sorted[metric].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pc_sorted["rank"], pc_sorted[metric], color="#c0c0c0", width=1.0, edgecolor="none")
    ax.axhline(ni_mean, color="#333333", linewidth=1.2, linestyle="--", label=f"NI mean: {ni_mean:.2f}")

    if highlight_practices:
        for pracno, colour in highlight_practices:
            row = pc_sorted[pc_sorted["Practice"].str.strip() == str(pracno).strip()]
            if not row.empty:
                r = row.iloc[0]
                display_name = r.get("PracticeName", pracno)
                if isinstance(display_name, str):
                    display_name = display_name.strip()
                ax.bar(r["rank"], r[metric], color=colour, width=2.5, zorder=5)
                ax.annotate(
                    display_name,
                    (r["rank"], r[metric]),
                    textcoords="offset points",
                    xytext=(5, 8),
                    fontsize=8,
                    color=colour,
                    fontweight="bold",
                )

    label = "Items per registered patient" if metric == "ItemsPerCapita" else "Cost (£) per registered patient"
    ax.set_xlabel("Practice rank")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def trust_bar_chart(pc, title="", metric="ItemsPerCapita"):
    """Bar chart of per-capita rate by HSC Trust."""
    trust = pc.groupby("Trust").agg(
        TotalItems=("TotalItems", "sum"),
        TotalCost=("TotalCost", "sum"),
    ).reset_index()
    trust_pop = pc.drop_duplicates("Practice").groupby("Trust")["RegisteredPatients"].sum()
    trust["Pop"] = trust["Trust"].map(trust_pop)
    trust["ItemsPerCapita"] = trust["TotalItems"] / trust["Pop"]
    trust["CostPerCapita"] = trust["TotalCost"] / trust["Pop"]
    trust = trust.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    ax.barh(trust["Trust"], trust[metric], color=colours[:len(trust)])
    label = "Items per registered patient" if metric == "ItemsPerCapita" else "Cost (£) per registered patient"
    ax.set_xlabel(label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def lcg_summary(pc):
    """Summary stats by LCG."""
    lcg = pc.groupby("LCG").agg(
        Practices=("Practice", "nunique"),
        TotalItems=("TotalItems", "sum"),
        TotalCost=("TotalCost", "sum"),
    ).reset_index()
    lcg_pop = pc.drop_duplicates("Practice").groupby("LCG")["RegisteredPatients"].sum()
    lcg["RegisteredPatients"] = lcg["LCG"].map(lcg_pop)
    lcg["ItemsPerCapita"] = lcg["TotalItems"] / lcg["RegisteredPatients"]
    lcg["CostPerCapita"] = lcg["TotalCost"] / lcg["RegisteredPatients"]
    return lcg.sort_values("ItemsPerCapita", ascending=False)


def practice_detail(pc, pracno, ni_mean):
    """Metrics for a specific practice vs NI mean."""
    row = pc[pc["Practice"].str.strip() == str(pracno).strip()]
    if row.empty:
        return None
    r = row.iloc[0]
    pct_diff = ((r["ItemsPerCapita"] - ni_mean) / ni_mean) * 100
    rank = (pc["ItemsPerCapita"] <= r["ItemsPerCapita"]).sum()
    return {
        "Practice": r["PracticeName"],
        "LCG": r["LCG"],
        "Registered patients": f"{int(r['RegisteredPatients']):,}",
        "Total items": f"{int(r['TotalItems']):,}",
        "Items per capita": f"{r['ItemsPerCapita']:.2f}",
        "NI mean": f"{ni_mean:.2f}",
        "vs NI mean": f"{pct_diff:+.1f}%",
        "Rank": f"{rank} / {len(pc)}",
    }


# ── Consistent LCG colour palette ─────────────────────────────────────
LCG_COLOURS = {
    "Belfast": "#2196F3",
    "Northern": "#4CAF50",
    "South Eastern": "#FF9800",
    "Southern": "#9C27B0",
    "Western": "#F44336",
}

DEP_COLOURS = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#66bb6a", 5: "#1e88e5"}
QUINTILE_LABELS = {1: "Q1\nMost deprived", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5\nLeast deprived"}
QUINTILE_LABELS_FLAT = {1: "Q1 (most deprived)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (least deprived)"}

def _scatter_by_colour(ax, data, metric, colour_by):
    """Plot scatter points coloured by deprivation quintile or LCG."""
    if colour_by == "LCG":
        for lcg_name in sorted(data["LCG"].dropna().unique()):
            lcg_d = data[data["LCG"] == lcg_name]
            ax.scatter(lcg_d["Ward_Dep_Rank"], lcg_d[metric],
                       color=LCG_COLOURS.get(lcg_name, "#999"),
                       alpha=0.6, s=30, label=lcg_name)
    else:
        for q in sorted(data["DepQuintile"].dropna().unique()):
            qd = data[data["DepQuintile"] == q]
            ax.scatter(qd["Ward_Dep_Rank"], qd[metric],
                       color=DEP_COLOURS.get(int(q), "#999"),
                       alpha=0.6, s=30,
                       label=QUINTILE_LABELS.get(int(q), f"Q{int(q)}"))


def _scatter_highlight_practice(ax, data, metric, pracno, pracno_to_label):
    """Overlay a highlighted practice on a scatter plot."""
    row = data[data["Practice"].str.strip() == str(pracno).strip()]
    if not row.empty:
        r = row.iloc[0]
        display = pracno_to_label.get(str(pracno).strip(), pracno)
        if isinstance(display, str) and "(" in display:
            display = display.split("(")[0].strip()
        ax.scatter(r["Ward_Dep_Rank"], r[metric],
                   color="#000000", s=180, marker="*", zorder=10,
                   edgecolors="#e53935", linewidths=1.5)
        ax.annotate(display, (r["Ward_Dep_Rank"], r[metric]),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=7, color="#e53935", fontweight="bold")


def _scatter_trend_line(ax, data, metric):
    """Add a linear trend line to a scatter."""
    if len(data) >= 10:
        from numpy.polynomial.polynomial import polyfit
        x = data["Ward_Dep_Rank"].values
        y = data[metric].values
        b, m_coef = polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, b + m_coef * x_line, color="#333", linewidth=1.5, linestyle="--")


# ════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════

def main():
    st.title("Northern Ireland GP Prescribing Explorer")
    st.caption("Data: OpenDataNI GP Prescribing · Open Government Licence · All data relates to GP practices in Northern Ireland")

    # ── load or download data ───────────────────────────────────────────
    merged, practices = load_data()

    if merged is None:
        st.info("Bundled data not found. Downloading the latest 3 months from OpenDataNI — this may take 1–3 minutes…")
        progress = st.progress(0, "Starting download…")
        try:
            merged, practices = download_data_from_opendatani(
                latest_n=3, progress_bar=progress
            )
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            merged.to_pickle(CACHE_FILE)
            practices.to_pickle(CACHE_PRACTICES)
            progress.progress(1.0, "Data loaded!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            progress.empty()
            st.error(f"Failed to download data: {e}")
            return

    # Show data period and flag incomplete months
    import calendar as cal_mod
    if "Month" in merged.columns and "Year" in merged.columns:
        practices_per_month = merged.groupby(["Year", "Month"])["Practice"].nunique().reset_index()
        practices_per_month.columns = ["Year", "Month", "Practices"]
        max_prac = practices_per_month["Practices"].max()

        parts = []
        used_months = []
        for _, row in practices_per_month.sort_values(["Year", "Month"]).iterrows():
            y, mo, n = int(row["Year"]), int(row["Month"]), int(row["Practices"])
            label = f"{cal_mod.month_abbr[mo]} {y}"
            if n < max_prac * 0.8:
                label += f" (incomplete – {n} practices)"
            else:
                used_months.append(label)
            parts.append(label)

        st.caption(f"Data period: **{', '.join(parts)}** · Monthly average using {len(used_months)} complete month{'s' if len(used_months) != 1 else ''}")

    # ════════════════════════════════════════════════════════════════════
    # RESTRUCTURED SIDEBAR
    # ════════════════════════════════════════════════════════════════════
    st.sidebar.header("Data & Analysis")

    # 1. UNIFIED DRUG / CHAPTER SELECTOR
    # Build a single list: therapeutic areas first, then BNF chapters
    _ts_lcg_check = load_timeseries_lcg()
    _unified_options = []  # list of (key, label) tuples
    # — Therapeutic areas (prefix "ta:" to distinguish from chapters)
    for _ta_key in THERAPEUTIC_AREAS:
        _unified_options.append(("ta:" + _ta_key, _ta_key))
    # — BNF chapters (prefix "ch:")
    if _ts_lcg_check is not None:
        _avail_chapters = sorted(_ts_lcg_check["bnf_chapter"].unique())
        for _ch in _avail_chapters:
            if _ch in BNF_CHAPTERS and _ch != 0:
                _lbl = f"Ch {_ch}: {BNF_CHAPTERS[_ch]}"
                if _ch in STARPU_CHAPTERS:
                    _lbl += " ★"
                _unified_options.append((f"ch:{_ch}", _lbl))

    _unified_keys = [k for k, _ in _unified_options]
    _unified_labels = {k: v for k, v in _unified_options}

    _sidebar_selection = st.sidebar.selectbox(
        "Prescribing area",
        _unified_keys,
        format_func=lambda x, _m=_unified_labels: _m.get(x, str(x)),
        index=0,
        key="sidebar_prescribing_area",
    )

    # Derive area_name, selected_chapter, and area from the unified selection
    if _sidebar_selection.startswith("ta:"):
        area_name = _sidebar_selection[3:]
        area = THERAPEUTIC_AREAS[area_name]
        # Map therapeutic area to its parent BNF chapter for STAR-PU lookups
        selected_chapter = TA_TO_CHAPTER.get(area_name, 0)
    else:
        # BNF chapter selected
        selected_chapter = int(_sidebar_selection[3:])
        area_name = "All prescribing"
        area = THERAPEUTIC_AREAS["All prescribing"]

    # Show description for therapeutic areas
    if area_name != "All prescribing":
        st.sidebar.caption(area["description"])

    # 2. INDIVIDUAL DRUG (overrides therapeutic area)
    all_drugs = sorted(merged["VTM_NM"].dropna().unique().tolist())
    sidebar_drug = st.sidebar.selectbox(
        "Individual drug (overrides selection above)",
        all_drugs,
        index=None,
        placeholder="Start typing to search…",
        key="sidebar_drug",
    )
    if sidebar_drug:
        st.sidebar.caption(f"Showing: **{sidebar_drug}**")

    # 3. METRIC
    metric = st.sidebar.radio(
        "Metric",
        ["ItemsPerCapita", "CostPerCapita"],
        format_func=lambda x: "Items" if x == "ItemsPerCapita" else "Cost (£)",
    )
    label_metric = "Items per patient" if metric == "ItemsPerCapita" else "Cost (£) per patient"

    # 4. RATE TYPE (Raw vs Standardised) — only for BNF chapters with STAR-PU
    _has_starpu = selected_chapter in STARPU_CHAPTERS
    if _has_starpu and _sidebar_selection.startswith("ch:"):
        _rate_opts = ["Raw (per capita)", "Standardised (per STAR-PU)"]
        ts_rate = st.sidebar.radio(
            "Rate type",
            _rate_opts,
            horizontal=True,
            key="ts_rate",
        )
        use_starpu = "Standardised" in ts_rate
    else:
        if _sidebar_selection.startswith("ch:") and not _has_starpu:
            st.sidebar.info("ℹ️ No STAR-PU weighting for this chapter — raw rates only")
        use_starpu = False

    # 6. SMOOTHING (for time series)
    smoothing = st.sidebar.selectbox(
        "Smoothing (time series)",
        ["Monthly (raw)", "3-month rolling average", "12-month rolling average"],
        index=2,
        key="ts_smoothing",
    )
    smooth_window = 1
    if "3-month" in smoothing:
        smooth_window = 3
    elif "12-month" in smoothing:
        smooth_window = 12

    # 7. PRACTICE HIGHLIGHTING (Find practices by + Highlight)
    st.sidebar.divider()
    st.sidebar.header("Practices")

    # Practice lookup helper
    _prac_display = (
        practices["Address1"].str.strip()
        if "Address1" in practices.columns
        else practices["PracticeName"].str.strip()
    )
    practices["PracNo"] = practices["PracNo"].astype(str).str.strip()
    _pracno_str = practices["PracNo"]
    practices["_label"] = (
        _prac_display
        + "  (" + practices["Postcode"].str.strip() + ", "
        + practices["LCG"].str.strip() + ", #"
        + _pracno_str + ")"
    )
    label_to_pracno = dict(zip(practices["_label"], _pracno_str))
    pracno_to_label = dict(zip(_pracno_str, practices["_label"]))

    # Full list of all practice labels (always available for multiselect)
    all_labels = sorted(label_to_pracno.keys())

    # Finder: helps you locate practices, then add them to the highlight list
    find_by = st.sidebar.radio(
        "Find practices by",
        ["Name", "Postcode area", "LCG / Trust", "Practice number"],
        horizontal=True,
        key="find_by",
    )

    if find_by == "Postcode area":
        bt_areas = sorted(practices["Postcode"].str.extract(r"(BT\d+)", expand=False).dropna().unique())
        selected_bt = st.sidebar.selectbox("Postcode area", bt_areas, key="postcode_area")
        filtered = practices[practices["Postcode"].str.startswith(selected_bt)]
        finder_labels = sorted(filtered["_label"].unique())
    elif find_by == "LCG / Trust":
        selected_lcg = st.sidebar.selectbox("LCG area", sorted(practices["LCG"].dropna().unique()), key="lcg_area")
        filtered = practices[practices["LCG"] == selected_lcg]
        finder_labels = sorted(filtered["_label"].unique())
    elif find_by == "Practice number":
        prac_num = st.sidebar.text_input("Practice number", "", key="prac_num_input")
        if prac_num.strip():
            filtered = practices[practices["PracNo"] == prac_num.strip()]
            finder_labels = sorted(filtered["_label"].unique())
        else:
            finder_labels = all_labels
    else:
        finder_labels = all_labels

    # Quick-add: pick from filtered list to add to highlights
    _quick_add = st.sidebar.selectbox(
        "Add a practice",
        finder_labels,
        index=None,
        placeholder="Pick from list above…",
        key="quick_add_practice",
    )

    # Multiselect with FULL list so selections persist across finder changes
    _current_default = st.session_state.get("_highlight_labels_persistent", [])
    # If user just picked one via quick-add, include it
    if _quick_add and _quick_add not in _current_default:
        _current_default = _current_default + [_quick_add]

    highlight_labels = st.sidebar.multiselect(
        "Selected practices",
        all_labels,
        default=_current_default,
        key="highlight_labels",
        help="Up to 5 practices shown across charts. Use 'Add a practice' above or type here.",
    )
    # Persist so they survive finder switches
    st.session_state["_highlight_labels_persistent"] = highlight_labels
    highlight_pracnos = [label_to_pracno.get(l, l) for l in highlight_labels]

    # 8. ABOUT EXPANDER (brief)
    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ About this dashboard"):
        st.markdown(
            "**Explores:** GP prescribing variation across NI practices, "
            "linked to deprivation and QOF outcomes.\n\n"
            "**Data:** OpenDataNI prescribing (Apr 2013–Jan 2026), "
            "QOF 2022/23, NIMDM 2017, STAR-PU 2023.\n\n"
            "See the **Overview** tab for full methodology."
        )

    # 9. REFRESH DATA
    if st.sidebar.button("Refresh data from OpenDataNI", key="refresh_btn"):
        for f in [CACHE_FILE, CACHE_PRACTICES]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_data.clear()
        st.rerun()

    # ── compute per-capita (for cross-sectional views) ───────────────────
    if sidebar_drug:
        drug_filter = lambda df, drug=sidebar_drug: df[
            df["VTM_NM"].str.strip().str.lower() == drug.strip().lower()
        ]
        pc = per_cap(merged, drug_filter)
        display_name = sidebar_drug
    else:
        pc = per_cap(merged, area["filter"])
        display_name = area_name
    ni_mean = pc[metric].mean()

    # ════════════════════════════════════════════════════════════════════
    # LOAD ALL DATA
    # ════════════════════════════════════════════════════════════════════
    qof_df = load_qof()
    prev_df = load_prevalence()
    ts_lcg = load_timeseries_lcg()
    ts_practice = load_timeseries_practice()

    # ════════════════════════════════════════════════════════════════════
    # NEW 4-TAB STRUCTURE
    # ════════════════════════════════════════════════════════════════════
    tab_names = ["Overview", "Practices", "NI Trends", "Practice Profile"]
    if qof_df is not None or prev_df is not None:
        tab_names.append("QOF")
    tabs = st.tabs(tab_names)
    tab_overview = tabs[0]
    tab_practices = tabs[1]
    tab_trends = tabs[2]
    tab_profile = tabs[3]
    if qof_df is not None or prev_df is not None:
        tab_qof = tabs[4]
    else:
        tab_qof = None

    # ════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.header(f"{display_name} – Overview")

        # ── Summary metrics ──────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Practices", f"{len(pc)}")
        col2.metric("Total items (monthly avg)", f"{int(pc['TotalItems'].sum()):,}")
        col3.metric("NI mean per capita", f"{ni_mean:.2f}")
        col4.metric("Total cost (monthly avg)", f"£{pc['TotalCost'].sum():,.0f}")

        # ── Caterpillar (practice ranking) ───────────────────────────────
        st.subheader("Practice ranking")
        colours = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]
        highlight = [(n, colours[i % len(colours)]) for i, n in enumerate(highlight_pracnos[:5])]
        fig = caterpillar_chart(pc, highlight,
                                title=f"{display_name} – {label_metric.lower()} by practice rank",
                                metric=metric)
        st.pyplot(fig)
        plt.close(fig)

        # ── Distribution ─────────────────────────────────────────────────
        st.subheader("Distribution across practices")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.hist(pc[metric].dropna(), bins=40, color="#42a5f5", edgecolor="white", alpha=0.85)
        ax2.axvline(ni_mean, color="#333", linewidth=1.2, linestyle="--", label=f"Mean: {ni_mean:.2f}")
        ax2.set_xlabel(label_metric)
        ax2.set_ylabel("Number of practices")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Trust / LCG chart ────────────────────────────────────────────
        st.subheader("By Trust / LCG")
        fig3 = trust_bar_chart(pc, title=f"{display_name} per capita by Trust", metric=metric)
        st.pyplot(fig3)
        plt.close(fig3)

        # ── LCG summary table ────────────────────────────────────────────
        lcg = lcg_summary(pc)
        display_cols = ["LCG", "Practices", "RegisteredPatients", "TotalItems",
                        "ItemsPerCapita", "TotalCost", "CostPerCapita"]
        st.dataframe(
            lcg[display_cols].style.format({
                "RegisteredPatients": "{:,.0f}",
                "TotalItems": "{:,.0f}",
                "ItemsPerCapita": "{:.2f}",
                "TotalCost": "£{:,.0f}",
                "CostPerCapita": "£{:.2f}",
            }),
            use_container_width=True,
        )

        # ── Highlighted practice cards ───────────────────────────────────
        if highlight_pracnos:
            st.subheader("Highlighted practices")
            cols = st.columns(min(len(highlight_pracnos), 3))
            for i, pno in enumerate(highlight_pracnos):
                detail = practice_detail(pc, pno, ni_mean)
                if detail:
                    with cols[i % 3]:
                        st.markdown(f"**{detail['Practice']}**")
                        st.markdown(f"LCG: {detail['LCG']}")
                        st.markdown(f"Patients: {detail['Registered patients']}")
                        st.markdown(f"Items/capita: **{detail['Items per capita']}** ({detail['vs NI mean']} vs NI)")
                        st.markdown(f"Rank: {detail['Rank']}")

        # ── About / Methodology (collapsible at bottom) ───────────────────
        st.divider()
        with st.expander("📖 Full methodology & data sources", expanded=False):
            st.markdown("""
This dashboard explores variation in GP prescribing across Northern Ireland
practices. It links prescribing data to practice-level deprivation and
QOF (Quality and Outcomes Framework) clinical achievement statistics.

### Data sources

| Source | Period | Licence |
|--------|--------|---------|
| GP Prescribing Data | April 2013 – January 2026 (154 monthly files from OpenDataNI) | Open Government Licence |
| GP Practice List Sizes | April 2023 reference file | Open Government Licence |
| QOF Clinical Achievement | 2022/23 | HSCB |
| NI Multiple Deprivation Measure | 2017 | NISRA |
| STAR-PU 2023 Weightings | Published November 2024 (DoH NI ref PFR2024_02) | DoH NI |
| Registered Patients by Practice, Gender & Age Group | 2014–2025 | BSO |
| GMS Statistics Annual Tables (LCG-level demographics) | 2014–2025 | BSO |

### How prescribing rates are calculated

Two measures of prescribing rate are available:

**1. Items per registered patient (raw rate):** For each practice,
prescribing is expressed as items per registered patient per month
(or cost per patient per month). Only months where at least 80% of
practices reported data are included. This simple per-capita measure
does not account for differences in age and sex structure between practices.

**2. Items per STAR-PU (age-sex standardised rate):** STAR-PU (Specific
Therapeutic Group Age-Sex Related Prescribing Units) adjusts for the
expected level of prescribing given a practice's or area's age-sex
profile. The NI-specific 2023 STAR-PU weightings (DoH NI, November 2024)
assign different weights to each age-sex band for each BNF chapter,
reflecting how much prescribing is typically needed. For example, a
practice with many elderly patients will have a higher STAR-PU denominator,
so its standardised rate reflects prescribing *relative to what is
expected* for its population mix.

STAR-PU weights are available for 10 BNF chapters (1: GI, 2: Cardiovascular,
3: Respiratory, 4: CNS, 5: Infections, 6: Endocrine, 7: Obs/Gynae/UT,
9: Nutrition, 10: MSK, 13: Skin) and 24 BNF sections/paragraphs.

### Therapeutic area drug groupings

The drug lists are based on NICE and BNF guidance. Some deliberate
exclusions are made where a drug has major indications outside the
therapeutic area, which would distort practice-level comparisons.

See the **About** tab in the original dashboard for full details on
drug groupings, deprivation mapping, and limitations.
""")


    # ════════════════════════════════════════════════════════════════════
    # TAB 2: PRACTICES
    # ════════════════════════════════════════════════════════════════════
    with tab_practices:
        st.header("Practice Analysis")

        if not highlight_pracnos:
            st.info("Select practices using the **Practices** section in the sidebar to view their analysis here.")

        # ── Per-practice details ───────────────────────────────────────
        _practice_colours_tab = ["#e53935", "#2563eb", "#16a34a", "#d97706", "#7c3aed"]
        for _hp_idx, _hp_pno in enumerate(highlight_pracnos[:5]):
            selected_pracno = str(_hp_pno).strip()
            selected_label = pracno_to_label.get(selected_pracno, selected_pracno)

            if _hp_idx > 0:
                st.divider()
            st.subheader(f"{selected_label}")

            prac_matches = practices[practices["PracNo"] == selected_pracno]
            if not prac_matches.empty:
                prac_row = prac_matches.iloc[0]
                dep_q = prac_row.get("DepQuintile", None)
                dep_label = QUINTILE_LABELS_FLAT.get(
                    int(dep_q) if pd.notna(dep_q) else None, "—"
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("LCG", prac_row.get("LCG", "—"))
                c2.metric("Trust", prac_row.get("Trust", "—"))
                c3.metric("Registered patients", f"{int(prac_row.get('RegisteredPatients', 0)):,}")
                c4.metric("Deprivation", dep_label)
            else:
                st.warning("Practice details not found in practice list.")

            # ── Caterpillar: Where this practice sits ────────────────────
            st.markdown(f"**Where this practice sits – {display_name}**")
            highlight_this = [(selected_pracno, _practice_colours_tab[_hp_idx % 5])]
            fig_cat = caterpillar_chart(
                pc, highlight_this,
                title=f"{display_name} – all NI practices",
                metric=metric,
            )
            st.pyplot(fig_cat)
            plt.close(fig_cat)

            # ── Performance across all therapeutic areas ──────────────────
            st.markdown("**Performance across therapeutic areas**")
            prac_display = selected_label.split("(")[0].strip() if selected_label else selected_pracno

            rows = []
            for _ta_iter_name in THERAPEUTIC_AREAS:
                ta_pc = per_cap_by_name(merged, _ta_iter_name)
                ta_mean = ta_pc[metric].mean()
                prac = ta_pc[ta_pc["Practice"].str.strip() == str(selected_pracno).strip()]
                if not prac.empty:
                    val = prac.iloc[0][metric]
                    pct = ((val - ta_mean) / ta_mean) * 100
                    rank = int((ta_pc[metric] <= val).sum())
                    rows.append({
                        "Therapeutic area": _ta_iter_name,
                        metric: val,
                        "NI mean": ta_mean,
                        "vs NI mean": pct,
                        "Rank": f"{rank}/{len(ta_pc)}",
                    })

            if rows:
                results = pd.DataFrame(rows)

                fig, ax = plt.subplots(figsize=(10, 5))
                x = range(len(results))
                w = 0.35
                ax.bar([i - w / 2 for i in x], results[metric], w,
                       label=prac_display, color="#1e88e5")
                ax.bar([i + w / 2 for i in x], results["NI mean"], w,
                       label="NI mean", color="#bdbdbd")
                ax.set_xticks(list(x))
                ax.set_xticklabels(results["Therapeutic area"], rotation=35, ha="right", fontsize=8)
                ax.set_ylabel(label_metric)
                ax.set_title(f"{prac_display} vs NI average")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(
                    results.style.format({
                        metric: "{:.2f}",
                        "NI mean": "{:.2f}",
                        "vs NI mean": "{:+.1f}%",
                    }).map(
                        lambda v: "color: #e53935" if isinstance(v, (int, float)) and v > 20
                        else ("color: #43a047" if isinstance(v, (int, float)) and v < -20 else ""),
                        subset=["vs NI mean"],
                    ),
                    use_container_width=True,
                )

        # ── Deprivation scatter (shown once, highlights all selected practices) ──
        if highlight_pracnos:
            st.divider()
            st.subheader("Practice-level deprivation analysis")
            st.caption("Deprivation quintiles based on NIMDM 2017 ward-level scores (1 = most deprived, 5 = least deprived)")

            has_dep = "DepQuintile" in pc.columns and pc["DepQuintile"].notna().any()
            if not has_dep:
                st.warning("Deprivation data not available. Refresh data to include NIMDM linkage.")
            elif "Ward_Dep_Rank" in pc.columns:
                scatter_data = pc.dropna(subset=["Ward_Dep_Rank", metric])

                ctrl1, ctrl2 = st.columns(2)
                with ctrl1:
                    colour_by = st.radio(
                        "Colour practices by",
                        ["Deprivation quintile", "LCG"],
                        horizontal=True,
                        key="practices_tab_colour",
                    )
                with ctrl2:
                    lcg_options = ["All Northern Ireland"] + sorted(pc["LCG"].dropna().unique().tolist())
                    selected_dep_lcg = st.selectbox("Filter by LCG", lcg_options, key="practices_tab_lcg")

                if selected_dep_lcg != "All Northern Ireland":
                    scatter_data = scatter_data[scatter_data["LCG"] == selected_dep_lcg]
                    area_label = selected_dep_lcg
                else:
                    area_label = "Northern Ireland"

                fig_scat, ax_scat = plt.subplots(figsize=(10, 5))
                _scatter_by_colour(ax_scat, scatter_data, metric, colour_by)
                _scatter_trend_line(ax_scat, scatter_data, metric)

                # Highlight all selected practices
                for _hp_idx2, _hp_pno2 in enumerate(highlight_pracnos[:5]):
                    _scatter_highlight_practice(
                        ax_scat, scatter_data, metric,
                        str(_hp_pno2).strip(), pracno_to_label,
                    )

                ax_scat.set_xlabel("Ward deprivation rank (1 = most deprived)")
                ax_scat.set_ylabel(label_metric)
                ax_scat.set_title(f"{display_name} – prescribing vs deprivation ({area_label})")
                ax_scat.legend(fontsize=8)
                fig_scat.tight_layout()
                st.pyplot(fig_scat)
                plt.close(fig_scat)

                from scipy import stats
                tau, p_val = stats.kendalltau(scatter_data["Ward_Dep_Rank"], scatter_data[metric])
                sig = "***" if p_val < 0.0005 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                st.caption(f"Kendall's τ = {tau:.3f} (p = {p_val:.4f}) {sig} · Negative τ = higher prescribing in more deprived areas")


    # ════════════════════════════════════════════════════════════════════
    # TAB 3: NI TRENDS (national story only — no practice lines)
    # ════════════════════════════════════════════════════════════════════
    import datetime as _dt
    _covid_start = _dt.datetime(2020, 3, 1)
    _covid_end = _dt.datetime(2021, 6, 1)

    # Load therapeutic area & STAR-PU data (shared between NI Trends and Practice Profile)
    ta_ni = load_ta_ni()
    ta_practice = load_ta_practice()
    starpu_prac = load_starpu_practice()

    # Determine view mode: therapeutic area vs BNF chapter
    _use_ta = (area_name != "All prescribing") and ta_ni is not None
    _ta_name = area_name if area_name != "All prescribing" else None
    if _use_ta and _ta_name and ta_ni is not None:
        _ta_available = _ta_name in ta_ni["therapeutic_area"].unique()
        if not _ta_available:
            _use_ta = False

    if ts_lcg is not None:
        with tab_trends:
            st.header("NI Prescribing Trends")
            st.caption("Monthly data from April 2013 to January 2026 (154 months)")

            if _use_ta and _ta_name and ta_ni is not None:
                # ══════════════════════════════════════════════════════════
                # THERAPEUTIC AREA — NI trend only
                # ══════════════════════════════════════════════════════════
                ta_data = ta_ni[ta_ni["therapeutic_area"] == _ta_name].sort_values("date")

                if starpu_prac is not None:
                    ni_pop = starpu_prac[starpu_prac["bnf_chapter"] == 1].groupby("year")["total_population"].sum().reset_index()
                    ni_pop.columns = ["year", "ni_population"]
                    ta_data = ta_data.merge(ni_pop, on="year", how="left")
                    if ta_data["ni_population"].isna().any():
                        _fill_pop = ni_pop["ni_population"].iloc[0] if len(ni_pop) > 0 else None
                        if _fill_pop:
                            ta_data["ni_population"] = ta_data["ni_population"].fillna(_fill_pop)
                    if metric == "ItemsPerCapita":
                        ta_data["rate"] = ta_data["total_items"] / ta_data["ni_population"] * 1000
                        ta_rate_label = "Items per 1,000 patients"
                    else:
                        ta_data["rate"] = ta_data["total_cost"] / ta_data["ni_population"] * 1000
                        ta_rate_label = "Cost (£) per 1,000 patients"
                else:
                    ta_data["rate"] = ta_data["total_items"] if metric == "ItemsPerCapita" else ta_data["total_cost"]
                    ta_rate_label = "Total items" if metric == "ItemsPerCapita" else "Total cost (£)"

                st.subheader(f"Northern Ireland – {_ta_name}")
                fig1, ax1 = plt.subplots(figsize=(12, 4))
                if smooth_window > 1:
                    ta_data["rate_smooth"] = ta_data["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                    ax1.plot(ta_data["date"], ta_data["rate"], color="#2563eb", linewidth=0.4, alpha=0.3)
                    ax1.plot(ta_data["date"], ta_data["rate_smooth"], color="#2563eb", linewidth=2)
                else:
                    ax1.plot(ta_data["date"], ta_data["rate"], color="#2563eb", linewidth=1.5)
                ax1.set_ylabel(ta_rate_label, fontsize=10)
                ax1.set_xlabel("")
                ax1.set_ylim(bottom=0)
                ax1.grid(axis="y", alpha=0.3)
                ax1.spines["top"].set_visible(False)
                ax1.spines["right"].set_visible(False)
                ax1.axvspan(_covid_start, _covid_end, alpha=0.08, color="red", label="COVID-19")
                ax1.legend(fontsize=8, loc="upper left")
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                st.caption(
                    "Rates are per 1,000 registered patients (raw, not age-sex standardised). "
                    "Drug group definitions match the therapeutic areas in the sidebar."
                )

            else:
                # ══════════════════════════════════════════════════════════
                # BNF CHAPTER — NI trend + LCG comparison
                # ══════════════════════════════════════════════════════════
                if selected_chapter == 0:
                    lcg_data = ts_lcg.groupby(["lcg", "date", "year", "month"]).agg(
                        total_items=("total_items", "sum"),
                        total_cost=("total_cost", "sum"),
                        starpu=("starpu", "sum"),
                    ).reset_index()
                else:
                    lcg_data = ts_lcg[ts_lcg["bnf_chapter"] == selected_chapter].copy()

                ts_metric_val = "items" if metric == "ItemsPerCapita" else "cost"

                if ts_metric_val == "items":
                    lcg_data["rate"] = lcg_data["total_items"] / lcg_data["starpu"]
                    rate_label = "Items per STAR-PU"
                else:
                    lcg_data["rate"] = lcg_data["total_cost"] / lcg_data["starpu"]
                    rate_label = "Cost (£) per STAR-PU"

                ni_data = lcg_data.groupby(["date", "year", "month"]).agg(
                    total_items=("total_items", "sum"),
                    total_cost=("total_cost", "sum"),
                    starpu=("starpu", "sum"),
                ).reset_index()
                if ts_metric_val == "items":
                    ni_data["rate"] = ni_data["total_items"] / ni_data["starpu"]
                else:
                    ni_data["rate"] = ni_data["total_cost"] / ni_data["starpu"]
                ni_data = ni_data.sort_values("date")

                chapter_title = "All prescribing" if selected_chapter == 0 else BNF_CHAPTERS.get(selected_chapter, f"Chapter {selected_chapter}")

                # ── NI-wide trend ──
                st.subheader(f"Northern Ireland – {chapter_title}")
                fig1, ax1 = plt.subplots(figsize=(12, 4))
                if smooth_window > 1:
                    ni_data["rate_smooth"] = ni_data["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                    ax1.plot(ni_data["date"], ni_data["rate"], color="#2563eb", linewidth=0.4, alpha=0.3)
                    ax1.plot(ni_data["date"], ni_data["rate_smooth"], color="#2563eb", linewidth=2)
                else:
                    ax1.plot(ni_data["date"], ni_data["rate"], color="#2563eb", linewidth=1.5)
                ax1.set_ylabel(rate_label, fontsize=10)
                ax1.set_xlabel("")
                ax1.set_ylim(bottom=0)
                ax1.grid(axis="y", alpha=0.3)
                ax1.spines["top"].set_visible(False)
                ax1.spines["right"].set_visible(False)
                ax1.axvspan(_covid_start, _covid_end, alpha=0.08, color="red", label="COVID-19")
                ax1.legend(fontsize=8, loc="upper left")
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                # ── LCG comparison ──
                st.subheader("By Local Commissioning Group")
                lcg_colours = {
                    "Belfast": "#e11d48",
                    "Northern": "#2563eb",
                    "South Eastern": "#16a34a",
                    "Southern": "#d97706",
                    "Western": "#7c3aed",
                }
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                for lcg_name in sorted(lcg_data["lcg"].unique()):
                    lcg_subset = lcg_data[lcg_data["lcg"] == lcg_name].sort_values("date")
                    colour = lcg_colours.get(lcg_name, "#666666")
                    if smooth_window > 1:
                        lcg_subset = lcg_subset.copy()
                        lcg_subset["rate_smooth"] = lcg_subset["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                        ax2.plot(lcg_subset["date"], lcg_subset["rate"], color=colour, linewidth=0.3, alpha=0.15)
                        ax2.plot(lcg_subset["date"], lcg_subset["rate_smooth"],
                                 label=lcg_name, color=colour, linewidth=1.8, alpha=0.9)
                    else:
                        ax2.plot(lcg_subset["date"], lcg_subset["rate"],
                                 label=lcg_name, color=colour, linewidth=1.2, alpha=0.85)

                ax2.set_ylabel(rate_label, fontsize=10)
                ax2.set_xlabel("")
                ax2.set_ylim(bottom=0)
                ax2.grid(axis="y", alpha=0.3)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                ax2.axvspan(_covid_start, _covid_end, alpha=0.08, color="red")
                ax2.legend(fontsize=9, loc="best")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

                st.caption(
                    "★ Chapters marked with ★ have STAR-PU age-sex weightings available. "
                    "STAR-PU standardised rates adjust for the expected prescribing given a population's "
                    "age and sex profile, allowing fairer comparison between areas with different demographics."
                )
    else:
        with tab_trends:
            st.info("Time-series data not available. Trends will appear once data is loaded.")

    # ════════════════════════════════════════════════════════════════════
    # TAB 4: PRACTICE PROFILE (practice vs NI, federation bands, 6-chapter panel)
    # ════════════════════════════════════════════════════════════════════
    with tab_profile:
        st.header("Practice Profile")

        if not highlight_pracnos:
            st.info("Select practices using **Highlight practices** in the sidebar to see their profile here.")
        elif ts_practice is None:
            st.info("Time-series data not available.")
        else:
            # Build info for each highlighted practice
            _profile_colours = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#7c3aed"]
            _profile_infos = []
            for _pi, _pno in enumerate(highlight_pracnos[:5]):
                _pno_int = int(_pno) if str(_pno).isdigit() else _pno
                _pno_str = str(_pno).strip()
                _pr = practices[practices["PracNo"] == _pno_str]
                if _pr.empty:
                    _pr = practices[practices["PracNo"] == str(_pno_int)]
                if _pr.empty:
                    continue
                _fed = _pr["Federation"].values[0].strip() if "Federation" in _pr.columns else None
                _disp = _pr["Address1"].values[0].strip() if "Address1" in _pr.columns else _pno_str
                if _fed:
                    _fp = practices[practices["Federation"].str.strip() == _fed]["PracNo"].tolist()
                    _fp_int = [int(p) if str(p).isdigit() else p for p in _fp]
                else:
                    _fp_int = []
                _profile_infos.append({
                    "pracno_int": _pno_int,
                    "display": _disp,
                    "federation": _fed,
                    "fed_pracnos": _fp_int,
                    "colour": _profile_colours[_pi % len(_profile_colours)],
                })

            if not _profile_infos:
                st.warning("Could not find data for the selected practice(s).")
            else:
                # ── Section 1: Selected area — practice vs NI ──
                if _use_ta and _ta_name and ta_practice is not None and starpu_prac is not None:
                    ta_prac = ta_practice[ta_practice["therapeutic_area"] == _ta_name].copy()
                    ta_chapter = TA_TO_CHAPTER.get(_ta_name)
                    ta_data_ni = ta_ni[ta_ni["therapeutic_area"] == _ta_name].sort_values("date") if ta_ni is not None else None

                    if ta_chapter and not ta_prac.empty:
                        sp_ch = starpu_prac[starpu_prac["bnf_chapter"] == ta_chapter][["year", "practice", "total_population"]].copy()
                        ta_prac = ta_prac.merge(sp_ch, on=["year", "practice"], how="left")

                        # Compute NI rate for overlay
                        if ta_data_ni is not None and starpu_prac is not None:
                            ni_pop = starpu_prac[starpu_prac["bnf_chapter"] == 1].groupby("year")["total_population"].sum().reset_index()
                            ni_pop.columns = ["year", "ni_population"]
                            ta_data_ni = ta_data_ni.merge(ni_pop, on="year", how="left")
                            if ta_data_ni["ni_population"].isna().any():
                                _fill_pop = ni_pop["ni_population"].iloc[0] if len(ni_pop) > 0 else None
                                if _fill_pop:
                                    ta_data_ni["ni_population"] = ta_data_ni["ni_population"].fillna(_fill_pop)
                            if metric == "ItemsPerCapita":
                                ta_data_ni["rate"] = ta_data_ni["total_items"] / ta_data_ni["ni_population"] * 1000
                                _prac_rate_label = "Items per 1,000 patients"
                            else:
                                ta_data_ni["rate"] = ta_data_ni["total_cost"] / ta_data_ni["ni_population"] * 1000
                                _prac_rate_label = "Cost (£) per 1,000 patients"

                        st.subheader(f"{_ta_name} – practice vs NI")
                        fig_ta, ax_ta = plt.subplots(figsize=(12, 5))
                        colours_ta = plt.cm.Set1(np.linspace(0, 1, max(len(highlight_pracnos), 8)))

                        for idx_p, pno in enumerate(highlight_pracnos[:5]):
                            pno_int = int(pno) if str(pno).isdigit() else pno
                            ps = ta_prac[ta_prac["practice"] == pno_int].sort_values("date").copy()
                            if ps.empty or ps["total_population"].isna().all():
                                continue
                            if metric == "ItemsPerCapita":
                                ps["rate"] = ps["total_items"] / ps["total_population"] * 1000
                            else:
                                ps["rate"] = ps["total_cost"] / ps["total_population"] * 1000
                            plabel = pracno_to_label.get(str(pno), str(pno))
                            if len(plabel) > 40:
                                plabel = plabel[:37] + "…"
                            if smooth_window > 1:
                                ps["rate_smooth"] = ps["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                                ax_ta.plot(ps["date"], ps["rate"], color=colours_ta[idx_p], linewidth=0.3, alpha=0.15)
                                ax_ta.plot(ps["date"], ps["rate_smooth"], label=plabel, color=colours_ta[idx_p], linewidth=1.8)
                            else:
                                ax_ta.plot(ps["date"], ps["rate"], label=plabel, color=colours_ta[idx_p], linewidth=1.2)

                        # NI average
                        if ta_data_ni is not None and "rate" in ta_data_ni.columns:
                            if smooth_window > 1:
                                ta_data_ni["rate_smooth"] = ta_data_ni["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                                ax_ta.plot(ta_data_ni["date"], ta_data_ni["rate_smooth"],
                                          color="#999999", linewidth=1.5, linestyle=":", label="NI average")
                            else:
                                ax_ta.plot(ta_data_ni["date"], ta_data_ni["rate"],
                                          color="#999999", linewidth=1.5, linestyle=":", label="NI average")

                        ax_ta.set_ylabel(_prac_rate_label, fontsize=10)
                        ax_ta.set_xlabel("")
                        ax_ta.set_ylim(bottom=0)
                        ax_ta.grid(axis="y", alpha=0.3)
                        ax_ta.spines["top"].set_visible(False)
                        ax_ta.spines["right"].set_visible(False)
                        ax_ta.axvspan(_covid_start, _covid_end, alpha=0.08, color="red")
                        ax_ta.legend(fontsize=7, loc="best")
                        fig_ta.tight_layout()
                        st.pyplot(fig_ta)
                        plt.close(fig_ta)

                elif not _use_ta and ts_practice is not None:
                    # BNF chapter practice overlay
                    if selected_chapter == 0:
                        prac_data = ts_practice.groupby(["practice", "date", "year", "month"]).agg(
                            total_items=("total_items", "sum"),
                            total_cost=("total_cost", "sum"),
                            starpu=("starpu", "sum"),
                            total_population=("total_population", "first"),
                        ).reset_index()
                    else:
                        prac_data = ts_practice[ts_practice["bnf_chapter"] == selected_chapter].copy()

                    ts_metric_val = "items" if metric == "ItemsPerCapita" else "cost"
                    chapter_title = "All prescribing" if selected_chapter == 0 else BNF_CHAPTERS.get(selected_chapter, f"Chapter {selected_chapter}")

                    st.subheader(f"{chapter_title} – practice vs NI")
                    fig_ch, ax_ch = plt.subplots(figsize=(12, 5))
                    colours_ch = plt.cm.Set1(np.linspace(0, 1, max(len(highlight_pracnos), 8)))

                    # NI average
                    ni_agg = prac_data.groupby("date").agg(
                        total_items=("total_items", "sum"),
                        total_cost=("total_cost", "sum"),
                        starpu=("starpu", "sum"),
                    ).reset_index().sort_values("date")
                    if ts_metric_val == "items":
                        ni_agg["rate"] = ni_agg["total_items"] / ni_agg["starpu"]
                        _ch_rate_label = "Items per STAR-PU"
                    else:
                        ni_agg["rate"] = ni_agg["total_cost"] / ni_agg["starpu"]
                        _ch_rate_label = "Cost (£) per STAR-PU"

                    for idx_p, pno in enumerate(highlight_pracnos[:5]):
                        pno_int = int(pno) if str(pno).isdigit() else pno
                        prac_subset = prac_data[prac_data["practice"] == pno_int].sort_values("date").copy()
                        if prac_subset.empty:
                            continue
                        if ts_metric_val == "items":
                            prac_subset["rate"] = prac_subset["total_items"] / prac_subset["starpu"]
                        else:
                            prac_subset["rate"] = prac_subset["total_cost"] / prac_subset["starpu"]
                        plabel = pracno_to_label.get(str(pno), str(pno))
                        if len(plabel) > 40:
                            plabel = plabel[:37] + "…"
                        if smooth_window > 1:
                            prac_subset["rate_smooth"] = prac_subset["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                            ax_ch.plot(prac_subset["date"], prac_subset["rate"], color=colours_ch[idx_p], linewidth=0.3, alpha=0.15)
                            ax_ch.plot(prac_subset["date"], prac_subset["rate_smooth"], label=plabel, color=colours_ch[idx_p], linewidth=1.8)
                        else:
                            ax_ch.plot(prac_subset["date"], prac_subset["rate"], label=plabel, color=colours_ch[idx_p], linewidth=1.2)

                    if smooth_window > 1:
                        ni_agg["rate_smooth"] = ni_agg["rate"].rolling(window=smooth_window, min_periods=smooth_window).mean()
                        ax_ch.plot(ni_agg["date"], ni_agg["rate_smooth"], color="#999999", linewidth=1.5, linestyle=":", label="NI average")
                    else:
                        ax_ch.plot(ni_agg["date"], ni_agg["rate"], color="#999999", linewidth=1.5, linestyle=":", label="NI average")

                    ax_ch.set_ylabel(_ch_rate_label, fontsize=10)
                    ax_ch.set_xlabel("")
                    ax_ch.set_ylim(bottom=0)
                    ax_ch.grid(axis="y", alpha=0.3)
                    ax_ch.spines["top"].set_visible(False)
                    ax_ch.spines["right"].set_visible(False)
                    ax_ch.axvspan(_covid_start, _covid_end, alpha=0.08, color="red")
                    ax_ch.legend(fontsize=7, loc="best")
                    fig_ch.tight_layout()
                    st.pyplot(fig_ch)
                    plt.close(fig_ch)

                # ── Section 2: 6-chapter STAR-PU panel ──
                st.markdown("---")
                st.subheader("Standardised time series across key chapters")
                st.caption(
                    "Practice vs GP Federation peers (10th–90th percentile band) "
                    "and NI average across six major BNF chapters."
                )

                profile_chapters = [4, 2, 1, 6, 3, 13]
                chapter_names_map = {4: "CNS", 2: "Cardiovascular", 1: "GI", 6: "Endocrine", 3: "Respiratory", 13: "Skin"}

                profile_ts_metric = "items" if metric == "ItemsPerCapita" else "cost"
                profile_smooth_w = smooth_window
                rate_col = "items_per_starpu" if profile_ts_metric == "items" else "cost_per_starpu"
                rate_label_profile = "Items per STAR-PU" if profile_ts_metric == "items" else "Cost (£) per STAR-PU"

                def _smooth(series, w):
                    if w > 1:
                        return series.rolling(window=w, min_periods=w).mean()
                    return series

                fig_profile, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
                axes_flat = axes.flatten()

                for idx, ch in enumerate(profile_chapters):
                    ax = axes_flat[idx]
                    ch_label = chapter_names_map.get(ch, BNF_CHAPTERS.get(ch, f"Ch {ch}"))

                    ch_data = ts_practice[ts_practice["bnf_chapter"] == ch].copy()
                    if ch_data.empty:
                        ax.set_title(ch_label, fontsize=11, fontweight="bold")
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        continue

                    if rate_col not in ch_data.columns:
                        if profile_ts_metric == "items":
                            ch_data["items_per_starpu"] = ch_data["total_items"] / ch_data["starpu"]
                        else:
                            ch_data["cost_per_starpu"] = ch_data["total_cost"] / ch_data["starpu"]

                    # NI average
                    ni_ch = ch_data.groupby("date").agg(
                        total=("total_items" if profile_ts_metric == "items" else "total_cost", "sum"),
                        starpu_sum=("starpu", "sum"),
                    ).reset_index()
                    ni_ch["rate"] = ni_ch["total"] / ni_ch["starpu_sum"]
                    ni_ch = ni_ch.sort_values("date")
                    ax.plot(ni_ch["date"], _smooth(ni_ch["rate"], profile_smooth_w),
                            color="#999999", linewidth=1.5, linestyle=":", label="NI average")

                    # Plot each highlighted practice with its federation band
                    _fed_bands_drawn = set()
                    for pinfo in _profile_infos:
                        if pinfo["federation"] and pinfo["federation"] not in _fed_bands_drawn and pinfo["fed_pracnos"]:
                            fed_ch = ch_data[ch_data["practice"].isin(pinfo["fed_pracnos"])]
                            if not fed_ch.empty:
                                fed_agg = fed_ch.groupby("date")[rate_col].agg(
                                    p10=lambda x: x.quantile(0.1),
                                    p90=lambda x: x.quantile(0.9),
                                ).reset_index().sort_values("date")
                                p10_s = _smooth(fed_agg["p10"], profile_smooth_w)
                                p90_s = _smooth(fed_agg["p90"], profile_smooth_w)
                                ax.fill_between(fed_agg["date"], p10_s, p90_s,
                                               alpha=0.1, color=pinfo["colour"],
                                               label=f"{pinfo['federation']} (10th–90th)")
                        _fed_bands_drawn.add(pinfo["federation"])

                        prac_ch = ch_data[ch_data["practice"] == pinfo["pracno_int"]].sort_values("date")
                        if not prac_ch.empty:
                            ax.plot(prac_ch["date"],
                                    _smooth(prac_ch[rate_col], profile_smooth_w),
                                    color=pinfo["colour"], linewidth=2,
                                    label=pinfo["display"][:30])

                    ax.set_title(ch_label, fontsize=11, fontweight="bold")
                    ax.set_ylim(bottom=0)
                    ax.grid(axis="y", alpha=0.3)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.axvspan(_covid_start, _covid_end, alpha=0.06, color="red")
                    if idx == 0:
                        ax.legend(fontsize=7, loc="best")
                    if idx >= 4:
                        ax.tick_params(axis="x", rotation=30, labelsize=8)
                    ax.set_ylabel(rate_label_profile, fontsize=8)

                _title_names = [p["display"][:20] for p in _profile_infos]
                fig_profile.suptitle(
                    " vs ".join(_title_names) + " vs NI",
                    fontsize=12, fontweight="bold", y=1.01
                )
                fig_profile.tight_layout()
                st.pyplot(fig_profile)
                plt.close(fig_profile)

                _fed_list = list(dict.fromkeys(p["federation"] for p in _profile_infos if p["federation"]))
                st.caption(
                    f"Federations: {', '.join(_fed_list)} · "
                    f"Shaded bands = 10th–90th percentile of federation peers · "
                    f"Grey dotted = NI average"
                )

    # ════════════════════════════════════════════════════════════════════
    # TAB 4: QOF (Optional)
    # ════════════════════════════════════════════════════════════════════
    if tab_qof is not None:
        with tab_qof:
            st.header("QOF & Disease Prevalence")
            # Top-level choice: QOF achievement or disease prevalence
            qof_view = st.radio(
                "Compare prescribing with",
                ["QOF achievement", "Disease prevalence"],
                horizontal=True,
                key="tab_qof_view",
            )

            ctrl_col1, ctrl_col2 = st.columns(2)
            with ctrl_col1:
                qof_colour_by = st.radio(
                    "Colour practices by",
                    ["Deprivation quintile", "LCG"],
                    horizontal=True,
                    key="tab_qof_colour_by",
                )
            with ctrl_col2:
                lcg_options = ["All"] + sorted(LCG_COLOURS.keys())
                qof_lcg_filter = st.selectbox(
                    "Filter to LCG",
                    lcg_options,
                    index=0,
                    key="tab_qof_lcg_filter",
                )

            ta_pc = pc  # uses sidebar therapeutic area or individual drug
            qof_chart_label = display_name

            # ────────────────────────────────────────────────────────────
            # VIEW A: QOF Achievement
            # ────────────────────────────────────────────────────────────
            if qof_view == "QOF achievement" and qof_df is not None:
                st.subheader("Prescribing vs QOF Clinical Achievement")
                st.caption(
                    "QOF Clinical Achievement Statistics 2022/23 · "
                    "Each indicator measures a specific clinical action or outcome."
                )

                # Build indicator options
                qof_indicators = qof_df[
                    ~qof_df["QOF_Indicator"].str.contains("Total")
                ].drop_duplicates("QOF_Indicator").sort_values(
                    ["QOF_Domain", "QOF_Indicator"]
                )
                indicator_options = []
                indicator_to_code = {}
                for _, r in qof_indicators.iterrows():
                    label = f"{r['QOF_Indicator']} – {r['QOF_Description']}"
                    indicator_options.append(label)
                    indicator_to_code[label] = r["QOF_Indicator"]

                default_idx = 0
                for i, opt in enumerate(indicator_options):
                    if opt.startswith("DM008"):
                        default_idx = i
                        break

                selected_indicator_label = st.selectbox(
                    "QOF indicator",
                    indicator_options,
                    index=default_idx,
                    key="tab_qof_indicator",
                )
                selected_code = indicator_to_code[selected_indicator_label]

                qof_ind_df = qof_df[qof_df["QOF_Indicator"] == selected_code].copy()
                qof_ind_df = qof_ind_df[qof_ind_df["QOF_Achievement"].notna()]

                # Placeholder for summary metrics (shown after LCG filter)
                _qof_metrics_slot = st.empty()

                # Merge prescribing with QOF
                scatter_df = ta_pc.merge(
                    qof_ind_df[["Practice", "QOF_Achievement", "QOF_Register"]],
                    on="Practice",
                    how="inner",
                )
                scatter_df = scatter_df[scatter_df["QOF_Achievement"].notna()].copy()
                scatter_df["X_val"] = scatter_df["QOF_Achievement"] * 100
                x_label = f"{selected_code} achievement (%)"
                short_desc = selected_indicator_label.split(" – ", 1)[-1]
                x_label_full = f"{x_label}\n{short_desc}"
                chart_title = f"{qof_chart_label} prescribing vs {selected_code} achievement"
                dep_y_label = f"{selected_code} achievement (%)"
                dep_title = f"{selected_code} achievement by deprivation quintile"

            # ────────────────────────────────────────────────────────────
            # VIEW B: Disease Prevalence
            # ────────────────────────────────────────────────────────────
            elif qof_view == "Disease prevalence" and prev_df is not None:
                st.subheader("Prescribing vs Disease Prevalence")
                st.caption(
                    "Disease prevalence estimated from QOF register sizes (2022/23) "
                    "and April 2023 practice list sizes. "
                    "Prevalence = disease register / registered patients."
                )

                # Domain picker
                prev_domains = sorted(prev_df["QOF_Domain"].unique())
                default_domain_idx = 0
                for i, d in enumerate(prev_domains):
                    if d == "Diabetes":
                        default_domain_idx = i
                        break

                selected_domain = st.selectbox(
                    "Disease area",
                    prev_domains,
                    index=default_domain_idx,
                    key="tab_prev_domain",
                )

                domain_prev = prev_df[prev_df["QOF_Domain"] == selected_domain].copy()
                domain_prev["Prevalence_Pct"] = domain_prev["Prevalence"] * 100

                # Placeholder for summary metrics (shown after LCG filter)
                _prev_metrics_slot = st.empty()

                st.caption(
                    f"Based on {selected_domain} QOF register indicator: "
                    f"{domain_prev['RegisterIndicator'].iloc[0]}"
                )

                # Merge prescribing with prevalence
                scatter_df = ta_pc.merge(
                    domain_prev[["Practice", "Prevalence_Pct"]],
                    on="Practice",
                    how="inner",
                )
                scatter_df = scatter_df[scatter_df["Prevalence_Pct"].notna()].copy()
                scatter_df["X_val"] = scatter_df["Prevalence_Pct"]
                x_label = f"{selected_domain} prevalence (%)"
                x_label_full = x_label
                chart_title = f"{qof_chart_label} prescribing vs {selected_domain} prevalence"
                dep_y_label = f"{selected_domain} prevalence (%)"
                dep_title = f"{selected_domain} prevalence by deprivation quintile"

            else:
                scatter_df = pd.DataFrame()
                if qof_view == "QOF achievement" and qof_df is None:
                    st.warning("QOF achievement data not available.")
                elif qof_view == "Disease prevalence" and prev_df is None:
                    st.warning("Disease prevalence data not available.")

            # ── Apply LCG filter ────────────────────────────────────────
            if len(scatter_df) > 0 and qof_lcg_filter != "All":
                scatter_df = scatter_df[scatter_df["LCG"] == qof_lcg_filter].copy()

            # ── Fill in summary metrics (now that LCG filter is applied) ─
            if len(scatter_df) > 0:
                _lcg_note = f" ({qof_lcg_filter})" if qof_lcg_filter != "All" else ""
                if qof_view == "QOF achievement" and qof_df is not None:
                    with _qof_metrics_slot.container():
                        n_p = len(scatter_df)
                        m_ach = scatter_df["X_val"].mean()
                        med_ach = scatter_df["X_val"].median()
                        min_ach = scatter_df["X_val"].min()
                        m_reg = scatter_df["QOF_Register"].mean() if "QOF_Register" in scatter_df.columns else None
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Practices" + _lcg_note, f"{n_p}")
                        c2.metric("Mean achievement", f"{m_ach:.1f}%")
                        c3.metric("Median achievement", f"{med_ach:.1f}%")
                        c4.metric("Min achievement", f"{min_ach:.1f}%")
                        if m_reg and not pd.isna(m_reg):
                            st.caption(f"Mean register size: {m_reg:.0f} patients per practice")
                elif qof_view == "Disease prevalence" and prev_df is not None:
                    with _prev_metrics_slot.container():
                        n_p = len(scatter_df)
                        m_prev = scatter_df["X_val"].mean()
                        med_prev = scatter_df["X_val"].median()
                        mx_prev = scatter_df["X_val"].max()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Practices" + _lcg_note, f"{n_p}")
                        c2.metric("Mean prevalence", f"{m_prev:.1f}%")
                        c3.metric("Median prevalence", f"{med_prev:.1f}%")
                        c4.metric("Max prevalence", f"{mx_prev:.1f}%")

            # ── Common scatter + stats section ─────────────────────────
            if len(scatter_df) > 0 and len(scatter_df) < 5:
                st.warning(
                    f"Too few practices with matching data ({len(scatter_df)})."
                )
            elif len(scatter_df) >= 5:
                fig, ax = plt.subplots(figsize=(10, 6))

                if qof_colour_by == "LCG":
                    for lcg_name in sorted(scatter_df["LCG"].dropna().unique()):
                        lcg_d = scatter_df[scatter_df["LCG"] == lcg_name]
                        ax.scatter(
                            lcg_d["X_val"], lcg_d[metric],
                            color=LCG_COLOURS.get(lcg_name, "#999"),
                            alpha=0.6, s=30, label=lcg_name,
                        )
                else:
                    for q in sorted(scatter_df["DepQuintile"].dropna().unique()):
                        qd = scatter_df[scatter_df["DepQuintile"] == q]
                        ax.scatter(
                            qd["X_val"], qd[metric],
                            color=DEP_COLOURS.get(int(q), "#999"),
                            alpha=0.6, s=30,
                            label=QUINTILE_LABELS.get(int(q), f"Q{int(q)}"),
                        )

                # Highlight practices from sidebar
                for pno in highlight_pracnos[:5]:
                    row = scatter_df[
                        scatter_df["Practice"].str.strip() == str(pno).strip()
                    ]
                    if not row.empty:
                        r = row.iloc[0]
                        display = pracno_to_label.get(str(pno).strip(), pno)
                        if isinstance(display, str) and "(" in display:
                            display = display.split("(")[0].strip()
                        ax.scatter(
                            r["X_val"], r[metric],
                            color="#000000", s=180, marker="*", zorder=10,
                            edgecolors="#e53935", linewidths=1.5,
                        )
                        ax.annotate(
                            display, (r["X_val"], r[metric]),
                            textcoords="offset points", xytext=(6, 8),
                            fontsize=7, color="#e53935", fontweight="bold",
                        )

                # Trend line
                if len(scatter_df) >= 10:
                    from numpy.polynomial.polynomial import polyfit
                    x_vals = scatter_df["X_val"].values
                    y_vals = scatter_df[metric].values
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if mask.sum() >= 10:
                        b, m_coef = polyfit(x_vals[mask], y_vals[mask], 1)
                        x_line = np.linspace(
                            x_vals[mask].min(), x_vals[mask].max(), 100
                        )
                        ax.plot(
                            x_line, b + m_coef * x_line,
                            color="#333", linewidth=1.5, linestyle="--",
                        )

                ax.set_xlabel(x_label_full)
                ax.set_ylabel(label_metric)
                _lcg_suffix = f" ({qof_lcg_filter})" if qof_lcg_filter != "All" else ""
                ax.set_title(chart_title + _lcg_suffix)
                ax.legend(fontsize=8, loc="best")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Kendall's tau correlation
                from scipy.stats import kendalltau
                valid = scatter_df[["X_val", metric]].dropna()
                if len(valid) >= 10:
                    tau, p_val = kendalltau(valid["X_val"], valid[metric])
                    sig = "significant" if p_val < 0.05 else "not significant"
                    st.caption(
                        f"Kendall's \u03c4 = {tau:.3f} (p = {p_val:.4f}) \u2014 "
                        f"{sig} at 5% level \u00b7 {len(valid)} practices"
                    )

                # ── Boxplots: by LCG or deprivation quintile ──────────
                st.divider()
                box_data_df = scatter_df[scatter_df["X_val"].notna()].copy()

                if qof_colour_by == "LCG" and len(box_data_df) > 0:
                    # Boxplot by LCG
                    _box_by = "LCG"
                    _box_title = dep_y_label + " by LCG"
                    st.subheader(_box_title)

                    fig_box, ax_box = plt.subplots(figsize=(8, 5))
                    lcg_names = sorted(box_data_df["LCG"].dropna().unique())
                    b_data = [
                        box_data_df[box_data_df["LCG"] == lcg]["X_val"].values
                        for lcg in lcg_names
                    ]
                    bp = ax_box.boxplot(
                        b_data, labels=lcg_names, patch_artist=True,
                    )
                    for patch, lcg in zip(bp["boxes"], lcg_names):
                        patch.set_facecolor(LCG_COLOURS.get(lcg, "#ccc"))
                        patch.set_alpha(0.7)
                    ax_box.set_ylabel(dep_y_label)
                    ax_box.set_title(_box_title)
                    fig_box.tight_layout()
                    st.pyplot(fig_box)
                    plt.close(fig_box)

                    lcg_summ = box_data_df.groupby("LCG").agg(
                        Practices=("Practice", "nunique"),
                        Mean=("X_val", "mean"),
                        Median=("X_val", "median"),
                        Min=("X_val", "min"),
                        Max=("X_val", "max"),
                    ).reset_index()
                    lcg_summ.columns = [
                        "LCG", "Practices",
                        "Mean (%)", "Median (%)", "Min (%)", "Max (%)",
                    ]
                    st.dataframe(
                        lcg_summ.style.format({
                            "Mean (%)": "{:.1f}",
                            "Median (%)": "{:.1f}",
                            "Min (%)": "{:.1f}",
                            "Max (%)": "{:.1f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                elif qof_colour_by != "LCG" and len(box_data_df) > 0:
                    # Boxplot by deprivation quintile
                    st.subheader(dep_title)

                    fig_box, ax_box = plt.subplots(figsize=(8, 5))
                    quintiles = sorted(box_data_df["DepQuintile"].dropna().unique())
                    b_data = [
                        box_data_df[box_data_df["DepQuintile"] == q]["X_val"].values
                        for q in quintiles
                    ]
                    bp = ax_box.boxplot(
                        b_data,
                        labels=[
                            QUINTILE_LABELS.get(int(q), f"Q{int(q)}")
                            for q in quintiles
                        ],
                        patch_artist=True,
                    )
                    for patch, q in zip(bp["boxes"], quintiles):
                        patch.set_facecolor(DEP_COLOURS.get(int(q), "#ccc"))
                        patch.set_alpha(0.7)
                    ax_box.set_ylabel(dep_y_label)
                    ax_box.set_title(dep_title)
                    fig_box.tight_layout()
                    st.pyplot(fig_box)
                    plt.close(fig_box)

                    dep_summary = box_data_df.groupby("DepQuintile").agg(
                        Practices=("Practice", "nunique"),
                        Mean=("X_val", "mean"),
                        Median=("X_val", "median"),
                        Min=("X_val", "min"),
                        Max=("X_val", "max"),
                    ).reset_index()
                    dep_summary["DepQuintile"] = dep_summary["DepQuintile"].map(
                        lambda q: QUINTILE_LABELS_FLAT.get(int(q), f"Q{int(q)}")
                    )
                    dep_summary.columns = [
                        "Deprivation quintile", "Practices",
                        "Mean (%)", "Median (%)", "Min (%)", "Max (%)",
                    ]
                    st.dataframe(
                        dep_summary.style.format({
                            "Mean (%)": "{:.1f}",
                            "Median (%)": "{:.1f}",
                            "Min (%)": "{:.1f}",
                            "Max (%)": "{:.1f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )


if __name__ == "__main__":
    main()
