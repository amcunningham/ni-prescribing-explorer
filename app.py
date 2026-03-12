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

    # ── sidebar ─────────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    area_name = st.sidebar.selectbox(
        "Therapeutic area",
        list(THERAPEUTIC_AREAS.keys()),
        index=0,
    )
    area = THERAPEUTIC_AREAS[area_name]
    st.sidebar.caption(area["description"])

    # Individual drug override
    all_drugs = sorted(merged["VTM_NM"].dropna().unique().tolist())
    sidebar_drug = st.sidebar.selectbox(
        "Or pick an individual drug",
        all_drugs,
        index=None,
        placeholder="Start typing to search…",
        key="sidebar_drug",
    )
    if sidebar_drug:
        st.sidebar.caption(f"Showing: **{sidebar_drug}** (overrides therapeutic area)")

    metric = st.sidebar.radio(
        "Metric",
        ["ItemsPerCapita", "CostPerCapita"],
        format_func=lambda x: "Items per patient" if x == "ItemsPerCapita" else "Cost per patient (£)",
    )

    label_metric = "Items per patient" if metric == "ItemsPerCapita" else "Cost (£) per patient"

    # Practice lookup helper — prefer Address1 (surgery name) over partner names
    _prac_display = (
        practices["Address1"].str.strip()
        if "Address1" in practices.columns
        else practices["PracticeName"].str.strip()
    )
    practices["_label"] = (
        _prac_display
        + "  (" + practices["Postcode"].str.strip() + ", "
        + practices["LCG"].str.strip() + ", #"
        + practices["PracNo"].str.strip() + ")"
    )
    label_to_pracno = dict(zip(practices["_label"], practices["PracNo"].str.strip()))
    pracno_to_label = dict(zip(practices["PracNo"].str.strip(), practices["_label"]))

    # Find practice by…
    find_by = st.sidebar.radio(
        "Find practices by",
        ["Name", "Postcode area", "LCG / Trust", "Practice number"],
        horizontal=True,
    )

    if find_by == "Name":
        all_labels = sorted(label_to_pracno.keys())
    elif find_by == "Postcode area":
        bt_areas = sorted(practices["Postcode"].str.extract(r"(BT\d+)", expand=False).dropna().unique())
        selected_bt = st.sidebar.selectbox("Postcode area", bt_areas)
        filtered = practices[practices["Postcode"].str.startswith(selected_bt)]
        all_labels = sorted(filtered["_label"].unique())
    elif find_by == "LCG / Trust":
        selected_lcg = st.sidebar.selectbox("LCG area", sorted(practices["LCG"].dropna().unique()))
        filtered = practices[practices["LCG"] == selected_lcg]
        all_labels = sorted(filtered["_label"].unique())
    else:
        prac_num = st.sidebar.text_input("Practice number", "")
        if prac_num.strip():
            filtered = practices[practices["PracNo"].str.strip() == prac_num.strip()]
            all_labels = sorted(filtered["_label"].unique())
        else:
            all_labels = sorted(label_to_pracno.keys())

    highlight_labels = st.sidebar.multiselect(
        "Highlight practices",
        all_labels,
        default=[],
        help="Select up to 5 practices to highlight on charts",
    )
    highlight_pracnos = [label_to_pracno.get(l, l) for l in highlight_labels]

    # Data management
    st.sidebar.divider()
    if st.sidebar.button("Refresh data from OpenDataNI"):
        for f in [CACHE_FILE, CACHE_PRACTICES]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_data.clear()
        st.rerun()

    with st.sidebar.expander("About this dashboard"):
        st.markdown(
            "Explores variation in GP prescribing across Northern Ireland "
            "practices, linked to deprivation and QOF clinical outcomes.\n\n"
            "**Data sources:** OpenDataNI prescribing data, QOF Clinical "
            "Achievement Statistics 2022/23, April 2023 practice list sizes.\n\n"
            "Drug groupings are based on NICE/BNF guidance with some "
            "deliberate exclusions — see the **About** tab for full details "
            "and rationale."
        )

    # ── compute per-capita ──────────────────────────────────────────────
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
    # TABS
    # ════════════════════════════════════════════════════════════════════
    # Load QOF data
    qof_df = load_qof()
    prev_df = load_prevalence()

    # Load time-series data
    ts_lcg = load_timeseries_lcg()
    ts_practice = load_timeseries_practice()

    tab_names = ["Northern Ireland", "Trust / LCG", "Deprivation", "Practice"]
    if qof_df is not None or prev_df is not None:
        tab_names.append("QOF / Prevalence")
    if ts_lcg is not None:
        tab_names.append("Time Series")
    tab_names.append("About")
    tabs = st.tabs(tab_names)
    tab_ni, tab_area, tab_dep, tab_prac = tabs[0], tabs[1], tabs[2], tabs[3]
    _next_idx = 4
    if qof_df is not None or prev_df is not None:
        tab_qof = tabs[_next_idx]
        _next_idx += 1
    else:
        tab_qof = None
    if ts_lcg is not None:
        tab_ts = tabs[_next_idx]
        _next_idx += 1
    else:
        tab_ts = None
    tab_about = tabs[_next_idx]

    # ── TAB 1: Northern Ireland overview ────────────────────────────────
    with tab_ni:
        st.header(f"{display_name} – Northern Ireland overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Practices", f"{len(pc)}")
        col2.metric("Total items (monthly avg)", f"{int(pc['TotalItems'].sum()):,}")
        col3.metric("NI mean per capita", f"{ni_mean:.2f}")
        col4.metric("Total cost (monthly avg)", f"£{pc['TotalCost'].sum():,.0f}")

        # Caterpillar
        colours = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]
        highlight = [(n, colours[i % len(colours)]) for i, n in enumerate(highlight_pracnos[:5])]
        fig = caterpillar_chart(pc, highlight,
                                title=f"{display_name} – {label_metric.lower()} by practice rank",
                                metric=metric)
        st.pyplot(fig)
        plt.close(fig)

        # Distribution
        st.subheader("Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.hist(pc[metric].dropna(), bins=40, color="#42a5f5", edgecolor="white", alpha=0.85)
        ax2.axvline(ni_mean, color="#333", linewidth=1.2, linestyle="--", label=f"Mean: {ni_mean:.2f}")
        ax2.set_xlabel(label_metric)
        ax2.set_ylabel("Number of practices")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Highlighted practice cards
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

    # ── TAB 2: Trust / LCG ─────────────────────────────────────────────
    with tab_area:
        st.header(f"{display_name} – by Trust / LCG")

        fig = trust_bar_chart(pc, title=f"{display_name} per capita by Trust", metric=metric)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("LCG summary")
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

        # Trust-level caterpillar
        st.subheader("Practice variation within Trusts")
        selected_trust = st.selectbox("Focus on Trust",
                                       ["All Northern Ireland"] + sorted(pc["Trust"].dropna().unique().tolist()),
                                       key="tab_area_trust")
        if selected_trust != "All Northern Ireland":
            pc_trust = pc[pc["Trust"] == selected_trust]
        else:
            pc_trust = pc

        colours = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]
        highlight = [(n, colours[i % len(colours)]) for i, n in enumerate(highlight_pracnos[:5])]
        fig2 = caterpillar_chart(
            pc_trust, highlight,
            title=f"{display_name} – {selected_trust if selected_trust != 'All Northern Ireland' else 'NI'} practices",
            metric=metric,
        )
        st.pyplot(fig2)
        plt.close(fig2)

    # ── TAB 3: Deprivation ──────────────────────────────────────────────
    with tab_dep:
        st.header(f"{display_name} – by deprivation")
        st.caption("Deprivation quintiles based on NIMDM 2017 ward-level scores (1 = most deprived, 5 = least deprived)")

        has_dep = "DepQuintile" in pc.columns and pc["DepQuintile"].notna().any()
        if not has_dep:
            st.warning("Deprivation data not available. Refresh data to include NIMDM linkage.")
        else:
            # ── Controls row ────────────────────────────────────────────
            ctrl1, ctrl2 = st.columns(2)
            with ctrl1:
                lcg_options = ["All Northern Ireland"] + sorted(pc["LCG"].dropna().unique().tolist())
                selected_dep_lcg = st.selectbox("Filter by LCG", lcg_options, key="tab_dep_lcg")
            with ctrl2:
                colour_by = st.radio("Colour practices by",
                                     ["Deprivation quintile", "LCG"],
                                     horizontal=True,
                                     key="tab_dep_colour_by")

            if selected_dep_lcg != "All Northern Ireland":
                pc_dep = pc[pc["LCG"] == selected_dep_lcg]
                area_label = selected_dep_lcg
            else:
                pc_dep = pc
                area_label = "Northern Ireland"

            # ── Quintile bar chart ──────────────────────────────────────
            dep_summary = pc_dep.dropna(subset=["DepQuintile"]).groupby("DepQuintile").agg(
                Practices=("Practice", "nunique"),
                MeanRate=(metric, "mean"),
                MedianRate=(metric, "median"),
                TotalItems=("TotalItems", "sum"),
                TotalCost=("TotalCost", "sum"),
                Patients=("RegisteredPatients", "sum"),
            ).reset_index()
            dep_summary["DepQuintile"] = dep_summary["DepQuintile"].astype(int)
            dep_summary["Label"] = dep_summary["DepQuintile"].map(QUINTILE_LABELS)

            fig_dep, ax_dep = plt.subplots(figsize=(8, 4.5))
            bar_colours = [DEP_COLOURS.get(int(q), "#999") for q in dep_summary["DepQuintile"]]
            ax_dep.bar(dep_summary["Label"], dep_summary["MeanRate"], color=bar_colours, edgecolor="white")
            ni_mean_line = pc[metric].mean()
            ax_dep.axhline(ni_mean_line, color="#333", linewidth=1.2, linestyle="--",
                           label=f"NI mean: {ni_mean_line:.2f}")
            ax_dep.set_ylabel(label_metric)
            ax_dep.set_title(f"{display_name} – mean {label_metric.lower()} by deprivation quintile ({area_label})")
            ax_dep.legend()
            fig_dep.tight_layout()
            st.pyplot(fig_dep)
            plt.close(fig_dep)

            # Ratio
            q1_rows = dep_summary[dep_summary["DepQuintile"] == 1]["MeanRate"]
            q5_rows = dep_summary[dep_summary["DepQuintile"] == 5]["MeanRate"]
            if not q1_rows.empty and not q5_rows.empty:
                q1_mean = q1_rows.values[0]
                q5_mean = q5_rows.values[0]
                if q5_mean > 0:
                    ratio = q1_mean / q5_mean
                    st.metric("Q1:Q5 ratio (most vs least deprived)", f"{ratio:.2f}",
                              help="Ratio > 1 means higher prescribing in deprived areas.")
            elif len(dep_summary) < 5:
                st.info(f"Only {len(dep_summary)} quintile(s) represented in {area_label}.")

            # Quintile summary table
            st.subheader("Quintile summary")
            display_dep = dep_summary[["Label", "Practices", "Patients", "MeanRate", "MedianRate",
                                        "TotalItems", "TotalCost"]].copy()
            display_dep.columns = ["Quintile", "Practices", "Patients",
                                   f"Mean {label_metric}", f"Median {label_metric}",
                                   "Total items", "Total cost"]
            st.dataframe(
                display_dep.style.format({
                    "Patients": "{:,.0f}",
                    f"Mean {label_metric}": "{:.3f}",
                    f"Median {label_metric}": "{:.3f}",
                    "Total items": "{:,.0f}",
                    "Total cost": "£{:,.0f}",
                }),
                use_container_width=True,
            )

            # ── Practice-level scatter ──────────────────────────────────
            st.subheader(f"Practice-level scatter ({area_label})")

            if "Ward_Dep_Rank" in pc_dep.columns:
                scatter_data = pc_dep.dropna(subset=["Ward_Dep_Rank", metric])

                fig_scat, ax_scat = plt.subplots(figsize=(10, 5))
                _scatter_by_colour(ax_scat, scatter_data, metric, colour_by)
                _scatter_trend_line(ax_scat, scatter_data, metric)

                # Highlight selected practices
                for pno in highlight_pracnos[:5]:
                    _scatter_highlight_practice(ax_scat, scatter_data, metric, pno, pracno_to_label)

                ax_scat.set_xlabel("Ward deprivation rank (1 = most deprived)")
                ax_scat.set_ylabel(label_metric)
                ax_scat.set_title(f"{display_name} – prescribing vs deprivation ({area_label})")
                ax_scat.legend(fontsize=8)
                fig_scat.tight_layout()
                st.pyplot(fig_scat)
                plt.close(fig_scat)

                # Correlation stat
                from scipy import stats
                tau, p_val = stats.kendalltau(scatter_data["Ward_Dep_Rank"], scatter_data[metric])
                sig = "***" if p_val < 0.0005 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                st.caption(f"Kendall's τ = {tau:.3f} (p = {p_val:.4f}) {sig} · Negative τ = higher prescribing in more deprived areas")

            # ── Correlation summary (always NI-wide) ────────────────────
            st.divider()
            with st.expander("Deprivation correlations across all therapeutic areas (all NI)", expanded=False):
                st.caption(
                    "Kendall's τ for each therapeutic area. Negative values mean higher "
                    "prescribing in more deprived areas. *** p < 0.0005 (Bonferroni-corrected)."
                )

                from scipy import stats as sp_stats
                corr_rows = []
                for ta_name in THERAPEUTIC_AREAS:
                    ta_pc = per_cap_by_name(merged, ta_name)
                    if "Ward_Dep_Rank" not in ta_pc.columns:
                        continue
                    ta_data = ta_pc.dropna(subset=["Ward_Dep_Rank", metric])
                    if len(ta_data) < 10:
                        continue
                    tau_val, p_val = sp_stats.kendalltau(ta_data["Ward_Dep_Rank"], ta_data[metric])
                    q1 = ta_data[ta_data["DepQuintile"] == 1][metric].mean() if "DepQuintile" in ta_data.columns else None
                    q5 = ta_data[ta_data["DepQuintile"] == 5][metric].mean() if "DepQuintile" in ta_data.columns else None
                    ratio = q1 / q5 if q5 and q5 > 0 else None
                    corr_rows.append({
                        "Therapeutic area": ta_name,
                        "Kendall's τ": tau_val,
                        "p-value": p_val,
                        "Sig": "***" if p_val < 0.0005 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")),
                        "Q1 mean": q1,
                        "Q5 mean": q5,
                        "Q1:Q5 ratio": ratio,
                        "Practices": len(ta_data),
                    })

                if corr_rows:
                    corr_df = pd.DataFrame(corr_rows).sort_values("Kendall's τ")

                    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                    colours_corr = ["#e53935" if p < 0.0005 else "#bdbdbd" for p in corr_df["p-value"]]
                    ax_corr.barh(
                        corr_df["Therapeutic area"],
                        corr_df["Kendall's τ"],
                        color=colours_corr,
                        edgecolor="white",
                        height=0.6,
                    )
                    ax_corr.axvline(0, color="#333", linewidth=0.8)
                    ax_corr.set_xlabel("Kendall's τ (negative = higher prescribing in more deprived areas)")
                    ax_corr.set_title("Prescribing and deprivation: correlations by therapeutic area")
                    fig_corr.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)
                    st.caption("Red bars = statistically significant (p < 0.0005, Bonferroni-corrected)")

                    display_corr = corr_df[["Therapeutic area", "Kendall's τ", "p-value", "Sig",
                                             "Q1 mean", "Q5 mean", "Q1:Q5 ratio", "Practices"]].copy()
                    st.dataframe(
                        display_corr.style.format({
                            "Kendall's τ": "{:.3f}",
                            "p-value": "{:.4f}",
                            "Q1 mean": "{:.3f}",
                            "Q5 mean": "{:.3f}",
                            "Q1:Q5 ratio": "{:.2f}",
                        }).map(
                            lambda v: "color: #e53935; font-weight: bold" if v == "***" else "",
                            subset=["Sig"],
                        ),
                        use_container_width=True,
                    )

    # ── TAB 4: Practice deep-dive ───────────────────────────────────────
    with tab_prac:
        st.header("Practice deep-dive")

        selected_label = st.selectbox("Select a practice", all_labels, key="tab_prac_select")
        selected_pracno = label_to_pracno.get(selected_label, selected_label) if selected_label else None

        if selected_pracno:
            st.subheader(f"{selected_label}")

            prac_matches = practices[practices["PracNo"].str.strip() == str(selected_pracno).strip()]
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

            # Caterpillar chart
            st.subheader(f"{display_name} – where this practice sits")
            highlight_this = [(selected_pracno, "#e53935")]
            fig_cat = caterpillar_chart(
                pc, highlight_this,
                title=f"{display_name} – all NI practices (selected in red)",
                metric=metric,
            )
            st.pyplot(fig_cat)
            plt.close(fig_cat)

            with st.expander("Performance across all therapeutic areas", expanded=True):
                prac_display = selected_label.split("(")[0].strip() if selected_label else selected_pracno

                rows = []
                for ta_name in THERAPEUTIC_AREAS:
                    ta_pc = per_cap_by_name(merged, ta_name)
                    ta_mean = ta_pc[metric].mean()
                    prac = ta_pc[ta_pc["Practice"].str.strip() == str(selected_pracno).strip()]
                    if not prac.empty:
                        val = prac.iloc[0][metric]
                        pct = ((val - ta_mean) / ta_mean) * 100
                        rank = int((ta_pc[metric] <= val).sum())
                        rows.append({
                            "Therapeutic area": ta_name,
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


    # ── TAB 5: QOF Achievement / Disease Prevalence ─────────────────────
    if tab_qof is not None:
        with tab_qof:
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


    # ── TAB: Time Series ─────────────────────────────────────────────────
    if tab_ts is not None:
      with tab_ts:
        st.header("Prescribing trends over time")
        st.caption("Monthly data from April 2013 to January 2026 (154 months)")

        # ── Time Series controls ──
        ts_col1, ts_col2, ts_col3 = st.columns(3)

        # Chapter selector
        available_chapters = sorted(ts_lcg["bnf_chapter"].unique())
        chapter_options = {0: "All prescribing (all chapters combined)"}
        for ch in available_chapters:
            if ch in BNF_CHAPTERS and ch != 0:
                has_starpu = ch in STARPU_CHAPTERS
                label = f"Ch {ch}: {BNF_CHAPTERS[ch]}"
                if has_starpu:
                    label += " *"
                chapter_options[ch] = label

        with ts_col1:
            selected_chapter = st.selectbox(
                "BNF chapter",
                list(chapter_options.keys()),
                format_func=lambda x: chapter_options[x],
                index=0,
                key="ts_chapter",
            )
            if selected_chapter == 0:
                st.caption("Showing total across all BNF chapters")
            elif selected_chapter in STARPU_CHAPTERS:
                st.caption("STAR-PU standardised rate available for this chapter")
            else:
                st.caption("No STAR-PU weighting for this chapter — raw rates only")

        with ts_col2:
            ts_metric = st.radio(
                "Metric",
                ["items", "cost"],
                format_func=lambda x: "Items" if x == "items" else "Cost (£)",
                horizontal=True,
                key="ts_metric",
            )

        with ts_col3:
            rate_type_options = ["Raw (per capita)"]
            if selected_chapter in STARPU_CHAPTERS:
                rate_type_options.append("Standardised (per STAR-PU)")
            ts_rate = st.radio(
                "Rate type",
                rate_type_options,
                horizontal=True,
                key="ts_rate",
            )
            use_starpu = "Standardised" in ts_rate

        # ── Filter and aggregate data ──
        if selected_chapter == 0:
            # Aggregate across all chapters
            lcg_data = ts_lcg.groupby(["lcg", "date", "year", "month"]).agg(
                total_items=("total_items", "sum"),
                total_cost=("total_cost", "sum"),
                starpu=("starpu", "sum"),
            ).reset_index()
            if ts_practice is not None:
                prac_data = ts_practice.groupby(["practice", "date", "year", "month"]).agg(
                    total_items=("total_items", "sum"),
                    total_cost=("total_cost", "sum"),
                    starpu=("starpu", "sum"),
                    total_population=("total_population", "first"),
                ).reset_index()
        else:
            lcg_data = ts_lcg[ts_lcg["bnf_chapter"] == selected_chapter].copy()
            if ts_practice is not None:
                prac_data = ts_practice[ts_practice["bnf_chapter"] == selected_chapter].copy()

        # Compute rate columns for LCG data
        if ts_metric == "items":
            if use_starpu:
                lcg_data["rate"] = lcg_data["total_items"] / lcg_data["starpu"]
                rate_label = "Items per STAR-PU"
            else:
                # For raw per capita at LCG level, we need population — use starpu as proxy
                # or compute from total items / a rough per-capita approach
                # Actually the LCG data doesn't have total_population, so for "all chapters"
                # aggregated we'll use items_per_starpu as the available rate
                if "items_per_starpu" in lcg_data.columns:
                    lcg_data["rate"] = lcg_data["total_items"] / lcg_data["starpu"]
                else:
                    lcg_data["rate"] = lcg_data["total_items"] / lcg_data["starpu"]
                rate_label = "Items per STAR-PU" if use_starpu else "Items per STAR-PU (population-adjusted)"
        else:
            if use_starpu:
                lcg_data["rate"] = lcg_data["total_cost"] / lcg_data["starpu"]
                rate_label = "Cost (£) per STAR-PU"
            else:
                lcg_data["rate"] = lcg_data["total_cost"] / lcg_data["starpu"]
                rate_label = "Cost (£) per STAR-PU"

        # NI-wide aggregate
        ni_data = lcg_data.groupby(["date", "year", "month"]).agg(
            total_items=("total_items", "sum"),
            total_cost=("total_cost", "sum"),
            starpu=("starpu", "sum"),
        ).reset_index()
        if ts_metric == "items":
            ni_data["rate"] = ni_data["total_items"] / ni_data["starpu"]
        else:
            ni_data["rate"] = ni_data["total_cost"] / ni_data["starpu"]
        ni_data = ni_data.sort_values("date")

        chapter_title = "All prescribing" if selected_chapter == 0 else BNF_CHAPTERS.get(selected_chapter, f"Chapter {selected_chapter}")

        # ── Chart 1: NI-wide trend ──
        st.subheader(f"Northern Ireland – {chapter_title}")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(ni_data["date"], ni_data["rate"], color="#2563eb", linewidth=1.5)
        ax1.set_ylabel(rate_label, fontsize=10)
        ax1.set_xlabel("")
        ax1.grid(axis="y", alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        # Shade COVID period
        import datetime
        covid_start = datetime.datetime(2020, 3, 1)
        covid_end = datetime.datetime(2021, 6, 1)
        ax1.axvspan(covid_start, covid_end, alpha=0.08, color="red", label="COVID-19 period")
        ax1.legend(fontsize=8, loc="upper left")
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ── Chart 2: LCG comparison ──
        st.subheader(f"By Local Commissioning Group")
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
            ax2.plot(lcg_subset["date"], lcg_subset["rate"],
                     label=lcg_name, color=colour, linewidth=1.2, alpha=0.85)

        ax2.set_ylabel(rate_label, fontsize=10)
        ax2.set_xlabel("")
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.axvspan(covid_start, covid_end, alpha=0.08, color="red")
        ax2.legend(fontsize=9, loc="best")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Chart 3: Practice-level detail (optional) ──
        if ts_practice is not None:
            st.subheader("Practice-level trends")
            st.caption("Select practices to compare their prescribing trends over time.")

            # Build practice selector using existing practices dataframe
            prac_labels_ts = {}
            if "practices" in dir() or practices is not None:
                for _, row in practices.iterrows():
                    pno = str(row["PracNo"]).strip()
                    pname = row.get("_label", row.get("PracticeName", pno))
                    prac_labels_ts[pname] = int(pno) if pno.isdigit() else pno

            selected_prac_labels = st.multiselect(
                "Select practices to compare",
                sorted(prac_labels_ts.keys()),
                max_selections=8,
                key="ts_practice_select",
            )

            if selected_prac_labels:
                selected_prac_nos = [prac_labels_ts[lbl] for lbl in selected_prac_labels]

                fig3, ax3 = plt.subplots(figsize=(12, 5))
                colours3 = plt.cm.Set1(np.linspace(0, 1, max(len(selected_prac_nos), 8)))

                for idx, pno in enumerate(selected_prac_nos):
                    prac_subset = prac_data[prac_data["practice"] == pno].sort_values("date")
                    if prac_subset.empty:
                        continue

                    # Compute rate
                    if ts_metric == "items":
                        if use_starpu:
                            prac_subset = prac_subset.copy()
                            prac_subset["rate"] = prac_subset["total_items"] / prac_subset["starpu"]
                        else:
                            prac_subset = prac_subset.copy()
                            prac_subset["rate"] = prac_subset["total_items"] / prac_subset["total_population"]
                    else:
                        if use_starpu:
                            prac_subset = prac_subset.copy()
                            prac_subset["rate"] = prac_subset["total_cost"] / prac_subset["starpu"]
                        else:
                            prac_subset = prac_subset.copy()
                            prac_subset["rate"] = prac_subset["total_cost"] / prac_subset["total_population"]

                    label = selected_prac_labels[idx]
                    # Truncate long labels
                    if len(label) > 40:
                        label = label[:37] + "…"
                    ax3.plot(prac_subset["date"], prac_subset["rate"],
                             label=label, color=colours3[idx], linewidth=1.2)

                rate_label_prac = rate_label
                if not use_starpu:
                    rate_label_prac = "Items per patient" if ts_metric == "items" else "Cost (£) per patient"

                ax3.set_ylabel(rate_label_prac, fontsize=10)
                ax3.set_xlabel("")
                ax3.grid(axis="y", alpha=0.3)
                ax3.spines["top"].set_visible(False)
                ax3.spines["right"].set_visible(False)
                ax3.axvspan(covid_start, covid_end, alpha=0.08, color="red")
                ax3.legend(fontsize=7, loc="best")
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

        st.markdown("---")
        st.caption(
            "\\* Chapters marked with * have STAR-PU age-sex weightings available. "
            "STAR-PU standardised rates adjust for the expected prescribing given a population's "
            "age and sex profile, allowing fairer comparison between areas with different demographics."
        )


    # ── TAB: About ──────────────────────────────────────────────────────
    with tab_about:
        st.header("About this dashboard")

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
9: Nutrition, 10: MSK, 13: Skin) and 24 BNF sections/paragraphs
(including PPIs, statins, antihypertensives, antidepressants, analgesics,
antibiotics, diabetes drugs, and NSAIDs).

**Age-band mapping:** The STAR-PU weightings use 9 age bands per sex
(0–4, 5–15, 16–24, 25–44, 45–59, 60–64, 65–74, 75–84, 85+). At LCG
level, demographics are available in 7 bands — five match exactly, and
two combined bands (16–44, 45–64) are resolved using NI population-weighted
averages of the sub-band weights. At practice level, only 4 age bands
are available (<18, 18–44, 45–64, 65+), requiring coarser aggregation.
The LCG-level standardisation is therefore more precise.

### Therapeutic area drug groupings

The drug lists below are based on NICE and BNF guidance. Some deliberate
exclusions are made where a drug has major indications outside the
therapeutic area, which would distort practice-level comparisons.
""")

        # Dynamically generate drug list documentation from THERAPEUTIC_AREAS
        for ta_name, ta_info in THERAPEUTIC_AREAS.items():
            if ta_name == "All prescribing":
                continue
            desc = ta_info["description"]
            filt = ta_info["filter"]
            # Extract the drug names from the regex pattern in the lambda
            if filt is not None:
                import inspect
                src = inspect.getsource(filt)
                # Pull out the regex string between the first pair of quotes
                import re as _re
                match = _re.search(r'"([^"]+)"', src)
                if match:
                    drugs_regex = match.group(1)
                    drug_list = [d.strip() for d in drugs_regex.split("|")]
                    drug_list_str = ", ".join(drug_list)
                else:
                    drug_list_str = "(filter defined programmatically)"
            else:
                drug_list_str = "All items"

            st.markdown(f"**{ta_name}**")
            st.caption(desc)
            st.markdown(f"Drugs included: {drug_list_str}")
            st.markdown("")

        st.markdown("""
### Key exclusions and rationale

- **Antidepressants (excl. TCAs):** Tricyclic antidepressants (amitriptyline,
  nortriptyline, dosulepin, clomipramine, imipramine, lofepramine) are excluded
  because in primary care they are predominantly prescribed for neuropathic pain
  and migraine prophylaxis rather than depression. Including them would
  overcount antidepressant prescribing in practices with high pain caseloads.
  Dosulepin is also no longer recommended for initiation by NICE.

- **Opioids (excl. methadone):** Methadone is excluded because it is
  predominantly prescribed for opioid substitution therapy rather than pain
  management. Practices running substance misuse clinics would otherwise appear
  as very high opioid prescribers, distorting comparisons.

- **Antihypertensives:** Spironolactone is excluded despite being NICE NG136
  step 4 for resistant hypertension, because it has major indications in heart
  failure, primary aldosteronism and other conditions. Beta-blockers are
  excluded as they are no longer first-line for hypertension (NICE NG136) and
  have large overlap with angina, heart failure and rate control in AF.

- **UTI antibiotics:** This category captures antibiotics commonly used for
  UTIs, but several (ciprofloxacin, co-amoxiclav, amoxicillin) have broad
  indications beyond UTIs. The category is best interpreted as
  "antibiotics that include UTI among their common uses" rather than
  UTI-specific prescribing.

### Disease prevalence

Disease prevalence is estimated from QOF register sizes (2022/23) divided by
April 2023 registered patient counts. For each disease domain, the QOF
indicator with the largest denominator is used as the best proxy for the full
disease register. The exception is CKD, where CKD006NI is used rather than
CKD005NI (whose denominator is the total adult population, not CKD patients).

### Deprivation

Practice-level deprivation is based on the NI Multiple Deprivation Measure
2017, mapped at ward level using the practice postcode. Quintile 1 = most
deprived, Quintile 5 = least deprived.

**Important caveats about the deprivation measure:**

- **Ecological assignment:** Deprivation is assigned based on the ward
  containing the practice *postcode* (i.e. the surgery location), not the
  actual deprivation profile of the registered patient population. Patients
  frequently cross ward boundaries, so a practice in a moderately deprived
  ward may serve many patients from more or less deprived surrounding areas.
  This is a well-recognised limitation of ecological deprivation measures.

- **Compressed rank range:** GP practices in NI span ward deprivation ranks
  1–178 out of 462 wards. The most affluent 62% of wards have no GP
  practice postcode assigned to them (because practices tend to be located
  in town centres and health centres rather than residential areas). The
  quintile labels therefore represent divisions within the more deprived
  end of the spectrum, not the full range of NI deprivation.

- **Belfast effect:** Belfast has high practice density, and practices in
  "moderate deprivation" wards often serve mixed catchments including
  nearby deprived areas. This is particularly visible in quintile 4, which
  has a disproportionate share of Belfast practices and shows higher
  prescribing than expected from its nominal deprivation level.

- **Age-sex standardisation steepens the gradient:** When prescribing rates
  are standardised using STAR-PU, the deprivation gradient becomes steeper
  (not flatter). This is because less deprived areas tend to have older
  populations, which inflates their raw per-capita rates. After adjusting
  for expected prescribing given the age-sex mix, the more deprived areas
  stand out even more clearly as prescribing above expectation.

### Practice names

Practices are identified by their surgery or clinic name (Address1 field from
the April 2023 reference file) rather than the senior partner's name, because
partner names change frequently — 98 of 305 practices had different partner
names between the 2023 and 2025 datasets.

### Limitations

- Prescribing data spans April 2013 to January 2026; QOF and prevalence
  data is from 2022/23. Practices that closed between these dates (13
  practices) are excluded from QOF/prevalence analyses.
- Prevalence estimates are approximate — they use QOF indicator denominators
  as proxies for disease registers, which exclude exception-reported patients.
- Deprivation is assigned at ward level based on the practice postcode and
  may not reflect the actual deprivation profile of a practice's registered
  population (see Deprivation section above for details).
- The "individual drug" search matches on VTM (Virtual Therapeutic Moiety)
  name, which groups all formulations and strengths of a drug together.
- STAR-PU standardisation uses a single set of 2023 weightings applied to
  all years. Prescribing patterns by age-sex group may have shifted over
  the 2013–2026 period, but the same weights provide a consistent standard
  for comparison.
- Practice-level demographics use 4 coarse age bands (<18, 18–44, 45–64,
  65+) which require aggregation of the 9 finer STAR-PU age bands.
  LCG-level demographics use 7 bands and give more precise standardisation.
- For April 2013 – March 2014, the 2014 demographic data is used as the
  denominator. For January 2026, the 2025 demographics are carried forward.
""")

        st.markdown("---")
        st.caption(
            "Built by Anne Marie Cunningham with AI assistance (Claude, Anthropic). "
            "Source code on [GitHub](https://github.com/amcunningham/ni-prescribing-explorer)."
        )


if __name__ == "__main__":
    main()
