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
    "Antidepressants": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "sertraline|citalopram|fluoxetine|mirtazapine|venlafaxine|duloxetine|amitriptyline|paroxetine|escitalopram|trazodone|dosulepin|nortriptyline|clomipramine|imipramine|lofepramine",
            case=False, na=False
        )],
        "description": "SSRIs, SNRIs, TCAs and other antidepressants (NICE CG90)",
    },
    "Gabapentinoids": {
        "filter": lambda df: df[df["VTM_NM"].str.contains("gabapentin|pregabalin", case=False, na=False)],
        "description": "Gabapentin and pregabalin – controlled drugs since April 2019 (NICE NG193)",
    },
    "Opioids": {
        "filter": lambda df: df[df["VTM_NM"].str.contains(
            "morphine|codeine|tramadol|oxycodone|fentanyl|buprenorphine|dihydrocodeine|co-codamol|co-dydramol|tapentadol|methadone|pethidine",
            case=False, na=False
        )],
        "description": "Opioid analgesics (NICE NG193 – chronic primary pain)",
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

# ── QOF: suggested therapeutic area per indicator ──────────────────────
# When a user picks a QOF indicator, suggest the most clinically relevant
# therapeutic area to plot against (user can override).
QOF_SUGGESTED_TA = {
    "DM008": "Diabetes (non-insulin)",
    "DM009": "Diabetes (non-insulin)",
    "DM006": "Diabetes (non-insulin)",
    "DM012": "Diabetes (non-insulin)",
    "DM022NI": "Statins",
    "DM023NI": "Statins",
    "DM024NI": "Antihypertensives",
    "CHD002": "Antihypertensives",
    "CHD003NI": "Statins",
    "CHD005": "Anticoagulants (oral)",
    "HYP003NI": "Antihypertensives",
    "HYP007": "Antihypertensives",
    "AST003": "All prescribing",
    "COPD003": "All prescribing",
    "COPD005NI": "All prescribing",
    "MH002": "Antidepressants",
    "MH003": "Antidepressants",
    "MH007": "Antidepressants",
    "MH0011NI": "Antidepressants",
    "MH0012NI": "Antidepressants",
    "HF003": "All prescribing",
    "HF004": "All prescribing",
    "STIA005NI": "Statins",
    "STIA007": "Anticoagulants (oral)",
    "STIA010NI": "Antihypertensives",
    "STIA011NI": "Antihypertensives",
    "AF007": "Anticoagulants (oral)",
    "AFOO6NI": "Anticoagulants (oral)",
    "CKD006NI": "Antihypertensives",
    "CKD007NI": "Antihypertensives",
}


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

    tab_names = ["Northern Ireland", "Trust / LCG", "Deprivation", "Practice"]
    if qof_df is not None or prev_df is not None:
        tab_names.append("QOF / Prevalence")
    tabs = st.tabs(tab_names)
    tab_ni, tab_area, tab_dep, tab_prac = tabs[0], tabs[1], tabs[2], tabs[3]
    tab_qof = tabs[4] if (qof_df is not None or prev_df is not None) else None

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

                # Summary metrics
                n_practices = len(qof_ind_df)
                mean_ach = qof_ind_df["QOF_Achievement"].mean()
                median_ach = qof_ind_df["QOF_Achievement"].median()
                min_ach = qof_ind_df["QOF_Achievement"].min()
                mean_reg = qof_ind_df["QOF_Register"].mean()
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Practices with data", f"{n_practices}")
                col_s2.metric("Mean achievement", f"{mean_ach:.1%}")
                col_s3.metric("Median achievement", f"{median_ach:.1%}")
                col_s4.metric("Min achievement", f"{min_ach:.1%}")

                if mean_reg and not pd.isna(mean_reg):
                    st.caption(
                        f"Mean register size (denominator): "
                        f"{mean_reg:.0f} patients per practice"
                    )

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

                # Summary metrics
                n_practices = len(domain_prev)
                mean_prev = domain_prev["Prevalence_Pct"].mean()
                median_prev = domain_prev["Prevalence_Pct"].median()
                max_prev = domain_prev["Prevalence_Pct"].max()
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Practices with data", f"{n_practices}")
                col_s2.metric("Mean prevalence", f"{mean_prev:.1f}%")
                col_s3.metric("Median prevalence", f"{median_prev:.1f}%")
                col_s4.metric("Max prevalence", f"{max_prev:.1f}%")

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
