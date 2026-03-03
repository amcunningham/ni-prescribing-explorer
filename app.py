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
    page_title="NI GP Prescribing Explorer",
    page_icon="💊",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
CACHE_FILE = os.path.join(APP_DIR, ".cache", "merged.pkl")
CACHE_PRACTICES = os.path.join(APP_DIR, ".cache", "practices.pkl")
PARQUET_PRESCRIBING = os.path.join(DATA_DIR, "prescribing.parquet")
PARQUET_PRACTICES = os.path.join(DATA_DIR, "practices.parquet")

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
    """Compute per-capita items by practice for a therapeutic area."""
    if area_filter is None:
        df = merged
    else:
        df = area_filter(merged)

    agg = (
        df.groupby(["Practice", "PracticeName", "LCG", "Trust", "RegisteredPatients"])
        .agg(TotalItems=("TotalItems", "sum"), TotalCost=("ActualCost", "sum"))
        .reset_index()
    )
    agg["ItemsPerCapita"] = agg["TotalItems"] / agg["RegisteredPatients"]
    agg["CostPerCapita"] = agg["TotalCost"] / agg["RegisteredPatients"]
    return agg.sort_values("ItemsPerCapita", ascending=False)


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
        for name, colour in highlight_practices:
            row = pc_sorted[pc_sorted["PracticeName"].str.contains(name, case=False, na=False)]
            if not row.empty:
                r = row.iloc[0]
                ax.bar(r["rank"], r[metric], color=colour, width=2.5, zorder=5)
                ax.annotate(
                    name,
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


def practice_detail(pc, practice_name, ni_mean):
    """Metrics for a specific practice vs NI mean."""
    row = pc[pc["PracticeName"].str.contains(practice_name, case=False, na=False)]
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


# ════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════

def main():
    st.title("💊 NI GP Prescribing Explorer")
    st.caption("Data: OpenDataNI GP Prescribing · Open Government Licence")

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

    # Show data period
    months_in_data = ""
    if "Month" in merged.columns:
        months = sorted(merged["Month"].dropna().unique())
        if len(months):
            months_in_data = f" ({', '.join(str(m) for m in months)})"

    # ── sidebar ─────────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    area_name = st.sidebar.selectbox(
        "Therapeutic area",
        list(THERAPEUTIC_AREAS.keys()),
        index=0,
    )
    area = THERAPEUTIC_AREAS[area_name]
    st.sidebar.caption(area["description"])

    metric = st.sidebar.radio(
        "Metric",
        ["ItemsPerCapita", "CostPerCapita"],
        format_func=lambda x: "Items per patient" if x == "ItemsPerCapita" else "Cost per patient (£)",
    )

    # Practice lookup helper – build combined labels for easier identification
    practices["_label"] = (
        practices["PracticeName"].str.strip()
        + "  (" + practices["Postcode"].str.strip() + ", "
        + practices["LCG"].str.strip() + ", #"
        + practices["PracNo"].str.strip() + ")"
    )
    label_to_name = dict(zip(practices["_label"], practices["PracticeName"].str.strip()))
    name_to_label = dict(zip(practices["PracticeName"].str.strip(), practices["_label"]))

    # Find practice by…
    find_by = st.sidebar.radio(
        "Find practices by",
        ["Name", "Postcode area", "LCG / Trust", "Practice number"],
        horizontal=True,
    )

    if find_by == "Name":
        all_labels = sorted(label_to_name.keys())
    elif find_by == "Postcode area":
        bt_areas = sorted(practices["Postcode"].str.extract(r"(BT\d+)", expand=False).dropna().unique())
        selected_bt = st.sidebar.selectbox("Postcode area", bt_areas)
        filtered = practices[practices["Postcode"].str.startswith(selected_bt)]
        all_labels = sorted(filtered["_label"].unique())
    elif find_by == "LCG / Trust":
        selected_lcg = st.sidebar.selectbox("LCG area", sorted(practices["LCG"].dropna().unique()))
        filtered = practices[practices["LCG"] == selected_lcg]
        all_labels = sorted(filtered["_label"].unique())
    else:  # Practice number
        prac_num = st.sidebar.text_input("Practice number", "")
        if prac_num.strip():
            filtered = practices[practices["PracNo"].str.strip() == prac_num.strip()]
            all_labels = sorted(filtered["_label"].unique())
        else:
            all_labels = sorted(label_to_name.keys())

    highlight_labels = st.sidebar.multiselect(
        "Highlight practices",
        all_labels,
        default=[],
        help="Select up to 5 practices to highlight on charts",
    )
    # Convert labels back to names for internal use
    highlight_names = [label_to_name.get(l, l) for l in highlight_labels]

    view_level = st.sidebar.radio("View level", ["NI overview", "By Trust / LCG", "Practice deep-dive"])

    # Data management
    st.sidebar.divider()
    st.sidebar.caption("Data management")
    dm_col1, dm_col2 = st.sidebar.columns(2)
    if dm_col1.button("🔄 Refresh from OpenDataNI"):
        for f in [CACHE_FILE, CACHE_PRACTICES]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_data.clear()
        st.rerun()
    if dm_col2.button("📁 Load different files"):
        for f in [CACHE_FILE, CACHE_PRACTICES]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_data.clear()
        st.rerun()

    # ── compute per-capita ──────────────────────────────────────────────
    pc = per_cap(merged, area["filter"])
    ni_mean = pc[metric].mean()

    # ── NI overview ─────────────────────────────────────────────────────
    if view_level == "NI overview":
        st.header(f"{area_name} – NI overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Practices", f"{len(pc)}")
        col2.metric("Total items", f"{int(pc['TotalItems'].sum()):,}")
        col3.metric("NI mean per capita", f"{ni_mean:.2f}")
        col4.metric("Total cost", f"£{pc['TotalCost'].sum():,.0f}")

        # Caterpillar
        colours = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]
        highlight = [(n, colours[i % len(colours)]) for i, n in enumerate(highlight_names[:5])]
        fig = caterpillar_chart(pc, highlight, title=f"{area_name} – items per registered patient", metric=metric)
        st.pyplot(fig)
        plt.close(fig)

        # Distribution
        st.subheader("Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.hist(pc[metric].dropna(), bins=40, color="#42a5f5", edgecolor="white", alpha=0.85)
        ax2.axvline(ni_mean, color="#333", linewidth=1.2, linestyle="--", label=f"Mean: {ni_mean:.2f}")
        label = "Items per patient" if metric == "ItemsPerCapita" else "Cost (£) per patient"
        ax2.set_xlabel(label)
        ax2.set_ylabel("Number of practices")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Highlighted practice cards
        if highlight_names:
            st.subheader("Highlighted practices")
            cols = st.columns(min(len(highlight_names), 3))
            for i, name in enumerate(highlight_names):
                detail = practice_detail(pc, name, ni_mean)
                if detail:
                    with cols[i % 3]:
                        st.markdown(f"**{detail['Practice']}**")
                        st.markdown(f"LCG: {detail['LCG']}")
                        st.markdown(f"Patients: {detail['Registered patients']}")
                        st.markdown(f"Items/capita: **{detail['Items per capita']}** ({detail['vs NI mean']} vs NI)")
                        st.markdown(f"Rank: {detail['Rank']}")

    # ── Trust / LCG view ────────────────────────────────────────────────
    elif view_level == "By Trust / LCG":
        st.header(f"{area_name} – by Trust / LCG")

        fig = trust_bar_chart(pc, title=f"{area_name} per capita by Trust", metric=metric)
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
                                       ["All"] + sorted(pc["Trust"].dropna().unique().tolist()))
        if selected_trust != "All":
            pc_trust = pc[pc["Trust"] == selected_trust]
        else:
            pc_trust = pc

        colours = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]
        highlight = [(n, colours[i % len(colours)]) for i, n in enumerate(highlight_names[:5])]
        fig2 = caterpillar_chart(
            pc_trust, highlight,
            title=f"{area_name} – {selected_trust if selected_trust != 'All' else 'NI'} practices",
            metric=metric,
        )
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Practice deep-dive ──────────────────────────────────────────────
    elif view_level == "Practice deep-dive":
        st.header("Practice deep-dive")

        selected_label = st.selectbox("Select a practice", all_labels)
        selected_practice = label_to_name.get(selected_label, selected_label) if selected_label else None

        if selected_practice:
            st.subheader(f"{selected_label}")

            prac_matches = practices[practices["PracticeName"].str.strip() == selected_practice]
            if prac_matches.empty:
                # Fallback: try case-insensitive contains
                prac_matches = practices[practices["PracticeName"].str.contains(selected_practice, case=False, na=False)]
            if not prac_matches.empty:
                prac_row = prac_matches.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("LCG", prac_row.get("LCG", "—"))
                c2.metric("Trust", prac_row.get("Trust", "—"))
                c3.metric("Registered patients", f"{int(prac_row.get('RegisteredPatients', 0)):,}")
            else:
                st.warning("Practice details not found in practice list.")

            # Caterpillar chart for the selected therapeutic area, highlighting this practice
            st.subheader(f"{area_name} – where this practice sits")
            highlight_this = [(selected_practice, "#e53935")]
            fig_cat = caterpillar_chart(
                pc, highlight_this,
                title=f"{area_name} – all practices (selected practice in red)",
                metric=metric,
            )
            st.pyplot(fig_cat)
            plt.close(fig_cat)

            st.subheader("Performance across therapeutic areas")

            rows = []
            for ta_name, ta in THERAPEUTIC_AREAS.items():
                ta_pc = per_cap(merged, ta["filter"])
                ta_mean = ta_pc[metric].mean()
                prac = ta_pc[ta_pc["PracticeName"].str.strip() == selected_practice]
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

                # Bar chart: practice vs NI mean
                fig, ax = plt.subplots(figsize=(10, 5))
                x = range(len(results))
                w = 0.35
                label_metric = "Items/patient" if metric == "ItemsPerCapita" else "Cost/patient (£)"
                ax.bar([i - w / 2 for i in x], results[metric], w,
                       label=selected_practice, color="#1e88e5")
                ax.bar([i + w / 2 for i in x], results["NI mean"], w,
                       label="NI mean", color="#bdbdbd")
                ax.set_xticks(list(x))
                ax.set_xticklabels(results["Therapeutic area"], rotation=35, ha="right", fontsize=8)
                ax.set_ylabel(label_metric)
                ax.set_title(f"{selected_practice} vs NI average")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Table
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

            # Ranking in current area
            st.subheader(f"Ranking – {area_name}")
            colours_hl = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]
            highlight = [(selected_practice, "#e53935")]
            highlight += [(n, colours_hl[(i + 1) % len(colours_hl)])
                         for i, n in enumerate(highlight_names[:4]) if n != selected_practice]
            fig3 = caterpillar_chart(pc, highlight,
                                     title=f"{area_name} – {selected_practice} highlighted",
                                     metric=metric)
            st.pyplot(fig3)
            plt.close(fig3)


if __name__ == "__main__":
    main()
