#!/usr/bin/env python3
"""
Memory-efficient rebuild of therapeutic area parquet files.
Processes 154 CSV files incrementally using SQLite as intermediate storage.
"""

import os
import glob
import sqlite3
import pandas as pd
import gc
import re
from pathlib import Path

# Configuration
SOURCE_DIR = "/sessions/relaxed-intelligent-meitner/mnt/prescribing/NI/"
OUTPUT_DIR = "/sessions/relaxed-intelligent-meitner/mnt/ni-prescribing-explorer/data/"
DB_FILE = "/tmp/therapeutic_areas.db"

# Therapeutic area definitions (VTM_NM pattern matching)
THERAPEUTIC_AREAS = {
    "Statins": r"statin(?!.*nystatin)",
    "Ezetimibe": r"ezetimibe",
    "UTI antibiotics": r"nitrofurantoin|trimethoprim|pivmecillinam|fosfomycin|cefalexin|ciprofloxacin|co-amoxiclav|amoxicillin",
    "Antidepressants": r"sertraline|citalopram|fluoxetine|mirtazapine|venlafaxine|duloxetine|paroxetine|escitalopram|trazodone",
    "Gabapentinoids": r"gabapentin|pregabalin",
    "Opioids": r"morphine|oxycodone|fentanyl|buprenorphine|tramadol|codeine|dihydrocodeine|tapentadol|pethidine|methadone",
    "PPIs": r"omeprazole|lansoprazole|pantoprazole|rabeprazole|esomeprazole",
    "Diabetes (non-insulin)": r"metformin|gliclazide|glimepiride|pioglitazone|empagliflozin|dapagliflozin|canagliflozin|ertugliflozin|sitagliptin|saxagliptin|linagliptin|alogliptin|vildagliptin|semaglutide|liraglutide|dulaglutide|exenatide",
    "SGLT2 inhibitors": r"empagliflozin|dapagliflozin|canagliflozin|ertugliflozin",
    "GLP-1 agonists": r"semaglutide|liraglutide|dulaglutide|exenatide",
    "DPP-4 inhibitors": r"sitagliptin|saxagliptin|linagliptin|alogliptin|vildagliptin",
    "Anticoagulants": r"warfarin|rivaroxaban|apixaban|edoxaban|dabigatran",
    "Antihypertensives": r"amlodipine|ramipril|lisinopril|losartan|candesartan|valsartan|irbesartan|bisoprolol|atenolol|doxazosin|indapamide|bendroflumethiazide",
    "HRT": r"estradiol|conjugated|tibolone|raloxifene",
}


def classify_drug(vtm_nm):
    """Classify a drug by therapeutic area. Returns list of matching areas."""
    if not isinstance(vtm_nm, str) or vtm_nm.strip() == "-":
        return None

    vtm_lower = vtm_nm.lower()
    matches = []

    for area, pattern in THERAPEUTIC_AREAS.items():
        if re.search(pattern, vtm_lower, re.IGNORECASE):
            matches.append(area)

    return matches if matches else None


def init_database(db_path):
    """Initialize SQLite database with tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Practice-level table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS practice_data (
            practice TEXT,
            year INTEGER,
            month INTEGER,
            therapeutic_area TEXT,
            total_items REAL,
            total_cost REAL,
            total_quantity REAL,
            PRIMARY KEY (practice, year, month, therapeutic_area)
        )
    """)

    # NI-level aggregated table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ni_data (
            year INTEGER,
            month INTEGER,
            therapeutic_area TEXT,
            total_items REAL,
            total_cost REAL,
            total_quantity REAL,
            PRIMARY KEY (year, month, therapeutic_area)
        )
    """)

    conn.commit()
    conn.close()


def process_csv_file(csv_path, db_path):
    """Process a single CSV file and insert into database."""
    try:
        # Read CSV with all columns first (to handle encoding/parsing issues)
        df = pd.read_csv(csv_path, encoding='latin-1')

        # Normalize column names to handle case variations
        df.columns = [col.strip() for col in df.columns]
        col_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=col_map)

        # Find the practice column (could be 'practice' or 'PRACTICE')
        practice_col = next((col for col in df.columns if 'practice' in col.lower()), None)
        year_col = next((col for col in df.columns if col.lower() == 'year'), None)
        month_col = next((col for col in df.columns if col.lower() == 'month'), None)
        vtm_col = next((col for col in df.columns if col.lower() == 'vtm_nm'), None)
        items_col = next((col for col in df.columns if 'total items' in col.lower()), None)
        cost_col = next((col for col in df.columns if 'actual cost' in col.lower()), None)
        qty_col = next((col for col in df.columns if 'total quantity' in col.lower()), None)

        if not all([practice_col, year_col, month_col, vtm_col, items_col, cost_col, qty_col]):
            raise ValueError(f"Missing required columns in {csv_path}")

        # Keep only needed columns
        df = df[[practice_col, year_col, month_col, vtm_col, items_col, cost_col, qty_col]].copy()

        # Rename columns for consistency
        df.columns = ['practice', 'year', 'month', 'vtm_nm', 'total_items', 'total_cost', 'total_quantity']

        # Convert comma-separated numbers to float
        for col in ['total_items', 'total_cost', 'total_quantity']:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)

        # Convert year and month to int, handling NaN
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')

        # Drop rows with missing year or month
        df = df.dropna(subset=['year', 'month'])
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)

        # Classify each drug
        df['therapeutic_area'] = df['vtm_nm'].apply(classify_drug)

        # Explode to handle multiple classifications per drug
        df = df.explode('therapeutic_area')

        # Remove rows with no classification
        df = df[df['therapeutic_area'].notna()]

        # Group by practice, year, month, and therapeutic area
        agg_cols = {
            'total_items': 'sum',
            'total_cost': 'sum',
            'total_quantity': 'sum'
        }

        practice_grouped = df.groupby(
            ['practice', 'year', 'month', 'therapeutic_area'],
            as_index=False
        ).agg(agg_cols)

        # Insert into practice table
        conn = sqlite3.connect(db_path)
        practice_grouped.to_sql(
            'practice_data',
            conn,
            if_exists='append',
            index=False
        )
        conn.close()

        # Aggregate to NI level
        ni_grouped = df.groupby(
            ['year', 'month', 'therapeutic_area'],
            as_index=False
        ).agg(agg_cols)

        # Insert into NI table
        conn = sqlite3.connect(db_path)
        ni_grouped.to_sql(
            'ni_data',
            conn,
            if_exists='append',
            index=False
        )
        conn.close()

        return True

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return False


def main():
    print("Starting therapeutic area rebuild...")

    # Clean up old database if exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed old database: {DB_FILE}")

    # Initialize database
    init_database(DB_FILE)
    print(f"Initialized database: {DB_FILE}")

    # Find and process all CSV files
    csv_files = sorted(glob.glob(f"{SOURCE_DIR}/gp-prescribing-northern-ireland-*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for idx, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"[{idx}/{len(csv_files)}] Processing {filename}...", end=" ", flush=True)

        success = process_csv_file(csv_file, DB_FILE)
        if success:
            print("✓")
        else:
            print("✗")

        # Periodically clean up memory
        if idx % 10 == 0:
            gc.collect()
            print(f"  Memory cleanup after {idx} files")

    print("\nConverting SQLite to Parquet...")

    # Read practice data from database
    conn = sqlite3.connect(DB_FILE)
    practice_df = pd.read_sql_query(
        "SELECT practice, total_items, total_cost, total_quantity, year, month, therapeutic_area FROM practice_data",
        conn
    )
    conn.close()

    # Reorder columns as specified
    practice_df = practice_df[['practice', 'total_items', 'total_cost', 'total_quantity', 'year', 'month', 'therapeutic_area']]

    # Write practice parquet
    practice_output = os.path.join(OUTPUT_DIR, "therapeutic_area_practice_monthly.parquet")
    practice_df.to_parquet(practice_output, index=False, compression='snappy')
    print(f"Wrote {len(practice_df)} rows to {practice_output}")

    del practice_df
    gc.collect()

    # Read NI data from database
    conn = sqlite3.connect(DB_FILE)
    ni_df = pd.read_sql_query(
        "SELECT year, month, therapeutic_area, total_items, total_cost, total_quantity FROM ni_data",
        conn
    )
    conn.close()

    # Reorder columns as specified
    ni_df = ni_df[['year', 'month', 'therapeutic_area', 'total_items', 'total_cost', 'total_quantity']]

    # Write NI parquet
    ni_output = os.path.join(OUTPUT_DIR, "therapeutic_area_ni_monthly.parquet")
    ni_df.to_parquet(ni_output, index=False, compression='snappy')
    print(f"Wrote {len(ni_df)} rows to {ni_output}")

    # Clean up
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    print("\n✓ Rebuild complete!")
    print(f"  Practice file: {practice_output}")
    print(f"  NI file: {ni_output}")


if __name__ == "__main__":
    main()
