import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore', category=Warning)

# Create output directory
output_dir = Path('/sessions/relaxed-intelligent-meitner/mnt/ni-prescribing-explorer/data')
output_dir.mkdir(parents=True, exist_ok=True)

# Get all NI prescribing CSV files
csv_files = sorted(glob.glob('/sessions/relaxed-intelligent-meitner/mnt/prescribing/NI/gp-prescribing-northern-ireland-*.csv'))
print(f"Found {len(csv_files)} CSV files to process\n")

# ============================================================================
# STEP 1: Stream process to identify top 200 drugs
# ============================================================================
print("="*70)
print("STEP 1: Identifying top 200 drugs (scanning all files)")
print("="*70)

drug_totals = {}
file_count = 0

for csv_file in csv_files:
    file_count += 1

    df = pd.read_csv(csv_file, encoding='latin-1', low_memory=False,
                     dtype={'PRACTICE': 'Int64'})

    # Standardize columns
    df.columns = [col.upper() for col in df.columns]

    # Select and rename columns
    df = df[['PRACTICE', 'YEAR', 'MONTH', 'VTM_NM', 'TOTAL ITEMS', 'ACTUAL COST (£)', 'BNF CHAPTER']].copy()
    df.columns = ['practice', 'year', 'month', 'vtm_nm', 'total_items', 'actual_cost', 'bnf_chapter']

    # Convert types
    df['total_items'] = pd.to_numeric(df['total_items'], errors='coerce')
    df['actual_cost'] = pd.to_numeric(df['actual_cost'], errors='coerce')

    # Filter
    df = df[(df['vtm_nm'] != '-') & (df['vtm_nm'].notna())]
    df = df.dropna(subset=['year', 'month', 'total_items', 'actual_cost'])

    # Accumulate drug totals
    for drug, items in df.groupby('vtm_nm')['total_items'].sum().items():
        drug_totals[drug] = drug_totals.get(drug, 0) + items

    if file_count % 20 == 0:
        print(f"Scanned {file_count} files... ({len(drug_totals)} unique drugs so far)")

    del df
    gc.collect()

# Get top 200 drugs
top_200_drugs = sorted(drug_totals.items(), key=lambda x: x[1], reverse=True)[:200]
top_200_drug_names = set([drug for drug, _ in top_200_drugs])
print(f"\nIdentified {len(top_200_drug_names)} top drugs")
print(f"Top drug: {top_200_drugs[0][0]} with {top_200_drugs[0][1]:,.0f} items")
print(f"200th drug: {top_200_drugs[199][0]} with {top_200_drugs[199][1]:,.0f} items\n")

# ============================================================================
# STEP 2: Process files again to create both parquet files
# ============================================================================
print("="*70)
print("STEP 2: Creating parquet files (second pass through data)")
print("="*70)

ni_data_all = []
practice_data_all = []
file_count = 0

for csv_file in csv_files:
    file_count += 1

    df = pd.read_csv(csv_file, encoding='latin-1', low_memory=False)

    # Standardize columns
    df.columns = [col.upper() for col in df.columns]

    # Select and rename columns
    df = df[['PRACTICE', 'YEAR', 'MONTH', 'VTM_NM', 'TOTAL ITEMS', 'ACTUAL COST (£)', 'BNF CHAPTER']].copy()
    df.columns = ['practice', 'year', 'month', 'vtm_nm', 'total_items', 'actual_cost', 'bnf_chapter']

    # Convert types
    df['total_items'] = pd.to_numeric(df['total_items'], errors='coerce')
    df['actual_cost'] = pd.to_numeric(df['actual_cost'], errors='coerce')
    df['bnf_chapter'] = pd.to_numeric(df['bnf_chapter'], errors='coerce')
    df['practice'] = pd.to_numeric(df['practice'], errors='coerce')

    # Filter
    df = df[(df['vtm_nm'] != '-') & (df['vtm_nm'].notna())]
    df = df.dropna(subset=['year', 'month', 'total_items', 'actual_cost', 'practice'])

    # ===== NI-level data =====
    ni_agg = df.groupby(['year', 'month', 'vtm_nm']).agg({
        'total_items': 'sum',
        'actual_cost': 'sum',
        'bnf_chapter': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    ni_data_all.append(ni_agg)

    # ===== Practice-level data (top 200 drugs only) =====
    practice_data = df[df['vtm_nm'].isin(top_200_drug_names)].copy()
    if len(practice_data) > 0:
        practice_agg = practice_data.groupby(['practice', 'year', 'month', 'vtm_nm']).agg({
            'total_items': 'sum',
            'actual_cost': 'sum',
            'bnf_chapter': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        practice_data_all.append(practice_agg)

    if file_count % 20 == 0:
        print(f"Processed {file_count} files...")

    del df, ni_agg
    gc.collect()

print(f"\nProcessed all {file_count} files")

# ============================================================================
# Create FILE 1: NI-level aggregated drug timeseries
# ============================================================================
print("\n" + "="*70)
print("Creating FILE 1: drug_timeseries_ni.parquet")
print("="*70)

ni_timeseries = pd.concat(ni_data_all, ignore_index=True)
print(f"Combined {len(ni_data_all)} DataFrames")

ni_timeseries = ni_timeseries.groupby(['year', 'month', 'vtm_nm']).agg({
    'total_items': 'sum',
    'actual_cost': 'sum',
    'bnf_chapter': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
}).reset_index()

ni_timeseries = ni_timeseries[['year', 'month', 'vtm_nm', 'bnf_chapter', 'total_items', 'actual_cost']]

ni_parquet_path = output_dir / 'drug_timeseries_ni.parquet'
ni_timeseries.to_parquet(ni_parquet_path, index=False, compression='snappy')

ni_file_size = os.path.getsize(ni_parquet_path) / (1024**2)
print(f"\nFILE 1 SAVED: {ni_parquet_path}")
print(f"  Rows: {len(ni_timeseries):,}")
print(f"  Columns: {list(ni_timeseries.columns)}")
print(f"  File size: {ni_file_size:.2f} MB")
print(f"  Year range: {int(ni_timeseries['year'].min())} - {int(ni_timeseries['year'].max())}")
print(f"  Unique drugs: {ni_timeseries['vtm_nm'].nunique():,}")

del ni_timeseries, ni_data_all
gc.collect()

# ============================================================================
# Create FILE 2: Practice-level drug timeseries (top 200 drugs)
# ============================================================================
print("\n" + "="*70)
print("Creating FILE 2: drug_timeseries_practice.parquet")
print("="*70)

practice_timeseries = pd.concat(practice_data_all, ignore_index=True)
print(f"Combined {len(practice_data_all)} DataFrames")

practice_timeseries = practice_timeseries.groupby(['practice', 'year', 'month', 'vtm_nm']).agg({
    'total_items': 'sum',
    'actual_cost': 'sum',
    'bnf_chapter': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
}).reset_index()

practice_timeseries = practice_timeseries[['practice', 'year', 'month', 'vtm_nm', 'bnf_chapter', 'total_items', 'actual_cost']]

practice_parquet_path = output_dir / 'drug_timeseries_practice.parquet'
practice_timeseries.to_parquet(practice_parquet_path, index=False, compression='snappy')

practice_file_size = os.path.getsize(practice_parquet_path) / (1024**2)
print(f"\nFILE 2 SAVED: {practice_parquet_path}")
print(f"  Rows: {len(practice_timeseries):,}")
print(f"  Columns: {list(practice_timeseries.columns)}")
print(f"  File size: {practice_file_size:.2f} MB")
print(f"  Year range: {int(practice_timeseries['year'].min())} - {int(practice_timeseries['year'].max())}")
print(f"  Unique practices: {practice_timeseries['practice'].nunique():,}")
print(f"  Unique drugs (top 200): {practice_timeseries['vtm_nm'].nunique():,}")

print("\n" + "="*70)
print("SUCCESS - Both parquet files created!")
print("="*70)
