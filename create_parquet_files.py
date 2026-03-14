import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path('/sessions/relaxed-intelligent-meitner/mnt/ni-prescribing-explorer/data')
output_dir.mkdir(parents=True, exist_ok=True)

# Get all NI prescribing CSV files
csv_files = sorted(glob.glob('/sessions/relaxed-intelligent-meitner/mnt/prescribing/NI/gp-prescribing-northern-ireland-*.csv'))
print(f"Found {len(csv_files)} CSV files to process")
print()

# Process files
all_data = []
file_count = 0

for csv_file in csv_files:
    file_count += 1

    # Read CSV with latin-1 encoding
    df = pd.read_csv(csv_file, encoding='latin-1')

    # Standardize column names (handle case differences)
    df.columns = [col.upper() for col in df.columns]

    # Keep only relevant columns
    # Note: BNF Chapter is stored as an integer in the CSV
    df = df[['PRACTICE', 'YEAR', 'MONTH', 'VTM_NM', 'TOTAL ITEMS', 'ACTUAL COST (£)', 'BNF CHAPTER']].copy()

    # Rename columns to lowercase
    df.columns = ['practice', 'year', 'month', 'vtm_nm', 'total_items', 'actual_cost', 'bnf_chapter']

    # Convert columns to proper types
    df['total_items'] = pd.to_numeric(df['total_items'], errors='coerce')
    df['actual_cost'] = pd.to_numeric(df['actual_cost'], errors='coerce')
    df['bnf_chapter'] = pd.to_numeric(df['bnf_chapter'], errors='coerce')

    # Remove rows with missing values in key columns and filter out '-' values
    df = df[(df['vtm_nm'] != '-') & (df['vtm_nm'].notna())]
    df = df.dropna(subset=['year', 'month', 'total_items', 'actual_cost'])

    all_data.append(df)

    if file_count % 20 == 0:
        print(f"Processed {file_count} files...")

print(f"\nProcessed all {file_count} files")
print("Combining data...")

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)
print(f"Total rows in combined data: {len(combined_df):,}")

# ============================================================================
# FILE 1: NI-level aggregated drug timeseries
# ============================================================================
print("\n" + "="*70)
print("Creating FILE 1: drug_timeseries_ni.parquet (NI-level aggregation)")
print("="*70)

# Group by year, month, vtm_nm and aggregate
ni_timeseries = combined_df.groupby(['year', 'month', 'vtm_nm']).agg({
    'total_items': 'sum',
    'actual_cost': 'sum',
    'bnf_chapter': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common BNF Chapter
}).reset_index()

# Rename columns to match specification
ni_timeseries = ni_timeseries[['year', 'month', 'vtm_nm', 'bnf_chapter', 'total_items', 'actual_cost']]

# Save to parquet
ni_parquet_path = output_dir / 'drug_timeseries_ni.parquet'
ni_timeseries.to_parquet(ni_parquet_path, index=False)

ni_file_size = os.path.getsize(ni_parquet_path) / (1024**2)  # Convert to MB
print(f"\nFILE 1 SAVED: {ni_parquet_path}")
print(f"  Rows: {len(ni_timeseries):,}")
print(f"  Columns: {list(ni_timeseries.columns)}")
print(f"  File size: {ni_file_size:.2f} MB")
print(f"  Year range: {int(ni_timeseries['year'].min())} - {int(ni_timeseries['year'].max())}")
print(f"  Unique drugs: {ni_timeseries['vtm_nm'].nunique():,}")

# ============================================================================
# FILE 2: Practice-level drug timeseries (top 200 drugs only)
# ============================================================================
print("\n" + "="*70)
print("Creating FILE 2: drug_timeseries_practice.parquet (Practice-level)")
print("="*70)

# Find top 200 drugs by total items across all time
top_200_drugs = combined_df.groupby('vtm_nm')['total_items'].sum().nlargest(200).index.tolist()
print(f"Top 200 drugs identified: {len(top_200_drugs)} drugs")

# Filter to top 200 drugs only
practice_data_filtered = combined_df[combined_df['vtm_nm'].isin(top_200_drugs)].copy()

# Group by practice, year, month, vtm_nm and aggregate
practice_timeseries = practice_data_filtered.groupby(['practice', 'year', 'month', 'vtm_nm']).agg({
    'total_items': 'sum',
    'actual_cost': 'sum',
    'bnf_chapter': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common BNF Chapter
}).reset_index()

# Rename columns to match specification
practice_timeseries = practice_timeseries[['practice', 'year', 'month', 'vtm_nm', 'bnf_chapter', 'total_items', 'actual_cost']]

# Save to parquet
practice_parquet_path = output_dir / 'drug_timeseries_practice.parquet'
practice_timeseries.to_parquet(practice_parquet_path, index=False)

practice_file_size = os.path.getsize(practice_parquet_path) / (1024**2)  # Convert to MB
print(f"\nFILE 2 SAVED: {practice_parquet_path}")
print(f"  Rows: {len(practice_timeseries):,}")
print(f"  Columns: {list(practice_timeseries.columns)}")
print(f"  File size: {practice_file_size:.2f} MB")
print(f"  Year range: {int(practice_timeseries['year'].min())} - {int(practice_timeseries['year'].max())}")
print(f"  Unique practices: {practice_timeseries['practice'].nunique():,}")
print(f"  Unique drugs: {practice_timeseries['vtm_nm'].nunique():,}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total combined data rows processed: {len(combined_df):,}")
print(f"\nFile 1 (NI-level): {ni_file_size:.2f} MB with {len(ni_timeseries):,} rows")
print(f"File 2 (Practice-level, top 200 drugs): {practice_file_size:.2f} MB with {len(practice_timeseries):,} rows")
print("\nBoth files created successfully!")
